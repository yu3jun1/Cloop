"""Isolated next-stage experiments on Outcome-v2's patient-disjoint folds.

The formal test split and every earlier output directory are read-only. The
saved fold-local world models are reused for factual H1-H3 forecasts; no
counterfactual treatment effect is estimated.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import yaml
from torch import Tensor
from torch.nn import functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

from ..v1.artifacts import RunArtifacts, read_json, read_torch, upsert_jsonl, write_json, write_torch
from ..v1.config import load_config
from ..v1.data import DataBundle, DynamicsDataset, collate_dynamics, load_bundle, window_refs
from ..v1.engine import run_synthetic_suite
from ..v1.metrics import dynamics_summary, reliability_summary, survival_summary, spearman
from ..v1.outcome import PiecewiseExponentialHead
from ..v1_1.outcome_diagnostics import _load_base, _resolve_device, _seed_all, _sha256_json
from ..v2.outcome_v2 import _world_from_entry
from .trajectory import TrajectorySurvivalHead
from .trajectory_metrics import calibration_bins, integrated_brier, time_dependent_auc
from ..v1.world import EnsembleWorldModel, PersistenceDynamics


VARIANTS = ("persistence", "baseline", "rrt", "ensemble", "rrt_ensemble")
OUTCOMES = ("O0_final", "O1_trajectory", "O2_trajectory_uncertainty")
FEATURES = ("states", "actions", "uncertainty", "timestamps", "state_mask",
            "clinical", "clinical_mask", "history")
DEFAULTS: dict[str, Any] = {
    "base_run": "brainiac_main_v1_provenance",
    "world_run": "outcome_v2_clarity_v1",
    "run": "next_stage_v1",
    "planner_run": "next_stage_toy_v1",
    "seeds": [7, 17, 29],
    "max_horizon": 3,
    "outcome_world": "rrt_ensemble",
    "hidden_dim": 128,
    "nhead": 4,
    "layers": 2,
    "max_epochs": 60,
    "patience": 10,
    "batch_size": 32,
    "lr": 0.001,
    "weight_decay": 0.0001,
    "score_days": 365.0,
}


class NextStageError(RuntimeError):
    pass


def load_experiment_config(path: str | Path) -> dict[str, Any]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    raw = raw.get("next_stage", raw)
    if not isinstance(raw, dict) or set(raw) - set(DEFAULTS):
        raise NextStageError(f"unknown next-stage options: {sorted(set(raw) - set(DEFAULTS))}")
    result = {**copy.deepcopy(DEFAULTS), **raw}
    if (not str(result["run"]).startswith("next_stage_")
            or not str(result["planner_run"]).startswith("next_stage_")
            or len({result["run"], result["planner_run"], result["world_run"], result["base_run"]}) != 4):
        raise NextStageError("use distinct next_stage_ output names and read-only source runs")
    if (not result["seeds"] or len(set(result["seeds"])) != len(result["seeds"])
            or any(not isinstance(seed, int) for seed in result["seeds"])):
        raise NextStageError("seeds must be distinct integers")
    if result["outcome_world"] not in VARIANTS[1:]:
        raise NextStageError("outcome_world must be a trained dynamics variant")
    if (not 1 <= int(result["max_horizon"]) <= 3 or int(result["max_epochs"]) < 1
            or int(result["patience"]) < 1 or int(result["batch_size"]) < 1
            or int(result["hidden_dim"]) % int(result["nhead"])):
        raise NextStageError("invalid model, training, or horizon settings")
    return result


def _source(config: dict[str, Any], spec: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    base, cache, dev_ids, test_ids = _load_base(config, spec)
    source_artifacts = RunArtifacts(config["paths"]["output_root"], spec["world_run"], version="v2")
    source_run = read_json(source_artifacts.path("run.json"))
    if not source_run or source_run.get("schema_version") != "cloop_outcome_v2_clarity_v1":
        raise NextStageError("a completed Outcome-v2 source run is required")
    if source_run.get("base_data_signature") != base.get("data_signature"):
        raise NextStageError("world and base data signatures differ")
    source_models = read_torch(source_artifacts.path("models.pt"), safe=True)
    if source_models.get("schema_version") != "cloop_outcome_v2_models_v1":
        raise NextStageError("unexpected Outcome-v2 model schema")
    folded = source_run.get("folds", [])
    fold_values = [set(row["validation_ids"]) for row in folded]
    eligible = {str(row["patient_id"]) for row in cache["patients"]
                if str(row["patient_id"]) in set(dev_ids)
                and bool(torch.as_tensor(row["labels"]["valid"]).any())}
    if (not folded or set.union(*fold_values) != eligible
            or sum(len(group) for group in fold_values) != len(eligible)):
        raise NextStageError("Outcome-v2 folds do not cover eligible development patients exactly once")
    if any(set(row["training_ids"]) & set(row["validation_ids"]) or
           set(row["training_ids"] + row["validation_ids"]) & set(test_ids) for row in folded):
        raise NextStageError("fold patients overlap or contain formal test patients")
    return base, cache, source_run, source_models


def _fold_bundle(cache: dict[str, Any], config: dict[str, Any], source_models: dict[str, Any], fold: dict[str, Any], test_ids: list[str]) -> DataBundle:
    index = int(fold["fold"])
    preprocessing = source_models["folds"][f"fold{index}"]["preprocessing"]
    split = {"train": fold["training_ids"], "validation": fold["validation_ids"], "test": test_ids}
    return load_bundle(cache, config, split, preprocessing)


def _world(source_models: dict[str, Any], bundle: DataBundle, config: dict[str, Any],
           variant: str, fold: int, seed: int, device: torch.device) -> EnsembleWorldModel:
    if variant == "persistence":
        return EnsembleWorldModel([PersistenceDynamics()]).to(device)
    key = f"{variant}/fold{fold}/seed{seed}"
    entry = source_models["worlds"].get(key)
    if entry is None or not entry.get("complete"):
        raise NextStageError(f"fold-local world model missing: {key}")
    return _world_from_entry(bundle, config, entry, device)


def _refs(bundle: DataBundle, split: str, horizon: int, *, labels: bool) -> list[Any]:
    candidates = window_refs(bundle, split, mode="all_exact", exact_horizon=horizon,
                             max_horizon=horizon)
    if labels:
        candidates = [ref for ref in candidates if bool(bundle.trajectories[ref.patient_id].survival_valid[ref.target])]
    # First eligible target per patient and horizon gives an independent
    # patient-level evaluation set within each H stratum.
    seen: set[str] = set()
    result = []
    for ref in candidates:
        if ref.patient_id not in seen:
            seen.add(ref.patient_id)
            result.append(ref)
    return result


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if isinstance(value, Tensor) else value
            for key, value in batch.items()}


@torch.no_grad()
def _dynamics_records(world: EnsembleWorldModel, bundle: DataBundle, split: str,
                      variant: str, fold: int, seed: int, spec: dict[str, Any],
                      device: torch.device) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    world.eval()
    for horizon in range(1, int(spec["max_horizon"]) + 1):
        refs = window_refs(bundle, split, mode="all_exact", exact_horizon=horizon,
                           max_horizon=horizon)
        loader = DataLoader(DynamicsDataset(bundle, refs), batch_size=int(spec["batch_size"]),
                            collate_fn=collate_dynamics)
        for raw in loader:
            batch = _to_device(raw, device)
            rollout = world.rollout(batch["z0"], batch["actions"], batch["deltas"],
                                    batch["context"], batch["clinical_mask"], batch["history0"])
            mean = rollout.states.mean(0)
            pred = mean[:, horizon]
            error = ((pred - batch["target"]) ** 2).mean(-1)
            similarity = F.cosine_similarity(pred, batch["target"], dim=-1)
            u = rollout.states[:, :, horizon].var(0, unbiased=False).mean(-1)
            for i, pid in enumerate(raw["patient_id"]):
                start = int(raw["target_index"][i]) - horizon
                tr = bundle.trajectories[pid]
                error1 = float(((mean[i, 1] - tr.latents[start + 1].to(device)) ** 2).mean())
                current = float(error[i])
                row = {
                    "record_id": f"next:dyn:{fold}:{seed}:{variant}:{pid}:{start}:H{horizon}",
                    "kind": "factual_dynamics", "split": split, "fold": fold,
                    "seed": seed, "variant": variant, "patient_id": pid,
                    "horizon": horizon, "target_index": int(raw["target_index"][i]),
                    "mse": current,
                    "latent_change": float(((tr.latents[start + horizon] - tr.latents[start]) ** 2).mean()),
                    "cosine_similarity": float(similarity[i]),
                    "drift_from_h1": current - error1,
                    "disagreement": float(u[i]) if world.ensemble_size > 1 else None,
                }
                rows.append(row)
    return rows


def _dynamics_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = dynamics_summary(rows)
    for h in (1, 2, 3):
        group = [row for row in rows if row["horizon"] == h]
        if group:
            result[f"drift@{h}"] = float(np.mean([row["drift_from_h1"] for row in group]))
            uncertain = [row for row in group if row["disagreement"] is not None]
            if uncertain:
                result[f"uncertainty@{h}"] = reliability_summary(
                    [row["mse"] for row in uncertain],
                    [row["disagreement"] for row in uncertain],
                )
                ordered = sorted(uncertain, key=lambda row: row["disagreement"])
                bins = np.array_split(np.arange(len(ordered)), 3)
                result[f"uncertainty_tertiles@{h}"] = [
                    {"n": len(part), "mean_mse": float(np.mean([ordered[int(i)]["mse"] for i in part]))
                     if len(part) else None} for part in bins
                ]
    return result


def _uncertainty_associations(rows: list[dict[str, Any]], bundle: DataBundle,
                              spec: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    tau = float(spec["score_days"])
    max_days = float(config["outcome"]["edges_days"][-1])
    for horizon in range(1, int(spec["max_horizon"]) + 1):
        group = [row for row in rows if row["horizon"] == horizon and row["disagreement"] is not None]
        if not group:
            continue
        rho, reason = spearman([row["disagreement"] for row in group],
                               [row["latent_change"] for row in group])
        eligible = []
        seen: set[str] = set()
        for row in group:
            pid = row["patient_id"]
            tr = bundle.trajectories[pid]
            if pid not in seen and bool(tr.survival_valid[row["target_index"]]):
                eligible.append(row)
                seen.add(pid)
        train_refs = _refs(bundle, "train", horizon, labels=True)
        train_times = []
        train_events = []
        for ref in train_refs:
            tr = bundle.trajectories[ref.patient_id]
            raw_time = float(tr.survival_time[ref.target])
            train_times.append(min(raw_time, max_days))
            train_events.append(int(tr.survival_event[ref.target]) if raw_time <= max_days else 0)
        val_times = []
        val_events = []
        for row in eligible:
            tr = bundle.trajectories[row["patient_id"]]
            raw_time = float(tr.survival_time[row["target_index"]])
            val_times.append(min(raw_time, max_days))
            val_events.append(int(tr.survival_event[row["target_index"]]) if raw_time <= max_days else 0)
        result[f"H{horizon}"] = {
            "latent_change_spearman": rho, "latent_change_reason": reason,
            "survival_association_auc": time_dependent_auc(
                train_times, train_events, val_times, val_events,
                [row["disagreement"] for row in eligible], tau,
            ),
            "patients_with_survival_label": len(eligible),
            "interpretation": "associational only; uncertainty is not a treatment effect",
        }
    return result


@torch.no_grad()
def _outcome_rows(world: EnsembleWorldModel, bundle: DataBundle, split: str,
                  spec: dict[str, Any], device: torch.device) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    width = int(spec["max_horizon"]) + 1
    world.eval()
    for horizon in range(1, width):
        refs = _refs(bundle, split, horizon, labels=True)
        loader = DataLoader(DynamicsDataset(bundle, refs), batch_size=int(spec["batch_size"]),
                            collate_fn=collate_dynamics)
        for raw in loader:
            batch = _to_device(raw, device)
            rollout = world.rollout(batch["z0"], batch["actions"], batch["deltas"],
                                    batch["context"], batch["clinical_mask"], batch["history0"])
            mean = rollout.states.mean(0).cpu()
            variance = rollout.states.var(0, unbiased=False).mean(-1).cpu()
            for i, pid in enumerate(raw["patient_id"]):
                target_index = int(raw["target_index"][i])
                tr = bundle.trajectories[pid]
                states = torch.zeros(width, bundle.latent_dim)
                actions = torch.zeros(width, bundle.action_dim)
                uncertainty = torch.zeros(width)
                timestamps = torch.zeros(width)
                state_mask = torch.zeros(width, dtype=torch.bool)
                states[:horizon + 1] = mean[i]
                actions[1:horizon + 1] = raw["actions"][i]
                uncertainty[:horizon + 1] = variance[i]
                timestamps[1:horizon + 1] = raw["deltas"][i].cumsum(0)
                state_mask[:horizon + 1] = True
                rows.append({
                    "states": states, "actions": actions, "uncertainty": uncertainty,
                    "timestamps": timestamps, "state_mask": state_mask,
                    "clinical": tr.clinical, "clinical_mask": tr.clinical_mask,
                    "history": tr.histories[target_index],
                    "time": tr.survival_time[target_index], "event": tr.survival_event[target_index],
                    "patient_id": pid, "target": tr.timepoint_ids[target_index], "horizon": horizon,
                })
    counts = Counter(row["patient_id"] for row in rows)
    for row in rows:
        row["weight"] = torch.tensor(1.0 / counts[row["patient_id"]])
    return rows


def _batch(rows: list[dict[str, Any]], indices: Sequence[int], device: torch.device) -> dict[str, Any]:
    keys = (*FEATURES, "time", "event", "weight")
    return {key: torch.stack([rows[i][key] for i in indices]).to(device) for key in keys}


def _new_outcome(name: str, bundle: DataBundle, config: dict[str, Any], spec: dict[str, Any],
                 device: torch.device) -> torch.nn.Module:
    if name == "O0_final":
        return PiecewiseExponentialHead.from_config(
            config, bundle.latent_dim, bundle.clinical_dim, bundle.history_dim
        ).to(device)
    return TrajectorySurvivalHead(
        bundle.latent_dim, bundle.action_dim, bundle.clinical_dim, bundle.history_dim,
        hidden_dim=int(spec["hidden_dim"]), nhead=int(spec["nhead"]), layers=int(spec["layers"]),
        edges_days=config["outcome"]["edges_days"],
        use_uncertainty=name == "O2_trajectory_uncertainty",
    ).to(device)


def _rates(model: torch.nn.Module, name: str, batch: dict[str, Tensor]) -> Tensor:
    if name == "O0_final":
        last = batch["state_mask"].long().sum(1) - 1
        final = batch["states"][torch.arange(len(last), device=last.device), last]
        return model.rates(final, batch["clinical"], batch["clinical_mask"], batch["history"])
    return model.rates(*(batch[key] for key in FEATURES))


def _loss(model: torch.nn.Module, name: str, batch: dict[str, Tensor], *, weighted: bool) -> Tensor:
    from .trajectory import nll_from_rates
    return nll_from_rates(_rates(model, name, batch), model.edges, batch["time"], batch["event"],
                          batch["weight"] if weighted else None, "mean" if weighted else "none")


def _inner_patient_split(rows: list[dict[str, Any]], seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    patients = sorted({row["patient_id"] for row in rows},
                      key=lambda pid: hashlib.sha256(f"{seed}:{pid}".encode()).hexdigest())
    if len(patients) < 5:
        raise NextStageError("inner outcome early stopping needs at least five training patients")
    stop_count = max(1, round(len(patients) * 0.2))
    stop_ids = set(patients[:stop_count])
    fit = [row for row in rows if row["patient_id"] not in stop_ids]
    stop = [row for row in rows if row["patient_id"] in stop_ids]
    if not fit or not stop:
        raise NextStageError("inner patient split produced an empty partition")
    return fit, stop


def _train_outcome(name: str, rows: list[dict[str, Any]], validation: list[dict[str, Any]],
                   bundle: DataBundle, config: dict[str, Any], spec: dict[str, Any],
                   seed: int, device: torch.device) -> tuple[dict[str, Tensor], dict[str, Any]]:
    if not rows or not validation:
        raise NextStageError("outcome needs eligible training and validation trajectories")
    _seed_all(seed)
    model = _new_outcome(name, bundle, config, spec, device)
    optimizer = AdamW(model.parameters(), lr=float(spec["lr"]), weight_decay=float(spec["weight_decay"]))
    generator = torch.Generator().manual_seed(seed)
    best = math.inf
    best_epoch = 0
    best_state: dict[str, Tensor] = {}
    stale = 0
    history = []
    for epoch in range(1, int(spec["max_epochs"]) + 1):
        model.train()
        order = torch.randperm(len(rows), generator=generator).tolist()
        for offset in range(0, len(order), int(spec["batch_size"])):
            batch = _batch(rows, order[offset:offset + int(spec["batch_size"])], device)
            optimizer.zero_grad(set_to_none=True)
            loss = _loss(model, name, batch, weighted=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["training"]["grad_clip"]))
            optimizer.step()
        model.eval()
        numerator = denominator = 0.0
        with torch.no_grad():
            for offset in range(0, len(validation), int(spec["batch_size"])):
                batch = _batch(validation, list(range(offset, min(offset + int(spec["batch_size"]), len(validation)))), device)
                losses = _loss(model, name, batch, weighted=False)
                numerator += float((losses * batch["weight"]).sum())
                denominator += float(batch["weight"].sum())
        val_nll = numerator / denominator
        history.append({"epoch": epoch, "val_weighted_nll": val_nll})
        if val_nll < best:
            best, best_epoch, stale = val_nll, epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
        if stale >= int(spec["patience"]):
            break
    return best_state, {"best_epoch": best_epoch, "val_weighted_nll": best, "history": history}


@torch.no_grad()
def _evaluate_outcome(model: torch.nn.Module, name: str, rows: list[dict[str, Any]],
                      train_rows: list[dict[str, Any]], spec: dict[str, Any],
                      device: torch.device, fold: int, seed: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    predictions = []
    max_days = float(model.edges[-1])
    grid = sorted({float(day) for day in model.edges[1:].tolist()
                   if 0 < float(day) <= float(spec["score_days"])} | {float(spec["score_days"])})
    for offset in range(0, len(rows), int(spec["batch_size"])):
        selected = rows[offset:offset + int(spec["batch_size"])]
        batch = _batch(rows, list(range(offset, offset + len(selected))), device)
        rates = _rates(model, name, batch)
        from .trajectory import nll_from_rates, survival_from_rates
        nll = nll_from_rates(rates, model.edges, batch["time"], batch["event"], reduction="none")
        survival = survival_from_rates(rates, model.edges, float(spec["score_days"]))
        grid_survival = {day: survival_from_rates(rates, model.edges, day) for day in grid}
        for i, row in enumerate(selected):
            raw_time = float(row["time"])
            predictions.append({
                "record_id": f"next:out:{fold}:{seed}:{name}:{row['patient_id']}:H{row['horizon']}",
                "kind": "factual_outcome", "fold": fold, "seed": seed, "model": name,
                "patient_id": row["patient_id"], "target": row["target"], "horizon": row["horizon"],
                "time": min(raw_time, max_days),
                "event": int(row["event"]) if raw_time <= max_days else 0,
                "survival_probability": float(survival[i]), "survival_nll": float(nll[i]),
                "survival_by_day": {str(int(day)): float(grid_survival[day][i]) for day in grid},
            })
    result: dict[str, Any] = {}
    for h in range(1, int(spec["max_horizon"]) + 1):
        subset = [row for row in predictions if row["horizon"] == h]
        reference = [row for row in train_rows if row["horizon"] == h]
        if not subset or not reference:
            result[f"H{h}"] = {"available": False, "reason": "no_paired_eligible_landmarks"}
            continue
        train_times = [min(float(row["time"]), max_days) for row in reference]
        train_events = [int(row["event"]) if float(row["time"]) <= max_days else 0 for row in reference]
        summary = survival_summary(
            [row["time"] for row in subset], [row["event"] for row in subset],
            [row["survival_probability"] for row in subset], train_times, train_events,
            float(spec["score_days"]),
        )
        summary["survival_nll"] = float(np.mean([row["survival_nll"] for row in subset]))
        summary["patients"] = len({row["patient_id"] for row in subset})
        summary["integrated_brier"] = integrated_brier(
            train_times, train_events, [row["time"] for row in subset],
            [row["event"] for row in subset],
            {day: [row["survival_by_day"][str(int(day))] for row in subset] for day in grid},
            float(spec["score_days"]),
        )
        summary["time_dependent_auc"] = time_dependent_auc(
            train_times, train_events, [row["time"] for row in subset],
            [row["event"] for row in subset],
            [1.0 - row["survival_probability"] for row in subset],
            float(spec["score_days"]),
        )
        summary["calibration_tertiles"] = calibration_bins(
            [row["time"] for row in subset], [row["event"] for row in subset],
            [row["survival_probability"] for row in subset], float(spec["score_days"]),
        )
        result[f"H{h}"] = summary
    return result, predictions


def _initialize(config: dict[str, Any], spec: dict[str, Any], base: dict[str, Any],
                source: dict[str, Any], artifacts: RunArtifacts) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    source_path = RunArtifacts(config["paths"]["output_root"], spec["world_run"], version="v2").path("models.pt")
    digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
    implementation = hashlib.sha256()
    for filename in ("next_stage.py", "trajectory.py", "trajectory_metrics.py"):
        implementation.update(filename.encode())
        implementation.update((Path(__file__).parent / filename).read_bytes())
    implementation_sha256 = implementation.hexdigest()
    signature = _sha256_json({"config": spec, "base_data": base["data_signature"],
                              "world_signature": source["signature"], "world_models_sha256": digest,
                              "implementation_sha256": implementation_sha256})
    old = read_json(artifacts.path("run.json"))
    if old and old.get("signature") != signature:
        migration = old.get("layout_migration", {})
        if (migration.get("kind") != "versioned_paths_and_imports_only"
                or migration.get("original_signature") != old.get("signature")
                or migration.get("original_implementation_sha256") != old.get("implementation_sha256")
                or migration.get("runtime_signature") != signature
                or migration.get("runtime_implementation_sha256") != implementation_sha256):
            raise NextStageError("existing next-stage run has a different signature; choose a new run name")
    if old:
        return old, read_json(artifacts.path("metrics.json"), {}), read_torch(artifacts.path("models.pt"), safe=True)
    artifacts.root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "cloop_next_stage_v1", "signature": signature,
        "base_run": spec["base_run"], "world_run": spec["world_run"],
        "base_data_signature": base["data_signature"], "world_models_sha256": digest,
        "implementation_sha256": implementation_sha256,
        "formal_test_untouched": True, "evidence_scope": "development-fold observational factual",
        "limitations": ["world predictions on fold-training patients are in-sample",
                        "outer-fold survival comparison is observational and exploratory",
                        "planner scores are non-causal prognostic proxies"],
        "completed_tasks": [],
    }
    metrics: dict[str, Any] = {"dynamics": {}, "outcome": {}, "training": {}}
    models: dict[str, Any] = {"schema_version": "cloop_next_stage_models_v1", "outcomes": {}}
    write_json(artifacts.path("run.json"), manifest)
    write_json(artifacts.path("metrics.json"), metrics)
    write_torch(artifacts.path("models.pt"), models)
    return manifest, metrics, models


def run(config: dict[str, Any], spec: dict[str, Any], stage: str,
        *, selected_fold: int | None = None, selected_seed: int | None = None) -> Path:
    base, cache, source, source_models = _source(config, spec)
    artifacts = RunArtifacts(config["paths"]["output_root"], spec["run"], version="v3")
    manifest, metrics, models = _initialize(config, spec, base, source, artifacts)
    test_ids = list(base["split"]["test"])
    device = _resolve_device(config["project"]["device"])
    if selected_fold is not None and selected_fold not in {int(row["fold"]) for row in source["folds"]}:
        raise NextStageError("selected fold is absent from source run")
    for fold in source["folds"]:
        index = int(fold["fold"])
        if selected_fold is not None and index != selected_fold:
            continue
        bundle = _fold_bundle(cache, config, source_models, fold, test_ids)
        for seed in spec["seeds"]:
            if selected_seed is not None and seed != selected_seed:
                continue
            if stage in {"dynamics", "all"}:
                for variant in VARIANTS:
                    key = f"dynamics/{variant}/fold{index}/seed{seed}"
                    if key in manifest["completed_tasks"]:
                        continue
                    world = _world(source_models, bundle, config, variant, index, seed, device)
                    rows = _dynamics_records(world, bundle, "validation", variant, index, seed, spec, device)
                    metrics["dynamics"][key] = _dynamics_summary(rows)
                    if world.ensemble_size > 1:
                        metrics["dynamics"][key]["uncertainty_associations"] = _uncertainty_associations(rows, bundle, spec, config)
                    upsert_jsonl(artifacts.path("predictions.jsonl"), rows)
                    manifest["completed_tasks"].append(key)
                    write_json(artifacts.path("metrics.json"), metrics)
                    write_json(artifacts.path("run.json"), manifest)
                    print(f"completed {key}", flush=True)
            if stage in {"outcome", "all"}:
                pending = [name for name in OUTCOMES if f"outcome/{name}/fold{index}/seed{seed}" not in manifest["completed_tasks"]]
                if not pending:
                    continue
                world = _world(source_models, bundle, config, spec["outcome_world"], index, seed, device)
                train_rows = _outcome_rows(world, bundle, "train", spec, device)
                fit_rows, stop_rows = _inner_patient_split(train_rows, seed)
                val_rows = _outcome_rows(world, bundle, "validation", spec, device)
                for name in pending:
                    key = f"outcome/{name}/fold{index}/seed{seed}"
                    state, training = _train_outcome(name, fit_rows, stop_rows, bundle, config, spec, seed, device)
                    training["fit_patients"] = len({row["patient_id"] for row in fit_rows})
                    training["early_stop_patients"] = len({row["patient_id"] for row in stop_rows})
                    model = _new_outcome(name, bundle, config, spec, device)
                    model.load_state_dict(state)
                    summary, records = _evaluate_outcome(model, name, val_rows, train_rows, spec,
                                                         device, index, seed)
                    models["outcomes"][key] = {"state": state, "training_patient_hash":
                                                _sha256_json(sorted(bundle.split_ids["train"]))}
                    metrics["training"][key] = training
                    metrics["outcome"][key] = summary
                    write_torch(artifacts.path("models.pt"), models)
                    upsert_jsonl(artifacts.path("predictions.jsonl"), records)
                    manifest["completed_tasks"].append(key)
                    write_json(artifacts.path("metrics.json"), metrics)
                    write_json(artifacts.path("run.json"), manifest)
                    print(f"completed {key}", flush=True)
    artifacts.assert_flat()
    if stage in {"planner", "all"}:
        planner_artifacts = RunArtifacts(config["paths"]["output_root"], spec["planner_run"], version="v3")
        run_synthetic_suite(config, planner_artifacts, spec["planner_run"],
                            [selected_seed] if selected_seed is not None else spec["seeds"])
    return artifacts.root


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Isolated next-stage factual and toy experiments")
    parser.add_argument("--config", default="configs/v1/default.yaml")
    parser.add_argument("--paths", default="configs/v1/server.yaml")
    parser.add_argument("--experiment-config", default="configs/v3/next_stage.yaml")
    parser.add_argument("--stage", choices=("dynamics", "outcome", "planner", "all"), required=True)
    parser.add_argument("--fold", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config, args.paths, device=args.device)
        spec = load_experiment_config(args.experiment_config)
        if args.seed is not None and args.seed not in spec["seeds"]:
            raise NextStageError("selected seed is absent from experiment config")
        path = run(config, spec, args.stage, selected_fold=args.fold, selected_seed=args.seed)
        print(json.dumps({"run": str(path), "stage": args.stage}, ensure_ascii=False))
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        parser.exit(2, f"next-stage: {exc}\n")


if __name__ == "__main__":
    main()
