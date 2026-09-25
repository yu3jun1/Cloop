"""Cloop v4 development-fold experiments.

This runner reuses the frozen v2 fold-local dynamics models and never touches
the formal test split.  It adds velocity-aware outcome ablations, calibrated
ensemble uncertainty, a factual value proxy, and trajectory-value MPC.
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
from ..v1.data import DataBundle, DynamicsDataset, collate_dynamics, patient_state, window_refs
from ..v1.metrics import dynamics_summary, reliability_summary, spearman, survival_summary
from ..v1.outcome import PiecewiseExponentialHead
from ..v1_1.outcome_diagnostics import _resolve_device, _seed_all, _sha256_json
from ..v3.next_stage import (
    _fold_bundle,
    _inner_patient_split,
    _refs,
    _source,
    _to_device,
    _world,
)
from ..v3.trajectory import nll_from_rates, survival_from_rates
from ..v3.trajectory_metrics import calibration_bins, integrated_brier, time_dependent_auc
from .metrics import coverage_risk_curve, regression_calibration
from .planner import TrajectoryValueMPC, historical_candidates
from .trajectory import (
    DualHeadOutcomeModel,
    fixed_horizon_value_targets,
    pairwise_survival_ranking_loss,
)


DYNAMICS_VARIANTS = ("baseline", "ensemble", "rrt", "rrt_ensemble")
OUTCOMES = ("O0_final", "O1_trajectory", "O2_velocity", "O3_velocity_uncertainty")
FEATURES = (
    "states", "actions", "uncertainty", "intervals", "state_mask",
    "clinical", "clinical_mask", "history",
)
BATCH_KEYS = (*FEATURES, "latent_error", "time", "event", "weight")
PLANNING_METHODS = ("fixed_plan", "greedy", "mpc", "mpc_uncertainty")

DEFAULTS: dict[str, Any] = {
    "base_run": "brainiac_main_v1_provenance",
    "world_run": "outcome_v2_clarity_v1",
    "run": "next_experiment_v1",
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
    "ranking_weight": 0.1,
    "ranking_temperature": 0.1,
    "value_weight": 0.25,
    "uncertainty_calibration_weight": 0.1,
    "coverage_levels": [1.0, 0.9, 0.8, 0.6, 0.4],
    "calibration_bins": 5,
    "planner_horizon": 3,
    "planner_beam_width": 4,
    "planner_max_candidates": 6,
    "planner_lambda_uncertainty": 0.05,
    "planner_uncertainty_clip": 5.0,
}


class V4ExperimentError(RuntimeError):
    pass


def load_experiment_config(path: str | Path) -> dict[str, Any]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    raw = raw.get("v4", raw)
    if not isinstance(raw, dict):
        raise V4ExperimentError("v4 experiment config must be a mapping")
    unknown = set(raw) - set(DEFAULTS)
    if unknown:
        raise V4ExperimentError(f"unknown v4 options: {sorted(unknown)}")
    result = {**copy.deepcopy(DEFAULTS), **raw}
    if not str(result["run"]).startswith("next_experiment_"):
        raise V4ExperimentError("v4 output run must start with next_experiment_")
    if result["run"] in {result["base_run"], result["world_run"]}:
        raise V4ExperimentError("v4 output must be distinct from read-only source runs")
    if (not result["seeds"] or len(set(result["seeds"])) != len(result["seeds"])
            or any(not isinstance(seed, int) for seed in result["seeds"])):
        raise V4ExperimentError("seeds must be distinct integers")
    if result["outcome_world"] not in DYNAMICS_VARIANTS:
        raise V4ExperimentError("outcome_world must be a trained dynamics variant")
    if (not 1 <= int(result["max_horizon"]) <= 3
            or int(result["max_epochs"]) < 1
            or int(result["patience"]) < 1
            or int(result["batch_size"]) < 1
            or int(result["hidden_dim"]) % int(result["nhead"])
            or int(result["calibration_bins"]) < 1):
        raise V4ExperimentError("invalid model, training, calibration, or horizon setting")
    if (float(result["score_days"]) <= 0
            or any(float(result[key]) < 0 for key in (
                "ranking_weight", "value_weight", "uncertainty_calibration_weight",
                "planner_lambda_uncertainty",
            ))
            or float(result["ranking_temperature"]) <= 0
            or float(result["planner_uncertainty_clip"]) <= 0):
        raise V4ExperimentError("loss and planning scales are invalid")
    coverage = [float(value) for value in result["coverage_levels"]]
    if not coverage or any(value <= 0 or value > 1 for value in coverage):
        raise V4ExperimentError("coverage_levels must lie in (0,1]")
    result["coverage_levels"] = coverage
    if (int(result["planner_horizon"]) < 1
            or int(result["planner_beam_width"]) < 1
            or int(result["planner_max_candidates"]) < 1):
        raise V4ExperimentError("planner sizes must be positive")
    return result


@torch.no_grad()
def _dynamics_records(
    world: torch.nn.Module,
    bundle: DataBundle,
    split: str,
    variant: str,
    fold: int,
    seed: int,
    spec: dict[str, Any],
    device: torch.device,
) -> list[dict[str, Any]]:
    """Factual rollout records with the design-specified L2 drift."""
    rows: list[dict[str, Any]] = []
    world.eval()
    for horizon in range(1, int(spec["max_horizon"]) + 1):
        refs = window_refs(
            bundle, split, mode="all_exact", exact_horizon=horizon,
            max_horizon=horizon,
        )
        loader = DataLoader(
            DynamicsDataset(bundle, refs),
            batch_size=int(spec["batch_size"]),
            collate_fn=collate_dynamics,
        )
        for raw in loader:
            batch = _to_device(raw, device)
            rollout = world.rollout(
                batch["z0"], batch["actions"], batch["deltas"],
                batch["context"], batch["clinical_mask"], batch["history0"],
            )
            mean = rollout.states.mean(0)
            prediction = mean[:, horizon]
            mse = ((prediction - batch["target"]) ** 2).mean(-1)
            l2 = (prediction - batch["target"]).norm(dim=-1)
            similarity = F.cosine_similarity(prediction, batch["target"], dim=-1)
            disagreement = rollout.states[:, :, horizon].var(0, unbiased=False).mean(-1)
            for i, pid in enumerate(raw["patient_id"]):
                start = int(raw["target_index"][i]) - horizon
                trajectory = bundle.trajectories[pid]
                h1_error = float(
                    (mean[i, 1] - trajectory.latents[start + 1].to(device)).norm()
                )
                rows.append({
                    "record_id": f"v4:dyn:{fold}:{seed}:{variant}:{pid}:{start}:H{horizon}",
                    "kind": "factual_dynamics",
                    "split": split,
                    "fold": fold,
                    "seed": seed,
                    "variant": variant,
                    "patient_id": pid,
                    "horizon": horizon,
                    "target_index": int(raw["target_index"][i]),
                    "mse": float(mse[i]),
                    "l2_error": float(l2[i]),
                    "cosine_similarity": float(similarity[i]),
                    "drift_from_h1": float(l2[i]) - h1_error,
                    "disagreement": (
                        float(disagreement[i]) if world.ensemble_size > 1 else None
                    ),
                })
    return rows


def _dynamics_summary(rows: list[dict[str, Any]], spec: dict[str, Any]) -> dict[str, Any]:
    result = dynamics_summary(rows)
    for horizon in range(1, int(spec["max_horizon"]) + 1):
        group = [row for row in rows if row["horizon"] == horizon]
        if not group:
            continue
        result[f"l2@{horizon}"] = float(np.mean([row["l2_error"] for row in group]))
        result[f"drift@{horizon}"] = float(np.mean([row["drift_from_h1"] for row in group]))
        uncertain = [row for row in group if row["disagreement"] is not None]
        if uncertain:
            errors = [row["mse"] for row in uncertain]
            uncertainty = [row["disagreement"] for row in uncertain]
            result[f"uncertainty@{horizon}"] = reliability_summary(errors, uncertainty)
            result[f"uncertainty_scale@{horizon}"] = float(np.quantile(uncertainty, 0.9))
            result[f"coverage_risk@{horizon}"] = coverage_risk_curve(
                errors, uncertainty, spec["coverage_levels"],
            )
            result[f"raw_uncertainty_calibration@{horizon}"] = regression_calibration(
                uncertainty, errors, bins=int(spec["calibration_bins"]),
            )
    result["drift_definition"] = "L2(error_H) - L2(error_H1) from the same factual start"
    return result


@torch.no_grad()
def _outcome_rows(
    world: torch.nn.Module,
    bundle: DataBundle,
    split: str,
    spec: dict[str, Any],
    device: torch.device,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    width = int(spec["max_horizon"]) + 1
    world.eval()
    for horizon in range(1, width):
        refs = _refs(bundle, split, horizon, labels=True)
        loader = DataLoader(
            DynamicsDataset(bundle, refs),
            batch_size=int(spec["batch_size"]),
            collate_fn=collate_dynamics,
        )
        for raw in loader:
            batch = _to_device(raw, device)
            rollout = world.rollout(
                batch["z0"], batch["actions"], batch["deltas"],
                batch["context"], batch["clinical_mask"], batch["history0"],
            )
            mean = rollout.states.mean(0).cpu()
            variance = rollout.states.var(0, unbiased=False).mean(-1).cpu()
            for i, pid in enumerate(raw["patient_id"]):
                target_index = int(raw["target_index"][i])
                start = target_index - horizon
                trajectory = bundle.trajectories[pid]
                states = torch.zeros(width, bundle.latent_dim)
                actions = torch.zeros(width, bundle.action_dim)
                uncertainty = torch.zeros(width)
                intervals = torch.zeros(width)
                latent_error = torch.zeros(width)
                state_mask = torch.zeros(width, dtype=torch.bool)
                states[:horizon + 1] = mean[i, :horizon + 1]
                actions[1:horizon + 1] = raw["actions"][i, :horizon]
                uncertainty[:horizon + 1] = variance[i, :horizon + 1]
                intervals[1:horizon + 1] = raw["deltas"][i, :horizon]
                observed = trajectory.latents[start:target_index + 1]
                latent_error[:horizon + 1] = (
                    (states[:horizon + 1] - observed) ** 2
                ).mean(-1)
                state_mask[:horizon + 1] = True
                rows.append({
                    "states": states,
                    "actions": actions,
                    "uncertainty": uncertainty,
                    "intervals": intervals,
                    "state_mask": state_mask,
                    "clinical": trajectory.clinical,
                    "clinical_mask": trajectory.clinical_mask,
                    "history": trajectory.histories[target_index],
                    "latent_error": latent_error,
                    "time": trajectory.survival_time[target_index],
                    "event": trajectory.survival_event[target_index],
                    "patient_id": pid,
                    "target": trajectory.timepoint_ids[target_index],
                    "horizon": horizon,
                })
    counts = Counter(row["patient_id"] for row in rows)
    for row in rows:
        row["weight"] = torch.tensor(1.0 / counts[row["patient_id"]])
    return rows


def _batch(
    rows: list[dict[str, Any]], indices: Sequence[int], device: torch.device,
) -> dict[str, Any]:
    return {
        key: torch.stack([torch.as_tensor(rows[index][key]) for index in indices]).to(device)
        for key in BATCH_KEYS
    }


def _new_outcome(
    name: str,
    bundle: DataBundle,
    config: dict[str, Any],
    spec: dict[str, Any],
    device: torch.device,
) -> torch.nn.Module:
    if name not in OUTCOMES:
        raise V4ExperimentError(f"unknown outcome ablation: {name}")
    if name == "O0_final":
        return PiecewiseExponentialHead.from_config(
            config, bundle.latent_dim, bundle.clinical_dim, bundle.history_dim,
        ).to(device)
    return DualHeadOutcomeModel(
        bundle.latent_dim,
        bundle.action_dim,
        bundle.clinical_dim,
        bundle.history_dim,
        hidden_dim=int(spec["hidden_dim"]),
        nhead=int(spec["nhead"]),
        layers=int(spec["layers"]),
        edges_days=config["outcome"]["edges_days"],
        value_horizon_days=float(spec["score_days"]),
        use_velocity=name in {"O2_velocity", "O3_velocity_uncertainty"},
        use_uncertainty=name == "O3_velocity_uncertainty",
    ).to(device)


def _forward_outcome(
    model: torch.nn.Module, name: str, batch: dict[str, Tensor],
) -> tuple[Tensor, Any | None]:
    if name == "O0_final":
        last = batch["state_mask"].long().sum(1) - 1
        final = batch["states"][torch.arange(len(last), device=last.device), last]
        rates = model.rates(
            final, batch["clinical"], batch["clinical_mask"], batch["history"],
        )
        return rates, None
    output = model(*(batch[key] for key in FEATURES))
    return output.rates, output


def _loss_parts(
    model: torch.nn.Module,
    name: str,
    batch: dict[str, Tensor],
    spec: dict[str, Any],
    *,
    weighted: bool,
) -> dict[str, Tensor]:
    weights = batch["weight"] if weighted else None
    if name != "O0_final":
        return model.loss(
            *(batch[key] for key in FEATURES),
            days=batch["time"],
            events=batch["event"],
            latent_error=batch["latent_error"],
            weights=weights,
            ranking_weight=float(spec["ranking_weight"]),
            ranking_temperature=float(spec["ranking_temperature"]),
            value_weight=float(spec["value_weight"]),
            uncertainty_weight=float(spec["uncertainty_calibration_weight"]),
        )
    rates, _ = _forward_outcome(model, name, batch)
    nll = nll_from_rates(
        rates, model.edges, batch["time"], batch["event"], weights, "mean",
    )
    risks = 1.0 - survival_from_rates(rates, model.edges, float(spec["score_days"]))
    ranking = pairwise_survival_ranking_loss(
        risks,
        batch["time"],
        batch["event"],
        temperature=float(spec["ranking_temperature"]),
    )
    zero = nll * 0.0
    return {
        "total": nll + float(spec["ranking_weight"]) * ranking,
        "survival_nll": nll,
        "ranking": ranking,
        "value": zero,
        "uncertainty_calibration": zero,
    }


def _train_outcome(
    name: str,
    rows: list[dict[str, Any]],
    validation: list[dict[str, Any]],
    bundle: DataBundle,
    config: dict[str, Any],
    spec: dict[str, Any],
    seed: int,
    device: torch.device,
) -> tuple[dict[str, Tensor], dict[str, Any]]:
    if not rows or not validation:
        raise V4ExperimentError("outcome needs eligible training and validation trajectories")
    _seed_all(seed)
    model = _new_outcome(name, bundle, config, spec, device)
    optimizer = AdamW(
        model.parameters(), lr=float(spec["lr"]),
        weight_decay=float(spec["weight_decay"]),
    )
    generator = torch.Generator().manual_seed(seed)
    best = math.inf
    best_epoch = 0
    best_state: dict[str, Tensor] = {}
    stale = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, int(spec["max_epochs"]) + 1):
        model.train()
        order = torch.randperm(len(rows), generator=generator).tolist()
        for offset in range(0, len(order), int(spec["batch_size"])):
            indices = order[offset:offset + int(spec["batch_size"])]
            batch = _batch(rows, indices, device)
            optimizer.zero_grad(set_to_none=True)
            losses = _loss_parts(model, name, batch, spec, weighted=True)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(config["training"]["grad_clip"]),
            )
            optimizer.step()
        model.eval()
        totals: list[float] = []
        components: dict[str, list[float]] = {
            key: [] for key in (
                "survival_nll", "ranking", "value", "uncertainty_calibration",
            )
        }
        with torch.no_grad():
            for offset in range(0, len(validation), int(spec["batch_size"])):
                indices = list(range(offset, min(offset + int(spec["batch_size"]), len(validation))))
                batch = _batch(validation, indices, device)
                losses = _loss_parts(model, name, batch, spec, weighted=True)
                totals.append(float(losses["total"]))
                for key in components:
                    components[key].append(float(losses[key]))
        validation_loss = float(np.mean(totals))
        epoch_row = {"epoch": epoch, "validation_total": validation_loss}
        epoch_row.update({f"validation_{key}": float(np.mean(values)) for key, values in components.items()})
        history.append(epoch_row)
        if validation_loss < best:
            best, best_epoch, stale = validation_loss, epoch, 0
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            stale += 1
        if stale >= int(spec["patience"]):
            break
    return best_state, {
        "best_epoch": best_epoch,
        "validation_total": best,
        "history": history,
    }


@torch.no_grad()
def _evaluate_outcome(
    model: torch.nn.Module,
    name: str,
    rows: list[dict[str, Any]],
    train_rows: list[dict[str, Any]],
    spec: dict[str, Any],
    device: torch.device,
    fold: int,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    predictions: list[dict[str, Any]] = []
    max_days = float(model.edges[-1])
    score_days = float(spec["score_days"])
    grid = sorted({
        float(day) for day in model.edges[1:].tolist() if 0 < float(day) <= score_days
    } | {score_days})
    for offset in range(0, len(rows), int(spec["batch_size"])):
        selected = rows[offset:offset + int(spec["batch_size"])]
        batch = _batch(rows, list(range(offset, offset + len(selected))), device)
        rates, output = _forward_outcome(model, name, batch)
        nll = nll_from_rates(
            rates, model.edges, batch["time"], batch["event"], reduction="none",
        )
        survival = survival_from_rates(rates, model.edges, score_days)
        grid_survival = {
            day: survival_from_rates(rates, model.edges, day) for day in grid
        }
        for i, row in enumerate(selected):
            raw_time = float(row["time"])
            valid_steps = int(row["state_mask"].sum())
            record: dict[str, Any] = {
                "record_id": f"v4:out:{fold}:{seed}:{name}:{row['patient_id']}:H{row['horizon']}",
                "kind": "factual_outcome",
                "fold": fold,
                "seed": seed,
                "model": name,
                "patient_id": row["patient_id"],
                "target": row["target"],
                "horizon": row["horizon"],
                "time": min(raw_time, max_days),
                "event": int(row["event"]) if raw_time <= max_days else 0,
                "survival_probability": float(survival[i]),
                "survival_nll": float(nll[i]),
                "survival_by_day": {
                    str(int(day)): float(grid_survival[day][i]) for day in grid
                },
                "uncertainty_by_step": row["uncertainty"][:valid_steps].tolist(),
                "latent_error_by_step": row["latent_error"][:valid_steps].tolist(),
            }
            if output is not None:
                record["value_risk"] = float(output.value[i])
                record["attention_weights"] = output.attention_weights[i, :valid_steps].cpu().tolist()
                if output.calibrated_error is not None:
                    record["calibrated_error_by_step"] = (
                        output.calibrated_error[i, :valid_steps].cpu().tolist()
                    )
                    record["uncertainty_stats"] = output.uncertainty_stats[i].cpu().tolist()
            predictions.append(record)

    result: dict[str, Any] = {}
    for horizon in range(1, int(spec["max_horizon"]) + 1):
        subset = [row for row in predictions if row["horizon"] == horizon]
        reference = [row for row in train_rows if row["horizon"] == horizon]
        if not subset or not reference:
            result[f"H{horizon}"] = {
                "available": False, "reason": "no_paired_eligible_landmarks",
            }
            continue
        train_times = [min(float(row["time"]), max_days) for row in reference]
        train_events = [
            int(row["event"]) if float(row["time"]) <= max_days else 0
            for row in reference
        ]
        summary = survival_summary(
            [row["time"] for row in subset],
            [row["event"] for row in subset],
            [row["survival_probability"] for row in subset],
            train_times,
            train_events,
            score_days,
        )
        summary["survival_nll"] = float(np.mean([row["survival_nll"] for row in subset]))
        summary["patients"] = len({row["patient_id"] for row in subset})
        summary["integrated_brier"] = integrated_brier(
            train_times,
            train_events,
            [row["time"] for row in subset],
            [row["event"] for row in subset],
            {
                day: [row["survival_by_day"][str(int(day))] for row in subset]
                for day in grid
            },
            score_days,
        )
        summary["time_dependent_auc"] = time_dependent_auc(
            train_times,
            train_events,
            [row["time"] for row in subset],
            [row["event"] for row in subset],
            [1.0 - row["survival_probability"] for row in subset],
            score_days,
        )
        summary["calibration_tertiles"] = calibration_bins(
            [row["time"] for row in subset],
            [row["event"] for row in subset],
            [row["survival_probability"] for row in subset],
            score_days,
        )
        if name != "O0_final":
            label_days = torch.tensor([row["time"] for row in subset])
            label_events = torch.tensor([row["event"] for row in subset])
            targets, valid = fixed_horizon_value_targets(label_days, label_events, score_days)
            value = torch.tensor([row["value_risk"] for row in subset])
            summary["value_proxy"] = {
                "identifiable": int(valid.sum()),
                "brier": (
                    float(((value[valid] - targets[valid]) ** 2).mean())
                    if bool(valid.any()) else None
                ),
                "time_dependent_auc": time_dependent_auc(
                    train_times,
                    train_events,
                    [row["time"] for row in subset],
                    [row["event"] for row in subset],
                    [row["value_risk"] for row in subset],
                    score_days,
                ),
                "interpretation": "observational fixed-horizon adverse-outcome risk; not treatment effect",
            }
        if name == "O3_velocity_uncertainty":
            raw_u: list[float] = []
            calibrated: list[float] = []
            observed: list[float] = []
            for row in subset:
                raw_u.extend(row["uncertainty_by_step"][1:])
                calibrated.extend(row["calibrated_error_by_step"][1:])
                observed.extend(row["latent_error_by_step"][1:])
            rho, reason = spearman(raw_u, observed)
            summary["uncertainty_calibration"] = {
                "raw_disagreement_spearman": rho,
                "raw_disagreement_spearman_reason": reason,
                "calibrated": regression_calibration(
                    calibrated, observed, bins=int(spec["calibration_bins"]),
                ),
                "coverage_risk_curve": coverage_risk_curve(
                    observed, raw_u, spec["coverage_levels"],
                ),
                "selective_prediction_only": True,
            }
        result[f"H{horizon}"] = summary
    return result, predictions


def _implementation_sha256() -> str:
    digest = hashlib.sha256()
    directory = Path(__file__).parent
    for filename in ("next_experiment.py", "trajectory.py", "metrics.py", "planner.py"):
        digest.update(filename.encode())
        digest.update((directory / filename).read_bytes())
    # v4 deliberately reuses v3's immutable fold/data-loading boundary.
    digest.update((directory.parent / "v3" / "next_stage.py").read_bytes())
    return digest.hexdigest()


def _initialize(
    config: dict[str, Any],
    spec: dict[str, Any],
    base: dict[str, Any],
    source: dict[str, Any],
    artifacts: RunArtifacts,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    source_path = RunArtifacts(
        config["paths"]["output_root"], spec["world_run"], version="v2",
    ).path("models.pt")
    source_digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
    implementation = _implementation_sha256()
    signature = _sha256_json({
        "config": spec,
        "base_data": base["data_signature"],
        "world_signature": source["signature"],
        "world_models_sha256": source_digest,
        "implementation_sha256": implementation,
    })
    old = read_json(artifacts.path("run.json"))
    if old and old.get("signature") != signature:
        raise V4ExperimentError(
            "existing v4 run has a different signature; choose a new run name",
        )
    if old:
        return (
            old,
            read_json(artifacts.path("metrics.json"), {}),
            read_torch(artifacts.path("models.pt"), safe=True),
        )
    artifacts.root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "cloop_v4_next_experiment_v1",
        "signature": signature,
        "base_run": spec["base_run"],
        "world_run": spec["world_run"],
        "base_data_signature": base["data_signature"],
        "world_models_sha256": source_digest,
        "implementation_sha256": implementation,
        "formal_test_untouched": True,
        "evidence_scope": "development-fold observational factual",
        "value_definition": "observed fixed-horizon adverse-outcome risk",
        "limitations": [
            "fold-training world features are in-sample",
            "value is a prognostic proxy, not a causal treatment effect",
            "clinical planner comparisons are model-based and not outcome efficacy estimates",
        ],
        "completed_tasks": [],
    }
    metrics: dict[str, Any] = {
        "dynamics": {}, "outcome": {}, "training": {}, "planning": {},
    }
    models: dict[str, Any] = {
        "schema_version": "cloop_v4_models_v1", "outcomes": {},
    }
    write_json(artifacts.path("run.json"), manifest)
    write_json(artifacts.path("metrics.json"), metrics)
    write_torch(artifacts.path("models.pt"), models)
    return manifest, metrics, models


@torch.no_grad()
def _planning_task(
    world: torch.nn.Module,
    evaluator: DualHeadOutcomeModel,
    bundle: DataBundle,
    fold: int,
    seed: int,
    spec: dict[str, Any],
    uncertainty_scales: dict[int, float],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    candidates = historical_candidates(
        bundle.action_codec, max_candidates=int(spec["planner_max_candidates"]),
    )
    if not candidates:
        raise V4ExperimentError("planner has no supported train-catalog actions")
    rows: list[dict[str, Any]] = []
    for pid in bundle.split_ids["validation"]:
        trajectory = bundle.trajectories[pid]
        state = patient_state(bundle, pid, 0, state_version=0)
        actual = trajectory.action_objects[0] if trajectory.action_objects else None
        replay = (actual,) if actual is not None else ()
        method_results = {}
        for method in PLANNING_METHODS:
            method_candidates = (candidates[0],) if method == "fixed_plan" else candidates
            beam_width = 1 if method == "greedy" else int(spec["planner_beam_width"])
            penalty = (
                float(spec["planner_lambda_uncertainty"])
                if method == "mpc_uncertainty" else 0.0
            )
            planner = TrajectoryValueMPC(
                world,
                evaluator,
                bundle.action_codec,
                planned_interval_days=bundle.planned_interval_days,
                horizon=int(spec["planner_horizon"]),
                beam_width=beam_width,
                lambda_uncertainty=penalty,
                uncertainty_scales=uncertainty_scales,
                uncertainty_clip=float(spec["planner_uncertainty_clip"]),
                max_candidates=int(spec["planner_max_candidates"]),
            )
            decision = planner.plan(
                state, replay_actions=replay, candidates=method_candidates,
            )
            method_results[method] = decision
            rows.append({
                "record_id": f"v4:plan:{fold}:{seed}:{method}:{pid}",
                "kind": "model_based_planning",
                "fold": fold,
                "seed": seed,
                "method": method,
                "patient_id": pid,
                "recommended_action": decision.action.action_id,
                "plan": [action.action_id for action in decision.plan],
                "score": decision.score,
                "mean_predicted_risk": decision.mean_predicted_risk,
                "cumulative_predicted_risk": decision.cumulative_predicted_risk,
                "mean_disagreement": decision.mean_disagreement,
                "cumulative_uncertainty_penalty": decision.cumulative_uncertainty_penalty,
                "world_forwards": decision.world_forwards,
                "interpretation": "model-based prognostic proxy; not counterfactual clinical benefit",
            })
        rows[-1]["differs_from_unpenalized_mpc"] = (
            method_results["mpc_uncertainty"].action.action_id
            != method_results["mpc"].action.action_id
        )
    summary: dict[str, Any] = {}
    for method in PLANNING_METHODS:
        group = [row for row in rows if row["method"] == method]
        summary[method] = {
            "patients": len(group),
            "mean_cumulative_predicted_risk": float(np.mean([
                row["cumulative_predicted_risk"] for row in group
            ])),
            "mean_rollout_disagreement": float(np.mean([
                row["mean_disagreement"] for row in group
            ])),
            "mean_score": float(np.mean([row["score"] for row in group])),
            "mean_world_forwards": float(np.mean([row["world_forwards"] for row in group])),
        }
    uncertain = [row for row in rows if row["method"] == "mpc_uncertainty"]
    summary["mpc_uncertainty"]["selection_change_rate_vs_mpc"] = float(np.mean([
        bool(row.get("differs_from_unpenalized_mpc")) for row in uncertain
    ]))
    summary["evidence_boundary"] = (
        "self-model scores and disagreement only; no historical counterfactual outcome comparison"
    )
    return summary, rows


def run(
    config: dict[str, Any],
    spec: dict[str, Any],
    stage: str,
    *,
    selected_fold: int | None = None,
    selected_seed: int | None = None,
) -> Path:
    base, cache, source, source_models = _source(config, spec)
    artifacts = RunArtifacts(config["paths"]["output_root"], spec["run"], version="v4")
    manifest, metrics, models = _initialize(config, spec, base, source, artifacts)
    test_ids = list(base["split"]["test"])
    device = _resolve_device(config["project"]["device"])
    valid_folds = {int(row["fold"]) for row in source["folds"]}
    if selected_fold is not None and selected_fold not in valid_folds:
        raise V4ExperimentError("selected fold is absent from source run")
    for fold in source["folds"]:
        index = int(fold["fold"])
        if selected_fold is not None and index != selected_fold:
            continue
        bundle = _fold_bundle(cache, config, source_models, fold, test_ids)
        for seed in spec["seeds"]:
            if selected_seed is not None and seed != selected_seed:
                continue
            if stage in {"dynamics", "all"}:
                for variant in DYNAMICS_VARIANTS:
                    key = f"dynamics/{variant}/fold{index}/seed{seed}"
                    if key in manifest["completed_tasks"]:
                        continue
                    world = _world(source_models, bundle, config, variant, index, seed, device)
                    records = _dynamics_records(
                        world, bundle, "validation", variant, index, seed, spec, device,
                    )
                    metrics["dynamics"][key] = _dynamics_summary(records, spec)
                    upsert_jsonl(artifacts.path("predictions.jsonl"), records)
                    manifest["completed_tasks"].append(key)
                    write_json(artifacts.path("metrics.json"), metrics)
                    write_json(artifacts.path("run.json"), manifest)
                    print(f"completed {key}", flush=True)
            if stage in {"outcome", "all"}:
                pending = [
                    name for name in OUTCOMES
                    if f"outcome/{name}/fold{index}/seed{seed}" not in manifest["completed_tasks"]
                ]
                if pending:
                    world = _world(
                        source_models, bundle, config, spec["outcome_world"], index, seed, device,
                    )
                    train_rows = _outcome_rows(world, bundle, "train", spec, device)
                    fit_rows, stop_rows = _inner_patient_split(train_rows, seed)
                    validation_rows = _outcome_rows(world, bundle, "validation", spec, device)
                    for name in pending:
                        key = f"outcome/{name}/fold{index}/seed{seed}"
                        state, training = _train_outcome(
                            name, fit_rows, stop_rows, bundle, config, spec, seed, device,
                        )
                        training["fit_patients"] = len({row["patient_id"] for row in fit_rows})
                        training["early_stop_patients"] = len({
                            row["patient_id"] for row in stop_rows
                        })
                        model = _new_outcome(name, bundle, config, spec, device)
                        model.load_state_dict(state)
                        summary, records = _evaluate_outcome(
                            model, name, validation_rows, train_rows, spec, device, index, seed,
                        )
                        models["outcomes"][key] = {
                            "state": state,
                            "training_patient_hash": _sha256_json(
                                sorted(bundle.split_ids["train"]),
                            ),
                        }
                        metrics["training"][key] = training
                        metrics["outcome"][key] = summary
                        write_torch(artifacts.path("models.pt"), models)
                        upsert_jsonl(artifacts.path("predictions.jsonl"), records)
                        manifest["completed_tasks"].append(key)
                        write_json(artifacts.path("metrics.json"), metrics)
                        write_json(artifacts.path("run.json"), manifest)
                        print(f"completed {key}", flush=True)
            if stage in {"planner", "all"}:
                key = f"planning/fold{index}/seed{seed}"
                if key in manifest["completed_tasks"]:
                    continue
                outcome_key = f"outcome/O3_velocity_uncertainty/fold{index}/seed{seed}"
                if outcome_key not in models["outcomes"]:
                    raise V4ExperimentError(
                        f"planner requires completed O3 outcome model: {outcome_key}",
                    )
                dynamics_key = f"dynamics/{spec['outcome_world']}/fold{index}/seed{seed}"
                if dynamics_key not in metrics["dynamics"]:
                    raise V4ExperimentError(
                        f"planner requires completed dynamics calibration: {dynamics_key}",
                    )
                evaluator = _new_outcome(
                    "O3_velocity_uncertainty", bundle, config, spec, device,
                )
                evaluator.load_state_dict(models["outcomes"][outcome_key]["state"])
                world = _world(
                    source_models, bundle, config, spec["outcome_world"], index, seed, device,
                )
                uncertainty_scales = {
                    horizon: float(metrics["dynamics"][dynamics_key].get(
                        f"uncertainty_scale@{horizon}", 0.0,
                    ))
                    for horizon in range(1, int(spec["planner_horizon"]) + 1)
                }
                summary, records = _planning_task(
                    world, evaluator, bundle, index, seed, spec, uncertainty_scales,
                )
                metrics["planning"][key] = summary
                upsert_jsonl(artifacts.path("predictions.jsonl"), records)
                manifest["completed_tasks"].append(key)
                write_json(artifacts.path("metrics.json"), metrics)
                write_json(artifacts.path("run.json"), manifest)
                print(f"completed {key}", flush=True)
    artifacts.assert_flat()
    return artifacts.root


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Cloop v4 uncertainty-aware trajectory and planning experiments",
    )
    parser.add_argument("--config", default="configs/v1/default.yaml")
    parser.add_argument("--paths", default="configs/v1/server.yaml")
    parser.add_argument("--experiment-config", default="configs/v4/next_experiment.yaml")
    parser.add_argument(
        "--stage", choices=("dynamics", "outcome", "planner", "all"), required=True,
    )
    parser.add_argument("--fold", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config, args.paths, device=args.device)
        spec = load_experiment_config(args.experiment_config)
        if args.seed is not None and args.seed not in spec["seeds"]:
            raise V4ExperimentError("selected seed is absent from experiment config")
        path = run(
            config,
            spec,
            args.stage,
            selected_fold=args.fold,
            selected_seed=args.seed,
        )
        print(json.dumps({"run": str(path), "stage": args.stage}, ensure_ascii=False))
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        parser.exit(2, f"v4 experiment: {exc}\n")


if __name__ == "__main__":
    main()
