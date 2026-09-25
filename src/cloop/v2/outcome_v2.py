"""Development-only, fold-local transition-aware frozen outcome experiment.

Run ``python -m cloop.v2.outcome_v2 --stage outcome|world|op|all``. The formal
test split and all older run artifacts are read-only. OP uses world models
trained on the same CV training patients and preprocessing as its evaluator.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import yaml
from torch import Tensor, nn
from torch.nn import functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from ..v1.artifacts import RunArtifacts, read_json, read_torch, upsert_jsonl, write_json, write_torch
from ..v1.config import load_config
from ..v1.data import DataBundle, DynamicsDataset, collate_dynamics, window_refs
from ..v1.metrics import mean_std, reliability_summary, spearman, survival_summary
from ..v1.outcome import PiecewiseExponentialHead
from ..v1_1.outcome_diagnostics import (
    TrainOnlyPCA, _build_fold_bundle, _eligible_latents, _extract_scalar,
    _load_base, _nll_from_rates, _patient_has_eligible_label,
    _resolve_device, _seed_all, _sha256_json, make_stratified_patient_folds,
)
from ..v1.world import EnsembleWorldModel, OneStepDynamics, terminal_states, variant_spec


EXPERIMENTS = ("O0", "O1", "O0_matched", "O1_matched", "OT_observed")
VARIANTS = ("baseline", "rrt", "ensemble", "rrt_ensemble")
DEFAULTS: dict[str, Any] = {
    "base_run": "brainiac_main_v1_provenance",
    "run": "outcome_v2_clarity_v1",
    "folds": 5,
    "fallback_folds": 3,
    "fold_seed": 1701,
    "seeds": [7, 17, 29],
    "pca_dim": 64,
    "residual_hidden_dim": 32,
    "variants": list(VARIANTS),
    "primary_landmark": "first_eligible_per_patient",
}


class OutcomeV2Error(RuntimeError):
    pass


def load_v2_config(path: str | Path) -> dict[str, Any]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    raw = raw.get("outcome_v2", raw)
    if not isinstance(raw, dict):
        raise OutcomeV2Error("outcome_v2 config must be a mapping")
    unknown = set(raw) - set(DEFAULTS)
    if unknown:
        raise OutcomeV2Error(f"unknown outcome_v2 options: {sorted(unknown)}")
    result = {**copy.deepcopy(DEFAULTS), **raw}
    result["seeds"] = [int(x) for x in result["seeds"]]
    result["variants"] = [str(x) for x in result["variants"]]
    if (not result["seeds"] or len(set(result["seeds"])) != len(result["seeds"])
            or not result["variants"] or set(result["variants"]) != set(VARIANTS)):
        raise OutcomeV2Error("use distinct seeds and all four registered dynamics variants")
    if int(result["pca_dim"]) != 64:
        raise OutcomeV2Error("registered Outcome-v2 uses PCA64")
    if result["primary_landmark"] != "first_eligible_per_patient":
        raise OutcomeV2Error("unknown primary landmark policy")
    if not str(result["run"]).startswith("outcome_v2_"):
        raise OutcomeV2Error("Outcome-v2 run name must start with outcome_v2_")
    if str(result["run"]) == str(result["base_run"]):
        raise OutcomeV2Error("Outcome-v2 run must differ from its base run")
    return result


def _state_hash(state: dict[str, Tensor]) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(key.encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


class V2Dataset(Dataset[dict[str, Any]]):
    """Every transition row uses target survival and target treatment history."""

    def __init__(self, bundle: DataBundle, split: str, *, transition: bool, first_only: bool = False):
        self.bundle = bundle
        self.refs: list[tuple[str, int, int]] = []
        self.counts: dict[str, int] = {}
        transition_refs = (
            {(ref.patient_id, ref.target): ref.start for ref in window_refs(bundle, split, mode="one_step")}
            if transition else {}
        )
        for pid in bundle.split_ids[split]:
            tr = bundle.trajectories[pid]
            local = []
            for target in range(tr.length):
                if not bool(tr.survival_valid[target]):
                    continue
                if transition:
                    source = transition_refs.get((pid, target))
                    if source is None:
                        continue
                else:
                    source = target
                local.append((pid, int(source), target))
            if first_only:
                local = local[:1]
            self.counts[pid] = len(local)
            self.refs.extend(local)

    def __len__(self) -> int:
        return len(self.refs)

    def __getitem__(self, index: int) -> dict[str, Any]:
        pid, source, target = self.refs[index]
        tr = self.bundle.trajectories[pid]
        return {
            "z_source": tr.latents[source].float(),
            "z_target": tr.latents[target].float(),
            "clinical": tr.clinical.float(),
            "clinical_mask": tr.clinical_mask.float(),
            "history_source": tr.histories[source].float(),
            "history_target": tr.histories[target].float(),
            "action": (tr.actions[source].float() if source < target
                       else torch.zeros(self.bundle.action_dim, dtype=torch.float32)),
            "delta_days": (tr.days[target] - tr.days[source]).float(),
            "time": tr.survival_time[target].float(),
            "event": tr.survival_event[target].long(),
            "weight": torch.tensor(1.0 / self.counts[pid], dtype=torch.float32),
            "patient_id": pid,
            "source_timepoint": tr.timepoint_ids[source],
            "target_timepoint": tr.timepoint_ids[target],
            "source_index": source,
            "target_index": target,
        }

    def sample_hash(self) -> str:
        return _sha256_json([(pid, source, target) for pid, source, target in self.refs])


class ResidualEvaluator(nn.Module):
    """Frozen clinical O0 plus a zero-initialized MRI residual."""

    def __init__(self, bundle: DataBundle, config: dict[str, Any], pca_dim: int,
                 residual_hidden_dim: int, *, transition: bool):
        super().__init__()
        self.clinical = PiecewiseExponentialHead.from_config(
            config, bundle.latent_dim, bundle.clinical_dim, bundle.history_dim,
            clinical_only=True,
        )
        dimension = pca_dim * (3 if transition else 1)
        self.residual = nn.Sequential(
            nn.LayerNorm(dimension), nn.Linear(dimension, residual_hidden_dim),
            nn.SiLU(), nn.Linear(residual_hidden_dim, len(config["outcome"]["edges_days"]) - 1),
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def load_clinical(self, state: dict[str, Tensor]) -> None:
        self.clinical.load_state_dict(state)
        for parameter in self.clinical.parameters():
            parameter.requires_grad_(False)

    def rates(self, representation: Tensor, clinical: Tensor, mask: Tensor, history: Tensor) -> Tensor:
        zero_z = torch.zeros(clinical.shape[0], self.clinical.latent_dim,
                              dtype=clinical.dtype, device=clinical.device)
        base = self.clinical.network(self.clinical._features(zero_z, clinical, mask, history))
        return F.softplus(base + self.residual(representation)) / 365.0


def transition_representation(pca: TrainOnlyPCA, source: Tensor, target: Tensor) -> Tensor:
    pre, post = pca.transform(source), pca.transform(target)
    return torch.cat((pre, post, post - pre), dim=-1)


def _rates(experiment: str, model: nn.Module, batch: dict[str, Any], pca: TrainOnlyPCA) -> Tensor:
    if experiment.startswith("O0"):
        return model.rates(batch["z_target"], batch["clinical"], batch["clinical_mask"],
                           batch["history_target"])
    representation = (
        transition_representation(pca, batch["z_source"], batch["z_target"])
        if experiment == "OT_observed" else pca.transform(batch["z_target"])
    )
    return model.rates(representation, batch["clinical"], batch["clinical_mask"],
                       batch["history_target"])


def _survival(rates: Tensor, edges: Tensor, horizon: float) -> Tensor:
    left = edges[:-1].to(rates.device)
    widths = (edges[1:] - edges[:-1]).to(rates.device)
    exposure = torch.minimum(torch.clamp(horizon - left, min=0), widths)
    return torch.exp(-(rates * exposure).sum(-1))


def _move(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if isinstance(value, Tensor) else value
            for key, value in batch.items()}


@torch.no_grad()
def _validation_nll(experiment: str, model: nn.Module, dataset: V2Dataset,
                    pca: TrainOnlyPCA, config: dict[str, Any], device: torch.device) -> float:
    model.eval()
    total = weight_total = 0.0
    for raw in DataLoader(dataset, batch_size=int(config["training"]["batch_size"])):
        batch = _move(raw, device)
        loss = _nll_from_rates(_rates(experiment, model, batch, pca), model.clinical.edges
                               if isinstance(model, ResidualEvaluator) else model.edges,
                               batch["time"], batch["event"], reduction="none")
        total += float((loss * batch["weight"]).sum().cpu())
        weight_total += float(batch["weight"].sum().cpu())
    if weight_total <= 0:
        raise OutcomeV2Error("validation fold has no eligible samples")
    return total / weight_total


def _train_outcome(experiment: str, model: nn.Module, train: V2Dataset, validation: V2Dataset,
                   pca: TrainOnlyPCA, config: dict[str, Any], seed: int,
                   device: torch.device) -> tuple[dict[str, Tensor], int, list[dict[str, float]]]:
    if not train or not validation:
        raise OutcomeV2Error(f"{experiment} needs nonempty training and validation cohorts")
    _seed_all(seed)
    model.to(device)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = AdamW(parameters, lr=float(config["outcome"]["lr"]),
                      weight_decay=float(config["outcome"]["weight_decay"]))
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(train, batch_size=int(config["training"]["batch_size"]), shuffle=True,
                        generator=generator, num_workers=int(config["training"]["num_workers"]))
    best_state: dict[str, Tensor] = {}
    best_epoch, best_score, stale = 0, float("inf"), 0
    history: list[dict[str, float]] = []
    for epoch in range(1, int(config["outcome"]["max_epochs"]) + 1):
        model.train()
        total = weight_total = 0.0
        for raw in loader:
            batch = _move(raw, device)
            optimizer.zero_grad(set_to_none=True)
            edges = model.clinical.edges if isinstance(model, ResidualEvaluator) else model.edges
            loss = _nll_from_rates(_rates(experiment, model, batch, pca), edges,
                                   batch["time"], batch["event"], batch["weight"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, float(config["training"]["grad_clip"]))
            optimizer.step()
            weight = float(batch["weight"].sum().cpu())
            total += float(loss.detach().cpu()) * weight
            weight_total += weight
        score = _validation_nll(experiment, model, validation, pca, config, device)
        history.append({"epoch": float(epoch), "train_nll": total / weight_total, "val_nll": score})
        if score < best_score:
            best_score, best_epoch, stale = score, epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
        if stale >= int(config["outcome"]["patience"]):
            break
    if not best_state:
        raise OutcomeV2Error(f"{experiment} produced no checkpoint")
    return best_state, best_epoch, history


def _new_outcome(experiment: str, bundle: DataBundle, config: dict[str, Any],
                 v2: dict[str, Any], clinical_state: dict[str, Tensor] | None = None) -> nn.Module:
    if experiment.startswith("O0"):
        return PiecewiseExponentialHead.from_config(
            config, bundle.latent_dim, bundle.clinical_dim, bundle.history_dim,
            clinical_only=True,
        )
    model = ResidualEvaluator(bundle, config, int(v2["pca_dim"]),
                              int(v2["residual_hidden_dim"]),
                              transition=experiment == "OT_observed")
    if clinical_state is None:
        raise OutcomeV2Error(f"{experiment} requires its matching clinical checkpoint")
    model.load_clinical(clinical_state)
    return model


def _label_reference(dataset: V2Dataset, max_days: float) -> tuple[list[float], list[int]]:
    times, events = [], []
    for pid, _, target in dataset.refs:
        tr = dataset.bundle.trajectories[pid]
        raw = float(tr.survival_time[target])
        times.append(min(raw, max_days))
        events.append(int(tr.survival_event[target]) if raw <= max_days else 0)
    return times, events


@torch.no_grad()
def _evaluate_rates(model: nn.Module, dataset: V2Dataset, config: dict[str, Any],
                    device: torch.device, rate_fn: Any,
                    train_reference: V2Dataset) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    edges = model.clinical.edges if isinstance(model, ResidualEvaluator) else model.edges
    horizon = float(config["outcome"]["score_horizon_days"])
    max_days = float(edges[-1])
    rows: list[dict[str, Any]] = []
    for raw in DataLoader(dataset, batch_size=int(config["training"]["batch_size"])):
        batch = _move(raw, device)
        rates = rate_fn(batch)
        probabilities = _survival(rates, edges, horizon)
        losses = _nll_from_rates(rates, edges, batch["time"], batch["event"], reduction="none")
        for i in range(len(probabilities)):
            time = float(batch["time"][i].cpu())
            rows.append({
                "patient_id": str(raw["patient_id"][i]),
                "source_timepoint": str(raw["source_timepoint"][i]),
                "target_timepoint": str(raw["target_timepoint"][i]),
                "time": min(time, max_days),
                "event": int(batch["event"][i].cpu()) if time <= max_days else 0,
                "survival365": float(probabilities[i].cpu()),
                "nll": float(losses[i].cpu()),
            })
    if not rows:
        raise OutcomeV2Error("primary evaluation has no samples")
    train_times, train_events = _label_reference(train_reference, max_days)
    summary = survival_summary(
        [row["time"] for row in rows], [row["event"] for row in rows],
        [row["survival365"] for row in rows], train_times, train_events, horizon,
    )
    summary.update(available=True, survival_nll=float(np.mean([row["nll"] for row in rows])),
                   patients=len({row["patient_id"] for row in rows}),
                   observations=len(rows), landmark="first_eligible_per_patient")
    return summary, rows

def _folds_for_v2(cache: dict[str, Any], dev_ids: list[str], v2: dict[str, Any]) -> list[list[str]]:
    for count in (int(v2["folds"]), int(v2["fallback_folds"])):
        if count not in (3, 5):
            raise OutcomeV2Error("only registered 5-fold or 3-fold CV is supported")
        folds = make_stratified_patient_folds(
            dev_ids, cache, requested_folds=count, fallback_folds=count, seed=int(v2["fold_seed"]),
        )
        if len(folds) == count:
            return folds
    raise OutcomeV2Error("insufficient support for registered CV folds")


def _fold_resources(cache: dict[str, Any], config: dict[str, Any], v2: dict[str, Any],
                    folds: list[list[str]], fold: int, formal_test: list[str],
                    models: dict[str, Any]) -> tuple[DataBundle, dict[str, V2Dataset], TrainOnlyPCA]:
    train_ids = sorted(pid for index, group in enumerate(folds) if index != fold for pid in group)
    bundle, preprocessing = _build_fold_bundle(cache, config, train_ids, folds[fold], formal_test)
    datasets = {}
    for split in ("train", "validation"):
        for cohort, transition in (("all", False), ("matched", True)):
            datasets[f"{split}/{cohort}"] = V2Dataset(bundle, split, transition=transition)
            datasets[f"{split}/{cohort}/primary"] = V2Dataset(
                bundle, split, transition=transition, first_only=True,
            )
    if any(len(datasets[f"{split}/matched/primary"]) == 0 for split in ("train", "validation")):
        raise OutcomeV2Error(f"fold {fold} has no eligible observed transition")
    key = f"fold{fold}"
    train_hash = _sha256_json(train_ids)
    if key in models["folds"]:
        entry = models["folds"][key]
        if entry["training_patient_hash"] != train_hash:
            raise OutcomeV2Error("stored fold training patient hash changed")
        pca = TrainOnlyPCA.from_state_dict(entry["pca"])
    else:
        pca = TrainOnlyPCA.fit(_eligible_latents(bundle, "train"), int(v2["pca_dim"]))
        models["folds"][key] = {
            "preprocessing": preprocessing, "pca": pca.state_dict(),
            "training_patient_hash": train_hash,
            "pca_fit_scope": "training_fold_eligible_MRI_landmarks_only",
        }
    return bundle, datasets, pca


def _summary_and_pairs(tasks: dict[str, Any]) -> dict[str, Any]:
    names = list(EXPERIMENTS) + [f"OP_{variant}" for variant in VARIANTS]
    indexed = {(row["experiment"], int(row["fold"]), int(row["seed"])): row["primary"]
               for row in tasks.values() if "primary" in row}
    summaries = {}
    for name in names:
        fold_means = {}
        for fold in sorted({fold for experiment, fold, _ in indexed if experiment == name}):
            fold_means[fold] = {}
            for metric in ("c_index", "brier365", "nll"):
                values = [_extract_scalar(value, metric) for (experiment, f, _), value in indexed.items()
                          if experiment == name and f == fold]
                values = [value for value in values if value is not None]
                if values:
                    fold_means[fold][metric] = float(np.mean(values))
        summaries[name] = {metric: mean_std([row[metric] for row in fold_means.values() if metric in row])
                           for metric in ("c_index", "brier365", "nll")}
    comparisons = {
        "O1_vs_O0": ("O0", "O1"),
        "OT_vs_O1_matched": ("O1_matched", "OT_observed"),
        "OP_rrt_vs_OP_baseline": ("OP_baseline", "OP_rrt"),
        "OP_rrt_ensemble_vs_OP_rrt": ("OP_rrt", "OP_rrt_ensemble"),
    }
    paired = {}
    for name, (base, candidate) in comparisons.items():
        fold_rows = []
        for fold in sorted({fold for experiment, fold, _ in indexed if experiment == base}):
            seed_rows = []
            for seed in sorted({seed for experiment, f, seed in indexed if experiment == base and f == fold}):
                first, second = indexed[(base, fold, seed)], indexed.get((candidate, fold, seed))
                if second is None:
                    continue
                seed_rows.append({
                    metric: (_extract_scalar(second, metric) - _extract_scalar(first, metric)
                             if _extract_scalar(second, metric) is not None
                             and _extract_scalar(first, metric) is not None else None)
                    for metric in ("c_index", "brier365", "nll")
                })
            if seed_rows:
                fold_rows.append({"fold": fold, **{
                    metric: (float(np.mean([row[metric] for row in seed_rows if row[metric] is not None]))
                             if any(row[metric] is not None for row in seed_rows) else None)
                    for metric in ("c_index", "brier365", "nll")
                }})
        paired[name] = {
            "folds": fold_rows,
            "metrics": {metric: mean_std([row[metric] for row in fold_rows if row[metric] is not None])
                        for metric in ("c_index", "brier365", "nll")},
            "direction_counts": {
                metric: sum((row[metric] > 0 if metric == "c_index" else row[metric] < 0)
                            for row in fold_rows if row[metric] is not None)
                for metric in ("c_index", "brier365", "nll")
            },
        }
    gaps = {}
    for variant in VARIANTS:
        per_fold = {}
        for row in tasks.values():
            if row.get("experiment") != f"OP_{variant}":
                continue
            fold = int(row["fold"])
            bucket = per_fold.setdefault(fold, {"c_index": [], "brier365": [], "nll": []})
            for metric, value in row["predicted_minus_observed_gap"].items():
                if value is not None:
                    bucket[metric].append(value)
        gaps[variant] = {
            metric: mean_std([float(np.mean(bucket[metric])) for bucket in per_fold.values()
                              if bucket[metric]])
            for metric in ("c_index", "brier365", "nll")
        }
    gate = paired["OT_vs_O1_matched"]
    n = len(gate["folds"])
    improved = [metric for metric, count in gate["direction_counts"].items()
                if count > n / 2 and gate["metrics"][metric]["n"] == n]
    return {
        "experiments": summaries, "paired": paired, "predicted_to_observed_gap": gaps,
        "gate_b": {"passed": n >= 3 and len(improved) >= 2,
                   "improved_metrics": improved, "evaluable_folds": n,
                   "rule": "OT improves >=2 primary metrics in a majority of folds"},
    }


def _save(artifacts: RunArtifacts, manifest: dict[str, Any],
          models: dict[str, Any], metrics: dict[str, Any]) -> None:
    metrics["summary"] = _summary_and_pairs(metrics["tasks"])
    write_torch(artifacts.path("models.pt"), models)
    write_json(artifacts.path("metrics.json"), metrics)
    manifest["completed_tasks"] = sorted(metrics["tasks"])
    write_json(artifacts.path("run.json"), manifest)


def _initialize(config: dict[str, Any], v2: dict[str, Any]) -> tuple[Any, ...]:
    if config["data"]["protocol"] != "main_v1":
        raise OutcomeV2Error("Outcome-v2 requires main_v1")
    if [float(x) for x in config["outcome"]["edges_days"]] != [0, 90, 180, 365, 730]:
        raise OutcomeV2Error("Outcome-v2 requires registered survival edges")
    base, cache, dev_ids, formal_test = _load_base(config, v2)
    folds = _folds_for_v2(cache, dev_ids, v2)
    artifacts = RunArtifacts(config["paths"]["output_root"], str(v2["run"]), version="v2")
    signature = _sha256_json({
        "base_data_signature": base["data_signature"], "v2": v2,
        "data": config["data"], "world": config["world"], "outcome": config["outcome"],
        "training": config["training"], "evaluation": config["evaluation"],
    })
    manifest = read_json(artifacts.path("run.json"))
    if manifest:
        if manifest.get("schema_version") != "cloop_outcome_v2_clarity_v1" or manifest.get("signature") != signature:
            raise OutcomeV2Error("run schema/config changed; select a new outcome_v2_ run")
    else:
        if artifacts.root.exists() and any(artifacts.root.iterdir()):
            raise OutcomeV2Error("nonempty output directory has no Outcome-v2 run manifest")
        artifacts.root.mkdir(parents=True, exist_ok=True)
        manifest = {
            "schema_version": "cloop_outcome_v2_clarity_v1", "run_name": v2["run"],
            "base_run": v2["base_run"], "base_data_signature": base["data_signature"],
            "signature": signature, "formal_test_untouched": True,
            "formal_test_ids_sha256": _sha256_json(sorted(formal_test)),
            "development_ids_sha256": _sha256_json(sorted(dev_ids)),
            "folds": [{"fold": i, "validation_ids": group,
                       "training_ids": sorted(pid for j, other in enumerate(folds) if j != i for pid in other)}
                      for i, group in enumerate(folds)],
            "seeds": v2["seeds"], "completed_tasks": [],
            "evidence_scope": "observational prognostic evaluation; no causal treatment effect",
            "planner_integration": "pending Gate B and predicted-transition evidence",
        }
        write_json(artifacts.path("run.json"), manifest)
    models = (read_torch(artifacts.path("models.pt"), safe=True)
              if artifacts.path("models.pt").is_file() else {
                  "schema_version": "cloop_outcome_v2_models_v1",
                  "folds": {}, "outcomes": {}, "worlds": {},
              })
    metrics = read_json(artifacts.path("metrics.json"), {
        "schema_version": "cloop_outcome_v2_metrics_v1", "tasks": {}, "summary": {},
    })
    return artifacts, manifest, models, metrics, cache, folds, formal_test


def run_outcomes(config: dict[str, Any], v2: dict[str, Any], resources: tuple[Any, ...],
                 device: torch.device) -> None:
    artifacts, manifest, models, metrics, cache, folds, formal_test = resources
    for fold in range(len(folds)):
        bundle, datasets, pca = _fold_resources(cache, config, v2, folds, fold, formal_test, models)
        _save(artifacts, manifest, models, metrics)
        for experiment in EXPERIMENTS:
            cohort = "matched" if experiment.endswith("matched") or experiment == "OT_observed" else "all"
            train, validation = datasets[f"train/{cohort}"], datasets[f"validation/{cohort}"]
            primary, reference = datasets[f"validation/{cohort}/primary"], datasets[f"train/{cohort}/primary"]
            for seed in v2["seeds"]:
                key = f"{experiment}/fold{fold}/seed{seed}"
                if key in metrics["tasks"]:
                    continue
                clinical_state = None
                if not experiment.startswith("O0"):
                    base_name = "O0_matched" if cohort == "matched" else "O0"
                    clinical_state = models["outcomes"][f"{base_name}/fold{fold}/seed{seed}"]["state"]
                _seed_all(seed)
                model = _new_outcome(experiment, bundle, config, v2, clinical_state)
                state, best_epoch, history = _train_outcome(
                    experiment, model, train, validation, pca, config, seed, device,
                )
                model.load_state_dict(state)
                model.to(device).eval()
                result, rows = _evaluate_rates(
                    model, primary, config, device,
                    lambda batch: _rates(experiment, model, batch, pca), reference,
                )
                models["outcomes"][key] = {
                    "state": state, "state_sha256": _state_hash(state),
                    "training_sample_hash": train.sample_hash(),
                    "validation_sample_hash": primary.sample_hash(),
                }
                metrics["tasks"][key] = {
                    "experiment": experiment, "fold": fold, "seed": seed,
                    "primary": result, "best_epoch": best_epoch, "history": history,
                    "validation_sample_hash": primary.sample_hash(),
                }
                upsert_jsonl(artifacts.path("predictions.jsonl"), [
                    {**row, "record_id": f"outcome_v2:{key}:{row['patient_id']}:{row['target_timepoint']}",
                     "experiment": experiment, "fold": fold, "seed": seed}
                    for row in rows
                ])
                _save(artifacts, manifest, models, metrics)
                print(f"outcome_v2 {key} best_epoch={best_epoch} nll={result['survival_nll']:.6f}", flush=True)

def _world_member(bundle: DataBundle, config: dict[str, Any], device: torch.device) -> OneStepDynamics:
    return OneStepDynamics.from_config(
        config, bundle.latent_dim, bundle.action_dim, bundle.clinical_dim, bundle.history_dim,
    ).to(device)


@torch.no_grad()
def _world_validation(model: OneStepDynamics, bundle: DataBundle,
                      config: dict[str, Any], device: torch.device) -> float:
    model.eval()
    scores = []
    for horizon in (2, 3):
        refs = window_refs(bundle, "validation", mode="all_exact",
                           exact_horizon=horizon, max_horizon=horizon)
        if not refs:
            continue
        errors = []
        for raw in DataLoader(DynamicsDataset(bundle, refs),
                              batch_size=int(config["training"]["batch_size"]),
                              collate_fn=collate_dynamics):
            batch = _move(raw, device)
            rollout = EnsembleWorldModel([model]).rollout(
                batch["z0"], batch["actions"], batch["deltas"], batch["context"],
                batch["clinical_mask"], batch["history0"], batch["step_mask"],
            )
            terminal = terminal_states(rollout, batch["horizons"])[0]
            errors.extend(((terminal - batch["target"]) ** 2).mean(-1).cpu().tolist())
        scores.append(float(np.mean(errors)))
    if not scores:
        raise OutcomeV2Error("fold validation has no H2/H3 windows for world selection")
    return float(np.mean(scores))


def _train_world(bundle: DataBundle, config: dict[str, Any], variant: str,
                 seed: int, member_index: int, device: torch.device
                 ) -> tuple[dict[str, Tensor], int, list[dict[str, float]]]:
    recursive, member_count = variant_spec(variant, int(config["world"]["ensemble_size"]))
    refs = window_refs(bundle, "train", mode="max_available" if recursive else "one_step",
                       max_horizon=int(config["world"]["max_horizon"]))
    if not refs:
        raise OutcomeV2Error("fold training has no dynamics windows")
    member_seed = int(seed) if member_count == 1 else int(seed) * 1000 + member_index
    _seed_all(member_seed)
    model = _world_member(bundle, config, device)
    optimizer = AdamW(model.parameters(), lr=float(config["training"]["lr"]),
                      weight_decay=float(config["training"]["weight_decay"]))
    loader = DataLoader(
        DynamicsDataset(bundle, refs), batch_size=int(config["training"]["batch_size"]),
        shuffle=True, generator=torch.Generator().manual_seed(member_seed),
        num_workers=int(config["training"]["num_workers"]), collate_fn=collate_dynamics,
    )
    best_state: dict[str, Tensor] = {}
    best_epoch, best_score, stale = 0, float("inf"), 0
    history: list[dict[str, float]] = []
    for epoch in range(1, int(config["training"]["max_epochs"]) + 1):
        model.train()
        total = count = 0.0
        for raw in loader:
            batch = _move(raw, device)
            optimizer.zero_grad(set_to_none=True)
            rollout = EnsembleWorldModel([model]).rollout(
                batch["z0"], batch["actions"], batch["deltas"], batch["context"],
                batch["clinical_mask"], batch["history0"], batch["step_mask"],
            )
            terminal = terminal_states(rollout, batch["horizons"])[0]
            loss = ((terminal - batch["target"]) ** 2).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["training"]["grad_clip"]))
            optimizer.step()
            size = len(batch["z0"])
            total += float(loss.detach().cpu()) * size
            count += size
        score = _world_validation(model, bundle, config, device)
        history.append({"epoch": float(epoch), "train_mse": total / count, "val_long_mse": score})
        if score < best_score:
            best_score, best_epoch, stale = score, epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
        if stale >= int(config["training"]["patience"]):
            break
    if not best_state:
        raise OutcomeV2Error("world training produced no checkpoint")
    return best_state, best_epoch, history


def _assert_gate_b(metrics: dict[str, Any], folds: int, seeds: Sequence[int]) -> None:
    needed = {f"OT_observed/fold{fold}/seed{seed}"
              for fold in range(folds) for seed in seeds}
    if not needed.issubset(metrics["tasks"]):
        raise OutcomeV2Error("all fold/seed OT checkpoints are required before world/OP")
    if not metrics["summary"]["gate_b"]["passed"]:
        raise OutcomeV2Error("Gate B did not pass; transition route stops before world/OP")


def run_worlds(config: dict[str, Any], v2: dict[str, Any], resources: tuple[Any, ...],
               device: torch.device) -> None:
    artifacts, manifest, models, metrics, cache, folds, formal_test = resources
    _assert_gate_b(metrics, len(folds), v2["seeds"])
    for fold in range(len(folds)):
        bundle, _, _ = _fold_resources(cache, config, v2, folds, fold, formal_test, models)
        for variant in VARIANTS:
            _, member_count = variant_spec(variant, int(config["world"]["ensemble_size"]))
            for seed in v2["seeds"]:
                key = f"{variant}/fold{fold}/seed{seed}"
                entry = models["worlds"].setdefault(key, {
                    "member_states": [], "best_epochs": [], "member_seeds": [],
                    "member_histories": [], "complete": False,
                    "training_patient_hash": models["folds"][f"fold{fold}"]["training_patient_hash"],
                })
                if entry["training_patient_hash"] != models["folds"][f"fold{fold}"]["training_patient_hash"]:
                    raise OutcomeV2Error("world fold patient hash mismatch")
                for member in range(len(entry["member_states"]), member_count):
                    state, best_epoch, history = _train_world(
                        bundle, config, variant, seed, member, device,
                    )
                    entry["member_states"].append(state)
                    entry["best_epochs"].append(best_epoch)
                    entry["member_seeds"].append(
                        int(seed) if member_count == 1 else int(seed) * 1000 + member
                    )
                    entry["member_histories"].append(history)
                    entry["complete"] = len(entry["member_states"]) == member_count
                    _save(artifacts, manifest, models, metrics)
                    print(f"outcome_v2 world {key} member={member} best_epoch={best_epoch}", flush=True)


def _world_from_entry(bundle: DataBundle, config: dict[str, Any],
                      entry: dict[str, Any], device: torch.device) -> EnsembleWorldModel:
    if not entry.get("complete"):
        raise OutcomeV2Error("incomplete fold-specific world ensemble")
    members = []
    for state in entry["member_states"]:
        model = _world_member(bundle, config, device)
        model.load_state_dict(state)
        model.eval()
        members.append(model)
    return EnsembleWorldModel(members).to(device).eval()

@torch.no_grad()
def _evaluate_op(model: ResidualEvaluator, world: EnsembleWorldModel,
                 dataset: V2Dataset, train_reference: V2Dataset, pca: TrainOnlyPCA,
                 config: dict[str, Any], device: torch.device
                 ) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    model.eval()
    world.eval()
    rows: list[dict[str, Any]] = []
    horizon = float(config["outcome"]["score_horizon_days"])
    edges = model.clinical.edges
    max_days = float(edges[-1])
    for raw in DataLoader(dataset, batch_size=int(config["training"]["batch_size"])):
        batch = _move(raw, device)
        actions = batch["action"].unsqueeze(1)
        deltas = batch["delta_days"].unsqueeze(1)
        rollout = world.rollout(
            batch["z_source"], actions, deltas, batch["clinical"],
            batch["clinical_mask"], batch["history_source"],
            torch.ones_like(deltas, dtype=torch.bool),
        )
        member_states = terminal_states(
            rollout, torch.ones(len(batch["z_source"]), device=device, dtype=torch.long),
        )
        pre = pca.transform(batch["z_source"])
        observed_post = pca.transform(batch["z_target"])
        member_post = pca.transform(member_states)
        predicted_post = member_post.mean(0)
        representation = torch.cat((pre, predicted_post, predicted_post - pre), -1)
        observed_representation = torch.cat((pre, observed_post, observed_post - pre), -1)
        rates = model.rates(
            representation, batch["clinical"], batch["clinical_mask"], batch["history_target"],
        )
        observed_rates = model.rates(
            observed_representation, batch["clinical"], batch["clinical_mask"], batch["history_target"],
        )
        probabilities = _survival(rates, edges, horizon)
        predicted_nll = _nll_from_rates(
            rates, edges, batch["time"], batch["event"], reduction="none",
        )
        observed_nll = _nll_from_rates(
            observed_rates, edges, batch["time"], batch["event"], reduction="none",
        )
        observed_norm = (observed_post - pre).norm(dim=-1)
        predicted_norm = (predicted_post - pre).norm(dim=-1)
        latent_error = (predicted_post - observed_post).norm(dim=-1)
        uncertainty = ((member_post - predicted_post.unsqueeze(0)) ** 2).sum(-1).mean(0)
        for i in range(len(probabilities)):
            time = float(batch["time"][i].cpu())
            rows.append({
                "patient_id": str(raw["patient_id"][i]),
                "source_timepoint": str(raw["source_timepoint"][i]),
                "target_timepoint": str(raw["target_timepoint"][i]),
                "time": min(time, max_days),
                "event": int(batch["event"][i].cpu()) if time <= max_days else 0,
                "survival365": float(probabilities[i].cpu()),
                "nll": float(predicted_nll[i].cpu()),
                "observed_transition_norm": float(observed_norm[i].cpu()),
                "predicted_transition_norm": float(predicted_norm[i].cpu()),
                "latent_error": float(latent_error[i].cpu()),
                "outcome_error": float(abs(predicted_nll[i] - observed_nll[i]).cpu()),
                "disagreement": float(uncertainty[i].cpu()) if world.ensemble_size > 1 else None,
                "conditioning": "recorded_action_and_MRI_interval",
            })
    if not rows:
        raise OutcomeV2Error("OP evaluation has no matched samples")
    train_times, train_events = _label_reference(train_reference, max_days)
    summary = survival_summary(
        [row["time"] for row in rows], [row["event"] for row in rows],
        [row["survival365"] for row in rows], train_times, train_events, horizon,
    )
    summary.update(
        available=True, survival_nll=float(np.mean([row["nll"] for row in rows])),
        patients=len({row["patient_id"] for row in rows}), observations=len(rows),
        landmark="first_eligible_per_patient",
    )
    latent_rho, latent_reason = spearman(
        [row["latent_error"] for row in rows], [row["outcome_error"] for row in rows],
    )
    diagnostics: dict[str, Any] = {
        "observed_transition_norm_mean": float(np.mean([row["observed_transition_norm"] for row in rows])),
        "predicted_transition_norm_mean": float(np.mean([row["predicted_transition_norm"] for row in rows])),
        "latent_error_mean": float(np.mean([row["latent_error"] for row in rows])),
        "outcome_error_mean": float(np.mean([row["outcome_error"] for row in rows])),
        "latent_error_vs_outcome_error_spearman": latent_rho,
        "latent_error_vs_outcome_error_reason": latent_reason,
    }
    if world.ensemble_size > 1:
        diagnostics["ensemble_reliability"] = reliability_summary(
            [row["outcome_error"] for row in rows],
            [row["disagreement"] for row in rows],
            float(config["evaluation"]["selective_coverage"]),
        )
        diagnostics["ensemble_reliability"]["risk_definition"] = "absolute per-sample NLL gap vs observed transition"
        diagnostics["disagreement_definition"] = "mean member squared L2 distance in PCA space"
    return summary, rows, diagnostics


def run_op(config: dict[str, Any], v2: dict[str, Any], resources: tuple[Any, ...],
           device: torch.device) -> None:
    artifacts, manifest, models, metrics, cache, folds, formal_test = resources
    _assert_gate_b(metrics, len(folds), v2["seeds"])
    for fold in range(len(folds)):
        bundle, datasets, pca = _fold_resources(cache, config, v2, folds, fold, formal_test, models)
        primary = datasets["validation/matched/primary"]
        reference = datasets["train/matched/primary"]
        expected_sample_hash = primary.sample_hash()
        for seed in v2["seeds"]:
            ot_key = f"OT_observed/fold{fold}/seed{seed}"
            ot_entry = models["outcomes"][ot_key]
            if ot_entry["validation_sample_hash"] != expected_sample_hash:
                raise OutcomeV2Error("OT/OP patient or target-landmark mismatch")
            if _state_hash(ot_entry["state"]) != ot_entry["state_sha256"]:
                raise OutcomeV2Error("frozen OT checkpoint hash mismatch")
            clinical_state = models["outcomes"][f"O0_matched/fold{fold}/seed{seed}"]["state"]
            ot = _new_outcome("OT_observed", bundle, config, v2, clinical_state)
            ot.load_state_dict(ot_entry["state"])
            ot.to(device).eval()
            ot.requires_grad_(False)
            observed = metrics["tasks"][ot_key]["primary"]
            for variant in VARIANTS:
                key = f"OP_{variant}/fold{fold}/seed{seed}"
                if key in metrics["tasks"]:
                    continue
                world_key = f"{variant}/fold{fold}/seed{seed}"
                if world_key not in models["worlds"]:
                    raise OutcomeV2Error(f"missing fold-local world {world_key}")
                world_entry = models["worlds"][world_key]
                if world_entry["training_patient_hash"] != models["folds"][f"fold{fold}"]["training_patient_hash"]:
                    raise OutcomeV2Error("world was trained on different fold patients")
                world = _world_from_entry(bundle, config, world_entry, device)
                result, rows, diagnostics = _evaluate_op(
                    ot, world, primary, reference, pca, config, device,
                )
                gap = {}
                for metric in ("c_index", "brier365", "nll"):
                    predicted_value, observed_value = _extract_scalar(result, metric), _extract_scalar(observed, metric)
                    gap[metric] = (predicted_value - observed_value
                                   if predicted_value is not None and observed_value is not None else None)
                metrics["tasks"][key] = {
                    "experiment": f"OP_{variant}", "fold": fold, "seed": seed,
                    "primary": result, "predicted_minus_observed_gap": gap,
                    "diagnostics": diagnostics, "validation_sample_hash": expected_sample_hash,
                    "frozen_ot_sha256": ot_entry["state_sha256"],
                    "world_key": world_key,
                }
                upsert_jsonl(artifacts.path("predictions.jsonl"), [
                    {**row, "record_id": f"outcome_v2:{key}:{row['patient_id']}:{row['target_timepoint']}",
                     "experiment": f"OP_{variant}", "fold": fold, "seed": seed,
                     "frozen_ot_sha256": ot_entry["state_sha256"]}
                    for row in rows
                ])
                _save(artifacts, manifest, models, metrics)
                print(f"outcome_v2 {key} nll={result['survival_nll']:.6f}", flush=True)


class _Tee:
    def __init__(self, *streams: Any):
        self.streams = streams

    def write(self, value: str) -> int:
        for stream in self.streams:
            stream.write(value)
        return len(value)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Isolated CLARITY-inspired Outcome-v2 experiment")
    parser.add_argument("--config", default="configs/v1/default.yaml")
    parser.add_argument("--paths", default="configs/v1/server.yaml")
    parser.add_argument("--v2-config", default="configs/v2/outcome_v2.yaml")
    parser.add_argument("--run", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--stage", choices=("outcome", "world", "op", "all"), default="outcome")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_config(args.config, args.paths, device=args.device)
    v2 = load_v2_config(args.v2_config)
    if args.run:
        v2["run"] = args.run
    if not str(v2["run"]).startswith("outcome_v2_"):
        raise OutcomeV2Error("Outcome-v2 run must start with outcome_v2_")
    RunArtifacts(config["paths"]["output_root"], str(v2["run"]), version="v2")
    device = _resolve_device(config["project"]["device"])
    from ..v1.engine import validate_device_visibility
    validate_device_visibility(config, device)
    log_path = Path(config["paths"]["project_root"]) / "logs" / "v2" / f"{v2['run']}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", buffering=1) as handle:
        with redirect_stdout(_Tee(sys.stdout, handle)), redirect_stderr(_Tee(sys.stderr, handle)):
            print(f"[{datetime.now(timezone.utc).isoformat()}] outcome_v2 stage={args.stage}", flush=True)
            resources = _initialize(config, v2)
            if args.stage in ("outcome", "all"):
                run_outcomes(config, v2, resources, device)
            if args.stage in ("world", "all"):
                run_worlds(config, v2, resources, device)
            if args.stage in ("op", "all"):
                run_op(config, v2, resources, device)
            print(f"[{datetime.now(timezone.utc).isoformat()}] outcome_v2 completed", flush=True)
    print(json.dumps({"status": "ok", "run": v2["run"], "stage": args.stage,
                      "output": str(resources[0].root)}, indent=2))


if __name__ == "__main__":
    main()
