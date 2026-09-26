"""Same-architecture, independently trained CLARITY outcome comparison.

This entry point is deliberately isolated from the historical v4/v4.1 route.
For ``development_reuse`` it reads frozen Outcome-v2 worlds, creates matched
H1/H2/H3 forecasts, and trains one CLARITY SurvivalModule per
variant/fold/seed/horizon.  No outcome gradient reaches a world model.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import yaml
from torch import Tensor, nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from ..v1.artifacts import (
    RunArtifacts,
    read_json,
    read_torch,
    upsert_jsonl,
    write_json,
    write_text,
    write_torch,
)
from ..v1.config import load_config, strict_merge
from ..v1.data import (
    DataBundle,
    DynamicsDataset,
    WindowRef,
    collate_dynamics,
    load_bundle,
    sha256_file,
)
from ..v1.world import variant_spec
from ..v1_1.outcome_diagnostics import (
    _build_fold_bundle, _load_base, _resolve_device, _seed_all, _sha256_json,
)
from ..v2.outcome_v2 import _train_world, _world_from_entry
from .clarity_downstream_head import (
    ClarityOutcomeAdapter,
    fixed_time_survival_targets,
    outcome_loss,
    summarize_member_logits,
)
from .clarity_downstream_metrics import (
    aggregate_main,
    aggregate_uncertainty,
    build_report,
    evaluate_cohort,
    paired_comparisons,
    selective_ipcw,
)
from .next_stage import _fold_bundle, _refs, _source, _to_device, _world


SCHEMA = "cloop_clarity_outcome_v1"
MODEL_SCHEMA = "cloop_clarity_outcome_models_v1"
CACHE_SCHEMA = "cloop_clarity_forecasts_v1"
VARIANTS = ("baseline", "rrt", "ensemble", "rrt_ensemble")
ENSEMBLE_VARIANTS = ("ensemble", "rrt_ensemble")
UPSTREAM_COMMIT = "dadb82241a24f5ec5e4e4dc994e3116fd4a9da04"
UPSTREAM_BLOB = "21e984ee04745348b8fc5151e4650a63616a4567"

DEFAULTS: dict[str, Any] = {
    "run": "next_stage_clarity_outcome_v1",
    "protocol": "development_reuse",
    "base_run": "brainiac_main_v1_provenance",
    "world_run": "outcome_v2_clarity_v1",
    "variants": list(VARIANTS),
    "seeds": [7, 17, 29],
    "horizons": [1, 2, 3],
    "primary_horizon": 2,
    "head_scope": "per_horizon",
    "pair_mode": "observed_start_predicted_endpoint",
    "landmark": "first_eligible_per_patient_per_horizon",
    "label_anchor": "target_landmark",
    "administrative_censor_days": None,
    "feature_batch_size": 32,
    "token_mode": "single_global_token",
    "condition_mode": "clinical_mask_target_history_elapsed",
    "expected_latent_dim": 768,
    "main_aggregation": "mean_latent",
    "train_outcome_independently": True,
    "freeze_dynamics": True,
    "head": {
        "hidden_dim": 128,
        "attention_dim": 128,
        "num_twoway_layers": 2,
        "num_heads": 4,
        "dropout": 0.3,
    },
    "training": {
        "full_cohort_loss": True,
        "lr": 0.001,
        "weight_decay": 0.0001,
        "max_epochs": 300,
        "patience": 40,
        "grad_clip": 1.0,
        "outcome_split_seed": 1701,
        "early_stop_metric": "cox_bce",
        "cox_ties": "breslow",
        "cox_weight": 0.5,
        "bce_weight": 0.5,
        "probability_horizon_days": 365.0,
    },
    "uncertainty": {
        "enabled": True,
        "retrain_head": False,
        "compare_probability_aggregation": True,
        "score": "probability_std",
        "coverages": [1.0, 0.8, 0.6],
        "threshold_mode": "rank_curve",
        "random_reference_repeats": 200,
    },
    "evaluation": {
        "primary_metric": "c_index_risk",
        "companion_metric": "ipcw_brier365",
        "report_td_auc365": True,
        "aggregate_seeds_within_fold": True,
        "paired_bootstrap_repeats": 0,
    },
}


class ClarityExperimentError(RuntimeError):
    pass


def load_downstream_config(path: str | Path) -> dict[str, Any]:
    try:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ClarityExperimentError(f"cannot read downstream config: {exc}") from exc
    raw = raw.get("clarity_downstream", raw)
    if not isinstance(raw, dict):
        raise ClarityExperimentError("clarity_downstream config must be a mapping")
    try:
        spec = strict_merge(DEFAULTS, raw)
    except ValueError as exc:
        raise ClarityExperimentError(str(exc)) from exc
    if spec["protocol"] not in {"development_reuse", "original_holdout"}:
        raise ClarityExperimentError("protocol must be development_reuse or original_holdout")
    if not str(spec["run"]).startswith("next_stage_clarity_outcome_"):
        raise ClarityExperimentError("run must start with next_stage_clarity_outcome_")
    if len({spec["run"], spec["base_run"], spec["world_run"]}) != 3:
        raise ClarityExperimentError("new run and source runs must be distinct")
    if tuple(spec["variants"]) != VARIANTS:
        raise ClarityExperimentError(f"variants must be exactly {list(VARIANTS)}")
    if not spec["seeds"] or len(set(spec["seeds"])) != len(spec["seeds"]):
        raise ClarityExperimentError("seeds must be non-empty and distinct")
    if sorted(spec["horizons"]) != list(spec["horizons"]) or not set(spec["horizons"]).issubset({1, 2, 3}):
        raise ClarityExperimentError("horizons must be ordered unique values from [1,2,3]")
    fixed = {
        "head_scope": "per_horizon",
        "pair_mode": "observed_start_predicted_endpoint",
        "landmark": "first_eligible_per_patient_per_horizon",
        "label_anchor": "target_landmark",
        "token_mode": "single_global_token",
        "condition_mode": "clinical_mask_target_history_elapsed",
        "main_aggregation": "mean_latent",
        "train_outcome_independently": True,
        "freeze_dynamics": True,
    }
    bad = [key for key, value in fixed.items() if spec[key] != value]
    if bad:
        raise ClarityExperimentError("fixed experimental controls changed: " + ", ".join(bad))
    training = spec["training"]
    if (
        not training["full_cohort_loss"]
        or training["cox_ties"] != "breslow"
        or training["early_stop_metric"] != "cox_bce"
        or min(int(training["max_epochs"]), int(training["patience"])) < 1
        or min(float(training["lr"]), float(training["grad_clip"])) <= 0
        or min(float(training["cox_weight"]), float(training["bce_weight"])) < 0
        or float(training["cox_weight"]) + float(training["bce_weight"]) <= 0
    ):
        raise ClarityExperimentError("invalid fixed full-cohort Cox+BCE training config")
    head = spec["head"]
    if (
        int(head["attention_dim"]) % int(head["num_heads"])
        or int(head["hidden_dim"]) < 2
        or int(head["num_twoway_layers"]) < 1
        or not 0 <= float(head["dropout"]) < 1
    ):
        raise ClarityExperimentError("invalid CLARITY head configuration")
    if spec["uncertainty"]["retrain_head"]:
        raise ClarityExperimentError("uncertainty must remain a forward-only ablation")
    if spec["uncertainty"]["threshold_mode"] not in {"rank_curve", "fixed_threshold"}:
        raise ClarityExperimentError("invalid uncertainty threshold_mode")
    if any(not 0 < float(x) <= 1 for x in spec["uncertainty"]["coverages"]):
        raise ClarityExperimentError("uncertainty coverages must be in (0,1]")
    return spec


def _git_commit(project_root: str | Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=project_root, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _tensor_state_hash(state: dict[str, Tensor]) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _architecture_hash(model: nn.Module) -> str:
    return _sha256_json({
        key: {"shape": list(value.shape), "dtype": str(value.dtype)}
        for key, value in model.state_dict().items()
    })


def _condition_dim(bundle: DataBundle) -> int:
    return 2 * bundle.clinical_dim + bundle.history_dim + 1


def _head_config(spec: dict[str, Any]) -> dict[str, Any]:
    return {
        "hidden_dim": int(spec["head"]["hidden_dim"]),
        "attention_dim": int(spec["head"]["attention_dim"]),
        "num_twoway_layers": int(spec["head"]["num_twoway_layers"]),
        "num_heads": int(spec["head"]["num_heads"]),
        "dropout": float(spec["head"]["dropout"]),
    }


def _new_head(
    latent_dim: int,
    condition_dim: int,
    spec: dict[str, Any],
    device: torch.device,
) -> ClarityOutcomeAdapter:
    return ClarityOutcomeAdapter(
        latent_dim=latent_dim,
        condition_dim=condition_dim,
        **_head_config(spec),
    ).to(device)


def _selected(values: Iterable[Any], selected: Any | None) -> list[Any]:
    values = list(values)
    if selected is None:
        return values
    if selected not in values:
        raise ClarityExperimentError(f"selected value is not registered: {selected}")
    return [selected]


def _cache_key(variant: str, fold: int, seed: int, horizon: int, split: str) -> str:
    return f"{variant}/fold{fold}/seed{seed}/H{horizon}/{split}"


def _head_key(variant: str, fold: int, seed: int, horizon: int) -> str:
    return f"{variant}/fold{fold}/seed{seed}/H{horizon}"


def _split_patient_ids(
    bundle: DataBundle,
    horizons: Sequence[int],
    fold: int,
    seed: int,
) -> tuple[list[str], list[str]]:
    eligible = sorted({
        ref.patient_id
        for horizon in horizons
        for ref in _refs(bundle, "train", int(horizon), labels=True)
    }, key=lambda pid: hashlib.sha256(f"{seed}:{fold}:{pid}".encode()).hexdigest())
    if len(eligible) < 5:
        raise ClarityExperimentError(f"fold {fold} needs at least five outcome-eligible training patients")
    stop_count = max(1, round(0.2 * len(eligible)))
    stop = sorted(eligible[:stop_count])
    fit = sorted(eligible[stop_count:])
    if not fit or not stop or set(fit) & set(stop):
        raise ClarityExperimentError("invalid outcome fit/stop patient split")
    return fit, stop


def _row_alignment_hash(rows: Sequence[dict[str, Any]]) -> str:
    return _sha256_json([{
        "patient_id": row["patient_id"],
        "start_index": int(row["start_index"]),
        "target_index": int(row["target_index"]),
        "horizon": int(row["horizon"]),
        "time": float(row["time"]),
        "event": int(row["event"]),
        "pre_sha256": hashlib.sha256(row["pre"].numpy().tobytes()).hexdigest(),
        "condition_sha256": hashlib.sha256(row["condition"].numpy().tobytes()).hexdigest(),
    } for row in rows])


@torch.no_grad()
def prepare_rows(
    world: nn.Module,
    bundle: DataBundle,
    split: str,
    refs: Sequence[WindowRef],
    feature_batch_size: int,
    *,
    protocol: str = "development_reuse",
    variant: str = "baseline",
    fold: int = 0,
    seed: int = 0,
    device: torch.device | None = None,
) -> list[dict[str, Any]]:
    """Freeze a world and extract the exact observed-start/predicted-endpoint rows."""
    device = device or torch.device("cpu")
    refs = list(refs)
    if not refs:
        return []
    horizons = {int(ref.horizon) for ref in refs}
    if len(horizons) != 1:
        raise ClarityExperimentError("prepare_rows requires a single horizon")
    horizon = next(iter(horizons))
    world.eval().requires_grad_(False)
    rows: list[dict[str, Any]] = []
    loader = DataLoader(
        DynamicsDataset(bundle, refs),
        batch_size=int(feature_batch_size),
        shuffle=False,
        collate_fn=collate_dynamics,
    )
    for raw in loader:
        batch = _to_device(raw, device)
        rollout = world.rollout(
            batch["z0"], batch["actions"], batch["deltas"], batch["context"],
            batch["clinical_mask"], batch["history0"], batch["step_mask"],
        )
        member_post = rollout.states[:, :, horizon, :]
        post_mean = member_post.mean(0)
        latent_mse = ((post_mean - batch["target"]) ** 2).mean(-1)
        disagreement = member_post.var(0, unbiased=False).mean(-1)
        for index, patient_id in enumerate(raw["patient_id"]):
            target_index = int(raw["target_index"][index])
            start_index = target_index - horizon
            trajectory = bundle.trajectories[str(patient_id)]
            elapsed = trajectory.days[target_index] - trajectory.days[start_index]
            if not bool(torch.isfinite(elapsed)) or float(elapsed) <= 0:
                raise ClarityExperimentError("forecast row has invalid elapsed time")
            elapsed_feature = torch.log1p(elapsed.float()) / math.log(366.0)
            condition = torch.cat((
                trajectory.clinical,
                trajectory.clinical_mask,
                trajectory.histories[target_index],
                elapsed_feature.reshape(1),
            )).float()
            pre = batch["z0"][index].detach().cpu().float().contiguous()
            members = member_post[:, index].detach().cpu().float().contiguous()
            true_post = batch["target"][index].detach().cpu().float().contiguous()
            rows.append({
                "record_id": (
                    f"clarity_feature:{protocol}:{variant}:fold{fold}:seed{seed}:H{horizon}:"
                    f"{patient_id}:{start_index}:{target_index}"
                ),
                "variant": variant,
                "fold": int(fold),
                "seed": int(seed),
                "split": split,
                "patient_id": str(patient_id),
                "start_index": start_index,
                "target_index": target_index,
                "start_timepoint": str(raw["start_timepoint"][index]),
                "target_timepoint": str(raw["target_timepoint"][index]),
                "horizon": horizon,
                "pre": pre,
                "post_mean": post_mean[index].detach().cpu().float().contiguous(),
                "post_members": members,
                "true_post": true_post,
                "condition": condition.cpu().contiguous(),
                "time": float(trajectory.survival_time[target_index]),
                "event": int(trajectory.survival_event[target_index]),
                "latent_mse": float(latent_mse[index].cpu()),
                "latent_disagreement": float(disagreement[index].cpu()),
                "uncertainty_available": int(members.shape[0]) > 1,
            })
    return rows


def _initialize_artifacts(
    config: dict[str, Any],
    spec: dict[str, Any],
) -> tuple[RunArtifacts, dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    artifacts = RunArtifacts(config["paths"]["output_root"], str(spec["run"]), version="v3")
    signature = _sha256_json(spec)
    existing = read_json(artifacts.path("run.json"))
    if existing is not None and existing.get("signature") != signature:
        raise ClarityExperimentError("output run already exists with a different configuration")
    run = existing or {
        "schema_version": SCHEMA,
        "run": spec["run"],
        "signature": signature,
        "repository_commit": _git_commit(config["paths"]["project_root"]),
        "clarity_source_commit": UPSTREAM_COMMIT,
        "clarity_survival_module_blob": UPSTREAM_BLOB,
        "protocol": spec["protocol"],
        "evidence_scope": (
            "development comparison; source worlds selected on source validation"
            if spec["protocol"] == "development_reuse"
            else "locked train/validation/test confirmation"
        ),
        "world_train_feature_mode": "in_sample",
        "formal_test_untouched": True,
        "source_world_selected_on_report_cohort": spec["protocol"] == "development_reuse",
        "administrative_censor_days": spec["administrative_censor_days"],
        "config": copy.deepcopy(spec),
        "patient_splits": {},
        "sample_key_hashes": {},
        "completed_tasks": [],
    }
    if run.get("schema_version") != SCHEMA:
        raise ClarityExperimentError("unexpected downstream run schema")
    models = (
        read_torch(artifacts.path("models.pt"), safe=True)
        if artifacts.path("models.pt").exists()
        else {"schema_version": MODEL_SCHEMA, "heads": {}, "worlds": {}}
    )
    cache = (
        read_torch(artifacts.path("last.pt"), safe=True)
        if artifacts.path("last.pt").exists()
        else {"schema_version": CACHE_SCHEMA, "rows": {}, "alignment": {}}
    )
    metrics = read_json(artifacts.path("metrics.json"), {
        "schema_version": SCHEMA,
        "main": {"tasks": {}, "summary": {}},
        "paired": {},
        "uncertainty": {"tasks": {}},
        "training": {},
        "limitations": [
            "The development-reuse source worlds were selected using their source-fold validation cohorts.",
            "Outcome fit features are in-sample world forecasts whereas report features are out-of-sample.",
            "Single-token global latents preserve the CLARITY computation but not its original multi-token MRI representation.",
            "This is factual prognostic evaluation and does not establish causal treatment benefit.",
        ],
    })
    if models.get("schema_version") != MODEL_SCHEMA or cache.get("schema_version") != CACHE_SCHEMA:
        raise ClarityExperimentError("unexpected downstream model/cache schema")
    return artifacts, run, models, cache, metrics


def _save_run(artifacts: RunArtifacts, run: dict[str, Any]) -> None:
    run["completed_tasks"] = sorted(set(run.get("completed_tasks", [])))
    write_json(artifacts.path("run.json"), run)


def _mark(run: dict[str, Any], task: str) -> None:
    run.setdefault("completed_tasks", []).append(task)


def _run_prepare(
    config: dict[str, Any],
    spec: dict[str, Any],
    artifacts: RunArtifacts,
    run: dict[str, Any],
    models: dict[str, Any],
    cache: dict[str, Any],
    device: torch.device,
    *,
    selected_fold: int | None,
    selected_seed: int | None,
    selected_horizon: int | None,
    selected_variant: str | None,
    evaluate_test: bool,
) -> None:
    if spec["protocol"] == "original_holdout":
        return _run_prepare_holdout(
            config, spec, artifacts, run, models, cache, device,
            selected_fold=selected_fold, selected_seed=selected_seed,
            selected_horizon=selected_horizon, selected_variant=selected_variant,
            evaluate_test=evaluate_test,
        )
    base, source_cache, source_run, source_models = _source(config, spec)
    source_path = RunArtifacts(
        config["paths"]["output_root"], spec["world_run"], version="v2"
    ).path("models.pt")
    if "source_world_sha256" not in run:
        run["source_world_sha256"] = sha256_file(source_path)
    run["base_data_signature"] = base["data_signature"]
    run["source_world_signature"] = source_run.get("signature")
    formal_test = [str(x) for x in base["split"]["test"]]
    folds = source_run["folds"]
    for fold_index in _selected(range(len(folds)), selected_fold):
        fold = folds[fold_index]
        bundle = _fold_bundle(source_cache, config, source_models, fold, formal_test)
        if bundle.latent_dim != int(spec["expected_latent_dim"]):
            raise ClarityExperimentError(
                f"latent_dim {bundle.latent_dim} != expected {spec['expected_latent_dim']}"
            )
        fit_ids, stop_ids = _split_patient_ids(
            bundle, spec["horizons"], fold_index, int(spec["training"]["outcome_split_seed"]),
        )
        report_ids = sorted({
            ref.patient_id
            for horizon in spec["horizons"]
            for ref in _refs(bundle, "validation", int(horizon), labels=True)
        })
        if set(fit_ids) & set(stop_ids) or (set(fit_ids) | set(stop_ids)) & set(report_ids):
            raise ClarityExperimentError("fit/stop/report patients overlap")
        split_manifest = {
            "fit_ids": fit_ids,
            "stop_ids": stop_ids,
            "report_ids": report_ids,
            "fit_ids_hash": _sha256_json(fit_ids),
            "stop_ids_hash": _sha256_json(stop_ids),
            "report_ids_hash": _sha256_json(report_ids),
            "preprocessing_train_ids_hash": _sha256_json(sorted(fold["training_ids"])),
        }
        prior_split = run["patient_splits"].get(f"fold{fold_index}")
        if prior_split is not None and prior_split != split_manifest:
            raise ClarityExperimentError("stored outcome patient split changed")
        run["patient_splits"][f"fold{fold_index}"] = split_manifest
        for seed in _selected(spec["seeds"], selected_seed):
            for variant in _selected(spec["variants"], selected_variant):
                world = _world(source_models, bundle, config, variant, fold_index, seed, device)
                for horizon in _selected(spec["horizons"], selected_horizon):
                    for split in ("train", "validation"):
                        key = _cache_key(variant, fold_index, seed, horizon, split)
                        if key in cache["rows"]:
                            continue
                        refs = _refs(bundle, split, int(horizon), labels=True)
                        rows = prepare_rows(
                            world, bundle, split, refs, int(spec["feature_batch_size"]),
                            protocol=spec["protocol"], variant=variant, fold=fold_index,
                            seed=seed, device=device,
                        )
                        alignment_key = f"fold{fold_index}/H{horizon}/{split}"
                        alignment = _row_alignment_hash(rows)
                        expected = cache["alignment"].get(alignment_key)
                        if expected is not None and expected != alignment:
                            raise ClarityExperimentError(
                                f"variant forecast labels/features are misaligned: {alignment_key}"
                            )
                        cache["alignment"][alignment_key] = alignment
                        cache["rows"][key] = rows
                        run["sample_key_hashes"][alignment_key] = alignment
                        print(f"prepared {key} rows={len(rows)}", flush=True)
    write_torch(artifacts.path("last.pt"), cache)
    _mark(run, "prepare")
    _save_run(artifacts, run)



def _rows_for_task(
    cache: dict[str, Any],
    run: dict[str, Any],
    variant: str,
    fold: int,
    seed: int,
    horizon: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    train_key = _cache_key(variant, fold, seed, horizon, "train")
    stop_split = "train" if run["protocol"] == "development_reuse" else "validation"
    report_split = "validation" if run["protocol"] == "development_reuse" else "test"
    stop_key = _cache_key(variant, fold, seed, horizon, stop_split)
    report_key = _cache_key(variant, fold, seed, horizon, report_split)
    if train_key not in cache["rows"] or stop_key not in cache["rows"]:
        raise ClarityExperimentError(f"forecast cache missing for {_head_key(variant, fold, seed, horizon)}")
    partition = run["patient_splits"].get(f"fold{fold}")
    if partition is None:
        raise ClarityExperimentError(f"patient partition missing for fold {fold}")
    fit_ids, stop_ids = set(partition["fit_ids"]), set(partition["stop_ids"])
    train_rows = cache["rows"][train_key]
    fit = [row for row in train_rows if row["patient_id"] in fit_ids]
    stop_rows = cache["rows"][stop_key]
    stop = [row for row in stop_rows if row["patient_id"] in stop_ids]
    report = cache["rows"].get(report_key, [])
    if {row["patient_id"] for row in fit} & {row["patient_id"] for row in stop}:
        raise ClarityExperimentError("fit and stop rows overlap")
    if ({row["patient_id"] for row in fit} | {row["patient_id"] for row in stop}) & {
        row["patient_id"] for row in report
    }:
        raise ClarityExperimentError("outcome training and report rows overlap")
    return fit, stop, report


def _stack_rows(rows: Sequence[dict[str, Any]], device: torch.device) -> dict[str, Tensor]:
    if not rows:
        raise ClarityExperimentError("cannot stack an empty outcome cohort")
    return {
        "pre": torch.stack([row["pre"] for row in rows]).to(device),
        "post": torch.stack([row["post_mean"] for row in rows]).to(device),
        "condition": torch.stack([row["condition"] for row in rows]).to(device),
        "time": torch.tensor([row["time"] for row in rows], dtype=torch.float32, device=device),
        "event": torch.tensor([row["event"] for row in rows], dtype=torch.long, device=device),
    }


def _snapshot(
    model: ClarityOutcomeAdapter,
    batch: dict[str, Tensor],
    spec: dict[str, Any],
) -> dict[str, Any]:
    model.eval()
    with torch.no_grad():
        risk, logit = model(batch["pre"], batch["post"], batch["condition"])
        losses = outcome_loss(
            risk, logit, batch["time"], batch["event"],
            tau=float(spec["training"]["probability_horizon_days"]),
            w_cox=float(spec["training"]["cox_weight"]),
            w_bce=float(spec["training"]["bce_weight"]),
        )
    return {
        "total": float(losses["total"].cpu()),
        "cox_partial_nll": float(losses["cox_partial_nll"].cpu()),
        "bce_identifiable": float(losses["bce_identifiable"].cpu()),
        "n_events": int(losses["n_events"]),
        "n_identifiable": int(losses["n_identifiable"]),
        "risk_variance": float(risk.var(unbiased=False).cpu()),
        "survival_logit_variance": float(logit.var(unbiased=False).cpu()),
    }


def _cpu_state(model: nn.Module) -> dict[str, Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def train_one_head(
    fit_rows: Sequence[dict[str, Any]],
    stop_rows: Sequence[dict[str, Any]],
    spec: dict[str, Any],
    seed: int,
    *,
    device: torch.device | None = None,
    init_state: dict[str, Tensor] | None = None,
) -> tuple[dict[str, Tensor], dict[str, Any]]:
    """Train one outcome on its own frozen forecasts using full-cohort loss."""
    device = device or torch.device("cpu")
    fit_rows, stop_rows = list(fit_rows), list(stop_rows)
    if not fit_rows or not stop_rows:
        raise ClarityExperimentError("head training requires non-empty fit and stop cohorts")
    latent_dim = int(fit_rows[0]["pre"].numel())
    condition_dim = int(fit_rows[0]["condition"].numel())
    fit = _stack_rows(fit_rows, device)
    stop = _stack_rows(stop_rows, device)
    if int(fit["event"].sum()) == 0:
        raise ClarityExperimentError("outcome fit cohort has no event")
    _, stop_valid = fixed_time_survival_targets(
        stop["time"], stop["event"], float(spec["training"]["probability_horizon_days"]),
    )
    if int(stop["event"].sum()) == 0 and not bool(stop_valid.any()):
        raise ClarityExperimentError("outcome stop cohort has no usable supervision")

    _seed_all(int(seed))
    model = _new_head(latent_dim, condition_dim, spec, device)
    if init_state is not None:
        model.load_state_dict(init_state)
    optimizer = AdamW(
        model.parameters(),
        lr=float(spec["training"]["lr"]),
        weight_decay=float(spec["training"]["weight_decay"]),
    )
    initial_stop = _snapshot(model, stop, spec)
    initial_fit = _snapshot(model, fit, spec)
    best_score = float(initial_stop["total"])
    best_epoch = 0
    best_state = _cpu_state(model)
    history: list[dict[str, Any]] = [{
        "epoch": 0,
        "train": initial_fit,
        "stop": initial_stop,
        "gradient_norm": None,
    }]
    stale = 0
    for epoch in range(1, int(spec["training"]["max_epochs"]) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        risk, logit = model(fit["pre"], fit["post"], fit["condition"])
        losses = outcome_loss(
            risk, logit, fit["time"], fit["event"],
            tau=float(spec["training"]["probability_horizon_days"]),
            w_cox=float(spec["training"]["cox_weight"]),
            w_bce=float(spec["training"]["bce_weight"]),
        )
        total = losses["total"]
        if not isinstance(total, Tensor) or not bool(torch.isfinite(total)):
            raise ClarityExperimentError("non-finite outcome training loss")
        total.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(spec["training"]["grad_clip"]),
        )
        if not bool(torch.isfinite(torch.as_tensor(gradient_norm))):
            raise ClarityExperimentError("non-finite outcome gradient")
        optimizer.step()
        train_snapshot = _snapshot(model, fit, spec)
        stop_snapshot = _snapshot(model, stop, spec)
        history.append({
            "epoch": epoch,
            "train": train_snapshot,
            "stop": stop_snapshot,
            "gradient_norm": float(torch.as_tensor(gradient_norm).cpu()),
        })
        score = float(stop_snapshot["total"])
        if score < best_score:
            best_score = score
            best_epoch = epoch
            best_state = _cpu_state(model)
            stale = 0
        else:
            stale += 1
        if stale >= int(spec["training"]["patience"]):
            break
    metadata = {
        "available": True,
        "best_epoch": best_epoch,
        "best_stop_loss": best_score,
        "epochs_ran": len(history) - 1,
        "fit_n": len(fit_rows),
        "fit_events": int(fit["event"].sum().item()),
        "stop_n": len(stop_rows),
        "stop_events": int(stop["event"].sum().item()),
        "history": history,
    }
    return best_state, metadata


def _initial_state(
    rows: Sequence[dict[str, Any]],
    spec: dict[str, Any],
    initialization_seed: int,
) -> tuple[dict[str, Tensor], str, str]:
    _seed_all(initialization_seed)
    model = _new_head(
        int(rows[0]["pre"].numel()), int(rows[0]["condition"].numel()),
        spec, torch.device("cpu"),
    )
    state = _cpu_state(model)
    return state, _tensor_state_hash(state), _architecture_hash(model)



def _run_train(
    spec: dict[str, Any],
    artifacts: RunArtifacts,
    run: dict[str, Any],
    models: dict[str, Any],
    cache: dict[str, Any],
    metrics: dict[str, Any],
    device: torch.device,
    *,
    selected_fold: int | None,
    selected_seed: int | None,
    selected_horizon: int | None,
    selected_variant: str | None,
) -> None:
    fold_values = sorted(int(key.removeprefix("fold")) for key in run["patient_splits"])
    if not fold_values:
        raise ClarityExperimentError("prepare must run before train")
    for fold in _selected(fold_values, selected_fold):
        changed = False
        for seed in _selected(spec["seeds"], selected_seed):
            for horizon in _selected(spec["horizons"], selected_horizon):
                baseline_fit, _, _ = _rows_for_task(cache, run, "baseline", fold, seed, horizon)
                if not baseline_fit:
                    raise ClarityExperimentError(f"no fit rows for fold{fold}/seed{seed}/H{horizon}")
                initialization_seed = int(seed) * 100_000 + int(fold) * 100 + int(horizon)
                init_state, init_hash, architecture_hash = _initial_state(
                    baseline_fit, spec, initialization_seed,
                )
                for variant in _selected(spec["variants"], selected_variant):
                    key = _head_key(variant, fold, seed, horizon)
                    if key in models["heads"]:
                        entry = models["heads"][key]
                        if entry.get("init_state_hash") != init_hash:
                            raise ClarityExperimentError(f"stored initialization changed: {key}")
                        continue
                    fit, stop, _ = _rows_for_task(cache, run, variant, fold, seed, horizon)
                    partition = run["patient_splits"][f"fold{fold}"]
                    base_entry = {
                        "variant": variant,
                        "fold": fold,
                        "seed": seed,
                        "horizon": horizon,
                        "init_state_hash": init_hash,
                        "architecture_hash": architecture_hash,
                        "fit_ids_hash": partition["fit_ids_hash"],
                        "stop_ids_hash": partition["stop_ids_hash"],
                        "world_key": f"{variant}/fold{fold}/seed{seed}",
                        "initialization_seed": initialization_seed,
                    }
                    try:
                        state, training = train_one_head(
                            fit, stop, spec, initialization_seed,
                            device=device, init_state=init_state,
                        )
                    except ClarityExperimentError as exc:
                        state = {}
                        training = {"available": False, "reason": str(exc)}
                    models["heads"][key] = {
                        **base_entry,
                        "available": bool(training["available"]),
                        "state": state,
                        "best_epoch": training.get("best_epoch"),
                        "final_state_hash": _tensor_state_hash(state) if state else None,
                    }
                    metrics["training"][key] = {**base_entry, **training}
                    print(
                        f"trained {key} available={training['available']} "
                        f"best_epoch={training.get('best_epoch')}", flush=True,
                    )
                    changed = True
        if changed:
            write_torch(artifacts.path("models.pt"), models)
            write_json(artifacts.path("metrics.json"), metrics)
    _mark(run, "train")
    _save_run(artifacts, run)


@torch.no_grad()
def evaluate_one_head(
    head: ClarityOutcomeAdapter,
    report_rows: Sequence[dict[str, Any]],
    reference_rows: Sequence[dict[str, Any]],
    spec: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    report_rows, reference_rows = list(report_rows), list(reference_rows)
    if not report_rows:
        return {"available": False, "reason": "empty_report_cohort"}, []
    device = next(head.parameters()).device
    batch = _stack_rows(report_rows, device)
    head.eval()
    risk, logit = head(batch["pre"], batch["post"], batch["condition"])
    survival = torch.sigmoid(logit)
    predictions = []
    for index, source in enumerate(report_rows):
        predictions.append({
            **source,
            "risk_score": float(risk[index].cpu()),
            "survival_logit365": float(logit[index].cpu()),
            "survival365": float(survival[index].cpu()),
        })
    result = evaluate_cohort(
        predictions, reference_rows,
        float(spec["training"]["probability_horizon_days"]),
    )
    return result, predictions


def _prediction_record(
    source: dict[str, Any],
    protocol: str,
    aggregation: str,
    head_key: str,
) -> dict[str, Any]:
    record_id = (
        f"clarity_out:{protocol}:{source['variant']}:fold{source['fold']}:"
        f"seed{source['seed']}:H{source['horizon']}:{source['patient_id']}:"
        f"{source['start_index']}:{source['target_index']}:{aggregation}"
    )
    output = {
        "record_id": record_id,
        "kind": "factual_downstream",
        "variant": source["variant"],
        "fold": int(source["fold"]),
        "seed": int(source["seed"]),
        "horizon": int(source["horizon"]),
        "patient_id": source["patient_id"],
        "start_index": int(source["start_index"]),
        "target_index": int(source["target_index"]),
        "start_timepoint": source["start_timepoint"],
        "target_timepoint": source["target_timepoint"],
        "label_anchor": "target_landmark",
        "time": float(source["time"]),
        "event": int(source["event"]),
        "risk_score": float(source["risk_score"]),
        "survival_logit365": float(source["survival_logit365"]),
        "survival365": float(source["survival365"]),
        "aggregation": aggregation,
        "latent_mse": float(source["latent_mse"]),
        "latent_disagreement": float(source["latent_disagreement"]),
        "head_key": head_key,
    }
    for field in (
        "member_survival365", "member_risk_score", "survival365_mean_probability",
        "survival365_mean_latent", "probability_std",
    ):
        if field in source:
            output[field] = source[field]
    return output


def _run_evaluate(
    spec: dict[str, Any],
    artifacts: RunArtifacts,
    run: dict[str, Any],
    models: dict[str, Any],
    cache: dict[str, Any],
    metrics: dict[str, Any],
    device: torch.device,
    *,
    selected_fold: int | None,
    selected_seed: int | None,
    selected_horizon: int | None,
    selected_variant: str | None,
) -> None:
    if not models["heads"]:
        raise ClarityExperimentError("train must run before evaluate")
    predictions = []
    fold_values = sorted(int(key.removeprefix("fold")) for key in run["patient_splits"])
    for fold in _selected(fold_values, selected_fold):
        for seed in _selected(spec["seeds"], selected_seed):
            for horizon in _selected(spec["horizons"], selected_horizon):
                for variant in _selected(spec["variants"], selected_variant):
                    key = _head_key(variant, fold, seed, horizon)
                    entry = models["heads"].get(key)
                    if entry is None:
                        raise ClarityExperimentError(f"head checkpoint missing: {key}")
                    base = {"variant": variant, "fold": fold, "seed": seed, "horizon": horizon}
                    if not entry.get("available"):
                        metrics["main"]["tasks"][key] = {
                            **base, "available": False,
                            "reason": metrics["training"].get(key, {}).get("reason", "head unavailable"),
                        }
                        continue
                    fit, _, report_rows = _rows_for_task(cache, run, variant, fold, seed, horizon)
                    if not report_rows:
                        metrics["main"]["tasks"][key] = {
                            **base, "available": False, "reason": "empty_report_cohort",
                        }
                        continue
                    head = _new_head(
                        int(report_rows[0]["pre"].numel()),
                        int(report_rows[0]["condition"].numel()), spec, device,
                    )
                    head.load_state_dict(entry["state"])
                    result, predicted = evaluate_one_head(head, report_rows, fit, spec)
                    metrics["main"]["tasks"][key] = {**base, **result}
                    predictions.extend(
                        _prediction_record(row, spec["protocol"], "mean_latent", key)
                        for row in predicted
                    )
                    print(
                        f"evaluated {key} c_index={result.get('c_index_risk', {}).get('value')} "
                        f"brier={result.get('ipcw_brier365', {}).get('value')}", flush=True,
                    )
    if predictions:
        upsert_jsonl(artifacts.path("predictions.jsonl"), predictions)
    metrics["main"]["summary"] = aggregate_main(metrics["main"]["tasks"])
    metrics["paired"] = paired_comparisons(metrics["main"]["tasks"])
    write_json(artifacts.path("metrics.json"), metrics)
    if run["protocol"] == "original_holdout":
        run["formal_test_evaluated_once"] = True
    _mark(run, "evaluate")
    _save_run(artifacts, run)



@torch.no_grad()
def _member_predictions(
    head: ClarityOutcomeAdapter,
    rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = list(rows)
    if not rows:
        return []
    device = next(head.parameters()).device
    pre = torch.stack([row["pre"] for row in rows]).to(device)
    condition = torch.stack([row["condition"] for row in rows]).to(device)
    post_mean = torch.stack([row["post_mean"] for row in rows]).to(device)
    members = torch.stack([row["post_members"] for row in rows]).transpose(0, 1).to(device)
    if members.ndim != 3 or members.shape[1] != len(rows):
        raise ClarityExperimentError("member forecast tensor must be [M,N,D]")
    if members.shape[0] < 2:
        raise ClarityExperimentError("uncertainty is unavailable for a single-member world")
    head.eval()
    mean_latent_risk, mean_latent_logit = head(pre, post_mean, condition)
    member_risk, member_logit = [], []
    for member_index in range(members.shape[0]):
        risk, logit = head(pre, members[member_index], condition)
        member_risk.append(risk)
        member_logit.append(logit)
    risk_tensor = torch.stack(member_risk)
    logit_tensor = torch.stack(member_logit)
    summary = summarize_member_logits(risk_tensor, logit_tensor)
    mean_probability = summary["mean_survival"].clamp(1e-7, 1.0 - 1e-7)
    equivalent_logit = torch.logit(mean_probability)
    result = []
    for index, source in enumerate(rows):
        result.append({
            **source,
            "risk_score": float(summary["mean_risk_score"][index].cpu()),
            "survival_logit365": float(equivalent_logit[index].cpu()),
            "survival365": float(mean_probability[index].cpu()),
            "member_survival365": [
                float(value) for value in summary["member_survival"][:, index].cpu()
            ],
            "member_risk_score": [
                float(value) for value in risk_tensor[:, index].cpu()
            ],
            "survival365_mean_probability": float(mean_probability[index].cpu()),
            "survival365_mean_latent": float(torch.sigmoid(mean_latent_logit[index]).cpu()),
            "risk_score_mean_latent": float(mean_latent_risk[index].cpu()),
            "probability_std": float(summary["probability_std"][index].cpu()),
        })
    return result


def _fixed_thresholds(
    stop_predictions: Sequence[dict[str, Any]],
    coverages: Sequence[float],
) -> dict[float, float]:
    if not stop_predictions:
        raise ClarityExperimentError("fixed uncertainty thresholds need outcome-stop predictions")
    values = torch.tensor(
        [float(row["probability_std"]) for row in stop_predictions], dtype=torch.float64,
    )
    return {
        float(coverage): float(torch.quantile(values, float(coverage)).item())
        for coverage in coverages
    }


def run_uncertainty(
    head: ClarityOutcomeAdapter,
    report_rows: Sequence[dict[str, Any]],
    stop_rows: Sequence[dict[str, Any]],
    reference_rows: Sequence[dict[str, Any]],
    spec: dict[str, Any],
    *,
    seed: int = 0,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Forward-only member aggregation and selective reliability analysis."""
    predicted = _member_predictions(head, report_rows)
    stop_predicted = _member_predictions(head, stop_rows)
    tau = float(spec["training"]["probability_horizon_days"])
    comparison = evaluate_cohort(predicted, reference_rows, tau)
    uncertainty = spec["uncertainty"]
    thresholds = (
        _fixed_thresholds(stop_predicted, uncertainty["coverages"])
        if uncertainty["threshold_mode"] == "fixed_threshold"
        else None
    )
    selective = selective_ipcw(
        predicted,
        reference_rows,
        uncertainty["coverages"],
        seed,
        random_reference_repeats=int(uncertainty["random_reference_repeats"]),
        threshold_mode=str(uncertainty["threshold_mode"]),
        thresholds=thresholds,
        tau=tau,
    )
    return {
        "available": True,
        "aggregation": "mean_probability",
        "inference_only": True,
        "uncertainty_definition": "dynamics-induced survival-probability disagreement",
        "mean_probability_metrics": comparison,
        "selective_ipcw": selective,
        "fixed_thresholds": thresholds,
    }, predicted


def _run_uncertainty(
    spec: dict[str, Any],
    artifacts: RunArtifacts,
    run: dict[str, Any],
    models: dict[str, Any],
    cache: dict[str, Any],
    metrics: dict[str, Any],
    device: torch.device,
    *,
    selected_fold: int | None,
    selected_seed: int | None,
    selected_horizon: int | None,
    selected_variant: str | None,
) -> None:
    if not bool(spec["uncertainty"]["enabled"]):
        _mark(run, "uncertainty_disabled")
        _save_run(artifacts, run)
        return
    requested_variants = _selected(spec["variants"], selected_variant)
    variants = [variant for variant in requested_variants if variant in ENSEMBLE_VARIANTS]
    prediction_records = []
    fold_values = sorted(int(key.removeprefix("fold")) for key in run["patient_splits"])
    for fold in _selected(fold_values, selected_fold):
        for seed in _selected(spec["seeds"], selected_seed):
            for horizon in _selected(spec["horizons"], selected_horizon):
                for variant in variants:
                    key = _head_key(variant, fold, seed, horizon)
                    entry = models["heads"].get(key)
                    if entry is None or not entry.get("available"):
                        metrics["uncertainty"]["tasks"][key] = {
                            "available": False, "reason": "trained head unavailable",
                            "variant": variant, "fold": fold, "seed": seed, "horizon": horizon,
                        }
                        continue
                    fit, stop, report_rows = _rows_for_task(
                        cache, run, variant, fold, seed, horizon,
                    )
                    if not report_rows:
                        metrics["uncertainty"]["tasks"][key] = {
                            "available": False, "reason": "empty report cohort",
                            "variant": variant, "fold": fold, "seed": seed, "horizon": horizon,
                        }
                        continue
                    head = _new_head(
                        int(report_rows[0]["pre"].numel()),
                        int(report_rows[0]["condition"].numel()), spec, device,
                    )
                    head.load_state_dict(entry["state"])
                    result, predicted = run_uncertainty(
                        head, report_rows, stop, fit, spec,
                        seed=int(seed) * 10_000 + int(fold) * 10 + int(horizon),
                    )
                    metrics["uncertainty"]["tasks"][key] = {
                        "variant": variant, "fold": fold, "seed": seed,
                        "horizon": horizon, **result,
                    }
                    prediction_records.extend(
                        _prediction_record(row, spec["protocol"], "mean_probability", key)
                        for row in predicted
                    )
                    print(f"uncertainty {key} rows={len(predicted)}", flush=True)
    if prediction_records:
        upsert_jsonl(artifacts.path("predictions.jsonl"), prediction_records)
    metrics["uncertainty"]["summary"] = aggregate_uncertainty(metrics["uncertainty"]["tasks"])
    write_json(artifacts.path("metrics.json"), metrics)
    _mark(run, "uncertainty")
    _save_run(artifacts, run)


def _run_report(
    artifacts: RunArtifacts,
    run: dict[str, Any],
    metrics: dict[str, Any],
) -> None:
    metrics["main"]["summary"] = aggregate_main(metrics["main"]["tasks"])
    metrics["uncertainty"]["summary"] = aggregate_uncertainty(metrics["uncertainty"]["tasks"])
    metrics["paired"] = paired_comparisons(metrics["main"]["tasks"])
    write_json(artifacts.path("metrics.json"), metrics)
    write_text(artifacts.path("report.md"), build_report(metrics, run))
    _mark(run, "report")
    _save_run(artifacts, run)
    artifacts.assert_flat()





def _holdout_bundle(
    config: dict[str, Any],
    spec: dict[str, Any],
    models: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], DataBundle]:
    base, source_cache, _, formal_test = _load_base(config, spec)
    split = {
        name: [str(value) for value in base["split"][name]]
        for name in ("train", "validation", "test")
    }
    if split["test"] != formal_test:
        raise ClarityExperimentError("base formal-test split changed")
    preprocessing = models.get("holdout_preprocessing")
    if preprocessing is None:
        bundle, preprocessing = _build_fold_bundle(
            source_cache, config, split["train"], split["validation"], split["test"],
        )
        models["holdout_preprocessing"] = preprocessing
    else:
        bundle = load_bundle(source_cache, config, split, preprocessing)
    return base, split, bundle


def _run_holdout_world(
    config: dict[str, Any],
    spec: dict[str, Any],
    artifacts: RunArtifacts,
    run: dict[str, Any],
    models: dict[str, Any],
    device: torch.device,
    *,
    selected_seed: int | None,
    selected_variant: str | None,
) -> None:
    if spec["protocol"] != "original_holdout":
        raise ClarityExperimentError("world stage is only valid for original_holdout")
    base, split, bundle = _holdout_bundle(config, spec, models)
    if bundle.latent_dim != int(spec["expected_latent_dim"]):
        raise ClarityExperimentError("holdout latent dimension differs from the locked config")
    training_hash = _sha256_json(sorted(split["train"]))
    run["base_data_signature"] = base["data_signature"]
    run["patient_splits"]["fold0"] = {
        "fit_ids": split["train"],
        "stop_ids": split["validation"],
        "report_ids": [],
        "fit_ids_hash": training_hash,
        "stop_ids_hash": _sha256_json(sorted(split["validation"])),
        "report_ids_hash": None,
        "preprocessing_train_ids_hash": training_hash,
    }
    for variant in _selected(spec["variants"], selected_variant):
        _, member_count = variant_spec(variant, int(config["world"]["ensemble_size"]))
        for seed in _selected(spec["seeds"], selected_seed):
            key = f"{variant}/fold0/seed{seed}"
            entry = models["worlds"].setdefault(key, {
                "member_states": [],
                "best_epochs": [],
                "member_seeds": [],
                "member_histories": [],
                "complete": False,
                "training_patient_hash": training_hash,
            })
            if entry["training_patient_hash"] != training_hash:
                raise ClarityExperimentError("holdout world training patient set changed")
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
                write_torch(artifacts.path("models.pt"), models)
                print(
                    f"holdout world {key} member={member} best_epoch={best_epoch}",
                    flush=True,
                )
    _mark(run, "world")
    _save_run(artifacts, run)


def _run_prepare_holdout(
    config: dict[str, Any],
    spec: dict[str, Any],
    artifacts: RunArtifacts,
    run: dict[str, Any],
    models: dict[str, Any],
    cache: dict[str, Any],
    device: torch.device,
    *,
    selected_fold: int | None,
    selected_seed: int | None,
    selected_horizon: int | None,
    selected_variant: str | None,
    evaluate_test: bool,
) -> None:
    if selected_fold not in (None, 0):
        raise ClarityExperimentError("original_holdout has one locked split, addressed as fold 0")
    if not models.get("worlds"):
        raise ClarityExperimentError("run --stage world before holdout prepare")
    base, split, bundle = _holdout_bundle(config, spec, models)
    training_hash = _sha256_json(sorted(split["train"]))
    run["base_data_signature"] = base["data_signature"]
    run["patient_splits"]["fold0"] = {
        "fit_ids": split["train"],
        "stop_ids": split["validation"],
        "report_ids": split["test"] if evaluate_test else [],
        "fit_ids_hash": training_hash,
        "stop_ids_hash": _sha256_json(sorted(split["validation"])),
        "report_ids_hash": _sha256_json(sorted(split["test"])) if evaluate_test else None,
        "preprocessing_train_ids_hash": training_hash,
    }
    requested_splits = ["train", "validation"] + (["test"] if evaluate_test else [])
    for seed in _selected(spec["seeds"], selected_seed):
        for variant in _selected(spec["variants"], selected_variant):
            world_key = f"{variant}/fold0/seed{seed}"
            entry = models["worlds"].get(world_key)
            if entry is None or not entry.get("complete"):
                raise ClarityExperimentError(f"holdout world missing or incomplete: {world_key}")
            world = _world_from_entry(bundle, config, entry, device)
            world.eval().requires_grad_(False)
            for horizon in _selected(spec["horizons"], selected_horizon):
                for split_name in requested_splits:
                    key = _cache_key(variant, 0, seed, horizon, split_name)
                    if key in cache["rows"]:
                        continue
                    refs = _refs(bundle, split_name, int(horizon), labels=True)
                    rows = prepare_rows(
                        world, bundle, split_name, refs, int(spec["feature_batch_size"]),
                        protocol=spec["protocol"], variant=variant, fold=0,
                        seed=seed, device=device,
                    )
                    alignment_key = f"fold0/H{horizon}/{split_name}"
                    alignment = _row_alignment_hash(rows)
                    expected = cache["alignment"].get(alignment_key)
                    if expected is not None and expected != alignment:
                        raise ClarityExperimentError(
                            f"holdout variant rows are misaligned: {alignment_key}"
                        )
                    cache["alignment"][alignment_key] = alignment
                    cache["rows"][key] = rows
                    run["sample_key_hashes"][alignment_key] = alignment
                    print(f"prepared {key} rows={len(rows)}", flush=True)
    if evaluate_test:
        run["formal_test_untouched"] = False
        run["formal_test_opened"] = True
    write_torch(artifacts.path("last.pt"), cache)
    write_torch(artifacts.path("models.pt"), models)
    _mark(run, "prepare")
    _save_run(artifacts, run)


def run_stage(
    config: dict[str, Any],
    spec: dict[str, Any],
    stage: str,
    fold: int | None = None,
    seed: int | None = None,
    horizon: int | None = None,
    variant: str | None = None,
    *,
    device: torch.device | None = None,
    evaluate_test: bool = False,
) -> Path:
    if stage not in {"prepare", "train", "evaluate", "uncertainty", "report", "all", "world"}:
        raise ClarityExperimentError(f"unknown stage: {stage}")
    if seed is not None and seed not in spec["seeds"]:
        raise ClarityExperimentError("selected seed is absent from experiment config")
    if horizon is not None and horizon not in spec["horizons"]:
        raise ClarityExperimentError("selected horizon is absent from experiment config")
    if variant is not None and variant not in spec["variants"]:
        raise ClarityExperimentError("selected variant is absent from experiment config")
    if spec["protocol"] == "development_reuse" and evaluate_test:
        raise ClarityExperimentError("development_reuse never evaluates the formal test split")
    if spec["protocol"] == "original_holdout" and stage in {"evaluate", "uncertainty"} and not evaluate_test:
        raise ClarityExperimentError("original_holdout test evaluation requires --evaluate-test")
    device = device or _resolve_device(config["project"]["device"])
    artifacts, run, models, cache, metrics = _initialize_artifacts(config, spec)
    if (
        spec["protocol"] == "original_holdout"
        and evaluate_test
        and stage in {"evaluate", "all"}
        and run.get("formal_test_evaluated_once")
    ):
        raise ClarityExperimentError("the locked formal test was already evaluated for this run")
    _save_run(artifacts, run)
    if stage == "world":
        _run_holdout_world(
            config, spec, artifacts, run, models, device,
            selected_seed=seed, selected_variant=variant,
        )
        return artifacts.root
    common = {
        "selected_fold": fold,
        "selected_seed": seed,
        "selected_horizon": horizon,
        "selected_variant": variant,
    }
    if stage in {"prepare", "all"}:
        _run_prepare(
            config, spec, artifacts, run, models, cache, device,
            evaluate_test=evaluate_test, **common,
        )
    if stage in {"train", "all"}:
        _run_train(spec, artifacts, run, models, cache, metrics, device, **common)
    evaluate_now = stage == "evaluate" or (
        stage == "all" and (
            spec["protocol"] == "development_reuse" or evaluate_test
        )
    )
    uncertainty_now = stage == "uncertainty" or (
        stage == "all" and (
            spec["protocol"] == "development_reuse" or evaluate_test
        )
    )
    if evaluate_now:
        _run_evaluate(spec, artifacts, run, models, cache, metrics, device, **common)
    if uncertainty_now:
        _run_uncertainty(spec, artifacts, run, models, cache, metrics, device, **common)
    if stage in {"report", "all"}:
        _run_report(artifacts, run, metrics)
    return artifacts.root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Independent CLARITY-outcome downstream comparison for Cloop dynamics",
    )
    parser.add_argument("--config", default="configs/v1/default.yaml")
    parser.add_argument("--paths", default="configs/v1/server.yaml")
    parser.add_argument(
        "--experiment-config", default="configs/v3/clarity_downstream.yaml",
    )
    parser.add_argument(
        "--stage",
        choices=("prepare", "train", "evaluate", "uncertainty", "report", "all", "world"),
        required=True,
    )
    parser.add_argument("--fold", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--horizon", type=int)
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--device", default=None)
    parser.add_argument("--evaluate-test", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config, args.paths, device=args.device)
        spec = load_downstream_config(args.experiment_config)
        device = _resolve_device(config["project"]["device"])
        from ..v1.engine import validate_device_visibility
        validate_device_visibility(config, device)
        path = run_stage(
            config, spec, args.stage, args.fold, args.seed, args.horizon, args.variant,
            device=device, evaluate_test=bool(args.evaluate_test),
        )
        print(json.dumps({
            "status": "ok",
            "run": spec["run"],
            "protocol": spec["protocol"],
            "stage": args.stage,
            "output": str(path),
        }, ensure_ascii=False, indent=2))
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        parser.exit(2, f"clarity-downstream: {exc}\n")


if __name__ == "__main__":
    main()
