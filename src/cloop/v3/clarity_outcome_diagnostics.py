"""Read-only checkpoint and predicted-post diagnostics for CLARITY outcomes.

This module deliberately does not train an outcome head or mutate the source
run.  It audits the saved training/checkpoint records and then evaluates each
fixed head with its normal predicted post latent, deranged predicted post
latents, and the observed post latent.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch
import yaml
from torch import Tensor

from ..v1.artifacts import (
    RunArtifacts,
    read_json,
    read_torch,
    upsert_jsonl,
    write_json,
    write_text,
)
from ..v1.config import load_config, strict_merge
from ..v1.data import sha256_file
from ..v1_1.outcome_diagnostics import _resolve_device, _sha256_json
from .clarity_downstream import (
    CACHE_SCHEMA,
    MODEL_SCHEMA,
    SCHEMA as SOURCE_SCHEMA,
    VARIANTS,
    ClarityExperimentError,
    _architecture_hash,
    _head_key,
    _new_head,
    _rows_for_task,
    _tensor_state_hash,
    evaluate_one_head,
)


SCHEMA = "cloop_clarity_outcome_diagnostics_v1"
DEFAULTS: dict[str, Any] = {
    "run": "next_stage_clarity_outcome_diagnostics_v1",
    "source_run": "next_stage_clarity_outcome_v1",
    "variants": list(VARIANTS),
    "horizons": [2],
    "permutation_repeats": 20,
    "permutation_seed_base": 26092026,
    "metric_reproduction_tolerance": 1e-7,
}


class ClarityDiagnosticError(RuntimeError):
    pass


def load_diagnostic_config(path: str | Path) -> dict[str, Any]:
    try:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ClarityDiagnosticError(f"cannot read diagnostic config: {exc}") from exc
    raw = raw.get("clarity_outcome_diagnostics", raw)
    if not isinstance(raw, dict):
        raise ClarityDiagnosticError("clarity_outcome_diagnostics config must be a mapping")
    try:
        spec = strict_merge(DEFAULTS, raw)
    except ValueError as exc:
        raise ClarityDiagnosticError(str(exc)) from exc
    if not str(spec["run"]).startswith("next_stage_clarity_outcome_diagnostics_"):
        raise ClarityDiagnosticError(
            "run must start with next_stage_clarity_outcome_diagnostics_"
        )
    if spec["run"] == spec["source_run"]:
        raise ClarityDiagnosticError("diagnostic and source runs must be distinct")
    if tuple(spec["variants"]) != VARIANTS:
        raise ClarityDiagnosticError(f"variants must be exactly {list(VARIANTS)}")
    horizons = spec["horizons"]
    if (
        not horizons
        or sorted(set(horizons)) != list(horizons)
        or not set(horizons).issubset({1, 2, 3})
    ):
        raise ClarityDiagnosticError("horizons must be ordered unique values from [1,2,3]")
    if int(spec["permutation_repeats"]) < 1:
        raise ClarityDiagnosticError("permutation_repeats must be positive")
    tolerance = float(spec["metric_reproduction_tolerance"])
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ClarityDiagnosticError("metric_reproduction_tolerance must be non-negative")
    return spec


def make_post_diagnostic_rows(
    rows: Sequence[dict[str, Any]],
    mode: str,
    permutation_seed: int = 0,
) -> list[dict[str, Any]]:
    """Return copied rows with only ``post_mean`` replaced.

    Shuffling is a patient-level derangement within exactly one
    variant/fold/seed/horizon task.  No labels, pre latents, or conditions are
    moved, and the input rows are never modified in place.
    """

    rows = list(rows)
    if not rows:
        raise ClarityDiagnosticError("diagnostic cohort is empty")
    if mode not in {"normal_pred", "shuffled_pred", "observed_post"}:
        raise ClarityDiagnosticError(f"unknown diagnostic mode: {mode}")
    patients = [str(row["patient_id"]) for row in rows]
    if len(set(patients)) != len(rows):
        raise ClarityDiagnosticError("a diagnostic task must contain one row per patient")
    scope = {
        (
            str(row["variant"]),
            int(row["fold"]),
            int(row["seed"]),
            int(row["horizon"]),
        )
        for row in rows
    }
    if len(scope) != 1:
        raise ClarityDiagnosticError(
            "post latents cannot be shuffled across variant/fold/seed/horizon"
        )
    if mode == "shuffled_pred" and len(rows) < 2:
        raise ClarityDiagnosticError("at least two patients are required for shuffling")

    original_index = torch.arange(len(rows))
    permutation = original_index.clone()
    if mode == "shuffled_pred":
        generator = torch.Generator().manual_seed(int(permutation_seed))
        for _ in range(1000):
            candidate = torch.randperm(len(rows), generator=generator)
            if bool((candidate != original_index).all()):
                permutation = candidate
                break
        else:
            raise ClarityDiagnosticError("failed to construct a post-latent derangement")

    output = []
    for index, row in enumerate(rows):
        source = rows[int(permutation[index])]
        if mode == "observed_post":
            post = row["true_post"]
            donor = row["patient_id"]
        else:
            post = source["post_mean"]
            donor = source["patient_id"]
        if post.shape != row["post_mean"].shape:
            raise ClarityDiagnosticError("replacement post latent has the wrong shape")
        if not bool(torch.isfinite(post).all()):
            raise ClarityDiagnosticError("replacement post latent is non-finite")
        new_row = dict(row)
        new_row["post_mean"] = post.detach().clone()
        new_row["latent_mse"] = float(
            (post - row["true_post"]).square().mean().cpu()
        )
        new_row["diagnostic_mode"] = mode
        new_row["post_donor_patient_id"] = str(donor)
        new_row["permutation_seed"] = (
            int(permutation_seed) if mode == "shuffled_pred" else None
        )
        new_row["record_id"] = (
            f"{row['record_id']}:diag:{mode}:"
            f"{permutation_seed if mode == 'shuffled_pred' else 'na'}"
        )
        output.append(new_row)
    return output


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _metric_value(metric: Any) -> float | None:
    if isinstance(metric, dict):
        metric = metric.get("value")
    return float(metric) if _finite_number(metric) else None


def _difference(first: float | None, second: float | None) -> float | None:
    if first is None or second is None:
        return None
    return float(first - second)


def _mean_sd(values: Sequence[float | None]) -> dict[str, Any]:
    clean = np.asarray(
        [float(value) for value in values if value is not None and math.isfinite(float(value))],
        dtype=float,
    )
    if not len(clean):
        return {"mean": None, "sd": None, "n": 0}
    return {
        "mean": float(clean.mean()),
        "sd": float(clean.std(ddof=1)) if len(clean) > 1 else 0.0,
        "n": int(len(clean)),
    }


def audit_training_task(
    key: str,
    training: dict[str, Any],
    model_entry: dict[str, Any] | None,
) -> dict[str, Any]:
    """Audit one saved training record against its actual checkpoint."""

    result: dict[str, Any] = {
        "variant": str(training.get("variant", key.split("/")[0])),
        "fold": training.get("fold"),
        "seed": training.get("seed"),
        "horizon": training.get("horizon"),
        "available": bool(training.get("available", False)),
        "errors": [],
        "review_reasons": [],
    }
    if not result["available"]:
        result["reason"] = training.get("reason", "unavailable_without_reason")
        if model_entry is not None and model_entry.get("available"):
            result["errors"].append("training_unavailable_but_checkpoint_available")
        result["checkpoint_consistent"] = not result["errors"]
        return result

    for field in ("fold", "seed", "horizon", "fit_n", "fit_events", "stop_n", "stop_events"):
        if not isinstance(training.get(field), int):
            result["errors"].append(f"missing_or_invalid_{field}")
    history = training.get("history")
    if not isinstance(history, list) or not history:
        result["errors"].append("missing_training_history")
        history = []
    indexed = {
        row.get("epoch"): row
        for row in history
        if isinstance(row, dict) and isinstance(row.get("epoch"), int)
    }
    epochs_ran = training.get("epochs_ran")
    if isinstance(epochs_ran, int):
        expected_epochs = list(range(epochs_ran + 1))
        if sorted(indexed) != expected_epochs:
            result["errors"].append("history_epochs_are_incomplete_or_nonsequential")
    else:
        result["errors"].append("missing_or_invalid_epochs_ran")

    best_epoch = training.get("best_epoch")
    initial = indexed.get(0)
    selected = indexed.get(best_epoch)
    last = indexed[max(indexed)] if indexed else None
    if initial is None:
        result["errors"].append("epoch0_snapshot_missing")
    if not isinstance(best_epoch, int) or selected is None:
        result["errors"].append("selected_epoch_snapshot_missing")

    gradient_values = []
    bad_gradient_epochs = []
    bad_snapshot_epochs = []
    for epoch, row in sorted(indexed.items()):
        if epoch > 0:
            value = row.get("gradient_norm")
            if _finite_number(value):
                gradient_values.append(float(value))
            else:
                bad_gradient_epochs.append(epoch)
        for split in ("train", "stop"):
            snapshot = row.get(split)
            if not isinstance(snapshot, dict) or any(
                not _finite_number(snapshot.get(field))
                for field in (
                    "total",
                    "cox_partial_nll",
                    "bce_identifiable",
                    "risk_variance",
                    "survival_logit_variance",
                )
            ):
                bad_snapshot_epochs.append({"epoch": epoch, "split": split})
    if bad_gradient_epochs:
        result["errors"].append("nonfinite_or_missing_gradient_norm")
    if bad_snapshot_epochs:
        result["errors"].append("nonfinite_or_missing_loss_snapshot")

    if model_entry is None:
        result["errors"].append("checkpoint_entry_missing")
    else:
        if not model_entry.get("available"):
            result["errors"].append("checkpoint_marked_unavailable")
        if model_entry.get("best_epoch") != best_epoch:
            result["errors"].append("log_checkpoint_best_epoch_mismatch")
        state = model_entry.get("state")
        if not isinstance(state, dict) or not state:
            result["errors"].append("checkpoint_state_missing")
        else:
            if not all(
                isinstance(value, Tensor) and bool(torch.isfinite(value).all())
                for value in state.values()
            ):
                result["errors"].append("checkpoint_state_nonfinite_or_invalid")
            actual_hash = _tensor_state_hash(state)
            result["actual_state_hash"] = actual_hash
            if actual_hash != model_entry.get("final_state_hash"):
                result["errors"].append("final_state_hash_mismatch")
            if best_epoch == 0 and actual_hash != model_entry.get("init_state_hash"):
                result["errors"].append("epoch0_checkpoint_differs_from_initialization")
            if isinstance(best_epoch, int) and best_epoch > 0 and actual_hash == model_entry.get(
                "init_state_hash"
            ):
                result["errors"].append("trained_checkpoint_equals_initialization")

    if selected is not None:
        selected_stop = selected.get("stop", {}).get("total")
        if not _finite_number(selected_stop) or not math.isclose(
            float(selected_stop),
            float(training.get("best_stop_loss", math.nan)),
            rel_tol=1e-6,
            abs_tol=1e-7,
        ):
            result["errors"].append("best_stop_loss_mismatch")
        finite_stop = [
            (epoch, float(row["stop"]["total"]))
            for epoch, row in indexed.items()
            if isinstance(row.get("stop"), dict)
            and _finite_number(row["stop"].get("total"))
        ]
        if finite_stop:
            expected_best = min(finite_stop, key=lambda pair: (pair[1], pair[0]))[0]
            if best_epoch != expected_best:
                result["errors"].append("best_epoch_is_not_minimum_stop_loss")

    def snapshot(row: dict[str, Any] | None, split: str) -> dict[str, Any] | None:
        if row is None or not isinstance(row.get(split), dict):
            return None
        return copy.deepcopy(row[split])

    result.update({
        "best_epoch": best_epoch,
        "checkpoint_category": "initialization" if best_epoch == 0 else "trained",
        "epochs_ran": epochs_ran,
        "fit_n": training.get("fit_n"),
        "fit_events": training.get("fit_events"),
        "stop_n": training.get("stop_n"),
        "stop_events": training.get("stop_events"),
        "epoch0": {"train": snapshot(initial, "train"), "stop": snapshot(initial, "stop")},
        "selected": {
            "epoch": best_epoch,
            "train": snapshot(selected, "train"),
            "stop": snapshot(selected, "stop"),
        },
        "last": {
            "epoch": last.get("epoch") if last else None,
            "train": snapshot(last, "train"),
            "stop": snapshot(last, "stop"),
        },
        "gradient_norm": {
            **_mean_sd(gradient_values),
            "min": min(gradient_values) if gradient_values else None,
            "max": max(gradient_values) if gradient_values else None,
            "nonfinite_or_missing_epochs": bad_gradient_epochs,
        },
        "invalid_snapshot_epochs": bad_snapshot_epochs,
    })
    if initial is not None and selected is not None:
        result["selected_improvement_vs_epoch0"] = {
            split: {
                field: _difference(
                    _metric_value(initial.get(split, {}).get(field)),
                    _metric_value(selected.get(split, {}).get(field)),
                )
                for field in ("total", "cox_partial_nll", "bce_identifiable")
            }
            for split in ("train", "stop")
        }
    if best_epoch == 0:
        result["review_reasons"].append("selected_initialization")
        if initial is not None and last is not None:
            initial_train = _metric_value(initial.get("train", {}).get("total"))
            last_train = _metric_value(last.get("train", {}).get("total"))
            initial_stop = _metric_value(initial.get("stop", {}).get("total"))
            last_stop = _metric_value(last.get("stop", {}).get("total"))
            if (
                initial_train is not None
                and last_train is not None
                and last_train < initial_train
                and initial_stop is not None
                and last_stop is not None
                and last_stop > initial_stop
            ):
                result["review_reasons"].append(
                    "train_improved_while_last_stop_worsened"
                )
    result["checkpoint_consistent"] = not result["errors"]
    return result


def _audit_summary(tasks: dict[str, dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for task in tasks.values():
        if isinstance(task.get("horizon"), int):
            grouped[(str(task["variant"]), int(task["horizon"]))].append(task)
    by_variant_horizon = {}
    for (variant, horizon), rows in sorted(grouped.items()):
        available = [row for row in rows if row.get("available")]
        reasons = Counter(
            reason for row in available for reason in row.get("review_reasons", [])
        )
        by_variant_horizon[f"{variant}/H{horizon}"] = {
            "variant": variant,
            "horizon": horizon,
            "total_tasks": len(rows),
            "available_tasks": len(available),
            "epoch0": sum(row.get("best_epoch") == 0 for row in available),
            "selected_trained": sum(
                isinstance(row.get("best_epoch"), int) and row["best_epoch"] > 0
                for row in available
            ),
            "checkpoint_consistent": sum(
                bool(row.get("checkpoint_consistent")) for row in rows
            ),
            "fit_n_range": _integer_range(row.get("fit_n") for row in available),
            "fit_events_range": _integer_range(
                row.get("fit_events") for row in available
            ),
            "stop_n_range": _integer_range(row.get("stop_n") for row in available),
            "stop_events_range": _integer_range(
                row.get("stop_events") for row in available
            ),
            "best_epoch_counts": dict(
                sorted(Counter(row.get("best_epoch") for row in available).items())
            ),
            "review_reason_counts": dict(sorted(reasons.items())),
        }
    return {
        "total_tasks": len(tasks),
        "available_tasks": sum(bool(row.get("available")) for row in tasks.values()),
        "epoch0": sum(row.get("best_epoch") == 0 for row in tasks.values()),
        "selected_trained": sum(
            isinstance(row.get("best_epoch"), int) and row["best_epoch"] > 0
            for row in tasks.values()
        ),
        "checkpoint_inconsistencies": sum(
            not bool(row.get("checkpoint_consistent")) for row in tasks.values()
        ),
        "review_reason_counts": dict(sorted(Counter(
            reason
            for row in tasks.values()
            for reason in row.get("review_reasons", [])
        ).items())),
        "by_variant_horizon": by_variant_horizon,
    }


def _integer_range(values: Sequence[int | None] | Any) -> list[int] | None:
    clean = [int(value) for value in values if isinstance(value, int)]
    return [min(clean), max(clean)] if clean else None


def audit_training(
    source_metrics: dict[str, Any],
    source_models: dict[str, Any],
) -> dict[str, Any]:
    training = source_metrics.get("training")
    if not isinstance(training, dict) or not training:
        raise ClarityDiagnosticError("source metrics have no training records")
    heads = source_models.get("heads")
    if not isinstance(heads, dict):
        raise ClarityDiagnosticError("source models have no head checkpoints")
    tasks = {
        key: audit_training_task(key, row, heads.get(key))
        for key, row in sorted(training.items())
        if isinstance(row, dict)
    }
    orphan_checkpoints = sorted(set(heads) - set(training))
    missing_records = sorted(set(training) - set(heads))
    return {
        "tasks": tasks,
        "summary": {
            **_audit_summary(tasks),
            "orphan_checkpoints": orphan_checkpoints,
            "training_records_without_checkpoint": missing_records,
        },
    }


def _prediction_changes(
    normal: Sequence[dict[str, Any]],
    changed: Sequence[dict[str, Any]],
) -> dict[str, float]:
    normal, changed = list(normal), list(changed)
    if len(normal) != len(changed):
        raise ClarityDiagnosticError("diagnostic prediction cohorts changed length")
    probability, risk = [], []
    for first, second in zip(normal, changed):
        if first["patient_id"] != second["patient_id"]:
            raise ClarityDiagnosticError("diagnostic prediction cohorts are misaligned")
        probability.append(abs(float(first["survival365"]) - float(second["survival365"])))
        risk.append(abs(float(first["risk_score"]) - float(second["risk_score"])))
    return {
        "mean_abs_probability_change": float(np.mean(probability)),
        "mean_abs_risk_change": float(np.mean(risk)),
    }


def _metric_reproduction(
    reproduced: dict[str, Any],
    source: dict[str, Any] | None,
    tolerance: float,
) -> dict[str, Any]:
    if source is None:
        return {"matches": False, "reason": "source_main_task_missing", "metrics": {}}
    comparisons = {}
    matches = bool(reproduced.get("available")) == bool(source.get("available"))
    for name in ("c_index_risk", "ipcw_brier365", "td_auc365"):
        actual, expected = _metric_value(reproduced.get(name)), _metric_value(source.get(name))
        difference = (
            abs(actual - expected) if actual is not None and expected is not None else None
        )
        metric_matches = (
            actual is None and expected is None
        ) or (
            difference is not None and difference <= tolerance
        )
        matches = matches and metric_matches
        comparisons[name] = {
            "source": expected,
            "reproduced": actual,
            "absolute_difference": difference,
            "matches": metric_matches,
        }
    for name in ("n", "events"):
        metric_matches = reproduced.get(name) == source.get(name)
        matches = matches and metric_matches
        comparisons[name] = {
            "source": source.get(name),
            "reproduced": reproduced.get(name),
            "matches": metric_matches,
        }
    return {"matches": matches, "metrics": comparisons}


def _diagnostic_prediction_record(
    row: dict[str, Any],
    head_key: str,
    checkpoint_category: str,
) -> dict[str, Any]:
    mode = str(row["diagnostic_mode"])
    permutation = row.get("permutation_seed")
    return {
        "record_id": (
            f"clarity_post_diag:{head_key}:{row['patient_id']}:"
            f"{mode}:{permutation if permutation is not None else 'na'}"
        ),
        "kind": "fixed_head_post_diagnostic",
        "head_key": head_key,
        "checkpoint_category": checkpoint_category,
        "variant": row["variant"],
        "fold": int(row["fold"]),
        "seed": int(row["seed"]),
        "horizon": int(row["horizon"]),
        "patient_id": str(row["patient_id"]),
        "post_donor_patient_id": str(row["post_donor_patient_id"]),
        "diagnostic_mode": mode,
        "permutation_seed": permutation,
        "time": float(row["time"]),
        "event": int(row["event"]),
        "risk_score": float(row["risk_score"]),
        "survival_logit365": float(row["survival_logit365"]),
        "survival365": float(row["survival365"]),
        "post_input_mse_to_observed": float(row["latent_mse"]),
    }


def _task_post_diagnostic(
    key: str,
    source_run: dict[str, Any],
    source_models: dict[str, Any],
    source_cache: dict[str, Any],
    source_metrics: dict[str, Any],
    audit_task: dict[str, Any],
    permutation_seeds: Sequence[int],
    tolerance: float,
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    entry = source_models["heads"].get(key)
    if entry is None or not entry.get("available"):
        return {
            "available": False,
            "reason": "trained_head_unavailable",
            "checkpoint_category": audit_task.get("checkpoint_category"),
        }, []
    variant = str(entry["variant"])
    fold, seed, horizon = int(entry["fold"]), int(entry["seed"]), int(entry["horizon"])
    fit, _, report = _rows_for_task(
        source_cache, source_run, variant, fold, seed, horizon
    )
    if not report:
        return {
            "available": False,
            "reason": "empty_report_cohort",
            "checkpoint_category": audit_task.get("checkpoint_category"),
        }, []

    source_spec = source_run["config"]
    head = _new_head(
        int(report[0]["pre"].numel()),
        int(report[0]["condition"].numel()),
        source_spec,
        device,
    )
    if _architecture_hash(head) != entry.get("architecture_hash"):
        raise ClarityDiagnosticError(f"head architecture changed for {key}")
    head.load_state_dict(entry["state"])
    head.eval().requires_grad_(False)

    normal_rows = make_post_diagnostic_rows(report, "normal_pred")
    normal_metrics, normal_predictions = evaluate_one_head(head, normal_rows, fit, source_spec)
    reproduction = _metric_reproduction(
        normal_metrics,
        source_metrics.get("main", {}).get("tasks", {}).get(key),
        tolerance,
    )
    if not reproduction["matches"]:
        return {
            "available": False,
            "reason": "normal_prediction_did_not_reproduce_source_metrics",
            "variant": variant,
            "fold": fold,
            "seed": seed,
            "horizon": horizon,
            "checkpoint_category": audit_task.get("checkpoint_category"),
            "normal": normal_metrics,
            "source_reproduction": reproduction,
        }, []

    checkpoint_category = str(audit_task["checkpoint_category"])
    records = [
        _diagnostic_prediction_record(row, key, checkpoint_category)
        for row in normal_predictions
    ]
    observed_rows = make_post_diagnostic_rows(report, "observed_post")
    observed_metrics, observed_predictions = evaluate_one_head(
        head, observed_rows, fit, source_spec
    )
    observed_change = _prediction_changes(normal_predictions, observed_predictions)
    observed_delta = {
        "c_index_improvement": _difference(
            _metric_value(observed_metrics.get("c_index_risk")),
            _metric_value(normal_metrics.get("c_index_risk")),
        ),
        "brier_improvement": _difference(
            _metric_value(normal_metrics.get("ipcw_brier365")),
            _metric_value(observed_metrics.get("ipcw_brier365")),
        ),
    }
    records.extend(
        _diagnostic_prediction_record(row, key, checkpoint_category)
        for row in observed_predictions
    )

    permutations = []
    for permutation_seed in permutation_seeds:
        shuffled_rows = make_post_diagnostic_rows(
            report, "shuffled_pred", int(permutation_seed)
        )
        shuffled_metrics, shuffled_predictions = evaluate_one_head(
            head, shuffled_rows, fit, source_spec
        )
        change = _prediction_changes(normal_predictions, shuffled_predictions)
        permutations.append({
            "permutation_seed": int(permutation_seed),
            "metrics": shuffled_metrics,
            "delta_c_index": _difference(
                _metric_value(normal_metrics.get("c_index_risk")),
                _metric_value(shuffled_metrics.get("c_index_risk")),
            ),
            "delta_brier": _difference(
                _metric_value(shuffled_metrics.get("ipcw_brier365")),
                _metric_value(normal_metrics.get("ipcw_brier365")),
            ),
            **change,
        })
        records.extend(
            _diagnostic_prediction_record(row, key, checkpoint_category)
            for row in shuffled_predictions
        )
    shuffle_summary = {
        name: _mean_sd([row.get(name) for row in permutations])
        for name in (
            "delta_c_index",
            "delta_brier",
            "mean_abs_probability_change",
            "mean_abs_risk_change",
        )
    }
    return {
        "available": True,
        "variant": variant,
        "fold": fold,
        "seed": seed,
        "horizon": horizon,
        "n": len(report),
        "events": sum(int(row["event"]) for row in report),
        "checkpoint_category": checkpoint_category,
        "best_epoch": audit_task.get("best_epoch"),
        "normal": normal_metrics,
        "source_reproduction": reproduction,
        "shuffled_pred": {
            "repeats": len(permutations),
            "summary": shuffle_summary,
            "permutations": permutations,
        },
        "observed_post": {
            "metrics": observed_metrics,
            "delta_vs_normal": observed_delta,
            "output_change_vs_normal": observed_change,
            "interpretation": (
                "reference substitution into a head trained on predicted latents; "
                "not a strict attainable upper bound"
            ),
        },
    }, records


def _task_value(task: dict[str, Any], name: str) -> float | None:
    if name == "normal_c_index":
        return _metric_value(task.get("normal", {}).get("c_index_risk"))
    if name == "normal_brier":
        return _metric_value(task.get("normal", {}).get("ipcw_brier365"))
    if name == "shuffle_delta_c_index":
        return _metric_value(
            task.get("shuffled_pred", {}).get("summary", {}).get("delta_c_index", {}).get("mean")
        )
    if name == "shuffle_delta_brier":
        return _metric_value(
            task.get("shuffled_pred", {}).get("summary", {}).get("delta_brier", {}).get("mean")
        )
    if name == "mean_abs_probability_change":
        return _metric_value(
            task.get("shuffled_pred", {}).get("summary", {}).get(
                "mean_abs_probability_change", {}
            ).get("mean")
        )
    if name == "mean_abs_risk_change":
        return _metric_value(
            task.get("shuffled_pred", {}).get("summary", {}).get(
                "mean_abs_risk_change", {}
            ).get("mean")
        )
    if name == "observed_c_index":
        return _metric_value(
            task.get("observed_post", {}).get("metrics", {}).get("c_index_risk")
        )
    if name == "observed_brier":
        return _metric_value(
            task.get("observed_post", {}).get("metrics", {}).get("ipcw_brier365")
        )
    if name == "observed_delta_c_index":
        return _metric_value(
            task.get("observed_post", {}).get("delta_vs_normal", {}).get(
                "c_index_improvement"
            )
        )
    if name == "observed_delta_brier":
        return _metric_value(
            task.get("observed_post", {}).get("delta_vs_normal", {}).get(
                "brier_improvement"
            )
        )
    raise KeyError(name)


POST_SUMMARY_METRICS = (
    "normal_c_index",
    "normal_brier",
    "shuffle_delta_c_index",
    "shuffle_delta_brier",
    "mean_abs_probability_change",
    "mean_abs_risk_change",
    "observed_c_index",
    "observed_brier",
    "observed_delta_c_index",
    "observed_delta_brier",
)


def _aggregate_post_group(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_fold: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_fold[int(row["fold"])].append(row)
    folds = []
    for fold, tasks in sorted(by_fold.items()):
        folds.append({
            "fold": fold,
            "seeds": sorted(int(task["seed"]) for task in tasks),
            **{
                name: (
                    float(np.mean(values))
                    if (values := [
                        value for task in tasks if (value := _task_value(task, name)) is not None
                    ])
                    else None
                )
                for name in POST_SUMMARY_METRICS
            },
        })
    return {
        "tasks": len(rows),
        "checkpoint_categories": dict(
            sorted(Counter(str(row["checkpoint_category"]) for row in rows).items())
        ),
        "normal_reproduced": sum(
            bool(row.get("source_reproduction", {}).get("matches")) for row in rows
        ),
        "folds": folds,
        "aggregate": {
            name: _mean_sd([row[name] for row in folds])
            for name in POST_SUMMARY_METRICS
        },
    }


def _post_summary(tasks: dict[str, dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for task in tasks.values():
        if task.get("available"):
            grouped[(str(task["variant"]), int(task["horizon"]))].append(task)
    return {
        "total_tasks": len(tasks),
        "available_tasks": sum(bool(task.get("available")) for task in tasks.values()),
        "normal_reproduction_failures": sum(
            task.get("reason") == "normal_prediction_did_not_reproduce_source_metrics"
            for task in tasks.values()
        ),
        "by_variant_horizon": {
            f"{variant}/H{horizon}": {
                "variant": variant,
                "horizon": horizon,
                **_aggregate_post_group(rows),
                "by_checkpoint_category": {
                    category: _aggregate_post_group([
                        row for row in rows
                        if row["checkpoint_category"] == category
                    ])
                    for category in ("trained", "initialization")
                    if any(row["checkpoint_category"] == category for row in rows)
                },
            }
            for (variant, horizon), rows in sorted(grouped.items())
        },
    }


def run_post_diagnostics(
    source_run: dict[str, Any],
    source_models: dict[str, Any],
    source_cache: dict[str, Any],
    source_metrics: dict[str, Any],
    audit: dict[str, Any],
    spec: dict[str, Any],
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    permutation_seeds = [
        int(spec["permutation_seed_base"]) + index
        for index in range(int(spec["permutation_repeats"]))
    ]
    selected_horizons = set(int(value) for value in spec["horizons"])
    selected_variants = set(str(value) for value in spec["variants"])
    tasks, predictions = {}, []
    for key, audit_task in sorted(audit["tasks"].items()):
        if (
            int(audit_task.get("horizon", -1)) not in selected_horizons
            or str(audit_task.get("variant")) not in selected_variants
        ):
            continue
        if not audit_task.get("available"):
            tasks[key] = {
                "available": False,
                "reason": audit_task.get("reason", "training_unavailable"),
                "checkpoint_category": audit_task.get("checkpoint_category"),
            }
            continue
        task, rows = _task_post_diagnostic(
            key,
            source_run,
            source_models,
            source_cache,
            source_metrics,
            audit_task,
            permutation_seeds,
            float(spec["metric_reproduction_tolerance"]),
            device,
        )
        tasks[key] = task
        predictions.extend(rows)
        print(
            f"post diagnostic {key} available={task.get('available')} "
            f"checkpoint={task.get('checkpoint_category')}",
            flush=True,
        )
    return {
        "permutation_seeds": permutation_seeds,
        "tasks": tasks,
        "summary": _post_summary(tasks),
    }, predictions


def _format_stat(value: dict[str, Any] | None, digits: int = 4) -> str:
    if not value or value.get("mean") is None:
        return "NA"
    return (
        f"{float(value['mean']):.{digits}f} ± {float(value['sd']):.{digits}f} "
        f"(k={int(value['n'])})"
    )


def build_diagnostic_report(
    run: dict[str, Any],
    metrics: dict[str, Any],
) -> str:
    audit = metrics["audit"]["summary"]
    post = metrics.get("post_diagnostic", {}).get("summary", {})
    diagnostic_horizons = set(int(value) for value in run["config"]["horizons"])
    diagnostic_audit_rows = [
        row for row in audit["by_variant_horizon"].values()
        if int(row["horizon"]) in diagnostic_horizons
    ]
    diagnostic_available = sum(row["available_tasks"] for row in diagnostic_audit_rows)
    diagnostic_epoch0 = sum(row["epoch0"] for row in diagnostic_audit_rows)
    train_stop_divergence = audit.get("review_reason_counts", {}).get(
        "train_improved_while_last_stop_worsened", 0
    )
    lines = [
        "# CLARITY outcome checkpoint and predicted-post diagnostics",
        "",
        f"- Source run: `{run['source_run']}` (read-only)",
        f"- Diagnostic horizons: {run['config']['horizons']}",
        f"- Fixed derangements per task: {run['config']['permutation_repeats']}",
        "- Outcome heads were not retrained; uncertainty filtering was not run.",
        "",
        "## A. Training and checkpoint audit",
        "",
        "| Variant | H | Available | Epoch 0 | Trained checkpoint | Fit n | Stop n | Checkpoint consistent |",
        "|---|---:|---:|---:|---:|---|---|---:|",
    ]
    for row in audit["by_variant_horizon"].values():
        fit_range = row["fit_n_range"]
        stop_range = row["stop_n_range"]
        lines.append(
            f"| {row['variant']} | {row['horizon']} | {row['available_tasks']} | "
            f"{row['epoch0']} | {row['selected_trained']} | "
            f"{fit_range[0]}–{fit_range[1]} | {stop_range[0]}–{stop_range[1]} | "
            f"{row['checkpoint_consistent']}/{row['total_tasks']} |"
        )
    lines.extend([
        "",
        f"Across all tasks, {audit['epoch0']}/{audit['available_tasks']} selected epoch 0 and "
        f"{audit['selected_trained']}/{audit['available_tasks']} selected a trained checkpoint. "
        f"Checkpoint/log inconsistencies: {audit['checkpoint_inconsistencies']}.",
        f"At the diagnostic horizon(s), {diagnostic_epoch0}/{diagnostic_available} selected epoch 0.",
        "",
        "Selecting epoch 0 means the initialized head won the recorded stop-loss comparison; "
        "it does not mean that optimization was skipped.",
        f"In {train_stop_divergence}/{audit['epoch0']} epoch-0 tasks, train loss improved while "
        "the last stop loss was worse than initialization.",
        "",
        "## B. Fixed-head predicted-post diagnostic",
        "",
        "Positive shuffle ΔC and ΔB mean shuffling made the metric worse. Positive observed ΔC "
        "and ΔB mean observed-post substitution improved the metric relative to normal predicted post.",
        "",
        "| Variant | H | Checkpoint | Normal C-index | Normal Brier | Shuffle ΔC | Shuffle ΔB | "
        "Observed C-index | Observed Brier | Mean abs Δprobability | Reproduced |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in post.get("by_variant_horizon", {}).values():
        for category, category_row in row["by_checkpoint_category"].items():
            aggregate = category_row["aggregate"]
            lines.append(
                f"| {row['variant']} | {row['horizon']} | {category} "
                f"(n={category_row['tasks']}) | "
                f"{_format_stat(aggregate['normal_c_index'])} | "
                f"{_format_stat(aggregate['normal_brier'])} | "
                f"{_format_stat(aggregate['shuffle_delta_c_index'])} | "
                f"{_format_stat(aggregate['shuffle_delta_brier'])} | "
                f"{_format_stat(aggregate['observed_c_index'])} | "
                f"{_format_stat(aggregate['observed_brier'])} | "
                f"{_format_stat(aggregate['mean_abs_probability_change'], 6)} | "
                f"{category_row['normal_reproduced']}/{category_row['tasks']} |"
            )
    lines.extend([
        "",
        "## Interpretation boundaries",
        "",
        "- Each diagnostic condition uses the task's own saved checkpoint; epoch-0 tasks are "
        "initialization sensitivity, not learned post utilization.",
        "- A shuffled post creates an unnatural pre/post/condition combination and is a sensitivity "
        "test, not a causal intervention.",
        "- Observed post is evaluated with a head trained on predicted latents and may be distribution "
        "shifted; it is a reference, not a strict upper bound.",
        "- Permutations are repeated perturbations of the same patients, not independent samples. "
        "They are averaged inside each task before seeds and folds are aggregated.",
        "- These are development-reuse diagnostics and do not open or redefine the formal test set.",
        "",
    ])
    return "\n".join(lines)


def _load_source(
    output_root: str | Path,
    source_run_name: str,
) -> tuple[RunArtifacts, dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    artifacts = RunArtifacts(output_root, source_run_name, version="v3")
    required = ("run.json", "metrics.json", "models.pt", "last.pt")
    missing = [name for name in required if not artifacts.path(name).exists()]
    if missing:
        raise ClarityDiagnosticError(
            "source run is incomplete; missing local artifacts: " + ", ".join(missing)
        )
    run = read_json(artifacts.path("run.json"))
    metrics = read_json(artifacts.path("metrics.json"))
    models = read_torch(artifacts.path("models.pt"), safe=True)
    cache = read_torch(artifacts.path("last.pt"), safe=True)
    if run.get("schema_version") != SOURCE_SCHEMA or metrics.get("schema_version") != SOURCE_SCHEMA:
        raise ClarityDiagnosticError("unexpected source run/metrics schema")
    if models.get("schema_version") != MODEL_SCHEMA:
        raise ClarityDiagnosticError("unexpected source model schema")
    if cache.get("schema_version") != CACHE_SCHEMA:
        raise ClarityDiagnosticError("unexpected source cache schema")
    if not isinstance(run.get("config"), dict):
        raise ClarityDiagnosticError("source run does not contain its experiment config")
    return artifacts, run, metrics, models, cache


def run_diagnostics(
    config: dict[str, Any],
    spec: dict[str, Any],
    *,
    device: torch.device | None = None,
) -> Path:
    """Run the read-only audit and fixed-head post diagnostic."""

    device = device or _resolve_device(config["project"]["device"])
    source_artifacts, source_run, source_metrics, source_models, source_cache = _load_source(
        config["paths"]["output_root"], str(spec["source_run"])
    )
    source_spec = source_run["config"]
    if tuple(source_spec.get("variants", [])) != VARIANTS:
        raise ClarityDiagnosticError("source run does not contain the four fixed variants")
    if not set(spec["horizons"]).issubset(set(source_spec.get("horizons", []))):
        raise ClarityDiagnosticError("diagnostic horizon is absent from source run")

    artifacts = RunArtifacts(
        config["paths"]["output_root"], str(spec["run"]), version="v3"
    )
    signature = _sha256_json({
        "schema_version": SCHEMA,
        "config": spec,
        "source_signature": source_run.get("signature"),
    })
    source_metrics_sha256 = sha256_file(source_artifacts.path("metrics.json"))
    source_run_sha256 = sha256_file(source_artifacts.path("run.json"))
    existing = read_json(artifacts.path("run.json"))
    if existing is not None and existing.get("signature") != signature:
        raise ClarityDiagnosticError(
            "diagnostic output already exists with a different configuration"
        )
    if existing is not None and (
        existing.get("source_metrics_sha256") != source_metrics_sha256
        or existing.get("source_run_sha256") != source_run_sha256
    ):
        raise ClarityDiagnosticError(
            "source run changed after this diagnostic output was created"
        )
    run = existing or {
        "schema_version": SCHEMA,
        "run": spec["run"],
        "signature": signature,
        "source_run": spec["source_run"],
        "source_schema_version": source_run["schema_version"],
        "source_signature": source_run.get("signature"),
        "source_repository_commit": source_run.get("repository_commit"),
        "source_metrics_sha256": source_metrics_sha256,
        "source_run_sha256": source_run_sha256,
        "source_protocol": source_run.get("protocol"),
        "read_only_source": True,
        "outcome_retrained": False,
        "uncertainty_analysis_run": False,
        "formal_test_untouched": bool(source_run.get("formal_test_untouched")),
        "config": copy.deepcopy(spec),
        "completed_tasks": [],
    }

    audit = audit_training(source_metrics, source_models)
    print(
        f"audited {audit['summary']['total_tasks']} training tasks; "
        f"epoch0={audit['summary']['epoch0']} "
        f"inconsistencies={audit['summary']['checkpoint_inconsistencies']}",
        flush=True,
    )
    post, predictions = run_post_diagnostics(
        source_run,
        source_models,
        source_cache,
        source_metrics,
        audit,
        spec,
        device,
    )
    metrics = {
        "schema_version": SCHEMA,
        "source_run": spec["source_run"],
        "audit": audit,
        "post_diagnostic": post,
        "limitations": [
            "Shuffled post latents create unnatural pre/post/condition combinations.",
            "Observed post substitution may be out of distribution for a head trained on predicted latents.",
            "Epoch-0 checkpoints diagnose initialization sensitivity rather than learned post utilization.",
            "The development-reuse protocol is not an independent formal-test confirmation.",
        ],
    }
    run["completed_tasks"] = ["audit", "post_diagnostic", "report"]
    write_json(artifacts.path("run.json"), run)
    write_json(artifacts.path("metrics.json"), metrics)
    if predictions:
        upsert_jsonl(artifacts.path("predictions.jsonl"), predictions)
    write_text(artifacts.path("report.md"), build_diagnostic_report(run, metrics))
    artifacts.assert_flat()
    return artifacts.root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit CLARITY outcome checkpoints and fixed-head predicted-post use",
    )
    parser.add_argument("--config", default="configs/v1/default.yaml")
    parser.add_argument("--paths", default="configs/v1/server.yaml")
    parser.add_argument(
        "--diagnostic-config",
        default="configs/v3/clarity_outcome_diagnostics.yaml",
    )
    parser.add_argument("--device", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config, args.paths, device=args.device)
        spec = load_diagnostic_config(args.diagnostic_config)
        device = _resolve_device(config["project"]["device"])
        from ..v1.engine import validate_device_visibility
        validate_device_visibility(config, device)
        path = run_diagnostics(config, spec, device=device)
        print(json.dumps({
            "status": "ok",
            "run": spec["run"],
            "source_run": spec["source_run"],
            "output": str(path),
        }, ensure_ascii=False, indent=2))
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        parser.exit(2, f"clarity-outcome-diagnostics: {exc}\n")


if __name__ == "__main__":
    main()
