"""Metrics, paired summaries, selective evaluation, and report rendering."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Sequence

import numpy as np
import torch
from torch.nn import functional as F

from ..v1.metrics import harrell_c_index, ipcw_brier
from .clarity_downstream_head import cox_breslow_nll, fixed_time_survival_targets
from .trajectory_metrics import time_dependent_auc


def _labels(rows: Sequence[dict[str, Any]]) -> tuple[list[float], list[int]]:
    return [float(row["time"]) for row in rows], [int(row["event"]) for row in rows]


def _value(metric: Any) -> float | None:
    if isinstance(metric, dict):
        metric = metric.get("value")
    if isinstance(metric, (int, float)) and math.isfinite(float(metric)):
        return float(metric)
    return None


def _mean_sd(values: Sequence[float]) -> dict[str, Any]:
    clean = np.asarray([float(x) for x in values if math.isfinite(float(x))], dtype=float)
    if not len(clean):
        return {"mean": None, "sd": None, "n": 0}
    return {
        "mean": float(clean.mean()),
        "sd": float(clean.std(ddof=1)) if len(clean) > 1 else 0.0,
        "n": int(len(clean)),
    }


def evaluate_cohort(
    prediction_rows: Sequence[dict[str, Any]],
    censor_reference: Sequence[dict[str, Any]],
    tau: float,
) -> dict[str, Any]:
    """Evaluate the Cox and fixed-time branches without conflating their directions."""
    rows = list(prediction_rows)
    reference = list(censor_reference)
    if not rows:
        return {"available": False, "reason": "empty_evaluation_cohort"}
    if not reference:
        return {"available": False, "reason": "empty_censoring_reference"}
    times, events = _labels(rows)
    train_times, train_events = _labels(reference)
    risks = [float(row["risk_score"]) for row in rows]
    logits = [float(row["survival_logit365"]) for row in rows]
    survival = [float(row["survival365"]) for row in rows]
    if not all(math.isfinite(x) for x in (*risks, *logits, *survival)):
        raise ValueError("non-finite outcome prediction")

    risk_tensor = torch.tensor(risks, dtype=torch.float64)
    time_tensor = torch.tensor(times, dtype=torch.float64)
    event_tensor = torch.tensor(events, dtype=torch.long)
    logit_tensor = torch.tensor(logits, dtype=torch.float64)
    labels, valid = fixed_time_survival_targets(time_tensor, event_tensor, tau)
    cox = cox_breslow_nll(risk_tensor, time_tensor, event_tensor)
    bce = (
        F.binary_cross_entropy_with_logits(logit_tensor[valid], labels[valid].to(logit_tensor))
        if bool(valid.any())
        else None
    )
    latent_mse = [float(row["latent_mse"]) for row in rows]
    return {
        "available": True,
        "n": len(rows),
        "events": int(sum(events)),
        "censored": int(len(rows) - sum(events)),
        "identifiable365": int(valid.sum().item()),
        "c_index_risk": harrell_c_index(times, events, risks),
        "ipcw_brier365": ipcw_brier(
            train_times, train_events, times, events, survival, tau,
        ),
        "td_auc365": time_dependent_auc(
            train_times, train_events, times, events,
            [1.0 - probability for probability in survival], tau,
        ),
        "cox_partial_nll": float(cox.item()),
        "bce_identifiable": float(bce.item()) if bce is not None else None,
        "latent_mse_same_cohort": float(np.mean(latent_mse)),
        "risk_definition": "raw CLARITY Cox risk score; larger means higher risk",
        "probability_definition": f"sigmoid(CLARITY survival logit) at {tau:g} days",
    }


def paired_comparisons(task_metrics: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Pair exact fold/seed/H tasks, then average seeds before folds."""
    indexed: dict[tuple[str, int, int, int], dict[str, Any]] = {}
    for task in task_metrics.values():
        if not task.get("available"):
            continue
        indexed[(
            str(task["variant"]), int(task["fold"]), int(task["seed"]), int(task["horizon"])
        )] = task
    comparisons = {
        "rrt_ensemble_minus_baseline": ("baseline", "rrt_ensemble"),
        "rrt_minus_baseline": ("baseline", "rrt"),
        "ensemble_minus_baseline": ("baseline", "ensemble"),
        "rrt_ensemble_minus_ensemble": ("ensemble", "rrt_ensemble"),
    }
    metric_names = (
        "c_index_risk", "ipcw_brier365", "td_auc365", "latent_mse_same_cohort",
    )
    result: dict[str, Any] = {}
    for name, (base, candidate) in comparisons.items():
        by_horizon: dict[str, Any] = {}
        horizons = sorted({key[3] for key in indexed if key[0] == base})
        for horizon in horizons:
            seed_differences: dict[int, list[dict[str, Any]]] = defaultdict(list)
            for variant, fold, seed, task_h in sorted(indexed):
                if variant != base or task_h != horizon:
                    continue
                first = indexed[(base, fold, seed, horizon)]
                second = indexed.get((candidate, fold, seed, horizon))
                if second is None:
                    continue
                deltas = {}
                for metric in metric_names:
                    a = _value(first.get(metric))
                    b = _value(second.get(metric))
                    deltas[metric] = b - a if a is not None and b is not None else None
                seed_differences[fold].append({"seed": seed, **deltas})
            fold_rows = []
            for fold, seed_rows in sorted(seed_differences.items()):
                fold_rows.append({
                    "fold": fold,
                    "seeds": [row["seed"] for row in seed_rows],
                    **{
                        metric: (
                            float(np.mean([row[metric] for row in seed_rows if row[metric] is not None]))
                            if any(row[metric] is not None for row in seed_rows) else None
                        )
                        for metric in metric_names
                    },
                })
            aggregate = {
                metric: _mean_sd([row[metric] for row in fold_rows if row[metric] is not None])
                for metric in metric_names
            }
            direction_counts = {
                "c_index_improved": sum(
                    row["c_index_risk"] is not None and row["c_index_risk"] > 0
                    for row in fold_rows
                ),
                "brier_improved": sum(
                    row["ipcw_brier365"] is not None and row["ipcw_brier365"] < 0
                    for row in fold_rows
                ),
                "latent_mse_improved": sum(
                    row["latent_mse_same_cohort"] is not None
                    and row["latent_mse_same_cohort"] < 0
                    for row in fold_rows
                ),
                "folds": len(fold_rows),
            }
            by_horizon[f"H{horizon}"] = {
                "base": base,
                "candidate": candidate,
                "folds": fold_rows,
                "aggregate": aggregate,
                "direction_counts": direction_counts,
            }
        result[name] = by_horizon
    return result


def selective_ipcw(
    prediction_rows: Sequence[dict[str, Any]],
    censor_reference: Sequence[dict[str, Any]],
    coverages: Sequence[float],
    seed: int,
    *,
    random_reference_repeats: int = 200,
    threshold_mode: str = "rank_curve",
    thresholds: dict[float, float] | None = None,
    tau: float = 365.0,
) -> list[dict[str, Any]]:
    """IPCW Brier after uncertainty selection, retaining all label statuses in ranking."""
    rows = list(prediction_rows)
    reference = list(censor_reference)
    if not rows:
        return []
    if threshold_mode not in {"rank_curve", "fixed_threshold"}:
        raise ValueError("unknown uncertainty threshold mode")
    train_times, train_events = _labels(reference)
    rng = np.random.default_rng(int(seed))
    result = []
    ordered = sorted(rows, key=lambda row: (float(row["probability_std"]), str(row["record_id"])))
    for requested in coverages:
        coverage = float(requested)
        if not 0 < coverage <= 1:
            raise ValueError("coverages must be in (0,1]")
        if threshold_mode == "rank_curve":
            retained_n = len(rows) if coverage == 1 else max(1, int(math.floor(coverage * len(rows))))
            retained = ordered[:retained_n]
            threshold = float(retained[-1]["probability_std"])
        else:
            if thresholds is None or coverage not in thresholds:
                raise ValueError("fixed_threshold requires a threshold for each coverage")
            threshold = float(thresholds[coverage])
            retained = [row for row in rows if float(row["probability_std"]) <= threshold]
        times, events = _labels(retained)
        probabilities = [float(row["survival365_mean_probability"]) for row in retained]
        score = ipcw_brier(train_times, train_events, times, events, probabilities, tau)

        random_values = []
        size = len(retained)
        if size and random_reference_repeats > 0:
            for _ in range(int(random_reference_repeats)):
                indices = rng.choice(len(rows), size=size, replace=False)
                sample = [rows[int(index)] for index in indices]
                random_score = ipcw_brier(
                    train_times,
                    train_events,
                    *(_labels(sample)),
                    [float(row["survival365_mean_probability"]) for row in sample],
                    tau,
                )
                value = _value(random_score)
                if value is not None:
                    random_values.append(value)
        result.append({
            "requested_coverage": coverage,
            "actual_coverage": len(retained) / len(rows),
            "retained": len(retained),
            "total": len(rows),
            "events": int(sum(int(row["event"]) for row in retained)),
            "censored": int(len(retained) - sum(int(row["event"]) for row in retained)),
            "threshold": threshold,
            "threshold_mode": threshold_mode,
            "ipcw_brier365": score,
            "random_reference": _mean_sd(random_values),
        })
    return result


def aggregate_main(task_metrics: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Average seeds within fold, then report mean/sample-SD across folds."""
    metrics = ("c_index_risk", "ipcw_brier365", "td_auc365", "latent_mse_same_cohort")
    grouped: dict[tuple[str, int], dict[int, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for task in task_metrics.values():
        if task.get("available"):
            grouped[(str(task["variant"]), int(task["horizon"]))][int(task["fold"])].append(task)
    output: dict[str, Any] = {}
    for (variant, horizon), folds in sorted(grouped.items()):
        fold_rows = []
        for fold, tasks in sorted(folds.items()):
            fold_rows.append({
                "fold": fold,
                "seeds": sorted(int(task["seed"]) for task in tasks),
                "n": int(tasks[0]["n"]),
                "events": int(tasks[0]["events"]),
                **{
                    metric: (
                        float(np.mean([value for task in tasks if (value := _value(task[metric])) is not None]))
                        if any(_value(task[metric]) is not None for task in tasks) else None
                    )
                    for metric in metrics
                },
            })
        output[f"{variant}/H{horizon}"] = {
            "variant": variant,
            "horizon": horizon,
            "folds": fold_rows,
            "aggregate": {
                metric: _mean_sd([row[metric] for row in fold_rows if row[metric] is not None])
                for metric in metrics
            },
            "patients_per_fold": [row["n"] for row in fold_rows],
            "events_per_fold": [row["events"] for row in fold_rows],
        }
    return output



def aggregate_uncertainty(task_metrics: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Average selective curves over seeds within fold and then over folds."""
    grouped: dict[tuple[str, int, float], dict[int, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for task in task_metrics.values():
        if not task.get("available"):
            continue
        for point in task.get("selective_ipcw", []):
            grouped[(
                str(task["variant"]),
                int(task["horizon"]),
                float(point["requested_coverage"]),
            )][int(task["fold"])].append(point)
    output: dict[str, Any] = {}
    for (variant, horizon, coverage), folds in sorted(grouped.items()):
        fold_rows = []
        for fold, points in sorted(folds.items()):
            brier = [_value(point["ipcw_brier365"]) for point in points]
            random_brier = [
                _value(point.get("random_reference", {}).get("mean"))
                for point in points
            ]
            fold_rows.append({
                "fold": fold,
                "actual_coverage": float(np.mean([
                    float(point["actual_coverage"]) for point in points
                ])),
                "retained": float(np.mean([int(point["retained"]) for point in points])),
                "events": float(np.mean([int(point["events"]) for point in points])),
                "ipcw_brier365": (
                    float(np.mean([value for value in brier if value is not None]))
                    if any(value is not None for value in brier) else None
                ),
                "random_ipcw_brier365": (
                    float(np.mean([value for value in random_brier if value is not None]))
                    if any(value is not None for value in random_brier) else None
                ),
            })
        key = f"{variant}/H{horizon}/coverage{coverage:g}"
        output[key] = {
            "variant": variant,
            "horizon": horizon,
            "requested_coverage": coverage,
            "folds": fold_rows,
            "actual_coverage": _mean_sd([row["actual_coverage"] for row in fold_rows]),
            "retained": _mean_sd([row["retained"] for row in fold_rows]),
            "events": _mean_sd([row["events"] for row in fold_rows]),
            "ipcw_brier365": _mean_sd([
                row["ipcw_brier365"] for row in fold_rows
                if row["ipcw_brier365"] is not None
            ]),
            "random_ipcw_brier365": _mean_sd([
                row["random_ipcw_brier365"] for row in fold_rows
                if row["random_ipcw_brier365"] is not None
            ]),
        }
    return output



def _fmt(summary: dict[str, Any]) -> str:
    mean = summary.get("mean")
    sd = summary.get("sd")
    count = int(summary.get("n", 0))
    return f"NA (k={count})" if mean is None else f"{mean:.4f} ± {sd:.4f} (k={count})"


def build_report(metrics: dict[str, Any], run: dict[str, Any]) -> str:
    """Render the auditable development/confirmation report."""
    lines = [
        "# Cloop downstream evaluation with the CLARITY outcome architecture",
        "",
        f"- Protocol: `{run.get('protocol')}`",
        f"- Evidence scope: {run.get('evidence_scope')}",
        "- Dynamics are frozen; each variant has its own independently trained outcome head.",
        "- H2 Cox-risk C-index is primary; H2 IPCW Brier@365 is the companion metric.",
        "",
        "## Main full-cohort results",
        "",
        "| Method | H | Patients/fold | Events/fold | Latent MSE | C-index (risk) | IPCW Brier@365 | TD-AUC@365 |",
        "|---|---:|---|---|---:|---:|---:|---:|",
    ]
    summary = metrics.get("main", {}).get("summary", {})
    for row in summary.values():
        aggregate = row["aggregate"]
        lines.append(
            f"| {row['variant']} | {row['horizon']} | "
            f"{row['patients_per_fold']} | {row['events_per_fold']} | "
            f"{_fmt(aggregate['latent_mse_same_cohort'])} | "
            f"{_fmt(aggregate['c_index_risk'])} | {_fmt(aggregate['ipcw_brier365'])} | "
            f"{_fmt(aggregate['td_auc365'])} |"
        )
    lines.extend(["", "## Pre-specified paired comparisons", ""])
    for comparison, horizons in metrics.get("paired", {}).items():
        h2 = horizons.get("H2")
        if not h2:
            continue
        counts = h2["direction_counts"]
        aggregate = h2["aggregate"]
        lines.append(
            f"- `{comparison}` at H2: ΔC-index {_fmt(aggregate['c_index_risk'])}; "
            f"ΔBrier {_fmt(aggregate['ipcw_brier365'])}; "
            f"C-index improved in {counts['c_index_improved']}/{counts['folds']} folds, "
            f"Brier improved in {counts['brier_improved']}/{counts['folds']} folds."
        )
    lines.extend([
        "",
        "## Uncertainty (separate inference-only analysis)",
        "",
        "Only ensemble variants are included. Probability disagreement is not a clinical "
        "confidence interval and selective results do not replace the full-cohort main table.",
        "",
        "| Method | H | Requested coverage | Actual coverage | Retained/fold | Events/fold | Selective Brier@365 | Random-retention Brier |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    uncertainty_summary = metrics.get("uncertainty", {}).get("summary", {})
    for row in uncertainty_summary.values():
        lines.append(
            f"| {row['variant']} | {row['horizon']} | {row['requested_coverage']:.2f} | "
            f"{_fmt(row['actual_coverage'])} | {_fmt(row['retained'])} | "
            f"{_fmt(row['events'])} | {_fmt(row['ipcw_brier365'])} | "
            f"{_fmt(row['random_ipcw_brier365'])} |"
        )
    lines.extend(["", "## Limitations", ""])
    for item in metrics.get("limitations", []):
        lines.append(f"- {item}")
    return "\n".join(lines) + "\n"
