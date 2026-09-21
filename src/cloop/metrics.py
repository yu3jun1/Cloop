"""Dynamics, reliability, survival, action, and aggregate metric definitions."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Iterable, Sequence

import numpy as np


def nullable(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def mean_std(values: Sequence[float]) -> dict[str, float | int | None]:
    finite = np.asarray([x for x in values if math.isfinite(float(x))], dtype=float)
    if len(finite) == 0:
        return {"mean": None, "std": None, "n": 0}
    return {
        "mean": float(finite.mean()),
        "std": float(finite.std(ddof=1)) if len(finite) > 1 else None,
        "n": int(len(finite)),
    }


def dynamics_summary(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_horizon: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_horizon[int(row["horizon"])].append(row)
    result: dict[str, Any] = {}
    for horizon in sorted(by_horizon):
        rows = by_horizon[horizon]
        mse = [float(row["mse"]) for row in rows]
        cosine = [float(row["cosine"]) for row in rows]
        patient_values: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            patient_values[str(row["patient_id"])].append(float(row["mse"]))
        patient_macro = [float(np.mean(v)) for v in patient_values.values()]
        result[f"mse@{horizon}"] = float(np.mean(mse)) if mse else None
        result[f"cosine@{horizon}"] = float(np.mean(cosine)) if cosine else None
        result[f"patient_macro_mse@{horizon}"] = float(np.mean(patient_macro)) if patient_macro else None
        result[f"n@{horizon}"] = len(rows)
        result[f"patients@{horizon}"] = len(patient_values)
    long_values = [result.get("mse@2"), result.get("mse@3")]
    result["long_mse"] = (
        float(sum(long_values) / 2) if all(value is not None for value in long_values) else None
    )
    return result


def paired_relative_improvement(base_by_seed: dict[str, float], candidate_by_seed: dict[str, float]) -> dict[str, Any]:
    seeds = sorted(set(base_by_seed) & set(candidate_by_seed))
    paired = {}
    for seed in seeds:
        base = float(base_by_seed[seed])
        if base <= 0 or not math.isfinite(base):
            continue
        paired[seed] = 100.0 * (base - float(candidate_by_seed[seed])) / base
    summary = mean_std(list(paired.values()))
    return {"per_seed": paired, **summary}


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2.0 + 1.0
        i = j
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> tuple[float | None, str | None]:
    x_array, y_array = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    finite = np.isfinite(x_array) & np.isfinite(y_array)
    x_array, y_array = x_array[finite], y_array[finite]
    if len(x_array) < 3:
        return None, "fewer_than_three_samples"
    if np.all(x_array == x_array[0]) or np.all(y_array == y_array[0]):
        return None, "constant_vector"
    value = float(np.corrcoef(_rankdata(x_array), _rankdata(y_array))[0, 1])
    return value, None


def reliability_summary(
    errors: Sequence[float], disagreement: Sequence[float], coverage: float = 0.8
) -> dict[str, Any]:
    error = np.asarray(errors, dtype=float)
    uncertainty = np.asarray(disagreement, dtype=float)
    finite = np.isfinite(error) & np.isfinite(uncertainty)
    error, uncertainty = error[finite], uncertainty[finite]
    n = len(error)
    if n == 0:
        return {"n": 0, "reason": "no_finite_samples"}
    rho, reason = spearman(uncertainty, error)
    order = np.argsort(uncertainty, kind="mergesort")
    third = int(math.floor(n / 3))
    low_high_ratio = None
    ratio_reason = None
    if third > 0:
        low = float(error[order[:third]].mean())
        high = float(error[order[-third:]].mean())
        if low > 0:
            low_high_ratio = high / low
        else:
            ratio_reason = "zero_low_tertile_error"
    else:
        ratio_reason = "fewer_than_three_samples"
    retained = int(math.floor(coverage * n))
    return {
        "n": n,
        "spearman": rho,
        "spearman_reason": reason,
        "high_low_error_ratio": low_high_ratio,
        "ratio_reason": ratio_reason,
        "risk@100": float(error.mean()),
        "risk@80": float(error[order[:retained]].mean()) if retained else None,
        "retained": retained,
        "coverage": retained / n,
    }


def harrell_c_index(
    times: Sequence[float], events: Sequence[int], risks: Sequence[float]
) -> dict[str, Any]:
    t = np.asarray(times, dtype=float)
    e = np.asarray(events, dtype=int)
    r = np.asarray(risks, dtype=float)
    finite = np.isfinite(t) & np.isfinite(r) & (t > 0) & np.isin(e, [0, 1])
    t, e, r = t[finite], e[finite], r[finite]
    concordant = tied = comparable = 0.0
    for i in range(len(t)):
        for j in range(i + 1, len(t)):
            if t[i] == t[j]:
                continue
            early, late = (i, j) if t[i] < t[j] else (j, i)
            if e[early] != 1:
                continue
            comparable += 1
            if r[early] > r[late]:
                concordant += 1
            elif r[early] == r[late]:
                tied += 1
    if comparable == 0:
        return {"value": None, "reason": "no_comparable_pairs", "comparable_pairs": 0}
    return {
        "value": float((concordant + 0.5 * tied) / comparable),
        "reason": None,
        "comparable_pairs": int(comparable),
    }


def _km_censoring(train_times: np.ndarray, train_events: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(train_times, kind="mergesort")
    times = train_times[order]
    censor = 1 - train_events[order]
    unique = np.unique(times)
    survival = 1.0
    values = []
    for time in unique:
        at_risk = int(np.sum(times >= time))
        censored = int(np.sum((times == time) & (censor == 1)))
        if at_risk:
            survival *= 1.0 - censored / at_risk
        values.append(survival)
    return unique, np.asarray(values)


def _step_value(times: np.ndarray, values: np.ndarray, point: float, *, left: bool = False) -> float:
    side = "left" if left else "right"
    idx = int(np.searchsorted(times, point, side=side)) - 1
    return 1.0 if idx < 0 else float(values[idx])


def ipcw_brier(
    train_times: Sequence[float],
    train_events: Sequence[int],
    eval_times: Sequence[float],
    eval_events: Sequence[int],
    survival_probability: Sequence[float],
    tau: float,
) -> dict[str, Any]:
    tt = np.asarray(train_times, dtype=float)
    te = np.asarray(train_events, dtype=int)
    et = np.asarray(eval_times, dtype=float)
    ee = np.asarray(eval_events, dtype=int)
    sp = np.asarray(survival_probability, dtype=float)
    train_ok = np.isfinite(tt) & (tt > 0) & np.isin(te, [0, 1])
    eval_ok = np.isfinite(et) & (et > 0) & np.isin(ee, [0, 1]) & np.isfinite(sp)
    tt, te, et, ee, sp = tt[train_ok], te[train_ok], et[eval_ok], ee[eval_ok], sp[eval_ok]
    if len(tt) == 0 or len(et) == 0:
        return {"value": None, "reason": "empty_train_or_evaluation"}
    if tau > float(tt.max()):
        return {"value": None, "reason": "tau_beyond_train_followup_support"}
    km_times, km_values = _km_censoring(tt, te)
    g_tau = _step_value(km_times, km_values, tau)
    terms = []
    for time, event, probability in zip(et, ee, sp):
        if time <= tau and event == 1:
            g = _step_value(km_times, km_values, float(time), left=True)
            if g <= 0:
                return {"value": None, "reason": "censoring_support_zero_at_event"}
            terms.append(float(probability**2 / g))
        elif time > tau:
            if g_tau <= 0:
                return {"value": None, "reason": "censoring_support_zero_at_tau"}
            terms.append(float((1.0 - probability) ** 2 / g_tau))
        # censored by tau contributes zero, matching the IPCW definition.
        else:
            terms.append(0.0)
    return {"value": float(np.mean(terms)), "reason": None, "n": len(terms)}


def survival_summary(
    times: Sequence[float],
    events: Sequence[int],
    survival_probability: Sequence[float],
    train_times: Sequence[float],
    train_events: Sequence[int],
    tau: float = 365.0,
) -> dict[str, Any]:
    risk = [1.0 - float(x) for x in survival_probability]
    return {
        "n": len(times),
        "events": int(sum(int(x) for x in events)),
        "censored": int(len(events) - sum(int(x) for x in events)),
        "risk_definition": f"1-S({int(tau)})",
        "c_index": harrell_c_index(times, events, risk),
        "ipcw_brier": ipcw_brier(train_times, train_events, times, events, survival_probability, tau),
    }


def action_set_metrics(actual: Iterable[str], predicted: Iterable[str]) -> dict[str, float]:
    actual_set, predicted_set = set(actual), set(predicted)
    intersection = len(actual_set & predicted_set)
    if not actual_set and not predicted_set:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "jaccard": 1.0, "both_empty": 1.0}
    precision = intersection / len(predicted_set) if predicted_set else 0.0
    recall = intersection / len(actual_set) if actual_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    union = len(actual_set | predicted_set)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "jaccard": intersection / union if union else 1.0,
        "both_empty": 0.0,
    }
