"""Censoring-aware survival diagnostics for next-stage development folds."""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

from ..v1.metrics import _km_censoring, _step_value, ipcw_brier


def time_dependent_auc(
    train_times: Sequence[float], train_events: Sequence[int],
    times: Sequence[float], events: Sequence[int], risks: Sequence[float], tau: float,
) -> dict[str, Any]:
    """Cumulative/dynamic AUC with inverse censoring weights for cases.

    Controls are subjects observed event-free beyond tau. Censored subjects
    before tau are excluded. Undefined support returns a reason, never zero.
    """
    tt, te = np.asarray(train_times, float), np.asarray(train_events, int)
    t, e, r = np.asarray(times, float), np.asarray(events, int), np.asarray(risks, float)
    valid_train = np.isfinite(tt) & (tt > 0) & np.isin(te, [0, 1])
    valid = np.isfinite(t) & (t > 0) & np.isfinite(r) & np.isin(e, [0, 1])
    tt, te, t, e, r = tt[valid_train], te[valid_train], t[valid], e[valid], r[valid]
    if len(tt) == 0 or tau > float(tt.max()):
        return {"value": None, "reason": "insufficient_train_followup_support"}
    cases = np.flatnonzero((t <= tau) & (e == 1))
    controls = np.flatnonzero(t > tau)
    if len(cases) == 0 or len(controls) == 0:
        return {"value": None, "reason": "no_cases_or_controls", "cases": len(cases), "controls": len(controls)}
    km_t, km_g = _km_censoring(tt, te)
    if _step_value(km_t, km_g, tau) <= 0:
        return {"value": None, "reason": "zero_censoring_support_at_tau"}
    weight = np.asarray([1.0 / _step_value(km_t, km_g, float(t[i]), left=True)
                         if _step_value(km_t, km_g, float(t[i]), left=True) > 0 else math.inf
                         for i in cases])
    if not np.isfinite(weight).all():
        return {"value": None, "reason": "zero_censoring_support_at_event"}
    concordance = (r[cases, None] > r[None, controls]).astype(float)
    concordance += 0.5 * (r[cases, None] == r[None, controls])
    value = float((concordance * weight[:, None]).sum() / (weight.sum() * len(controls)))
    return {"value": value, "reason": None, "cases": len(cases), "controls": len(controls)}


def integrated_brier(
    train_times: Sequence[float], train_events: Sequence[int],
    times: Sequence[float], events: Sequence[int],
    probabilities: dict[float, Sequence[float]], max_tau: float,
) -> dict[str, Any]:
    """Trapezoidal IPCW Brier integral on a declared grid from zero to tau."""
    grid = sorted(float(t) for t in probabilities if 0 < float(t) <= max_tau)
    if not grid or grid[-1] != max_tau:
        return {"value": None, "reason": "grid_does_not_end_at_tau"}
    points = [(0.0, 0.0)]
    for tau in grid:
        result = ipcw_brier(train_times, train_events, times, events, probabilities[tau], tau)
        if result["value"] is None:
            return {"value": None, "reason": result["reason"], "grid_days": grid}
        points.append((tau, result["value"]))
    area = sum((x1 - x0) * (y0 + y1) / 2 for (x0, y0), (x1, y1) in zip(points, points[1:]))
    return {"value": float(area / max_tau), "reason": None, "grid_days": grid,
            "brier_by_day": {str(int(x)): y for x, y in points[1:]}}


def calibration_bins(times: Sequence[float], events: Sequence[int],
                     survival: Sequence[float], tau: float, bins: int = 3) -> list[dict[str, Any]]:
    """Predicted risk versus within-bin Kaplan-Meier event risk at tau."""
    t, e, p = np.asarray(times, float), np.asarray(events, int), np.asarray(survival, float)
    order = np.argsort(1.0 - p, kind="mergesort")
    result = []
    for group in np.array_split(order, bins):
        if not len(group):
            result.append({"n": 0, "predicted_risk": None, "observed_km_risk": None})
            continue
        subgroup_t, subgroup_e = t[group], e[group]
        at_risk = len(group)
        km_survival = 1.0
        for day in sorted(set(subgroup_t[subgroup_t <= tau])):
            deaths = int(((subgroup_t == day) & (subgroup_e == 1)).sum())
            if at_risk:
                km_survival *= 1.0 - deaths / at_risk
            at_risk -= int((subgroup_t == day).sum())
        result.append({"n": len(group), "predicted_risk": float((1.0 - p[group]).mean()),
                       "observed_km_risk": float(1.0 - km_survival)})
    return result
