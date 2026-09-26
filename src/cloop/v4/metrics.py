"""Metrics specific to v4 uncertainty and selective-prediction experiments."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from ..v1.metrics import spearman


def coverage_risk_curve(
    errors: Sequence[float],
    uncertainty: Sequence[float],
    coverages: Sequence[float] = (1.0, 0.9, 0.8, 0.6, 0.4),
) -> list[dict[str, Any]]:
    """Risk after retaining the least-uncertain fraction of samples.

    This is a diagnostic only: samples are never discarded by the primary
    model or planner.
    """
    error = np.asarray(errors, dtype=float)
    score = np.asarray(uncertainty, dtype=float)
    if error.shape != score.shape or error.ndim != 1:
        raise ValueError("errors and uncertainty must be matching vectors")
    requested = [float(value) for value in coverages]
    if not requested or any(value <= 0 or value > 1 for value in requested):
        raise ValueError("coverages must lie in (0,1]")
    finite = np.isfinite(error) & np.isfinite(score)
    error, score = error[finite], score[finite]
    if not len(error):
        return [
            {"requested_coverage": value, "coverage": 0.0, "retained": 0,
             "risk": None, "uncertainty_threshold": None}
            for value in requested
        ]
    order = np.argsort(score, kind="mergesort")
    rows: list[dict[str, Any]] = []
    for coverage in requested:
        retained = max(1, int(np.floor(coverage * len(error))))
        selected = order[:retained]
        rows.append({
            "requested_coverage": coverage,
            "coverage": retained / len(error),
            "retained": retained,
            "risk": float(error[selected].mean()),
            "uncertainty_threshold": float(score[selected].max()),
        })
    return rows


def regression_calibration(
    predicted_error: Sequence[float],
    observed_error: Sequence[float],
    *,
    bins: int = 5,
) -> dict[str, Any]:
    """Quantile-bin calibration of predicted versus observed latent error."""
    predicted = np.asarray(predicted_error, dtype=float)
    observed = np.asarray(observed_error, dtype=float)
    if predicted.shape != observed.shape or predicted.ndim != 1:
        raise ValueError("predicted and observed error must be matching vectors")
    if bins < 1:
        raise ValueError("bins must be positive")
    finite = np.isfinite(predicted) & np.isfinite(observed)
    predicted, observed = predicted[finite], observed[finite]
    if not len(predicted):
        return {"n": 0, "mae": None, "spearman": None,
                "spearman_reason": "no_finite_samples", "bins": []}
    order = np.argsort(predicted, kind="mergesort")
    rows = []
    weighted_gap = 0.0
    for group in np.array_split(order, min(bins, len(order))):
        if not len(group):
            continue
        predicted_mean = float(predicted[group].mean())
        observed_mean = float(observed[group].mean())
        gap = abs(predicted_mean - observed_mean)
        weighted_gap += len(group) * gap
        rows.append({
            "n": len(group),
            "predicted_error": predicted_mean,
            "observed_error": observed_mean,
            "absolute_gap": gap,
        })
    rho, reason = spearman(predicted.tolist(), observed.tolist())
    return {
        "n": len(predicted),
        "mae": float(np.mean(np.abs(predicted - observed))),
        "bin_weighted_gap": weighted_gap / len(predicted),
        "spearman": rho,
        "spearman_reason": reason,
        "bins": rows,
    }

def survival_brier_errors(
    times: Sequence[float],
    events: Sequence[int],
    survival_probability: Sequence[float],
    horizon_days: float,
) -> tuple[list[float], list[int]]:
    """Return identifiable fixed-horizon Brier errors and source indices."""
    time = np.asarray(times, dtype=float)
    event = np.asarray(events, dtype=int)
    survival = np.asarray(survival_probability, dtype=float)
    if time.shape != event.shape or event.shape != survival.shape or time.ndim != 1:
        raise ValueError("survival error inputs must be matching vectors")
    if horizon_days <= 0:
        raise ValueError("horizon_days must be positive")
    if np.any((event != 0) & (event != 1)):
        raise ValueError("events must be binary")
    event_by_horizon = (event == 1) & (time <= horizon_days)
    known_survival = time > horizon_days
    identifiable = event_by_horizon | known_survival
    finite = np.isfinite(time) & np.isfinite(survival)
    selected = np.flatnonzero(identifiable & finite)
    target_survival = known_survival.astype(float)
    errors = (survival[selected] - target_survival[selected]) ** 2
    return errors.tolist(), selected.astype(int).tolist()


def outcome_uncertainty_calibration(
    times: Sequence[float],
    events: Sequence[int],
    survival_probability: Sequence[float],
    risk_variance: Sequence[float],
    horizon_days: float,
    *,
    bins: int = 5,
) -> dict[str, Any]:
    """Calibrate ensemble outcome-risk variance against survival Brier error."""
    variance = np.asarray(risk_variance, dtype=float)
    errors, indices = survival_brier_errors(
        times, events, survival_probability, horizon_days,
    )
    if variance.ndim != 1 or len(variance) != len(times):
        raise ValueError("risk_variance must contain one value per prediction")
    selected_variance = variance[np.asarray(indices, dtype=int)]
    result = regression_calibration(selected_variance.tolist(), errors, bins=bins)
    result.update({
        "identifiable": len(indices),
        "uncertainty_definition": "ensemble outcome-risk variance",
        "error_definition": "identifiable fixed-horizon survival Brier error",
    })
    return result


def survival_selective_prediction(
    times: Sequence[float],
    events: Sequence[int],
    survival_probability: Sequence[float],
    risk_uncertainty: Sequence[float],
    horizon_days: float,
    coverages: Sequence[float] = (1.0, 0.9, 0.8, 0.6, 0.4),
) -> list[dict[str, Any]]:
    """Coverage-risk curve ranked by outcome uncertainty, not latent error."""
    uncertainty = np.asarray(risk_uncertainty, dtype=float)
    errors, indices = survival_brier_errors(
        times, events, survival_probability, horizon_days,
    )
    if uncertainty.ndim != 1 or len(uncertainty) != len(times):
        raise ValueError("risk_uncertainty must contain one value per prediction")
    selected_uncertainty = uncertainty[np.asarray(indices, dtype=int)]
    return coverage_risk_curve(errors, selected_uncertainty.tolist(), coverages)
