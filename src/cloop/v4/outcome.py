"""Calibrated survival evaluation for trajectory ensembles.

There is one learned outcome: survival. The planning value interface is the
fixed-horizon risk 1 - S(t) and cannot learn a duplicated prognostic target.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..v1.outcome import OutcomeError
from ..v3.trajectory import nll_from_rates, survival_from_rates
from .trajectory_v4_1 import TrajectoryEncoder


def fixed_horizon_value_targets(
    days: Tensor, events: Tensor, horizon_days: float,
) -> tuple[Tensor, Tensor]:
    """Return fixed-horizon event targets and their identifiable-label mask."""
    if days.shape != events.shape or days.ndim != 1:
        raise OutcomeError("value labels must be matching [B] tensors")
    if horizon_days <= 0:
        raise OutcomeError("value horizon must be positive")
    if not bool(((events == 0) | (events == 1)).all()):
        raise OutcomeError("events must be binary")
    positive = (events == 1) & (days <= horizon_days)
    negative = days > horizon_days
    valid = positive | negative
    return positive.to(days.dtype), valid


def pairwise_survival_ranking_loss(
    risks: Tensor,
    days: Tensor,
    events: Tensor,
    *,
    patient_ids: Sequence[str] | Tensor | None = None,
    temperature: float = 1.0,
) -> Tensor:
    """Logistic ranking over comparable observations from different patients."""
    if risks.ndim != 1 or risks.shape != days.shape or days.shape != events.shape:
        raise OutcomeError("ranking inputs must be matching [B] tensors")
    if temperature <= 0:
        raise OutcomeError("ranking temperature must be positive")
    comparable = (events[:, None] == 1) & (days[:, None] < days[None, :])
    if patient_ids is not None:
        if isinstance(patient_ids, Tensor):
            if patient_ids.ndim != 1 or len(patient_ids) != len(risks):
                raise OutcomeError("patient_ids must contain one id per row")
            distinct = patient_ids[:, None] != patient_ids[None, :]
        else:
            if len(patient_ids) != len(risks):
                raise OutcomeError("patient_ids must contain one id per row")
            distinct = torch.tensor(
                [[left != right for right in patient_ids] for left in patient_ids],
                dtype=torch.bool,
                device=risks.device,
            )
        comparable = comparable & distinct
    if not bool(comparable.any()):
        return risks.sum() * 0.0
    margin = (risks[:, None] - risks[None, :]) / temperature
    return F.softplus(-margin[comparable]).mean()


def _weighted_mean(values: Tensor, weights: Tensor | None) -> Tensor:
    if weights is None:
        return values.mean()
    if weights.shape != values.shape:
        raise OutcomeError("weights must match per-sample losses")
    return (values * weights).sum() / weights.sum().clamp_min(1e-12)


@dataclass
class OutcomeOutput:
    rates: Tensor
    value: Tensor
    calibrated_error: Tensor | None
    attention_weights: Tensor
    trajectory_embedding: Tensor
    final_state_embedding: Tensor
    uncertainty_stats: Tensor


@dataclass
class OutcomeDistribution:
    member_rates: Tensor
    member_survival: Tensor
    member_risk: Tensor
    mean_survival: Tensor
    mean_risk: Tensor
    risk_variance: Tensor
    attention_weights: Tensor
    trajectory_embedding: Tensor
    uncertainty_stats: Tensor
    calibrated_error: Tensor | None


class TrajectoryOutcomeModel(nn.Module):
    """Trajectory encoder plus endpoint shortcut and one survival head."""

    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        clinical_dim: int,
        history_dim: int,
        *,
        hidden_dim: int = 128,
        nhead: int = 4,
        layers: int = 2,
        edges_days: Sequence[float] = (0, 90, 180, 365, 730),
        value_horizon_days: float = 365.0,
        use_velocity: bool = True,
        use_uncertainty: bool = False,
    ) -> None:
        super().__init__()
        edges = torch.tensor(edges_days, dtype=torch.float32)
        if len(edges) < 2 or edges[0] != 0 or bool((edges[1:] <= edges[:-1]).any()):
            raise OutcomeError("invalid hazard edges")
        if value_horizon_days <= 0 or value_horizon_days > float(edges[-1]):
            raise OutcomeError("value horizon must lie within hazard support")
        self.register_buffer("edges", edges)
        self.value_horizon_days = float(value_horizon_days)
        self.clinical_dim = int(clinical_dim)
        self.history_dim = int(history_dim)
        self.encoder = TrajectoryEncoder(
            latent_dim,
            action_dim,
            hidden_dim=hidden_dim,
            nhead=nhead,
            layers=layers,
            use_velocity=use_velocity,
            use_uncertainty=use_uncertainty,
        )
        self.final_state_projection = nn.Sequential(
            nn.LayerNorm(latent_dim), nn.Linear(latent_dim, hidden_dim), nn.GELU(),
        )
        context_dim = clinical_dim * 2 + history_dim
        joint_dim = hidden_dim * 2 + context_dim
        self.context_norm = nn.LayerNorm(joint_dim)
        self.survival_head = nn.Sequential(
            nn.Linear(joint_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, len(edges) - 1),
        )

    @property
    def use_velocity(self) -> bool:
        return self.encoder.use_velocity

    @property
    def use_uncertainty(self) -> bool:
        return self.encoder.use_uncertainty

    def forward(
        self,
        states: Tensor,
        actions: Tensor,
        uncertainty: Tensor,
        intervals: Tensor,
        state_mask: Tensor,
        clinical: Tensor,
        clinical_mask: Tensor,
        history: Tensor,
        state_source: Tensor | None = None,
    ) -> OutcomeOutput:
        batch = states.shape[0]
        if clinical.shape != (batch, self.clinical_dim) or clinical_mask.shape != clinical.shape:
            raise OutcomeError("clinical shape mismatch")
        if history.shape != (batch, self.history_dim):
            raise OutcomeError("history shape mismatch")
        encoded = self.encoder(
            states, actions, uncertainty, intervals, state_mask, state_source,
        )
        final_embedding = self.final_state_projection(encoded.final_state)
        context = torch.cat((clinical, clinical_mask, history), -1)
        joint = self.context_norm(torch.cat((
            encoded.embedding, final_embedding, context,
        ), -1))
        rates = F.softplus(self.survival_head(joint)) / 365.0
        value = 1.0 - survival_from_rates(
            rates, self.edges, self.value_horizon_days,
        )
        return OutcomeOutput(
            rates=rates,
            value=value,
            calibrated_error=encoded.calibrated_error,
            attention_weights=encoded.attention_weights,
            trajectory_embedding=encoded.embedding,
            final_state_embedding=final_embedding,
            uncertainty_stats=encoded.uncertainty_stats,
        )

    def rates(self, *features: Tensor) -> Tensor:
        return self(*features).rates

    def survival(self, *features: Tensor, days: float | Tensor) -> Tensor:
        return survival_from_rates(self.rates(*features), self.edges, days)

    def value(self, *features: Tensor) -> Tensor:
        """Return survival-derived fixed-horizon risk for the planner."""
        return 1.0 - self.survival(*features, days=self.value_horizon_days)

    @staticmethod
    def _repeat_feature(value: Tensor, members: int) -> Tensor:
        return value.unsqueeze(0).expand(members, *value.shape).reshape(
            members * value.shape[0], *value.shape[1:],
        )

    def outcome_distribution(
        self,
        member_states: Tensor,
        actions: Tensor,
        uncertainty: Tensor,
        intervals: Tensor,
        state_mask: Tensor,
        clinical: Tensor,
        clinical_mask: Tensor,
        history: Tensor,
        state_source: Tensor | None = None,
        *,
        days: float | Tensor | None = None,
    ) -> OutcomeDistribution:
        """Evaluate every ensemble trajectory before aggregating clinical risk."""
        if member_states.ndim != 4:
            raise OutcomeError("member_states must be [M,B,T,D]")
        members, batch, steps, _ = member_states.shape
        if members < 1 or actions.shape[:2] != (batch, steps):
            raise OutcomeError("ensemble trajectory/action shape mismatch")
        flat_states = member_states.reshape(members * batch, steps, -1)
        repeated = [
            self._repeat_feature(value, members)
            for value in (
                actions,
                uncertainty,
                intervals,
                state_mask,
                clinical,
                clinical_mask,
                history,
            )
        ]
        repeated_source = (
            None if state_source is None else self._repeat_feature(state_source, members)
        )
        output = self(flat_states, *repeated, repeated_source)
        member_rates = output.rates.reshape(members, batch, -1)
        query = self.value_horizon_days if days is None else days
        if isinstance(query, Tensor) and query.ndim > 0:
            if query.shape != (batch,):
                raise OutcomeError("distribution query days must be scalar or [B]")
            query = query.unsqueeze(0).expand(members, -1).reshape(-1)
        member_survival = survival_from_rates(
            member_rates.reshape(members * batch, -1), self.edges, query,
        ).reshape(members, batch)
        member_risk = 1.0 - member_survival
        calibrated = None
        if output.calibrated_error is not None:
            calibrated = output.calibrated_error.reshape(members, batch, steps).mean(0)
        return OutcomeDistribution(
            member_rates=member_rates,
            member_survival=member_survival,
            member_risk=member_risk,
            mean_survival=member_survival.mean(0),
            mean_risk=member_risk.mean(0),
            risk_variance=member_risk.var(0, unbiased=False),
            attention_weights=output.attention_weights.reshape(members, batch, steps).mean(0),
            trajectory_embedding=output.trajectory_embedding.reshape(
                members, batch, -1,
            ).mean(0),
            uncertainty_stats=output.uncertainty_stats.reshape(members, batch, -1).mean(0),
            calibrated_error=calibrated,
        )

    def value_distribution(self, *features: Tensor) -> OutcomeDistribution:
        return self.outcome_distribution(*features, days=self.value_horizon_days)

    def loss(
        self,
        *features: Tensor,
        days: Tensor,
        events: Tensor,
        patient_ids: Sequence[str] | Tensor | None = None,
        member_states: Tensor | None = None,
        weights: Tensor | None = None,
        ranking_weight: float = 0.1,
        uncertainty_weight: float = 0.1,
        ranking_temperature: float = 1.0,
        latent_error: Tensor | None = None,
    ) -> dict[str, Tensor]:
        if len(features) not in {8, 9}:
            raise OutcomeError("outcome features must contain 8 or 9 tensors")
        if latent_error is not None and member_states is not None:
            raise OutcomeError("legacy latent calibration and outcome propagation are exclusive")
        if member_states is None:
            output = self(*features)
            sample_nll = nll_from_rates(
                output.rates, self.edges, days, events, reduction="none",
            )
            risks = output.value
            risk_variance = torch.zeros_like(risks)
        else:
            distribution = self.outcome_distribution(member_states, *features[1:])
            members = member_states.shape[0]
            repeated_days = days.unsqueeze(0).expand(members, -1).reshape(-1)
            repeated_events = events.unsqueeze(0).expand(members, -1).reshape(-1)
            member_nll = nll_from_rates(
                distribution.member_rates.reshape(members * len(days), -1),
                self.edges,
                repeated_days,
                repeated_events,
                reduction="none",
            ).reshape(members, len(days))
            sample_nll = member_nll.mean(0)
            risks = distribution.mean_risk
            risk_variance = distribution.risk_variance
        survival_nll = _weighted_mean(sample_nll, weights)
        ranking = pairwise_survival_ranking_loss(
            risks,
            days,
            events,
            patient_ids=patient_ids,
            temperature=ranking_temperature,
        )
        uncertainty_auxiliary = risks.sum() * 0.0
        if member_states is not None:
            targets, identifiable = fixed_horizon_value_targets(
                days, events, self.value_horizon_days,
            )
            if bool(identifiable.any()):
                observed_brier = (risks.detach() - targets).square()
                per_sample = F.smooth_l1_loss(
                    risk_variance[identifiable],
                    observed_brier[identifiable],
                    reduction="none",
                )
                selected_weights = weights[identifiable] if weights is not None else None
                uncertainty_auxiliary = _weighted_mean(per_sample, selected_weights)
        total = (
            survival_nll
            + float(ranking_weight) * ranking
            + float(uncertainty_weight) * uncertainty_auxiliary
        )
        if latent_error is not None:
            calibration = total * 0.0
            if latent_error.shape != features[4].shape:
                raise OutcomeError("latent_error must match state_mask")
            if self.use_uncertainty and output.calibrated_error is not None:
                calibration_mask = features[4].clone()
                calibration_mask[:, 0] = False
                if bool(calibration_mask.any()):
                    calibration = F.smooth_l1_loss(
                        output.calibrated_error[calibration_mask],
                        latent_error[calibration_mask],
                    )
            compatible_total = (
                survival_nll
                + float(ranking_weight) * ranking
                + float(uncertainty_weight) * calibration
            )
            return {
                "total": compatible_total,
                "survival_nll": survival_nll,
                "ranking": ranking,
                "value": total * 0.0,
                "uncertainty_calibration": calibration,
            }
        return {
            "total": total,
            "survival_nll": survival_nll,
            "ranking": ranking,
            "uncertainty_auxiliary": uncertainty_auxiliary,
        }


# Compatibility name: there is no learned value head in v4.1.
DualHeadOutcomeModel = TrajectoryOutcomeModel


__all__ = [
    "DualHeadOutcomeModel",
    "OutcomeDistribution",
    "OutcomeOutput",
    "TrajectoryOutcomeModel",
    "fixed_horizon_value_targets",
    "pairwise_survival_ranking_loss",
]
