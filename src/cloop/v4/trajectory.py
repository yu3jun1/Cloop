"""Velocity-aware trajectory encoding and dual-head factual outcome model.

The action at token ``t`` is the action taken before state ``t``; token zero
therefore receives a zero action and a zero interval.  Ensemble disagreement
is not treated as disease risk.  It enters through a reliability gate and a
separate uncertainty representation, while the value head predicts an
observed fixed-horizon adverse-outcome risk (lower is better for planning).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..v1.outcome import OutcomeError
from ..v3.trajectory import nll_from_rates, survival_from_rates


def _masked_mean(values: Tensor, mask: Tensor, dim: int) -> Tensor:
    weights = mask.to(values.dtype)
    while weights.ndim < values.ndim:
        weights = weights.unsqueeze(-1)
    return (values * weights).sum(dim) / weights.sum(dim).clamp_min(1.0)


def fixed_horizon_value_targets(
    days: Tensor, events: Tensor, horizon_days: float,
) -> tuple[Tensor, Tensor]:
    """Return adverse-event targets and an identifiable-label mask.

    An event observed by the horizon is positive; follow-up beyond the horizon
    without such an event is negative.  Censoring before the horizon has no
    identifiable binary target and is excluded from the value loss.
    """
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
    temperature: float = 1.0,
) -> Tensor:
    """Logistic ranking loss over Harrell-comparable ordered pairs."""
    if risks.ndim != 1 or risks.shape != days.shape or days.shape != events.shape:
        raise OutcomeError("ranking inputs must be matching [B] tensors")
    if temperature <= 0:
        raise OutcomeError("ranking temperature must be positive")
    # Pair (i,j) is comparable when i has an observed event before j's time.
    comparable = (events[:, None] == 1) & (days[:, None] < days[None, :])
    if not bool(comparable.any()):
        return risks.sum() * 0.0
    margin = (risks[:, None] - risks[None, :]) / temperature
    return F.softplus(-margin[comparable]).mean()


@dataclass
class TrajectoryEncoding:
    embedding: Tensor
    attention_weights: Tensor
    calibrated_error: Tensor | None
    uncertainty_stats: Tensor


@dataclass
class OutcomeOutput:
    rates: Tensor
    value: Tensor
    calibrated_error: Tensor | None
    attention_weights: Tensor
    trajectory_embedding: Tensor
    uncertainty_stats: Tensor


class TrajectoryEncoder(nn.Module):
    """Masked Transformer with optional delta-latent and uncertainty paths.

    Core disease tokens are encoded independently of uncertainty.  When
    enabled, uncertainty contributes (1) a token value embedding, (2) a
    monotonically decreasing bias in the pooling attention, and (3) trajectory
    summary statistics.  This separation prevents disagreement from being
    interpreted directly as biological risk.
    """

    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        *,
        hidden_dim: int = 128,
        nhead: int = 4,
        layers: int = 2,
        use_velocity: bool = True,
        use_uncertainty: bool = False,
    ) -> None:
        super().__init__()
        if latent_dim < 1 or action_dim < 1:
            raise OutcomeError("latent_dim and action_dim must be positive")
        if hidden_dim < 1 or nhead < 1 or hidden_dim % nhead:
            raise OutcomeError("hidden_dim must be positive and divisible by nhead")
        self.latent_dim = int(latent_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.use_velocity = bool(use_velocity)
        self.use_uncertainty = bool(use_uncertainty)
        token_dim = latent_dim + action_dim + 1 + (latent_dim if use_velocity else 0)
        self.token_projection = nn.Sequential(
            nn.LayerNorm(token_dim), nn.Linear(token_dim, hidden_dim), nn.GELU(),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 2,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=layers, enable_nested_tensor=False,
        )
        self.pool_score = nn.Linear(hidden_dim, 1)
        if use_uncertainty:
            self.uncertainty_token = nn.Sequential(
                nn.Linear(1, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim),
            )
            self.uncertainty_summary = nn.Sequential(
                nn.LayerNorm(3), nn.Linear(3, hidden_dim), nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.error_calibrator = nn.Sequential(
                nn.Linear(1, max(4, hidden_dim // 4)), nn.SiLU(),
                nn.Linear(max(4, hidden_dim // 4), 1), nn.Softplus(),
            )
            # softplus(raw) is non-negative, making the reliability bias
            # monotonically non-increasing as disagreement grows.
            self.reliability_gate_raw = nn.Parameter(torch.tensor(0.0))
        else:
            self.uncertainty_token = None
            self.uncertainty_summary = None
            self.error_calibrator = None
            self.register_parameter("reliability_gate_raw", None)

    def _validate(
        self,
        states: Tensor,
        actions: Tensor,
        uncertainty: Tensor,
        intervals: Tensor,
        state_mask: Tensor,
    ) -> tuple[int, int]:
        if states.ndim != 3:
            raise OutcomeError("states must be [B,T,D]")
        batch, steps, dim = states.shape
        if dim != self.latent_dim or actions.shape != (batch, steps, self.action_dim):
            raise OutcomeError("trajectory state/action shape mismatch")
        if (uncertainty.shape != (batch, steps) or intervals.shape != (batch, steps)
                or state_mask.shape != (batch, steps) or state_mask.dtype != torch.bool):
            raise OutcomeError("trajectory uncertainty/interval/mask shape mismatch")
        if not bool(state_mask[:, 0].all()):
            raise OutcomeError("every trajectory needs an initial state")
        if steps > 1 and not bool((state_mask[:, :-1] >= state_mask[:, 1:]).all()):
            raise OutcomeError("state_mask must be right padded")
        valid = state_mask
        if not bool(torch.isfinite(states[valid]).all()) or not bool(torch.isfinite(actions[valid]).all()):
            raise OutcomeError("valid trajectory values must be finite")
        if not bool(torch.isfinite(intervals[valid]).all()) or bool((intervals[valid] < 0).any()):
            raise OutcomeError("valid intervals must be finite and non-negative")
        if self.use_uncertainty and (
            not bool(torch.isfinite(uncertainty[valid]).all())
            or bool((uncertainty[valid] < 0).any())
        ):
            raise OutcomeError("valid uncertainty must be finite and non-negative")
        return batch, steps

    def _uncertainty_statistics(
        self, uncertainty: Tensor, intervals: Tensor, mask: Tensor,
    ) -> Tensor:
        count = mask.sum(1)
        mean = _masked_mean(uncertainty, mask, 1)
        masked = uncertainty.masked_fill(~mask, -torch.inf)
        maximum = masked.max(1).values
        last = count - 1
        last_u = uncertainty.gather(1, last[:, None]).squeeze(1)
        elapsed = (intervals * mask.to(intervals.dtype)).sum(1)
        slope = torch.where(
            count > 1,
            (last_u - uncertainty[:, 0]) / elapsed.clamp_min(1.0) * 365.0,
            torch.zeros_like(last_u),
        )
        return torch.stack((mean, maximum, slope), -1)

    def forward(
        self,
        states: Tensor,
        actions: Tensor,
        uncertainty: Tensor,
        intervals: Tensor,
        state_mask: Tensor,
    ) -> TrajectoryEncoding:
        _, steps = self._validate(states, actions, uncertainty, intervals, state_mask)
        velocity = torch.zeros_like(states)
        if steps > 1:
            transition_mask = state_mask[:, 1:] & state_mask[:, :-1]
            velocity[:, 1:] = (states[:, 1:] - states[:, :-1]) * transition_mask[..., None]
        pieces = [states]
        if self.use_velocity:
            pieces.append(velocity)
        pieces.extend((actions, torch.log1p(intervals)[..., None] / 6.0))
        tokens = self.token_projection(torch.cat(pieces, -1))
        tokens = tokens.masked_fill(~state_mask[..., None], 0.0)
        encoded = self.encoder(tokens, src_key_padding_mask=~state_mask)
        logits = self.pool_score(encoded).squeeze(-1)

        calibrated_error: Tensor | None = None
        stats = torch.zeros(states.shape[0], 3, dtype=states.dtype, device=states.device)
        values = encoded
        if self.use_uncertainty:
            assert self.uncertainty_token is not None
            assert self.uncertainty_summary is not None
            assert self.error_calibrator is not None
            log_u = torch.log1p(uncertainty.clamp_min(0))[..., None]
            values = values + self.uncertainty_token(log_u)
            gate_strength = F.softplus(self.reliability_gate_raw)
            logits = logits - gate_strength * log_u.squeeze(-1)
            stats = self._uncertainty_statistics(uncertainty, intervals, state_mask)
            calibrated_error = self.error_calibrator(log_u).squeeze(-1)
            calibrated_error = calibrated_error.masked_fill(~state_mask, 0.0)

        logits = logits.masked_fill(~state_mask, -torch.inf)
        attention = torch.softmax(logits, 1)
        pooled = (attention[..., None] * values).sum(1)
        if self.use_uncertainty:
            assert self.uncertainty_summary is not None
            summary_input = torch.stack(
                (
                    torch.log1p(stats[:, 0].clamp_min(0)),
                    torch.log1p(stats[:, 1].clamp_min(0)),
                    torch.asinh(stats[:, 2]),
                ),
                -1,
            )
            pooled = pooled + self.uncertainty_summary(summary_input)
        return TrajectoryEncoding(pooled, attention, calibrated_error, stats)


class DualHeadOutcomeModel(nn.Module):
    """Trajectory encoder with survival and fixed-horizon value heads."""

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
        context_dim = clinical_dim * 2 + history_dim
        joint_dim = hidden_dim + context_dim
        self.context_norm = nn.LayerNorm(joint_dim)
        self.survival_head = nn.Sequential(
            nn.Linear(joint_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, len(edges) - 1),
        )
        self.value_head = nn.Sequential(
            nn.Linear(joint_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1),
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
    ) -> OutcomeOutput:
        batch = states.shape[0]
        if clinical.shape != (batch, self.clinical_dim) or clinical_mask.shape != clinical.shape:
            raise OutcomeError("clinical shape mismatch")
        if history.shape != (batch, self.history_dim):
            raise OutcomeError("history shape mismatch")
        encoded = self.encoder(states, actions, uncertainty, intervals, state_mask)
        context = torch.cat((clinical, clinical_mask, history), -1)
        joint = self.context_norm(torch.cat((encoded.embedding, context), -1))
        rates = F.softplus(self.survival_head(joint)) / 365.0
        value = torch.sigmoid(self.value_head(joint).squeeze(-1))
        return OutcomeOutput(
            rates=rates,
            value=value,
            calibrated_error=encoded.calibrated_error,
            attention_weights=encoded.attention_weights,
            trajectory_embedding=encoded.embedding,
            uncertainty_stats=encoded.uncertainty_stats,
        )

    def rates(self, *features: Tensor) -> Tensor:
        return self(*features).rates

    def value(self, *features: Tensor) -> Tensor:
        return self(*features).value

    def survival(self, *features: Tensor, days: float | Tensor) -> Tensor:
        return survival_from_rates(self.rates(*features), self.edges, days)

    def loss(
        self,
        *features: Tensor,
        days: Tensor,
        events: Tensor,
        latent_error: Tensor | None = None,
        weights: Tensor | None = None,
        ranking_weight: float = 0.1,
        value_weight: float = 0.25,
        uncertainty_weight: float = 0.1,
        ranking_temperature: float = 1.0,
    ) -> dict[str, Tensor]:
        output = self(*features)
        survival_nll = nll_from_rates(
            output.rates, self.edges, days, events, weights, reduction="mean",
        )
        risks = 1.0 - survival_from_rates(
            output.rates, self.edges, self.value_horizon_days,
        )
        ranking = pairwise_survival_ranking_loss(
            risks, days, events, temperature=ranking_temperature,
        )
        targets, valid_value = fixed_horizon_value_targets(
            days, events, self.value_horizon_days,
        )
        if bool(valid_value.any()):
            value_losses = F.binary_cross_entropy(
                output.value[valid_value], targets[valid_value], reduction="none",
            )
            if weights is not None:
                selected_weights = weights[valid_value]
                value_loss = (value_losses * selected_weights).sum() / selected_weights.sum().clamp_min(1e-12)
            else:
                value_loss = value_losses.mean()
        else:
            value_loss = output.value.sum() * 0.0

        calibration = output.rates.sum() * 0.0
        if self.use_uncertainty and latent_error is not None:
            if latent_error.shape != features[4].shape:
                raise OutcomeError("latent_error must match state_mask")
            assert output.calibrated_error is not None
            calibration_mask = features[4].clone()
            calibration_mask[:, 0] = False
            if bool(calibration_mask.any()):
                calibration = F.smooth_l1_loss(
                    output.calibrated_error[calibration_mask],
                    latent_error[calibration_mask],
                )
        total = (
            survival_nll
            + float(ranking_weight) * ranking
            + float(value_weight) * value_loss
            + float(uncertainty_weight) * calibration
        )
        return {
            "total": total,
            "survival_nll": survival_nll,
            "ranking": ranking,
            "value": value_loss,
            "uncertainty_calibration": calibration,
        }
