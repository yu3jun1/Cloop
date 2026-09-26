"""Temporal, source-aware encoding for observed and simulated trajectories."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..v1.outcome import OutcomeError


def _masked_mean(values: Tensor, mask: Tensor, dim: int) -> Tensor:
    weights = mask.to(values.dtype)
    while weights.ndim < values.ndim:
        weights = weights.unsqueeze(-1)
    return (values * weights).sum(dim) / weights.sum(dim).clamp_min(1.0)


class TemporalEmbedding(nn.Module):
    """Embed relative and cumulative elapsed days for every token."""

    def __init__(
        self,
        output_dim: int,
        *,
        hidden_dim: int | None = None,
        scale_days: float = 365.0,
    ) -> None:
        super().__init__()
        if output_dim < 1 or scale_days <= 0:
            raise OutcomeError("temporal embedding dimensions and scale must be positive")
        width = int(hidden_dim or max(4, output_dim))
        self.scale_days = float(scale_days)
        self.network = nn.Sequential(
            nn.Linear(2, width), nn.SiLU(), nn.Linear(width, output_dim),
        )

    def forward(self, intervals: Tensor, state_mask: Tensor) -> Tensor:
        if intervals.shape != state_mask.shape or state_mask.dtype != torch.bool:
            raise OutcomeError("temporal intervals and mask must be matching [B,T] tensors")
        valid_intervals = intervals.masked_fill(~state_mask, 0.0)
        if not bool(torch.isfinite(valid_intervals).all()) or bool((valid_intervals < 0).any()):
            raise OutcomeError("valid intervals must be finite and non-negative")
        cumulative = valid_intervals.cumsum(1)
        denominator = torch.log1p(torch.as_tensor(
            self.scale_days, dtype=intervals.dtype, device=intervals.device,
        ))
        temporal = torch.stack(
            (
                torch.log1p(valid_intervals) / denominator,
                torch.log1p(cumulative) / denominator,
            ),
            -1,
        )
        return self.network(temporal).masked_fill(~state_mask[..., None], 0.0)


@dataclass
class TrajectoryEncoding:
    embedding: Tensor
    final_state: Tensor
    attention_weights: Tensor
    calibrated_error: Tensor | None
    uncertainty_stats: Tensor


class TrajectoryEncoder(nn.Module):
    """Masked Transformer over state, dynamics, time, and state provenance."""

    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        *,
        hidden_dim: int = 128,
        nhead: int = 4,
        layers: int = 2,
        time_embedding_dim: int = 16,
        time_scale_days: float = 365.0,
        velocity_epsilon: float = 1e-6,
        use_velocity: bool = True,
        use_uncertainty: bool = False,
    ) -> None:
        super().__init__()
        if latent_dim < 1 or action_dim < 1:
            raise OutcomeError("latent_dim and action_dim must be positive")
        if hidden_dim < 1 or nhead < 1 or hidden_dim % nhead:
            raise OutcomeError("hidden_dim must be positive and divisible by nhead")
        if velocity_epsilon <= 0:
            raise OutcomeError("velocity_epsilon must be positive")
        self.latent_dim = int(latent_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.use_velocity = bool(use_velocity)
        self.use_uncertainty = bool(use_uncertainty)
        self.velocity_epsilon = float(velocity_epsilon)
        self.temporal_embedding = TemporalEmbedding(
            int(time_embedding_dim), scale_days=time_scale_days,
        )
        # O2 adds both latent change and time-normalized velocity.
        dynamics_dim = 2 * latent_dim if use_velocity else 0
        token_dim = latent_dim + dynamics_dim + action_dim + int(time_embedding_dim) + 2
        self.token_projection = nn.Sequential(
            nn.LayerNorm(token_dim), nn.Linear(token_dim, hidden_dim), nn.GELU(),
        )
        # 0 = observed; 1 = predicted.
        self.state_source_embedding = nn.Embedding(2, hidden_dim)
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
            self.reliability_gate_raw = nn.Parameter(torch.tensor(0.0))
        else:
            self.uncertainty_token = None
            self.uncertainty_summary = None
            self.error_calibrator = None
            self.register_parameter("reliability_gate_raw", None)

    @staticmethod
    def _default_sources(state_mask: Tensor) -> Tensor:
        source = torch.ones_like(state_mask, dtype=torch.long)
        source[:, 0] = 0
        return source.masked_fill(~state_mask, 0)

    def _validate(
        self,
        states: Tensor,
        actions: Tensor,
        uncertainty: Tensor,
        intervals: Tensor,
        state_mask: Tensor,
        state_source: Tensor | None,
    ) -> tuple[int, int, Tensor]:
        if states.ndim != 3:
            raise OutcomeError("states must be [B,T,D]")
        batch, steps, dim = states.shape
        if dim != self.latent_dim or actions.shape != (batch, steps, self.action_dim):
            raise OutcomeError("trajectory state/action shape mismatch")
        if (
            uncertainty.shape != (batch, steps)
            or intervals.shape != (batch, steps)
            or state_mask.shape != (batch, steps)
            or state_mask.dtype != torch.bool
        ):
            raise OutcomeError("trajectory uncertainty/interval/mask shape mismatch")
        if not bool(state_mask[:, 0].all()):
            raise OutcomeError("every trajectory needs an initial state")
        if steps > 1 and not bool((state_mask[:, :-1] >= state_mask[:, 1:]).all()):
            raise OutcomeError("state_mask must be right padded")
        valid = state_mask
        if (
            not bool(torch.isfinite(states[valid]).all())
            or not bool(torch.isfinite(actions[valid]).all())
        ):
            raise OutcomeError("valid trajectory values must be finite")
        if not bool(torch.isfinite(intervals[valid]).all()) or bool((intervals[valid] < 0).any()):
            raise OutcomeError("valid intervals must be finite and non-negative")
        if self.use_uncertainty and (
            not bool(torch.isfinite(uncertainty[valid]).all())
            or bool((uncertainty[valid] < 0).any())
        ):
            raise OutcomeError("valid uncertainty must be finite and non-negative")
        source = self._default_sources(state_mask) if state_source is None else state_source
        if source.shape != (batch, steps):
            raise OutcomeError("state_source must be [B,T]")
        source = source.to(device=states.device, dtype=torch.long)
        if not bool(((source[valid] == 0) | (source[valid] == 1)).all()):
            raise OutcomeError("state_source must use 0=observed and 1=predicted")
        return batch, steps, source

    def _uncertainty_statistics(
        self, uncertainty: Tensor, intervals: Tensor, mask: Tensor,
    ) -> Tensor:
        count = mask.sum(1)
        mean = _masked_mean(uncertainty, mask, 1)
        maximum = uncertainty.masked_fill(~mask, -torch.inf).max(1).values
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
        state_source: Tensor | None = None,
    ) -> TrajectoryEncoding:
        batch, steps, source = self._validate(
            states, actions, uncertainty, intervals, state_mask, state_source,
        )
        delta = torch.zeros_like(states)
        velocity = torch.zeros_like(states)
        if steps > 1:
            transition_mask = state_mask[:, 1:] & state_mask[:, :-1]
            changes = states[:, 1:] - states[:, :-1]
            delta[:, 1:] = changes * transition_mask[..., None]
            denominator = intervals[:, 1:].clamp_min(self.velocity_epsilon)[..., None]
            velocity[:, 1:] = changes / denominator * transition_mask[..., None]

        count = state_mask.sum(1)
        positions = torch.arange(steps, device=states.device).expand(batch, -1)
        terminal = positions == (count - 1)[:, None]
        observed = source == 0
        pieces = [states]
        if self.use_velocity:
            pieces.extend((delta, velocity))
        pieces.extend(
            (
                actions,
                self.temporal_embedding(intervals, state_mask),
                observed.to(states.dtype)[..., None],
                terminal.to(states.dtype)[..., None],
            )
        )
        tokens = self.token_projection(torch.cat(pieces, -1))
        tokens = tokens + self.state_source_embedding(source)
        tokens = tokens.masked_fill(~state_mask[..., None], 0.0)
        encoded = self.encoder(tokens, src_key_padding_mask=~state_mask)
        logits = self.pool_score(encoded).squeeze(-1)

        calibrated_error: Tensor | None = None
        stats = torch.zeros(batch, 3, dtype=states.dtype, device=states.device)
        values = encoded
        if self.use_uncertainty:
            assert self.uncertainty_token is not None
            assert self.uncertainty_summary is not None
            assert self.error_calibrator is not None
            safe_uncertainty = uncertainty.masked_fill(~state_mask, 0.0)
            log_u = torch.log1p(safe_uncertainty.clamp_min(0))[..., None]
            values = values + self.uncertainty_token(log_u)
            logits = logits - F.softplus(self.reliability_gate_raw) * log_u.squeeze(-1)
            stats = self._uncertainty_statistics(safe_uncertainty, intervals, state_mask)
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
        final_state = states[torch.arange(batch, device=states.device), count - 1]
        return TrajectoryEncoding(
            pooled, final_state, attention, calibrated_error, stats,
        )
