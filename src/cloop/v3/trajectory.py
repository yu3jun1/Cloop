"""Masked factual trajectories and a trajectory-conditioned survival head.

The action at token t is the action *before* state t. No future action is
attached to the initial token. All times are relative to the first MRI.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..v1.outcome import OutcomeError, event_bins, interval_exposure


def survival_from_rates(rates: Tensor, edges: Tensor, days: float | Tensor) -> Tensor:
    query = torch.as_tensor(days, dtype=rates.dtype, device=rates.device)
    if query.ndim == 0:
        query = query.expand(rates.shape[0])
    if query.shape != rates.shape[:-1]:
        raise OutcomeError("survival query must be scalar or [B]")
    query = query.clamp(min=0, max=float(edges[-1]))
    left = edges[:-1].to(rates.device)
    widths = (edges[1:] - edges[:-1]).to(rates.device)
    exposure = torch.minimum((query[:, None] - left).clamp_min(0), widths)
    return torch.exp(-(rates * exposure).sum(-1))


def nll_from_rates(
    rates: Tensor, edges: Tensor, days: Tensor, events: Tensor,
    weights: Tensor | None = None, reduction: str = "mean",
) -> Tensor:
    if rates.ndim != 2 or rates.shape[1] != len(edges) - 1:
        raise OutcomeError("rates must be [B,K]")
    if days.shape != rates.shape[:1] or events.shape != days.shape:
        raise OutcomeError("survival label shape mismatch")
    if not bool(((events == 0) | (events == 1)).all()):
        raise OutcomeError("events must be binary")
    exposure = interval_exposure(days, edges)
    active_event = events.float() * (days <= edges[-1]).float()
    selected = rates.gather(1, event_bins(days, edges)[:, None]).squeeze(1)
    losses = (rates * exposure).sum(-1) - active_event * selected.clamp_min(1e-12).log()
    if reduction == "none":
        return losses
    if reduction == "sum":
        return (losses * weights).sum() if weights is not None else losses.sum()
    if reduction != "mean":
        raise OutcomeError(f"unknown reduction: {reduction}")
    if weights is None:
        return losses.mean()
    return (losses * weights).sum() / weights.sum().clamp_min(1e-12)


class TrajectorySurvivalHead(nn.Module):
    """Temporal encoder with a piecewise-exponential hazard output.

    `use_uncertainty=False` implements the trajectory-only ablation. Tokens
    are masked before attention and the last valid token is pooled, so a
    sample's output is invariant to additional right padding.
    """

    def __init__(
        self, latent_dim: int, action_dim: int, clinical_dim: int,
        history_dim: int, *, hidden_dim: int = 128, nhead: int = 4,
        layers: int = 2, edges_days: Sequence[float] = (0, 90, 180, 365, 730),
        use_uncertainty: bool = False,
    ):
        super().__init__()
        if hidden_dim % nhead:
            raise OutcomeError("hidden_dim must be divisible by nhead")
        edges = torch.tensor(edges_days, dtype=torch.float32)
        if len(edges) < 2 or edges[0] != 0 or bool((edges[1:] <= edges[:-1]).any()):
            raise OutcomeError("invalid hazard edges")
        self.register_buffer("edges", edges)
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.clinical_dim = clinical_dim
        self.history_dim = history_dim
        self.use_uncertainty = use_uncertainty
        token_dim = latent_dim + action_dim + 1 + int(use_uncertainty)
        self.token = nn.Sequential(nn.LayerNorm(token_dim), nn.Linear(token_dim, hidden_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nhead, dim_feedforward=hidden_dim * 2,
            dropout=0.0, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers, enable_nested_tensor=False)
        context_dim = clinical_dim * 2 + history_dim
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim + context_dim),
            nn.Linear(hidden_dim + context_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, len(edges) - 1),
        )

    def rates(
        self, states: Tensor, actions: Tensor, uncertainty: Tensor,
        timestamps: Tensor, state_mask: Tensor, clinical: Tensor,
        clinical_mask: Tensor, history: Tensor,
    ) -> Tensor:
        batch, steps, dim = states.shape
        if dim != self.latent_dim or actions.shape != (batch, steps, self.action_dim):
            raise OutcomeError("trajectory state/action shape mismatch")
        if (uncertainty.shape != (batch, steps) or timestamps.shape != (batch, steps)
                or state_mask.shape != (batch, steps) or state_mask.dtype != torch.bool):
            raise OutcomeError("trajectory uncertainty/time/mask shape mismatch")
        if not bool(state_mask[:, 0].all()) or not bool((state_mask[:, :-1] >= state_mask[:, 1:]).all()):
            raise OutcomeError("state_mask must be nonempty and right padded")
        if clinical.shape != (batch, self.clinical_dim) or clinical_mask.shape != clinical.shape:
            raise OutcomeError("clinical shape mismatch")
        if history.shape != (batch, self.history_dim):
            raise OutcomeError("history shape mismatch")
        pieces = [states, actions, torch.log1p(timestamps.clamp_min(0))[..., None] / 6.0]
        if self.use_uncertainty:
            pieces.append(torch.log1p(uncertainty.clamp_min(0))[..., None])
        tokens = self.token(torch.cat(pieces, -1))
        encoded = self.encoder(tokens, src_key_padding_mask=~state_mask)
        last = state_mask.long().sum(1) - 1
        pooled = encoded[torch.arange(batch, device=states.device), last]
        context = torch.cat((clinical, clinical_mask, history), -1)
        return F.softplus(self.head(torch.cat((pooled, context), -1))) / 365.0

    def survival(self, *features: Tensor, days: float | Tensor) -> Tensor:
        return survival_from_rates(self.rates(*features), self.edges, days)

    def nll(
        self, *features: Tensor, days: Tensor, events: Tensor,
        weights: Tensor | None = None, reduction: str = "mean",
    ) -> Tensor:
        return nll_from_rates(self.rates(*features), self.edges, days, events, weights, reduction)
