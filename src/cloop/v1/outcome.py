"""Independent piecewise-exponential outcome/state-cost heads."""

from __future__ import annotations

from typing import Any, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class OutcomeError(ValueError):
    pass


def interval_exposure(times: Tensor, edges: Tensor) -> Tensor:
    """Observed duration in each (left,right] interval, administratively capped."""
    if edges.ndim != 1 or len(edges) < 2 or edges[0] != 0 or bool((edges[1:] <= edges[:-1]).any()):
        raise OutcomeError("edges must strictly increase from zero")
    if bool((times <= 0).any()) or not bool(torch.isfinite(times).all()):
        raise OutcomeError("survival times must be finite and positive")
    capped = times.clamp(max=float(edges[-1]))
    left = edges[:-1].to(times.device)
    widths = (edges[1:] - edges[:-1]).to(times.device)
    return torch.minimum(torch.clamp(capped.unsqueeze(-1) - left, min=0), widths)


def event_bins(times: Tensor, edges: Tensor) -> Tensor:
    # right=False places an event exactly at 90 in (0,90], exactly as registered.
    return torch.bucketize(times.clamp(max=float(edges[-1])), edges[1:].to(times.device), right=False).clamp(
        max=len(edges) - 2
    )


class PiecewiseExponentialHead(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        clinical_dim: int,
        history_dim: int,
        *,
        hidden_dim: int = 128,
        edges_days: Sequence[float] = (0, 90, 180, 365, 730),
        clinical_only: bool = False,
    ):
        super().__init__()
        edges = torch.tensor(edges_days, dtype=torch.float32)
        if edges[0] != 0 or bool((edges[1:] <= edges[:-1]).any()):
            raise OutcomeError("invalid piecewise-exponential edges")
        self.register_buffer("edges", edges)
        self.latent_dim = int(latent_dim)
        self.clinical_dim = int(clinical_dim)
        self.history_dim = int(history_dim)
        self.clinical_only = bool(clinical_only)
        input_dim = latent_dim + clinical_dim * 2 + history_dim
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, len(edges) - 1),
        )

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        latent_dim: int,
        clinical_dim: int,
        history_dim: int,
        *,
        clinical_only: bool = False,
    ) -> "PiecewiseExponentialHead":
        out = config["outcome"]
        return cls(
            latent_dim,
            clinical_dim,
            history_dim,
            hidden_dim=out["hidden_dim"],
            edges_days=out["edges_days"],
            clinical_only=clinical_only,
        )

    def _features(self, z: Tensor, clinical: Tensor, clinical_mask: Tensor, history: Tensor) -> Tensor:
        if z.shape[-1] != self.latent_dim:
            raise OutcomeError("latent shape mismatch")
        if clinical.shape != clinical_mask.shape or clinical.shape[-1] != self.clinical_dim:
            raise OutcomeError("clinical/mask shape mismatch")
        if history.shape[-1] != self.history_dim:
            raise OutcomeError("history shape mismatch")
        if self.clinical_only:
            z = torch.zeros_like(z)
        return torch.cat((z, clinical, clinical_mask, history), -1)

    def rates(self, z: Tensor, clinical: Tensor, clinical_mask: Tensor, history: Tensor) -> Tensor:
        return F.softplus(self.network(self._features(z, clinical, clinical_mask, history))) / 365.0

    def survival(
        self,
        z: Tensor,
        clinical: Tensor,
        clinical_mask: Tensor,
        history: Tensor,
        times: Tensor | float,
    ) -> Tensor:
        rates = self.rates(z, clinical, clinical_mask, history)
        query = torch.as_tensor(times, dtype=rates.dtype, device=rates.device)
        if query.ndim == 0:
            query = query.expand(rates.shape[:-1])
        query = torch.clamp(query, min=0, max=float(self.edges[-1]))
        left = self.edges[:-1].to(rates.device)
        widths = (self.edges[1:] - self.edges[:-1]).to(rates.device)
        exposure = torch.minimum(torch.clamp(query.unsqueeze(-1) - left, min=0), widths)
        return torch.exp(-(rates * exposure).sum(-1))

    def nll(
        self,
        z: Tensor,
        clinical: Tensor,
        clinical_mask: Tensor,
        history: Tensor,
        times: Tensor,
        events: Tensor,
        weights: Tensor | None = None,
        reduction: str = "mean",
    ) -> Tensor:
        if bool((times <= 0).any()) or not bool(torch.isfinite(times).all()):
            raise OutcomeError("survival times must be positive and finite")
        if not bool(((events == 0) | (events == 1)).all()):
            raise OutcomeError("events must be binary")
        rates = self.rates(z, clinical, clinical_mask, history)
        exposure = interval_exposure(times, self.edges)
        admin_event = events.float() * (times <= self.edges[-1]).float()
        bins = event_bins(times, self.edges)
        selected_rate = rates.gather(-1, bins.unsqueeze(-1)).squeeze(-1)
        losses = (rates * exposure).sum(-1) - admin_event * torch.log(selected_rate.clamp_min(1e-12))
        if weights is not None:
            losses = losses * weights
            if reduction == "mean":
                return losses.sum() / weights.sum().clamp_min(1e-12)
        if reduction == "none":
            return losses
        if reduction == "sum":
            return losses.sum()
        if reduction != "mean":
            raise OutcomeError(f"unknown reduction: {reduction}")
        return losses.mean()

    def state_cost(self, z: Tensor, clinical: Tensor, clinical_mask: Tensor, history: Tensor) -> Tensor:
        return 1.0 - self.survival(z, clinical, clinical_mask, history, 365.0)


class SyntheticCostHead(nn.Module):
    """Non-clinical state-cost regressor sharing the planner interface."""

    def __init__(self, latent_dim: int, clinical_dim: int, history_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(latent_dim + clinical_dim * 2 + history_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def state_cost(self, z: Tensor, clinical: Tensor, clinical_mask: Tensor, history: Tensor) -> Tensor:
        return self.network(torch.cat((z, clinical, clinical_mask, history), -1)).squeeze(-1)

    def forward(self, z: Tensor, clinical: Tensor, clinical_mask: Tensor, history: Tensor) -> Tensor:
        return self.state_cost(z, clinical, clinical_mask, history)

