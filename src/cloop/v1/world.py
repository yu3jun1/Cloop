"""One-step residual dynamics, recursive rollout, and independent ensembles."""

from __future__ import annotations

import math
from typing import Any, Sequence

import torch
from torch import Tensor, nn

from .data import update_history
from .types import Rollout


class WorldModelError(ValueError):
    pass


class OneStepDynamics(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        clinical_dim: int,
        history_dim: int,
        *,
        hidden_dim: int = 128,
        action_embed_dim: int = 32,
        time_embed_dim: int = 16,
        clinical_embed_dim: int = 16,
        history_embed_dim: int = 16,
        delta_scale_days: float = 365.0,
        use_context: bool = True,
        use_history: bool = True,
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.action_dim = int(action_dim)
        self.clinical_dim = int(clinical_dim)
        self.history_dim = int(history_dim)
        self.delta_scale_days = float(delta_scale_days)
        self.use_context = bool(use_context)
        self.use_history = bool(use_history)
        self.action_projection = nn.Sequential(nn.Linear(action_dim, action_embed_dim), nn.SiLU())
        self.time_projection = nn.Sequential(nn.Linear(1, time_embed_dim), nn.SiLU())
        token_dim = action_embed_dim + time_embed_dim
        if use_context:
            self.clinical_projection: nn.Module | None = nn.Sequential(
                nn.Linear(clinical_dim * 2, clinical_embed_dim), nn.SiLU()
            )
            token_dim += clinical_embed_dim
        else:
            self.clinical_projection = None
        if use_history:
            self.history_projection: nn.Module | None = nn.Sequential(
                nn.Linear(history_dim, history_embed_dim), nn.SiLU()
            )
            token_dim += history_embed_dim
        else:
            self.history_projection = None
        self.initial_hidden = nn.Linear(latent_dim, hidden_dim)
        self.gru = nn.GRUCell(token_dim, hidden_dim)
        self.residual = nn.Sequential(
            nn.Linear(latent_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        latent_dim: int,
        action_dim: int,
        clinical_dim: int,
        history_dim: int,
    ) -> "OneStepDynamics":
        world = config["world"]
        return cls(
            latent_dim,
            action_dim,
            clinical_dim,
            history_dim,
            hidden_dim=world["hidden_dim"],
            action_embed_dim=world["action_embed_dim"],
            time_embed_dim=world["time_embed_dim"],
            clinical_embed_dim=world["clinical_embed_dim"],
            history_embed_dim=world["history_embed_dim"],
            delta_scale_days=world["delta_scale_days"],
            use_context=world["use_context"],
            use_history=world["use_history"],
        )

    def forward(
        self,
        z: Tensor,
        action: Tensor,
        delta_days: Tensor,
        clinical: Tensor,
        clinical_mask: Tensor,
        history: Tensor,
    ) -> Tensor:
        if z.ndim != 2 or z.shape[-1] != self.latent_dim:
            raise WorldModelError("z must be [B,latent_dim]")
        if action.shape != (z.shape[0], self.action_dim):
            raise WorldModelError("action shape mismatch")
        delta_days = delta_days.reshape(-1)
        if delta_days.shape[0] != z.shape[0] or bool((delta_days <= 0).any()):
            raise WorldModelError("active dynamics steps require strictly positive delta_days")
        scaled_time = torch.log1p(delta_days).unsqueeze(-1) / math.log1p(self.delta_scale_days)
        features = [self.action_projection(action), self.time_projection(scaled_time)]
        if self.clinical_projection is not None:
            if clinical.shape != clinical_mask.shape or clinical.shape[-1] != self.clinical_dim:
                raise WorldModelError("clinical/mask shape mismatch")
            features.append(self.clinical_projection(torch.cat((clinical, clinical_mask), -1)))
        if self.history_projection is not None:
            if history.shape[-1] != self.history_dim:
                raise WorldModelError("history shape mismatch")
            features.append(self.history_projection(history))
        hidden = self.gru(torch.cat(features, -1), torch.tanh(self.initial_hidden(z)))
        return z + self.residual(torch.cat((z, hidden), -1))


class LegacyOneStepDynamics(nn.Module):
    """Exact Stage 1 architecture retained only for legacy numeric regression."""

    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        *,
        hidden_dim: int = 128,
        action_embed_dim: int = 32,
        time_embed_dim: int = 16,
        delta_scale_days: float = 365.0,
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.action_dim = int(action_dim)
        self.delta_scale_days = float(delta_scale_days)
        self.action_projection = nn.Sequential(
            nn.Linear(action_dim, action_embed_dim),
            nn.LayerNorm(action_embed_dim),
            nn.SiLU(),
        )
        self.time_projection = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        self.gru = nn.GRU(
            action_embed_dim + time_embed_dim,
            hidden_dim,
            batch_first=True,
        )
        self.network = nn.Sequential(
            nn.LayerNorm(latent_dim + hidden_dim),
            nn.Linear(latent_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        nn.init.normal_(self.network[-1].weight, std=0.01)
        nn.init.zeros_(self.network[-1].bias)

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        latent_dim: int,
        action_dim: int,
    ) -> "LegacyOneStepDynamics":
        world = config["world"]
        return cls(
            latent_dim,
            action_dim,
            hidden_dim=world["hidden_dim"],
            action_embed_dim=world["action_embed_dim"],
            time_embed_dim=world["time_embed_dim"],
            delta_scale_days=world["delta_scale_days"],
        )

    def forward(
        self,
        z: Tensor,
        action: Tensor,
        delta_days: Tensor,
        clinical: Tensor,
        clinical_mask: Tensor,
        history: Tensor,
    ) -> Tensor:
        del clinical, clinical_mask, history
        if z.ndim != 2 or z.shape[-1] != self.latent_dim:
            raise WorldModelError("z must be [B,latent_dim]")
        if action.shape != (z.shape[0], self.action_dim):
            raise WorldModelError("action shape mismatch")
        delta_days = delta_days.reshape(-1)
        if delta_days.shape[0] != z.shape[0] or bool((delta_days <= 0).any()):
            raise WorldModelError("active dynamics steps require strictly positive delta_days")
        scaled = torch.log1p(delta_days.clamp_min(0.0)) / math.log1p(self.delta_scale_days)
        sequence = torch.cat(
            (self.action_projection(action), self.time_projection(scaled[:, None])),
            dim=-1,
        )
        _, hidden = self.gru(sequence[:, None, :])
        return z + self.network(torch.cat((z, hidden[-1]), dim=-1))


class PersistenceDynamics(nn.Module):
    """D0: explicit no-training baseline."""

    def forward(
        self,
        z: Tensor,
        action: Tensor,
        delta_days: Tensor,
        clinical: Tensor,
        clinical_mask: Tensor,
        history: Tensor,
    ) -> Tensor:
        if bool((delta_days.reshape(-1) <= 0).any()):
            raise WorldModelError("active dynamics steps require positive time")
        return z


class EnsembleWorldModel(nn.Module):
    def __init__(self, members: Sequence[nn.Module]):
        super().__init__()
        if not members:
            raise WorldModelError("ensemble needs at least one member")
        self.members = nn.ModuleList(members)

    @property
    def ensemble_size(self) -> int:
        return len(self.members)

    def rollout(
        self,
        z0: Tensor,
        actions: Tensor,
        deltas: Tensor,
        clinical: Tensor,
        clinical_mask: Tensor,
        history0: Tensor,
        step_mask: Tensor | None = None,
    ) -> Rollout:
        if z0.ndim != 2 or actions.ndim != 3 or deltas.ndim != 2:
            raise WorldModelError("rollout expects z0[B,D], actions[B,K,A], deltas[B,K]")
        batch, steps = actions.shape[:2]
        if z0.shape[0] != batch or deltas.shape != (batch, steps):
            raise WorldModelError("rollout batch/step shape mismatch")
        if step_mask is None:
            step_mask = torch.ones(batch, steps, dtype=torch.bool, device=z0.device)
        if step_mask.shape != (batch, steps) or step_mask.dtype != torch.bool:
            raise WorldModelError("step_mask must be bool [B,K]")
        states = z0.unsqueeze(0).expand(self.ensemble_size, -1, -1).clone()
        history = history0.clone()
        saved = [states]
        for step in range(steps):
            active = step_mask[:, step]
            if bool(active.any()):
                if bool((deltas[active, step] <= 0).any()):
                    raise WorldModelError("non-positive delta on active rollout step")
                next_states = states.clone()
                for member_index, member in enumerate(self.members):
                    prediction = member(
                        states[member_index, active],
                        actions[active, step],
                        deltas[active, step],
                        clinical[active],
                        clinical_mask[active],
                        history[active],
                    )
                    next_states[member_index, active] = prediction
                states = next_states
                updated = update_history(history[active], actions[active, step], deltas[active, step])
                history = history.clone()
                history[active] = updated
            saved.append(states)
        return Rollout(states=torch.stack(saved, dim=2), step_mask=step_mask)

    def one_step_members(
        self,
        member_states: Tensor,
        action: Tensor,
        delta_days: float | Tensor,
        clinical: Tensor,
        clinical_mask: Tensor,
        history: Tensor,
    ) -> Tensor:
        """Advance [M,D] independently under one shared candidate action."""
        if member_states.ndim != 2 or member_states.shape[0] != self.ensemble_size:
            raise WorldModelError("member_states must be [M,D]")
        action = action.reshape(1, -1)
        clinical = clinical.reshape(1, -1)
        clinical_mask = clinical_mask.reshape(1, -1)
        history = history.reshape(1, -1)
        delta = torch.as_tensor([delta_days], dtype=member_states.dtype, device=member_states.device)
        predictions = []
        for index, member in enumerate(self.members):
            predictions.append(
                member(member_states[index : index + 1], action, delta, clinical, clinical_mask, history)[0]
            )
        return torch.stack(predictions)


def terminal_states(rollout: Rollout, horizons: Tensor) -> Tensor:
    """Gather each sample's own endpoint, returning [M,B,D]."""
    if horizons.ndim != 1 or horizons.shape[0] != rollout.states.shape[1]:
        raise WorldModelError("horizons must be [B]")
    member_count, batch, _, latent_dim = rollout.states.shape
    index = horizons.view(1, batch, 1, 1).expand(member_count, batch, 1, latent_dim)
    return rollout.states.gather(2, index).squeeze(2)


def terminal_mse(rollout: Rollout, horizons: Tensor, target: Tensor) -> Tensor:
    prediction = terminal_states(rollout, horizons)
    return ((prediction - target.unsqueeze(0)) ** 2).mean()


def ensemble_mean_and_disagreement(states: Tensor) -> tuple[Tensor, Tensor]:
    """Return mean [B,D] and U [B] with population denominator M*d."""
    if states.ndim < 2:
        raise WorldModelError("ensemble states must include member and latent dimensions")
    mean = states.mean(0)
    disagreement = ((states - mean.unsqueeze(0)) ** 2).mean(dim=(0, states.ndim - 1))
    return mean, disagreement


def variant_spec(variant: str, ensemble_size: int) -> tuple[bool, int]:
    if variant == "baseline":
        return False, 1
    if variant == "rrt":
        return True, 1
    if variant == "ensemble":
        return False, ensemble_size
    if variant == "rrt_ensemble":
        return True, ensemble_size
    raise WorldModelError(f"unknown dynamics variant: {variant}")
