"""Trajectory-value MPC with a soft ensemble-disagreement penalty."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor

from ..v1.data import ActionCodec, update_history
from ..v1.types import Action, PatientState
from ..v1.world import EnsembleWorldModel
from .trajectory import DualHeadOutcomeModel


class PlanningError(RuntimeError):
    pass


def historical_candidates(
    codec: ActionCodec,
    replay_actions: Sequence[Action] = (),
    *,
    max_candidates: int = 6,
    min_support: int | None = None,
) -> tuple[Action, ...]:
    """Merge recorded replay actions with the train-only action catalog."""
    if max_candidates < 1:
        raise PlanningError("max_candidates must be positive")
    threshold = codec.min_support if min_support is None else int(min_support)
    merged: dict[str, Action] = {}
    for action in (*replay_actions, *codec.catalog):
        if (action.action_id in codec.by_id and action.support_count >= threshold
                and len(action.token_ids) == len(set(action.token_ids))
                and all(0 <= token < codec.dim for token in action.token_ids)):
            merged.setdefault(action.action_id, action)
    ordered = sorted(
        merged.values(), key=lambda action: (-action.support_count, action.action_id),
    )
    return tuple(ordered[:max_candidates])


@dataclass(frozen=True)
class MPCResult:
    action: Action
    plan: tuple[Action, ...]
    score: float
    mean_predicted_risk: float
    mean_disagreement: float
    cumulative_predicted_risk: float
    cumulative_uncertainty_penalty: float
    expanded_nodes: int
    world_forwards: int


@dataclass
class _Node:
    member_states: Tensor
    mean_states: tuple[Tensor, ...]
    action_vectors: tuple[Tensor, ...]
    uncertainties: tuple[float, ...]
    intervals: tuple[float, ...]
    history: Tensor
    actions: tuple[Action, ...]
    weighted_risk: float
    weighted_penalty: float
    weight_total: float

    @property
    def score(self) -> float:
        if self.weight_total <= 0:
            return float("inf")
        return (self.weighted_risk + self.weighted_penalty) / self.weight_total


class TrajectoryValueMPC:
    """Beam-search MPC scored by factual value risk plus soft uncertainty.

    The value output is an observational fixed-horizon risk proxy, not a
    causal treatment-effect estimate.  No branch is removed solely because it
    is uncertain.
    """

    def __init__(
        self,
        world: EnsembleWorldModel,
        evaluator: DualHeadOutcomeModel,
        action_codec: ActionCodec,
        *,
        planned_interval_days: float,
        horizon: int = 3,
        beam_width: int = 4,
        lambda_uncertainty: float = 0.0,
        uncertainty_scales: dict[int, float] | None = None,
        uncertainty_clip: float = 5.0,
        discount_scale_days: float = 365.0,
        max_candidates: int = 6,
        min_action_support: int | None = None,
    ) -> None:
        if planned_interval_days <= 0:
            raise PlanningError("planned interval must be positive")
        if horizon < 1 or beam_width < 1:
            raise PlanningError("horizon and beam_width must be positive")
        if lambda_uncertainty < 0 or uncertainty_clip <= 0 or discount_scale_days <= 0:
            raise PlanningError("planner scales must be positive")
        if world.ensemble_size < 1:
            raise PlanningError("world must contain at least one member")
        self.world = world
        self.evaluator = evaluator
        self.codec = action_codec
        self.interval = float(planned_interval_days)
        self.horizon = int(horizon)
        self.beam_width = int(beam_width)
        self.lambda_uncertainty = float(lambda_uncertainty)
        self.uncertainty_scales = uncertainty_scales or {}
        self.uncertainty_clip = float(uncertainty_clip)
        self.discount_scale_days = float(discount_scale_days)
        self.max_candidates = int(max_candidates)
        self.min_action_support = min_action_support

    def _vector(self, action: Action, reference: Tensor) -> Tensor:
        vector = torch.zeros(self.codec.dim, dtype=reference.dtype, device=reference.device)
        if action.token_ids:
            vector[list(action.token_ids)] = 1.0
        return vector

    def _normalized_uncertainty(self, disagreement: float, depth: int) -> float:
        scale = float(self.uncertainty_scales.get(depth, 0.0))
        if scale > 0:
            return min(self.uncertainty_clip, disagreement / (scale + 1e-12))
        return disagreement

    def _trajectory_value(self, node: _Node, state: PatientState) -> float:
        steps = len(node.mean_states)
        device = node.mean_states[0].device
        zero_action = torch.zeros(self.codec.dim, dtype=node.mean_states[0].dtype, device=device)
        states = torch.stack(node.mean_states).unsqueeze(0)
        actions = torch.stack((zero_action, *node.action_vectors)).unsqueeze(0)
        uncertainty = torch.tensor(
            (0.0, *node.uncertainties), dtype=states.dtype, device=device,
        ).unsqueeze(0)
        intervals = torch.tensor(
            (0.0, *node.intervals), dtype=states.dtype, device=device,
        ).unsqueeze(0)
        mask = torch.ones(1, steps, dtype=torch.bool, device=device)
        return float(self.evaluator.value(
            states,
            actions,
            uncertainty,
            intervals,
            mask,
            state.clinical.to(device).unsqueeze(0),
            state.clinical_mask.to(device).unsqueeze(0),
            node.history.unsqueeze(0),
        )[0].detach().cpu())

    @torch.no_grad()
    def plan(
        self,
        state: PatientState,
        *,
        replay_actions: Sequence[Action] = (),
        candidates: Sequence[Action] | None = None,
    ) -> MPCResult:
        if not bool(torch.isfinite(state.z).all()):
            raise PlanningError("state latent must be finite")
        selected = tuple(candidates) if candidates is not None else historical_candidates(
            self.codec,
            replay_actions,
            max_candidates=self.max_candidates,
            min_support=self.min_action_support,
        )
        if not selected:
            raise PlanningError("no supported historical actions are available")
        invalid = [action.action_id for action in selected if action.action_id not in self.codec.by_id]
        if invalid:
            raise PlanningError(f"candidate actions are outside the catalog: {invalid}")

        parameter = next(self.world.parameters(), None)
        device = parameter.device if parameter is not None else state.z.device
        initial = state.z.to(device)
        root = _Node(
            member_states=initial.expand(self.world.ensemble_size, -1).clone(),
            mean_states=(initial,),
            action_vectors=(),
            uncertainties=(),
            intervals=(),
            history=state.history.to(device),
            actions=(),
            weighted_risk=0.0,
            weighted_penalty=0.0,
            weight_total=0.0,
        )
        self.world.eval()
        self.evaluator.eval()
        beam = [root]
        expanded_nodes = 0
        forwards = 0
        for depth in range(1, self.horizon + 1):
            expanded: list[_Node] = []
            for node in beam:
                for action in selected:
                    vector = self._vector(action, node.member_states)
                    next_members = self.world.one_step_members(
                        node.member_states,
                        vector,
                        self.interval,
                        state.clinical.to(device),
                        state.clinical_mask.to(device),
                        node.history,
                    )
                    forwards += self.world.ensemble_size
                    expanded_nodes += 1
                    mean_state = next_members.mean(0)
                    disagreement = float(
                        ((next_members - mean_state) ** 2).mean().detach().cpu()
                    )
                    history = update_history(node.history, vector, self.interval)
                    child = _Node(
                        member_states=next_members,
                        mean_states=node.mean_states + (mean_state,),
                        action_vectors=node.action_vectors + (vector,),
                        uncertainties=node.uncertainties + (disagreement,),
                        intervals=node.intervals + (self.interval,),
                        history=history,
                        actions=node.actions + (action,),
                        weighted_risk=node.weighted_risk,
                        weighted_penalty=node.weighted_penalty,
                        weight_total=node.weight_total,
                    )
                    risk = self._trajectory_value(child, state)
                    normalized_u = self._normalized_uncertainty(disagreement, depth)
                    weight = math.exp(-(depth * self.interval) / self.discount_scale_days)
                    child.weighted_risk += weight * risk
                    child.weighted_penalty += weight * self.lambda_uncertainty * normalized_u
                    child.weight_total += weight
                    expanded.append(child)
            expanded.sort(key=lambda item: (
                item.score, tuple(action.action_id for action in item.actions),
            ))
            beam = expanded[:self.beam_width]
        best = beam[0]
        mean_risk = best.weighted_risk / best.weight_total
        mean_penalty = best.weighted_penalty / best.weight_total
        mean_uncertainty = (
            sum(best.uncertainties) / len(best.uncertainties)
            if best.uncertainties else 0.0
        )
        return MPCResult(
            action=best.actions[0],
            plan=best.actions,
            score=best.score,
            mean_predicted_risk=mean_risk,
            mean_disagreement=mean_uncertainty,
            cumulative_predicted_risk=best.weighted_risk,
            cumulative_uncertainty_penalty=best.weighted_penalty,
            expanded_nodes=expanded_nodes,
            world_forwards=forwards,
        )
