"""Outcome-distribution MPC and staged closed-loop execution for Cloop v4.1."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

import torch
from torch import Tensor

from ..v1.data import ActionCodec, update_history
from ..v1.types import Action, PatientState
from ..v1.world import EnsembleWorldModel
from .outcome import TrajectoryOutcomeModel


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
        if (
            action.action_id in codec.by_id
            and action.support_count >= threshold
            and len(action.token_ids) == len(set(action.token_ids))
            and all(0 <= token < codec.dim for token in action.token_ids)
        ):
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
    mean_risk_uncertainty: float
    mean_risk_variance: float
    cumulative_predicted_risk: float
    cumulative_uncertainty_penalty: float
    expanded_nodes: int
    world_forwards: int

    @property
    def mean_disagreement(self) -> float:
        """Compatibility alias; the value is outcome-risk variance in v4.1."""
        return self.mean_risk_variance


@dataclass(frozen=True)
class ClosedLoopResult:
    method: str
    actions: tuple[Action, ...]
    decisions: tuple[MPCResult, ...]
    final_state: PatientState
    replans: int
    replanning_rate: float
    mean_predicted_risk: float
    mean_risk_uncertainty: float
    uncertainty_reduction: float
    world_forwards: int


@dataclass
class _Node:
    member_states: Tensor
    member_trajectory: tuple[Tensor, ...]
    action_vectors: tuple[Tensor, ...]
    intervals: tuple[float, ...]
    history: Tensor
    actions: tuple[Action, ...]
    weighted_risk: float
    weighted_risk_uncertainty: float
    weighted_risk_variance: float
    weight_total: float

    def score(self, penalty: float) -> float:
        if self.weight_total <= 0:
            return float("inf")
        return (
            self.weighted_risk + penalty * self.weighted_risk_uncertainty
        ) / self.weight_total


class TrajectoryValueMPC:
    """Beam-search MPC scored by mean clinical risk plus risk uncertainty."""

    def __init__(
        self,
        world: EnsembleWorldModel,
        evaluator: TrajectoryOutcomeModel,
        action_codec: ActionCodec,
        *,
        planned_interval_days: float,
        horizon: int = 3,
        beam_width: int = 4,
        lambda_uncertainty: float = 0.0,
        discount_scale_days: float = 365.0,
        max_candidates: int = 6,
        min_action_support: int | None = None,
    ) -> None:
        if planned_interval_days <= 0:
            raise PlanningError("planned interval must be positive")
        if horizon < 1 or beam_width < 1:
            raise PlanningError("horizon and beam_width must be positive")
        if lambda_uncertainty < 0 or discount_scale_days <= 0:
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
        self.discount_scale_days = float(discount_scale_days)
        self.max_candidates = int(max_candidates)
        self.min_action_support = min_action_support

    def _vector(self, action: Action, reference: Tensor) -> Tensor:
        vector = torch.zeros(self.codec.dim, dtype=reference.dtype, device=reference.device)
        if action.token_ids:
            vector[list(action.token_ids)] = 1.0
        return vector

    def _trajectory_risk(
        self, node: _Node, state: PatientState,
    ) -> tuple[float, float]:
        trajectory = torch.stack(node.member_trajectory, 1).unsqueeze(1)
        mean_states = trajectory.mean(0)
        steps = mean_states.shape[1]
        device = trajectory.device
        zero_action = torch.zeros(
            self.codec.dim, dtype=trajectory.dtype, device=device,
        )
        actions = torch.stack((zero_action, *node.action_vectors)).unsqueeze(0)
        uncertainty = trajectory.var(0, unbiased=False).mean(-1)
        intervals = torch.tensor(
            (0.0, *node.intervals), dtype=trajectory.dtype, device=device,
        ).unsqueeze(0)
        mask = torch.ones(1, steps, dtype=torch.bool, device=device)
        source = torch.ones(1, steps, dtype=torch.long, device=device)
        source[0, 0] = 0 if state.source == "observed" else 1
        distribution = self.evaluator.value_distribution(
            trajectory,
            actions,
            uncertainty,
            intervals,
            mask,
            state.clinical.to(device).unsqueeze(0),
            state.clinical_mask.to(device).unsqueeze(0),
            node.history.unsqueeze(0),
            source,
        )
        return (
            float(distribution.mean_risk[0].detach().cpu()),
            float(distribution.risk_variance[0].detach().cpu()),
        )

    @torch.no_grad()
    def plan(
        self,
        state: PatientState,
        *,
        replay_actions: Sequence[Action] = (),
        candidates: Sequence[Action] | None = None,
        horizon: int | None = None,
    ) -> MPCResult:
        if not bool(torch.isfinite(state.z).all()):
            raise PlanningError("state latent must be finite")
        planning_horizon = self.horizon if horizon is None else int(horizon)
        if planning_horizon < 1:
            raise PlanningError("planning horizon must be positive")
        selected = tuple(candidates) if candidates is not None else historical_candidates(
            self.codec,
            replay_actions,
            max_candidates=self.max_candidates,
            min_support=self.min_action_support,
        )
        if not selected:
            raise PlanningError("no supported historical actions are available")
        invalid = [
            action.action_id
            for action in selected
            if action.action_id not in self.codec.by_id
        ]
        if invalid:
            raise PlanningError(f"candidate actions are outside the catalog: {invalid}")

        parameter = next(self.world.parameters(), None)
        device = parameter.device if parameter is not None else state.z.device
        initial = state.z.to(device)
        member_states = initial.expand(self.world.ensemble_size, -1).clone()
        root = _Node(
            member_states=member_states,
            member_trajectory=(member_states,),
            action_vectors=(),
            intervals=(),
            history=state.history.to(device),
            actions=(),
            weighted_risk=0.0,
            weighted_risk_uncertainty=0.0,
            weighted_risk_variance=0.0,
            weight_total=0.0,
        )
        self.world.eval()
        self.evaluator.eval()
        beam = [root]
        expanded_nodes = 0
        forwards = 0
        for depth in range(1, planning_horizon + 1):
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
                    history = update_history(node.history, vector, self.interval)
                    child = _Node(
                        member_states=next_members,
                        member_trajectory=node.member_trajectory + (next_members,),
                        action_vectors=node.action_vectors + (vector,),
                        intervals=node.intervals + (self.interval,),
                        history=history,
                        actions=node.actions + (action,),
                        weighted_risk=node.weighted_risk,
                        weighted_risk_uncertainty=node.weighted_risk_uncertainty,
                        weighted_risk_variance=node.weighted_risk_variance,
                        weight_total=node.weight_total,
                    )
                    risk, variance = self._trajectory_risk(child, state)
                    risk_uncertainty = math.sqrt(max(0.0, variance))
                    weight = math.exp(
                        -(depth * self.interval) / self.discount_scale_days,
                    )
                    child.weighted_risk += weight * risk
                    child.weighted_risk_uncertainty += weight * risk_uncertainty
                    child.weighted_risk_variance += weight * variance
                    child.weight_total += weight
                    expanded.append(child)
            expanded.sort(
                key=lambda item: (
                    item.score(self.lambda_uncertainty),
                    tuple(action.action_id for action in item.actions),
                )
            )
            beam = expanded[:self.beam_width]
        best = beam[0]
        return MPCResult(
            action=best.actions[0],
            plan=best.actions,
            score=best.score(self.lambda_uncertainty),
            mean_predicted_risk=best.weighted_risk / best.weight_total,
            mean_risk_uncertainty=best.weighted_risk_uncertainty / best.weight_total,
            mean_risk_variance=best.weighted_risk_variance / best.weight_total,
            cumulative_predicted_risk=best.weighted_risk,
            cumulative_uncertainty_penalty=(
                self.lambda_uncertainty * best.weighted_risk_uncertainty
            ),
            expanded_nodes=expanded_nodes,
            world_forwards=forwards,
        )

    @torch.no_grad()
    def _simulated_observation(
        self, state: PatientState, action: Action,
    ) -> PatientState:
        parameter = next(self.world.parameters(), None)
        device = parameter.device if parameter is not None else state.z.device
        current = state.z.to(device).expand(self.world.ensemble_size, -1).clone()
        vector = self._vector(action, current)
        members = self.world.one_step_members(
            current,
            vector,
            self.interval,
            state.clinical.to(device),
            state.clinical_mask.to(device),
            state.history.to(device),
        )
        return PatientState(
            patient_key=state.patient_key,
            timepoint_key=f"{state.timepoint_key}+{state.state_version + 1}",
            observed_day=state.observed_day + self.interval,
            z=members.mean(0),
            clinical=state.clinical.to(device),
            clinical_mask=state.clinical_mask.to(device),
            history=update_history(state.history.to(device), vector, self.interval),
            source="imagined",
            state_version=state.state_version + 1,
        )

    @torch.no_grad()
    def execute(
        self,
        state: PatientState,
        *,
        stages: int,
        method: str = "mpc",
        replay_actions: Sequence[Action] = (),
        candidates: Sequence[Action] | None = None,
        observer: Callable[[PatientState, Action, int], PatientState] | None = None,
    ) -> ClosedLoopResult:
        """Execute open-loop, full-horizon closed-loop, or rolling-window MPC."""
        if stages < 1:
            raise PlanningError("execution stages must be positive")
        if method not in {"open_loop", "closed_loop", "mpc", "uncertainty_mpc"}:
            raise PlanningError(f"unknown execution method: {method}")
        selected = tuple(candidates) if candidates is not None else historical_candidates(
            self.codec,
            replay_actions,
            max_candidates=self.max_candidates,
            min_support=self.min_action_support,
        )
        current = state
        decisions: list[MPCResult] = []
        actions: list[Action] = []
        open_plan: tuple[Action, ...] = ()
        if method == "open_loop":
            decision = self.plan(
                current,
                replay_actions=replay_actions,
                candidates=selected,
                horizon=stages,
            )
            decisions.append(decision)
            open_plan = decision.plan
        for stage in range(stages):
            if method == "open_loop":
                action = open_plan[stage]
            else:
                window = stages - stage if method == "closed_loop" else self.horizon
                decision = self.plan(
                    current,
                    replay_actions=replay_actions,
                    candidates=selected,
                    horizon=window,
                )
                decisions.append(decision)
                action = decision.action
            actions.append(action)
            current = (
                observer(current, action, stage)
                if observer is not None
                else self._simulated_observation(current, action)
            )
        replans = max(0, len(decisions) - 1)
        possible_replans = max(1, stages - 1)
        uncertainties = [decision.mean_risk_uncertainty for decision in decisions]
        return ClosedLoopResult(
            method=method,
            actions=tuple(actions),
            decisions=tuple(decisions),
            final_state=current,
            replans=replans,
            replanning_rate=replans / possible_replans if stages > 1 else 0.0,
            mean_predicted_risk=float(sum(
                decision.mean_predicted_risk for decision in decisions
            ) / len(decisions)),
            mean_risk_uncertainty=float(sum(uncertainties) / len(uncertainties)),
            uncertainty_reduction=(
                uncertainties[0] - uncertainties[-1] if len(uncertainties) > 1 else 0.0
            ),
            world_forwards=sum(decision.world_forwards for decision in decisions),
        )

    def execute_open_loop(self, state: PatientState, *, stages: int, **kwargs: object) -> ClosedLoopResult:
        return self.execute(state, stages=stages, method="open_loop", **kwargs)

    def execute_closed_loop(
        self,
        state: PatientState,
        *,
        stages: int,
        mpc: bool = False,
        **kwargs: object,
    ) -> ClosedLoopResult:
        return self.execute(
            state, stages=stages, method="mpc" if mpc else "closed_loop", **kwargs,
        )
