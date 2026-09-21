"""Hard constraints, shared-action ensemble beam search, and receding-horizon decisions."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import Tensor

from .data import ActionCodec, update_history
from .types import Action, Decision, PatientState, PolicyAgent, PolicyRequest, StateCost
from .world import EnsembleWorldModel


class PlannerError(RuntimeError):
    pass


@dataclass
class PlanNode:
    member_states: Tensor
    history: Tensor
    elapsed_days: float
    path: tuple[Action, ...]
    weighted_cost: float
    weight_total: float
    last_mean_cost: float
    last_disagreement: float

    @property
    def score(self) -> float:
        return self.weighted_cost / self.weight_total if self.weight_total else float("inf")


class Planner:
    def __init__(
        self,
        world: EnsembleWorldModel | None,
        state_cost: StateCost | None,
        policy: PolicyAgent,
        action_codec: ActionCodec,
        *,
        planned_interval_days: float,
        interval_support_days: tuple[float, float] | None = None,
        horizon: int = 3,
        beam_width: int = 4,
        lambda_uncertainty: float = 0.0,
        uncertainty_scales: dict[int, float] | None = None,
        uncertainty_clip: float = 5.0,
        discount_scale_days: float = 365.0,
        max_candidates: int = 6,
        min_action_support: int = 2,
        mode: Literal["frequency", "greedy", "mpc"] = "mpc",
        seed: int = 17,
    ):
        if planned_interval_days <= 0:
            raise PlannerError("planned interval must be positive")
        if horizon < 1 or beam_width < 1:
            raise PlannerError("horizon and beam width must be positive")
        if mode != "frequency" and (world is None or state_cost is None):
            raise PlannerError("model-based planning requires world and state_cost")
        self.world = world
        self.state_cost = state_cost
        self.policy = policy
        self.codec = action_codec
        self.interval = float(planned_interval_days)
        self.interval_support = interval_support_days
        self.interval_extrapolation = bool(
            interval_support_days is not None
            and not (float(interval_support_days[0]) <= self.interval <= float(interval_support_days[1]))
        )
        self.horizon = 1 if mode == "greedy" else int(horizon)
        self.beam_width = int(beam_width)
        self.lambda_uncertainty = float(lambda_uncertainty)
        self.uncertainty_scales = uncertainty_scales or {}
        self.uncertainty_clip = float(uncertainty_clip)
        self.discount_scale_days = float(discount_scale_days)
        self.max_candidates = int(max_candidates)
        self.min_action_support = int(min_action_support)
        self.mode = mode
        self.seed = int(seed)
        self._cache: dict[tuple[Any, ...], Tensor] = {}
        self._cache_version: int | None = None

    def _request(self, state: PatientState, history: Tensor) -> PolicyRequest:
        visible = tuple(
            float(value) if float(mask) > 0 else None
            for value, mask in zip(state.clinical.detach().cpu(), state.clinical_mask.detach().cpu())
        )
        return PolicyRequest(
            observation_day=state.observed_day,
            clinical_summary=visible,
            executed_history_summary=tuple(float(x) for x in history.detach().cpu()),
            allowed_action_ids=tuple(action.action_id for action in self.codec.catalog),
            max_candidates=self.max_candidates,
            request_seed=self.seed,
            state_source=state.source,
            state_version=state.state_version,
        )

    def _valid(self, action: Action) -> tuple[bool, str | None]:
        if action.action_id not in self.codec.by_id:
            return False, "out_of_catalog"
        if len(action.token_ids) != len(set(action.token_ids)):
            return False, "duplicate_token"
        if any(token < 0 or token >= self.codec.dim for token in action.token_ids):
            return False, "invalid_token"
        if self.codec.unk_index is not None and self.codec.unk_index in action.token_ids:
            return False, "unknown_token"
        if action.support_count < self.min_action_support:
            return False, "insufficient_support"
        if action.known_empty and action.token_ids:
            return False, "invalid_empty_action"
        return True, None

    def _vector(self, action: Action, *, device: torch.device, dtype: torch.dtype) -> Tensor:
        vector = torch.zeros(self.codec.dim, device=device, dtype=dtype)
        if action.token_ids:
            vector[list(action.token_ids)] = 1.0
        return vector

    def _advance(self, state: PatientState, node: PlanNode, action: Action, depth: int) -> tuple[PlanNode, int]:
        assert self.world is not None and self.state_cost is not None
        vector = self._vector(action, device=node.member_states.device, dtype=node.member_states.dtype)
        key = (state.state_version, tuple(a.action_id for a in node.path), action.action_id, self.interval)
        if key in self._cache:
            next_states = self._cache[key]
            forwards = 0
        else:
            next_states = self.world.one_step_members(
                node.member_states,
                vector,
                self.interval,
                state.clinical.to(node.member_states.device),
                state.clinical_mask.to(node.member_states.device),
                node.history,
            )
            self._cache[key] = next_states
            forwards = self.world.ensemble_size
        member_count = next_states.shape[0]
        clinical = state.clinical.to(next_states.device).expand(member_count, -1)
        clinical_mask = state.clinical_mask.to(next_states.device).expand(member_count, -1)
        next_history = update_history(node.history, vector, self.interval)
        histories = next_history.expand(member_count, -1)
        member_cost = self.state_cost.state_cost(next_states, clinical, clinical_mask, histories)
        mean_cost = float(member_cost.mean().detach().cpu())
        mean_state = next_states.mean(0)
        disagreement = float(((next_states - mean_state) ** 2).mean().detach().cpu())
        scale = float(self.uncertainty_scales.get(depth, 0.0))
        if scale > 0:
            normalized_u = min(self.uncertainty_clip, disagreement / (scale + 1e-12))
        elif disagreement == 0:
            normalized_u = 0.0
        else:
            normalized_u = 0.0  # disabled, never fabricate a scale
        elapsed = node.elapsed_days + self.interval
        weight = math.exp(-elapsed / self.discount_scale_days)
        step_cost = mean_cost + self.lambda_uncertainty * normalized_u
        return (
            PlanNode(
                member_states=next_states,
                history=next_history,
                elapsed_days=elapsed,
                path=node.path + (action,),
                weighted_cost=node.weighted_cost + weight * step_cost,
                weight_total=node.weight_total + weight,
                last_mean_cost=mean_cost,
                last_disagreement=disagreement,
            ),
            forwards,
        )

    def plan(self, state: PatientState) -> Decision:
        started = time.perf_counter()
        # The rollout cache is scoped to one decision and is never reused across patients/observations.
        self._cache.clear()
        self._cache_version = state.state_version
        if not bool(torch.isfinite(state.z).all()) or not bool(torch.isfinite(state.clinical).all()):
            return Decision(None, "abstain", ("incomplete_or_nonfinite_state",), (), {}, state.state_version)
        initial_batch = self.policy.propose(self._request(state, state.history))
        valid_initial = []
        invalid_count = 0
        for action in initial_batch.candidates:
            valid, _ = self._valid(action)
            if valid:
                valid_initial.append(action)
            else:
                invalid_count += 1
        if not valid_initial:
            return Decision(
                None,
                "abstain",
                ("no_valid_candidate",),
                (),
                {},
                state.state_version,
                {"candidate_count": len(initial_batch.candidates), "invalid_count": invalid_count},
            )
        if self.mode == "frequency":
            selected = valid_initial[0]
            return Decision(
                selected,
                "recommend",
                ("frequency_baseline",),
                (selected,),
                {"score": None, "mean_cost": None, "disagreement": None},
                state.state_version,
                {
                    "candidate_count": len(initial_batch.candidates),
                    "valid_candidate_count": len(valid_initial),
                    "world_model_forwards": 0,
                    "beam_nodes": 0,
                    "planned_interval_extrapolation": self.interval_extrapolation,
                    "wall_time_ms": (time.perf_counter() - started) * 1000,
                },
            )
        assert self.world is not None
        member_states = state.z.to(next(self.world.parameters(), state.z).device).expand(
            self.world.ensemble_size, -1
        ).clone()
        root = PlanNode(member_states, state.history.to(member_states.device), 0.0, (), 0.0, 0.0, 0.0, 0.0)
        beam = [root]
        forwards = 0
        beam_nodes = 0
        calls = initial_batch.call_count
        for depth in range(1, self.horizon + 1):
            expanded: list[PlanNode] = []
            for node_index, node in enumerate(beam):
                if depth == 1 and node_index == 0:
                    candidates = valid_initial
                else:
                    batch = self.policy.propose(self._request(state, node.history))
                    calls += batch.call_count
                    candidates = [action for action in batch.candidates if self._valid(action)[0]]
                for action in candidates:
                    child, count = self._advance(state, node, action, depth)
                    forwards += count
                    beam_nodes += 1
                    expanded.append(child)
            if not expanded:
                return Decision(
                    None,
                    "abstain",
                    ("no_feasible_horizon_plan",),
                    (),
                    {},
                    state.state_version,
                    {"reached_depth": depth - 1, "world_model_forwards": forwards},
                )
            expanded.sort(key=lambda node: (node.score, tuple(action.action_id for action in node.path)))
            beam = expanded[: self.beam_width]
        best = beam[0]
        return Decision(
            recommended_action=best.path[0],
            status="recommend",
            reason_codes=("model_based_plan",),
            imagined_plan=best.path,
            score_components={
                "score": best.score,
                "mean_cost": best.last_mean_cost,
                "disagreement": best.last_disagreement,
            },
            state_version=state.state_version,
            diagnostics={
                "candidate_count": len(initial_batch.candidates),
                "valid_candidate_count": len(valid_initial),
                "invalid_count": invalid_count,
                "world_model_forwards": forwards,
                "beam_nodes": beam_nodes,
                "policy_calls": calls,
                "fallback": initial_batch.fallback,
                "planned_interval_extrapolation": self.interval_extrapolation,
                "wall_time_ms": (time.perf_counter() - started) * 1000,
            },
        )
