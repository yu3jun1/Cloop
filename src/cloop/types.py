"""Data-transfer types. Factual labels are deliberately separate from policy state."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from torch import Tensor


@dataclass(frozen=True)
class Action:
    action_id: str
    token_ids: tuple[int, ...]
    display_terms: tuple[str, ...]
    support_count: int
    known_empty: bool = False


@dataclass(frozen=True)
class PatientState:
    patient_key: str
    timepoint_key: str
    observed_day: float
    z: Tensor
    clinical: Tensor
    clinical_mask: Tensor
    history: Tensor
    source: Literal["observed", "imagined"]
    state_version: int


@dataclass(frozen=True)
class FactualTarget:
    next_z: Tensor
    actual_action: Action | None
    actual_delta_days: float
    survival_time: float | None
    event: int | None


@dataclass(frozen=True)
class PolicyRequest:
    observation_day: float
    clinical_summary: tuple[float | int | str | None, ...]
    executed_history_summary: tuple[float, ...]
    allowed_action_ids: tuple[str, ...]
    max_candidates: int
    request_seed: int
    state_source: Literal["observed", "imagined"]
    state_version: int


@dataclass(frozen=True)
class CandidateBatch:
    candidates: tuple[Action, ...]
    source: str
    parse_failures: int = 0
    duplicate_count: int = 0
    out_of_catalog_count: int = 0
    fallback: bool = False
    latency_ms: float = 0.0
    call_count: int = 0
    request_hash: str | None = None


@dataclass(frozen=True)
class Rollout:
    states: Tensor  # [M, B, K+1, D]
    step_mask: Tensor  # bool [B, K]


@dataclass(frozen=True)
class Decision:
    recommended_action: Action | None
    status: Literal["recommend", "abstain"]
    reason_codes: tuple[str, ...]
    imagined_plan: tuple[Action, ...]
    score_components: dict[str, float | None]
    state_version: int
    diagnostics: dict[str, Any] = field(default_factory=dict)


class PolicyAgent(Protocol):
    def propose(self, request: PolicyRequest) -> CandidateBatch: ...


class LLMProvider(Protocol):
    def generate_json(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        schema: dict[str, Any],
        temperature: float,
        timeout_s: float,
    ) -> dict[str, Any]: ...


class StateCost(Protocol):
    def state_cost(
        self, z: Tensor, clinical: Tensor, clinical_mask: Tensor, history: Tensor
    ) -> Tensor: ...

