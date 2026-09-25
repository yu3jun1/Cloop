"""Offline catalog policy and provider-neutral constrained LLM policy."""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Callable, Sequence

from .config import canonical_json
from .data import ActionCodec, DataError
from .types import CandidateBatch, LLMProvider, PolicyRequest


class PolicyError(RuntimeError):
    pass


def _plain_json(value: Any) -> Any:
    if isinstance(value, (tuple, list)):
        return [_plain_json(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _plain_json(item) for key, item in value.items()}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "numel") and callable(value.numel) and value.numel() == 1:
        return value.item()
    raise PolicyError(f"request contains a non-serializable value: {type(value).__name__}")


def request_hash(request: PolicyRequest) -> str:
    safe = {
        "observation_day": request.observation_day,
        "clinical_summary": request.clinical_summary,
        "executed_history_summary": request.executed_history_summary,
        "allowed_action_ids": request.allowed_action_ids,
        "max_candidates": request.max_candidates,
        "request_seed": request.request_seed,
        "state_source": request.state_source,
        "state_version": request.state_version,
    }
    return hashlib.sha256(canonical_json(_plain_json(safe)).encode()).hexdigest()


class CatalogPolicy:
    def __init__(self, codec: ActionCodec, max_candidates: int = 6):
        self.codec = codec
        self.max_candidates = int(max_candidates)

    def propose(self, request: PolicyRequest) -> CandidateBatch:
        start = time.perf_counter()
        allowed = set(request.allowed_action_ids)
        count = min(self.max_candidates, request.max_candidates)
        actions = tuple(action for action in self.codec.catalog if action.action_id in allowed)[:count]
        return CandidateBatch(
            candidates=actions,
            source="catalog",
            latency_ms=(time.perf_counter() - start) * 1000,
            request_hash=request_hash(request),
        )


class FakeProvider:
    """Deterministic provider for contract tests; it never performs I/O."""

    def __init__(self, candidate_ids: Sequence[str] | Callable[[dict[str, Any]], Sequence[str]], error: Exception | None = None):
        self.candidate_ids = candidate_ids
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def generate_json(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        schema: dict[str, Any],
        temperature: float,
        timeout_s: float,
    ) -> dict[str, Any]:
        call = {
            "model": model,
            "messages": messages,
            "schema": schema,
            "temperature": temperature,
            "timeout_s": timeout_s,
        }
        self.calls.append(call)
        if self.error is not None:
            raise self.error
        ids = self.candidate_ids(call) if callable(self.candidate_ids) else self.candidate_ids
        return {"candidate_ids": list(ids)}


class LLMPolicy:
    """The provider may only select catalog IDs; free-form treatments are rejected."""

    schema = {
        "type": "object",
        "properties": {"candidate_ids": {"type": "array", "items": {"type": "string"}}},
        "required": ["candidate_ids"],
        "additionalProperties": False,
    }

    def __init__(
        self,
        codec: ActionCodec,
        provider: LLMProvider,
        *,
        model: str,
        temperature: float = 0.0,
        timeout_s: float = 30.0,
        max_retries: int = 1,
        fallback: CatalogPolicy | None = None,
        allow_network: bool = False,
        provider_is_fake: bool = False,
    ):
        if not allow_network and not provider_is_fake:
            raise PolicyError("LLM policy is disabled because allow_network=false")
        self.codec = codec
        self.provider = provider
        self.model = model
        self.temperature = float(temperature)
        self.timeout_s = float(timeout_s)
        self.max_retries = int(max_retries)
        self.fallback = fallback

    def _messages(self, request: PolicyRequest) -> list[dict[str, Any]]:
        # No patient key, factual target, survival label, raw note, or latent vector appears here.
        payload = {
            "observation_day": request.observation_day,
            "clinical_summary": request.clinical_summary,
            "executed_history_summary": request.executed_history_summary,
            "allowed_action_ids": request.allowed_action_ids,
            "max_candidates": request.max_candidates,
            "request_seed": request.request_seed,
            "state_source": request.state_source,
        }
        return [
            {"role": "system", "content": "Select and rank only valid action IDs from the supplied catalog."},
            {"role": "user", "content": json.dumps(_plain_json(payload), separators=(",", ":"))},
        ]

    def propose(self, request: PolicyRequest) -> CandidateBatch:
        start = time.perf_counter()
        failures = 0
        calls = 0
        last_error = "unknown"
        for _ in range(self.max_retries + 1):
            calls += 1
            try:
                response = self.provider.generate_json(
                    model=self.model,
                    messages=self._messages(request),
                    schema=self.schema,
                    temperature=self.temperature,
                    timeout_s=self.timeout_s,
                )
                if not isinstance(response, dict):
                    raise PolicyError("provider response is not a JSON object")
                ids = response.get("candidate_ids")
                if not isinstance(ids, list) or any(not isinstance(x, str) for x in ids):
                    raise PolicyError("provider response does not match candidate_ids schema")
                allowed = set(request.allowed_action_ids)
                seen: set[str] = set()
                duplicate = outside = 0
                valid = []
                for action_id in ids:
                    if action_id in seen:
                        duplicate += 1
                        continue
                    seen.add(action_id)
                    if action_id not in allowed:
                        outside += 1
                        continue
                    try:
                        valid.append(self.codec.decode(action_id))
                    except DataError:
                        outside += 1
                    if len(valid) >= request.max_candidates:
                        break
                if not valid:
                    raise PolicyError("provider returned no valid in-catalog action")
                return CandidateBatch(
                    candidates=tuple(valid),
                    source="llm",
                    parse_failures=failures,
                    duplicate_count=duplicate,
                    out_of_catalog_count=outside,
                    latency_ms=(time.perf_counter() - start) * 1000,
                    call_count=calls,
                    request_hash=request_hash(request),
                )
            except (PolicyError, TimeoutError, ValueError, TypeError) as exc:
                failures += 1
                last_error = type(exc).__name__
        if self.fallback is not None:
            fallback = self.fallback.propose(request)
            return CandidateBatch(
                candidates=fallback.candidates,
                source="catalog_fallback",
                parse_failures=failures,
                fallback=True,
                latency_ms=(time.perf_counter() - start) * 1000,
                call_count=calls,
                request_hash=request_hash(request),
            )
        raise PolicyError(f"LLM proposal failed after {calls} calls: {last_error}")
