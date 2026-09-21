from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from cloop.data import ActionCodec, empty_history, patient_state, update_history
from cloop.engine import _decision_record
from cloop.planner import Planner
from cloop.policy import CatalogPolicy, FakeProvider, LLMPolicy, PolicyError
from cloop.synthetic import ToyEnv, ToyState
from cloop.types import Action, CandidateBatch, PatientState, PolicyRequest
from cloop.world import EnsembleWorldModel


class ActionDynamics(nn.Module):
    def __init__(self, weights, bias=0.0):
        super().__init__()
        self.register_buffer("weights", torch.tensor(weights, dtype=torch.float32))
        self.bias = float(bias)
        self.actions = []

    def forward(self, z, action, delta, clinical, mask, history):
        self.actions.append(action.detach().clone())
        return z + action @ self.weights.unsqueeze(-1) + self.bias


class SquareCost(nn.Module):
    def state_cost(self, z, clinical, clinical_mask, history):
        return z.square().mean(-1)


class InvalidPolicy:
    def propose(self, request):
        invalid = Action("OUTSIDE", (999,), ("invalid",), 0)
        return CandidateBatch((invalid,), "invalid")


def _setup_planner_state():
    codec = ActionCodec.fit([("category:a0",)] * 3 + [("category:a1",)] * 2, min_support=1)
    weights = [0.0] * codec.dim
    weights[codec.index["category:a0"]] = -1.0
    weights[codec.index["category:a1"]] = 1.0
    members = [ActionDynamics(weights, -0.05), ActionDynamics(weights, 0.05)]
    world = EnsembleWorldModel(members)
    policy = CatalogPolicy(codec, 2)
    state = PatientState(
        "P", "T0", 0.0, torch.tensor([2.0]), torch.zeros(1), torch.ones(1),
        empty_history(codec.dim), "observed", 0,
    )
    return codec, world, policy, state


def _planner(codec, world, policy, **kwargs):
    return Planner(
        world, SquareCost(), policy, codec, planned_interval_days=30.0,
        horizon=kwargs.pop("horizon", 1), beam_width=2, max_candidates=2,
        min_action_support=1, **kwargs,
    )


def test_catalog_works_without_keys_or_network(monkeypatch):
    monkeypatch.delenv("CLOOP_API_KEY", raising=False)
    codec, _, policy, state = _setup_planner_state()
    request = PolicyRequest(0, (0.0,), tuple(state.history), tuple(codec.by_id), 2, 17, "observed", 0)
    assert policy.propose(request).candidates
    with pytest.raises(PolicyError):
        LLMPolicy(codec, FakeProvider([]), model="x", allow_network=False)


def test_fake_provider_and_catalog_candidates_score_identically():
    codec, world, catalog, state = _setup_planner_state()
    catalog_decision = _planner(codec, world, catalog).plan(state)
    ids = [action.action_id for action in catalog.propose(_planner(codec, world, catalog)._request(state, state.history)).candidates]
    llm = LLMPolicy(codec, FakeProvider(ids), model="fake", allow_network=False, provider_is_fake=True)
    llm_decision = _planner(codec, world, llm).plan(state)
    assert catalog_decision.recommended_action == llm_decision.recommended_action
    assert catalog_decision.score_components["score"] == pytest.approx(llm_decision.score_components["score"])


def test_model_string_only_changes_provider_request():
    codec, world, catalog, state = _setup_planner_state()
    ids = tuple(codec.by_id)
    providers = [FakeProvider(ids), FakeProvider(ids)]
    decisions = []
    for model, provider in zip(("one", "two"), providers):
        policy = LLMPolicy(codec, provider, model=model, allow_network=False, provider_is_fake=True)
        decisions.append(_planner(codec, world, policy).plan(state))
    assert decisions[0].recommended_action == decisions[1].recommended_action
    assert providers[0].calls[0]["model"] == "one"
    assert providers[1].calls[0]["model"] == "two"


def test_duplicate_outside_and_timeout_are_explicit():
    codec, _, catalog, state = _setup_planner_state()
    valid = next(iter(codec.by_id))
    request = PolicyRequest(0, (), tuple(state.history), tuple(codec.by_id), 4, 17, "observed", 0)
    policy = LLMPolicy(
        codec, FakeProvider([valid, valid, "OUTSIDE"]), model="fake",
        allow_network=False, provider_is_fake=True,
    )
    batch = policy.propose(request)
    assert batch.duplicate_count == 1 and batch.out_of_catalog_count == 1
    timeout = LLMPolicy(
        codec, FakeProvider([], error=TimeoutError()), model="fake", max_retries=1,
        fallback=catalog, allow_network=False, provider_is_fake=True,
    ).propose(request)
    assert timeout.fallback and timeout.parse_failures == 2 and timeout.call_count == 2


def test_only_first_action_is_committed_and_history_is_not_mutated():
    codec, world, catalog, state = _setup_planner_state()
    before = state.history.clone()
    decision = _planner(codec, world, catalog, horizon=3).plan(state)
    assert len(decision.imagined_plan) == 3
    assert decision.recommended_action == decision.imagined_plan[0]
    assert torch.equal(state.history, before)


def test_new_observation_can_change_greedy_action_and_cache_is_decision_scoped():
    codec, world, catalog, positive = _setup_planner_state()
    planner = _planner(codec, world, catalog)
    first = planner.plan(positive)
    calls_after_first = sum(len(member.actions) for member in world.members)
    negative = PatientState(
        "P", "T1", 30.0, torch.tensor([-2.0]), positive.clinical, positive.clinical_mask,
        positive.history, "observed", 1,
    )
    second = planner.plan(negative)
    assert first.recommended_action != second.recommended_action
    assert sum(len(member.actions) for member in world.members) > calls_after_first


def test_h3_mpc_reobserves_and_replans_instead_of_following_fixed_plan():
    codec, world, catalog, initial = _setup_planner_state()
    planner = _planner(codec, world, catalog, horizon=3)
    first = planner.plan(initial)
    assert len(first.imagined_plan) == 3
    executed = first.recommended_action
    assert executed is not None
    action_vector = torch.zeros(codec.dim)
    action_vector[list(executed.token_ids)] = 1.0
    observed_history = update_history(initial.history, action_vector, 30.0)
    reobserved = PatientState(
        "P",
        "T1",
        30.0,
        torch.tensor([-2.0]),
        initial.clinical,
        initial.clinical_mask,
        observed_history,
        "observed",
        1,
    )
    replanned = planner.plan(reobserved)
    assert replanned.recommended_action is not None
    assert replanned.recommended_action != first.imagined_plan[1]


def test_all_invalid_candidates_abstain_before_world_call():
    codec, world, _, state = _setup_planner_state()
    decision = _planner(codec, world, InvalidPolicy()).plan(state)
    assert decision.status == "abstain"
    assert decision.reason_codes == ("no_valid_candidate",)
    assert all(not member.actions for member in world.members)


def test_every_member_receives_same_action_and_keeps_own_latent():
    codec, world, catalog, state = _setup_planner_state()
    _planner(codec, world, catalog, horizon=2).plan(state)
    assert torch.equal(world.members[0].actions[0], world.members[1].actions[0])


def test_historical_mismatch_record_is_not_counterfactual_mse():
    codec, world, catalog, state = _setup_planner_state()
    decision = _planner(codec, world, catalog).plan(state)
    actual = next(action for action in codec.catalog if action != decision.recommended_action)
    row = _decision_record(
        split="test", method="greedy", seed=17, state=state, actual=actual,
        decision=decision, catalog_ids=set(codec.by_id), candidate_ids=set(codec.by_id),
    )
    assert row["recommended_action"] != row["actual_action"]
    assert "mse" not in row
    assert row["evidence_level"] == "historical_action_agreement_only"


def test_decision_record_compares_against_the_prior_future_plan():
    codec, world, catalog, state = _setup_planner_state()
    decision = _planner(codec, world, catalog, horizon=3).plan(state)
    current_action = decision.recommended_action.action_id

    stable = _decision_record(
        split="test", method="mpc_rrt_ensemble", seed=17, state=state,
        actual=decision.recommended_action, decision=decision,
        catalog_ids=set(codec.by_id), candidate_ids=set(codec.by_id),
        previously_planned_action_for_this_stage=current_action,
        plan_revision_evaluable=True,
    )
    assert stable["previously_planned_action_for_this_stage"] == current_action
    assert stable["plan_revision_evaluable"] is True
    assert stable["plan_revised"] is False

    other_action = next(action_id for action_id in codec.by_id if action_id != current_action)
    revised = _decision_record(
        split="test", method="mpc_rrt_ensemble", seed=17, state=state,
        actual=decision.recommended_action, decision=decision,
        catalog_ids=set(codec.by_id), candidate_ids=set(codec.by_id),
        previously_planned_action_for_this_stage=other_action,
        plan_revision_evaluable=True,
    )
    assert revised["plan_revised"] is True

    greedy = _decision_record(
        split="test", method="greedy", seed=17, state=state,
        actual=decision.recommended_action, decision=decision,
        catalog_ids=set(codec.by_id), candidate_ids=set(codec.by_id),
    )
    assert greedy["plan_revision_evaluable"] is False
    assert greedy["plan_revised"] is None


def test_synthetic_environment_transition_uses_executed_action():
    first = ToyEnv(ToyState(3, 1, 0), [0]).step(0)[0]
    second = ToyEnv(ToyState(3, 1, 0), [0]).step(2)[0]
    assert first != second
