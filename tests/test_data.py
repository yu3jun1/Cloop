from __future__ import annotations

import copy

import torch

from cloop.data import (
    ActionCodec,
    Normalizer,
    _survival_label,
    align_interval_actions,
    collect_events,
    fit_preprocessing,
    load_bundle,
    split_patients,
    window_refs,
)
from cloop.types import Action, Decision


def test_patient_split_is_mutually_exclusive_reproducible_and_order_independent():
    ids = [f"P{i}" for i in range(40)]
    first = split_patients(ids, 17)
    second = split_patients(reversed(ids), 17)
    assert first == second
    assert not (set(first["train"]) & set(first["validation"]))
    assert not (set(first["train"]) & set(first["test"]))
    assert set().union(*map(set, first.values())) == set(ids)


def test_normalizer_and_vocab_are_train_only(config, tiny):
    cache, bundle = tiny
    original = fit_preprocessing(cache, config, bundle.split_ids)
    changed = copy.deepcopy(cache)
    test_pid = bundle.split_ids["test"][0]
    row = next(item for item in changed["patients"] if item["patient_id"] == test_pid)
    row["latents_raw"] += 10000
    row["action_terms"][0] = ["category:test-only-action"]
    after = fit_preprocessing(changed, config, bundle.split_ids)
    assert torch.equal(original["latent_normalizer"]["mean"], after["latent_normalizer"]["mean"])
    assert original["action_codec"]["vocab"] == after["action_codec"]["vocab"]


def test_interval_alignment_deduplicates_and_uses_timestamps():
    event = {"agent": "Drug X", "start_day": 25, "end_day": 75}
    timeline = [
        {"tp_id": "TP1", "mri_day": 0, "actions": {"chemotherapy": [event]}},
        {"tp_id": "TP2", "mri_day": 50, "actions": {"chemotherapy": [event]}},
        {"tp_id": "TP3", "mri_day": 100, "actions": {"chemotherapy": []}},
    ]
    events = collect_events(timeline)
    actions, known, audit = align_interval_actions(timeline, events)
    assert len(events) == 1
    assert known == [True, True]
    assert actions[0] == actions[1] == ("agent:drug x", "category:chemotherapy")
    assert audit["known_interval"] == 2


def test_undated_destination_event_breaks_continuity_without_becoming_history():
    timeline = [
        {"tp_id": "TP1", "mri_day": 0, "actions": {}},
        {
            "tp_id": "TP2",
            "mri_day": 50,
            "actions": {"chemotherapy": [{"agent": "x", "duration_unknown": True, "assigned_reason": "legacy"}]},
        },
        {"tp_id": "TP3", "mri_day": 100, "actions": {}},
    ]
    actions, known, _ = align_interval_actions(timeline, collect_events(timeline))
    assert actions[0] == ()
    assert known == [False, True]


def test_unknown_empty_and_abstain_are_distinct():
    codec = ActionCodec.fit([(), ("category:a",)], min_support=1)
    empty = next(action for action in codec.catalog if action.known_empty)
    unknown_action = None
    abstain = Decision(None, "abstain", ("no_valid_candidate",), (), {}, 1)
    assert empty.action_id == "EMPTY"
    assert unknown_action is None
    assert abstain.status == "abstain" and abstain.recommended_action is None


def test_future_fields_do_not_enter_state_or_catalog(config, tiny):
    cache, bundle = tiny
    train_vocab = set(bundle.action_codec.vocab)
    test = bundle.trajectories[bundle.split_ids["test"][0]]
    state_fields = set(test.__dict__)
    assert "survival_time" in state_fields  # trajectory storage is evaluator-only
    from cloop.data import patient_state

    visible = patient_state(bundle, test.patient_id, 0, 0)
    assert not hasattr(visible, "survival_time")
    assert not hasattr(visible, "event")
    assert "category:test-only" not in train_vocab


def test_shifted_death_label_is_excluded_and_regular_censoring_retained():
    shifted = {"survival": {"survival_from_tp_days": 12, "event_indicator": 1, "censoring_rule": "death_shifted_to_L_plus_1"}}
    regular = {"survival": {"survival_from_tp_days": 12, "event_indicator": 0, "censoring_rule": "no_death_last_mri"}}
    assert _survival_label(shifted, True)[2] is False
    assert _survival_label(regular, True)[2] is True


def test_continuity_break_prevents_rrt_window_crossing(config, tiny):
    cache, bundle = tiny
    pid = bundle.split_ids["train"][0]
    bundle.trajectories[pid].action_known[1] = False
    refs = [ref for ref in window_refs(bundle, "train", mode="max_available", max_horizon=3) if ref.patient_id == pid]
    assert all(not (ref.start <= 1 < ref.start + ref.horizon) for ref in refs)

