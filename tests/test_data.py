from __future__ import annotations

import copy
import json

import numpy as np
import pytest
import torch

from cloop.data import (
    ActionCodec,
    DataError,
    LegacyNormalizer,
    Normalizer,
    _survival_label,
    align_interval_actions,
    build_main_cache,
    collect_events,
    empty_history,
    fit_preprocessing,
    load_bundle,
    sha256_file,
    split_patients,
    update_history,
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


def test_history_completed_interval_count_uses_integer_semantics():
    history = empty_history(2)
    action = torch.tensor([1.0, 0.0])
    first = update_history(history, action, 30.0)
    second = update_history(first, action, 30.0)
    assert first[4].item() == 1.0
    assert second[4].item() == 2.0


def _main_cache_fixture(tmp_path, config, sources=None):
    latent_dir = tmp_path / "latents"
    latent_dir.mkdir()
    sources = sources or [
        "observed",
        "imputed_interior",
        "imputed_leading",
        "imputed_trailing",
    ]
    timeline = []
    for index, source in enumerate(sources, start=1):
        np.save(
            latent_dir / f"P1_Timepoint_{index}.npy",
            np.full(768, index, dtype=np.float32),
        )
        node = {
            "tp_id": f"TP{index}",
            "mri_day": float((index - 1) * 30),
            "actions": {},
        }
        if source is not None:
            node["mri_day_source"] = source
        timeline.append(node)
    timeline_path = tmp_path / "timeline.json"
    timeline_path.write_text(
        json.dumps({"patients": {"P1": {"timeline": timeline}}}),
        encoding="utf-8",
    )
    checkpoint = tmp_path / "BrainIAC.ckpt"
    checkpoint.write_bytes(b"frozen-brainiac-checkpoint")
    provenance_path = tmp_path / "provenance.json"
    provenance = {
        "encoder": "brainiac",
        "frozen": True,
        "adapter": None,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "output_dim": 768,
        "extraction_protocol": "brainiac_mean_v1",
        "source_commit": "abc123",
        "num_timepoints": 4,
    }
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    cfg = copy.deepcopy(config)
    cfg["paths"].update(
        timeline=str(timeline_path),
        latent_dir=str(latent_dir),
        latent_provenance=str(provenance_path),
        brainiac_checkpoint=str(checkpoint),
    )
    return cfg, provenance_path, provenance


def test_main_cache_verifies_latent_provenance_and_audits_mri_time(tmp_path, config):
    cfg, provenance_path, _ = _main_cache_fixture(tmp_path, config)
    cache, audit = build_main_cache(cfg)
    assert cache["encoder"]["provenance_manifest_sha256"] == sha256_file(provenance_path)
    assert cache["encoder"]["checkpoint_sha256"] == sha256_file(
        cfg["paths"]["brainiac_checkpoint"]
    )
    assert cache["patients"][0]["mri_day_sources"] == [
        "observed",
        "imputed_interior",
        "imputed_leading",
        "imputed_trailing",
    ]
    quality = audit["time_quality"]
    assert quality["total_timepoints"] == 4
    assert quality["timepoint_sources"] == {
        "observed": 1,
        "imputed_interior": 1,
        "imputed_leading": 1,
        "imputed_trailing": 1,
        "unknown": 0,
    }
    assert quality["transitions_involving_imputation"] == 3
    assert quality["windows"]["H2"]["involving_imputation"] == 2
    assert quality["windows"]["H3"]["involving_imputation"] == 1


@pytest.mark.parametrize("unknown_source", [None, "unknown", "unrecognized"])
def test_main_v1_requires_explicit_mri_day_source(
    tmp_path, config, unknown_source
):
    cfg, _, _ = _main_cache_fixture(
        tmp_path,
        config,
        ["observed", "imputed_interior", unknown_source, "imputed_trailing"],
    )
    with pytest.raises(DataError, match="found 1 unknown timepoints"):
        build_main_cache(cfg)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("encoder", "other"),
        ("frozen", False),
        ("adapter", "lora"),
        ("output_dim", 512),
    ],
)
def test_main_cache_rejects_invalid_latent_provenance(
    tmp_path, config, field, value
):
    cfg, provenance_path, provenance = _main_cache_fixture(tmp_path, config)
    provenance[field] = value
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    with pytest.raises(DataError, match="provenance"):
        build_main_cache(cfg)


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


def test_legacy_normalizer_matches_numpy_float64_reference(config, tiny):
    values = torch.tensor(
        [
            [0.10000001, 4.0, 7.0],
            [0.10000002, 8.0, 7.0],
            [0.10000003, 12.0, 7.0],
        ],
        dtype=torch.float32,
    )
    actual = LegacyNormalizer.fit(values, min_std=1e-6)
    array = values.numpy().astype(np.float64)
    expected_mean = array.mean(axis=0).astype(np.float32)
    expected_std64 = array.std(axis=0)
    expected_std = np.where(expected_std64 < 1e-6, 1.0, expected_std64).astype(
        np.float32
    )
    np.testing.assert_array_equal(actual.mean.numpy(), expected_mean)
    np.testing.assert_array_equal(actual.std.numpy(), expected_std)
    assert actual.low_variance_count == 2

    cache, bundle = tiny
    legacy = copy.deepcopy(cache)
    legacy["protocol"] = "legacy_stage1"
    legacy["legacy_action_vocab"] = sorted(
        {
            term
            for row in legacy["patients"]
            for terms in row["action_terms"]
            for term in terms
        }
    )
    preprocessing = fit_preprocessing(legacy, config, bundle.split_ids)
    assert (
        preprocessing["normalizer_kind"]
        == "legacy_numpy_float64_to_float32"
    )
