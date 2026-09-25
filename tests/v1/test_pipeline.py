from __future__ import annotations

import copy
import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch

import cloop.v1.engine as engine
from cloop.v1.artifacts import (
    ArtifactError,
    RunArtifacts,
    read_json,
    read_torch,
    upsert_jsonl,
    write_json,
    write_torch,
)
from cloop.v1.config import ConfigError, _validate, load_config
from cloop.v1.data import make_tiny_cache


def test_cpu_smoke_is_offline_and_complete(config):
    result = engine.smoke(config)
    assert result["status"] == "ok"
    assert result["device"] == "cpu"
    assert result["network_used"] is False
    assert result["ensemble_members"] == 2


def test_minimal_prepare_train_evaluate_report_is_flat(config, tmp_path, monkeypatch):
    cfg = copy.deepcopy(config)
    cfg["paths"]["cache_root"] = str(tmp_path / "cache")
    cfg["paths"]["output_root"] = str(tmp_path / "outputs")
    cfg["paths"]["project_root"] = str(Path(__file__).resolve().parents[2])
    cache = make_tiny_cache(num_patients=18, steps=5, latent_dim=8, seed=17)

    def fake_prepare(_config, path):
        write_torch(path, cache)
        return cache, False

    monkeypatch.setattr(engine, "prepare_cache", fake_prepare)
    artifacts = RunArtifacts(cfg["paths"]["output_root"], "mini")
    engine.prepare(cfg, artifacts, "mini")
    engine.train_dynamics(cfg, artifacts, ["rrt"], [17])
    models = read_torch(artifacts.path("models.pt"), safe=True)
    metrics = read_json(artifacts.path("metrics.json"))
    assert models["dynamics"]["rrt/17"]["member_seeds"] == [17]
    assert metrics["training"]["rrt/17"]["member_seeds"] == [17]
    assert metrics["training"]["rrt/17"]["members"][0]["member_seed"] == 17
    engine.evaluate_dynamics(cfg, artifacts, "validation", variants=["rrt"], seeds=[17])
    report = engine.generate_report(artifacts)
    assert report.exists()
    assert not any(path.is_dir() for path in artifacts.root.iterdir())
    assert len(list(artifacts.root.iterdir())) <= 6
    assert not artifacts.path("last.pt").exists()


def test_jsonl_upsert_and_metrics_are_idempotent(tmp_path):
    path = tmp_path / "predictions.jsonl"
    upsert_jsonl(path, [{"record_id": "b", "x": 1}, {"record_id": "a", "x": 1}])
    upsert_jsonl(path, [{"record_id": "a", "x": 2}])
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["record_id"] for row in rows] == ["a", "b"]
    assert rows[0]["x"] == 2


def test_rng_and_generator_restore_matches_continuation():
    random.seed(4)
    np.random.seed(4)
    torch.manual_seed(4)
    generator = torch.Generator().manual_seed(4)
    state = engine.capture_rng(generator)
    expected = (random.random(), np.random.rand(), torch.rand(1), torch.rand(1, generator=generator))
    engine.restore_rng(state, generator)
    actual = (random.random(), np.random.rand(), torch.rand(1), torch.rand(1, generator=generator))
    assert expected[0] == actual[0]
    assert expected[1] == actual[1]
    assert torch.equal(expected[2], actual[2])
    assert torch.equal(expected[3], actual[3])


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("data", "action_alignment", "legacy_source"),
        ("data", "unknown_interval_policy", "include"),
        ("data", "require_mri_day_provenance", False),
        ("planner", "interval_source", "fixed"),
        ("planner", "require_supported_actions", False),
        ("planner", "abstain_when_no_valid_action", False),
    ],
)
def test_config_rejects_fields_that_v1_does_not_implement(
    config, section, field, value
):
    cfg = copy.deepcopy(config)
    cfg[section][field] = value
    with pytest.raises(ConfigError):
        _validate(cfg)


def test_disabled_clinical_rules_are_reported_as_not_evaluated():
    row = {
        "status": "recommend",
        "metrics": {
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "jaccard": 1.0,
            "both_empty": False,
        },
        "recommended_action": "A",
        "patient_id": "P",
        "catalog_covers_actual": True,
        "candidate_contains_actual": True,
        "diagnostics": {},
    }
    summary = engine._replay_summary([row])
    assert summary["structural_rule_violation_rate"] is None
    assert (
        summary["structural_rule_violation_reason"]
        == "clinical_rules_disabled"
    )


def _freezable_run(config, tmp_path):
    cfg = copy.deepcopy(config)
    cfg["paths"]["output_root"] = str(tmp_path / "outputs")
    artifacts = RunArtifacts(cfg["paths"]["output_root"], "freeze")
    artifacts.root.mkdir(parents=True, exist_ok=True)
    manifest = engine._new_run_manifest(
        cfg, "freeze", torch.device("cpu")
    )
    manifest["data_signature"] = "test-data-signature"
    manifest["uncertainty_scales"] = {
        "rrt_ensemble/17": {"H1": 1.0, "H2": 1.0, "H3": 1.0}
    }
    write_json(artifacts.path("run.json"), manifest)
    write_torch(artifacts.path("models.pt"), {"frozen": torch.tensor([1.0])})
    return cfg, artifacts


def test_freeze_blocks_dynamics_training(config, tmp_path):
    cfg, artifacts = _freezable_run(config, tmp_path)
    engine.freeze_protocol(cfg, artifacts)
    manifest = read_json(artifacts.path("run.json"))
    assert manifest["frozen_models_sha256"] == engine.sha256_file(
        artifacts.path("models.pt")
    )
    with pytest.raises(engine.EngineError, match="protocol is frozen"):
        engine.train_dynamics(cfg, artifacts, ["rrt"], [17])


def test_freeze_blocks_force_task(config, tmp_path):
    cfg, artifacts = _freezable_run(config, tmp_path)
    engine.freeze_protocol(cfg, artifacts)
    with pytest.raises(engine.EngineError, match="protocol is frozen"):
        engine.train_dynamics(
            cfg, artifacts, ["rrt"], [17], force_task=True
        )


def test_modified_models_pt_breaks_test_evaluation(config, tmp_path):
    cfg, artifacts = _freezable_run(config, tmp_path)
    engine.freeze_protocol(cfg, artifacts)
    write_torch(
        artifacts.path("models.pt"), {"tampered": torch.tensor([2.0])}
    )
    with pytest.raises(engine.EngineError, match="changed after protocol freeze"):
        engine.evaluate_dynamics(
            cfg, artifacts, "test", variants=["rrt"], seeds=[17]
        )


def test_freeze_rejects_incomplete_last_checkpoint(config, tmp_path):
    cfg, artifacts = _freezable_run(config, tmp_path)
    write_torch(artifacts.path("last.pt"), {"epoch": 1})
    with pytest.raises(engine.EngineError, match="last.pt exists"):
        engine.freeze_protocol(cfg, artifacts)


def test_report_still_works_after_freeze(config, tmp_path):
    cfg, artifacts = _freezable_run(config, tmp_path)
    engine.freeze_protocol(cfg, artifacts)
    report = engine.generate_report(artifacts)
    assert report.is_file()
    assert "Protocol frozen: `True`" in report.read_text(encoding="utf-8")


def _assert_nested_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for first, second in zip(left, right):
            _assert_nested_equal(first, second)
    else:
        assert left == right


def _without_epoch_seconds(history):
    return [
        {key: value for key, value in row.items() if key != "epoch_seconds"}
        for row in history
    ]


def test_complete_epoch_resume_matches_continuous_training(
    config, tiny, tmp_path, monkeypatch
):
    _, bundle = tiny
    cfg = copy.deepcopy(config)
    cfg["training"].update(max_epochs=4, patience=10)
    device = torch.device("cpu")

    continuous_artifacts = RunArtifacts(tmp_path / "continuous", "run")
    continuous_state, continuous_best_epoch, continuous_history = (
        engine._train_world_member(
            cfg,
            bundle,
            "rrt",
            17,
            0,
            device,
            continuous_artifacts,
            resume=False,
        )
    )
    continuous_last = read_torch(
        continuous_artifacts.path("last.pt"), safe=True
    )

    resumed_artifacts = RunArtifacts(tmp_path / "resumed", "run")
    original_write_torch = engine.write_torch

    def interrupt_after_epoch_two(path, value):
        original_write_torch(path, value)
        if (
            Path(path).name == "last.pt"
            and value.get("task") == "dynamics"
            and value.get("epoch") == 2
        ):
            raise RuntimeError("simulated epoch-boundary interruption")

    monkeypatch.setattr(engine, "write_torch", interrupt_after_epoch_two)
    with pytest.raises(RuntimeError, match="epoch-boundary"):
        engine._train_world_member(
            cfg,
            bundle,
            "rrt",
            17,
            0,
            device,
            resumed_artifacts,
            resume=False,
        )
    monkeypatch.setattr(engine, "write_torch", original_write_torch)
    resumed_state, resumed_best_epoch, resumed_history = (
        engine._train_world_member(
            cfg,
            bundle,
            "rrt",
            17,
            0,
            device,
            resumed_artifacts,
            resume=True,
        )
    )
    resumed_last = read_torch(resumed_artifacts.path("last.pt"), safe=True)

    _assert_nested_equal(continuous_state, resumed_state)
    _assert_nested_equal(
        continuous_last["model_state"], resumed_last["model_state"]
    )
    _assert_nested_equal(
        continuous_last["optimizer_state"], resumed_last["optimizer_state"]
    )
    assert continuous_best_epoch == resumed_best_epoch
    assert _without_epoch_seconds(continuous_history) == _without_epoch_seconds(
        resumed_history
    )


def test_missing_formal_model_does_not_fallback(tmp_path):
    artifacts = RunArtifacts(tmp_path, "missing")
    with pytest.raises(engine.EngineError):
        engine._load_models(artifacts, required=True)


def test_strict_config_rejects_unknown_field(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("project:\n  unknown: true\n")
    with pytest.raises(ConfigError):
        load_config(path)


def test_run_artifact_rejects_subdirectories(tmp_path):
    artifacts = RunArtifacts(tmp_path, "run")
    (artifacts.root / "seed_17").mkdir(parents=True)
    with pytest.raises(ArtifactError):
        artifacts.assert_flat()
