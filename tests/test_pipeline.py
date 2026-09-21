from __future__ import annotations

import copy
import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch

import cloop.engine as engine
from cloop.artifacts import ArtifactError, RunArtifacts, read_torch, upsert_jsonl, write_torch
from cloop.config import ConfigError, load_config
from cloop.data import make_tiny_cache


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
    cfg["paths"]["project_root"] = str(Path(__file__).resolve().parents[1])
    cache = make_tiny_cache(num_patients=18, steps=5, latent_dim=8, seed=17)

    def fake_prepare(_config, path):
        write_torch(path, cache)
        return cache, False

    monkeypatch.setattr(engine, "prepare_cache", fake_prepare)
    artifacts = RunArtifacts(cfg["paths"]["output_root"], "mini")
    engine.prepare(cfg, artifacts, "mini")
    engine.train_dynamics(cfg, artifacts, ["rrt"], [17])
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

