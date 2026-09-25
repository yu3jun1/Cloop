from __future__ import annotations

import pytest
import torch

from cloop.v1.artifacts import RunArtifacts, write_json

from cloop.v3.next_stage import NextStageError, _batch, _dynamics_records, _initialize, _inner_patient_split, _new_outcome, _outcome_rows, _rates
from cloop.v3.trajectory import TrajectorySurvivalHead, nll_from_rates
from cloop.v3.trajectory_metrics import integrated_brier, time_dependent_auc
from cloop.v1.world import EnsembleWorldModel, PersistenceDynamics


def test_trajectory_padding_and_uncertainty_ablation():
    torch.manual_seed(3)
    model = TrajectorySurvivalHead(4, 2, 2, 3, hidden_dim=16, nhead=4, layers=1)
    model.eval()
    states = torch.randn(2, 4, 4)
    actions = torch.randn(2, 4, 2)
    uncertainty = torch.rand(2, 4)
    times = torch.tensor([[0., 30., 0., 0.], [0., 40., 80., 0.]])
    mask = torch.tensor([[True, True, False, False], [True, True, True, False]])
    clinical = torch.randn(2, 2)
    clinical_mask = torch.ones(2, 2)
    history = torch.randn(2, 3)
    first = model.rates(states, actions, uncertainty, times, mask, clinical, clinical_mask, history)
    states[0, 2:] += 1000
    states[1, 3:] += 1000
    actions[0, 2:] += 1000
    actions[1, 3:] += 1000
    uncertainty[0, 2:] += 1000
    uncertainty[1, 3:] += 1000
    second = model.rates(states, actions, uncertainty, times, mask, clinical, clinical_mask, history)
    assert torch.allclose(first[0], second[0], atol=1e-6)
    assert torch.allclose(first[1], second[1], atol=1e-6)
    assert torch.all(first > 0)
    assert torch.allclose(
        nll_from_rates(first, model.edges, torch.tensor([90., 400.]), torch.tensor([1, 0]), reduction="none"),
        nll_from_rates(second, model.edges, torch.tensor([90., 400.]), torch.tensor([1, 0]), reduction="none"),
    )


def test_factual_rows_and_matched_outcome_inputs(config, tiny):
    _, bundle = tiny
    spec = {"max_horizon": 3, "batch_size": 8, "hidden_dim": 16, "nhead": 4, "layers": 1}
    world = EnsembleWorldModel([PersistenceDynamics()])
    rows = _outcome_rows(world, bundle, "validation", spec, torch.device("cpu"))
    assert rows
    fit, stop = _inner_patient_split(_outcome_rows(world, bundle, "train", spec, torch.device("cpu")), 17)
    assert {r["patient_id"] for r in fit}.isdisjoint({r["patient_id"] for r in stop})
    batch = _batch(rows, [0], torch.device("cpu"))
    assert batch["state_mask"][0, 0]
    assert torch.all(batch["actions"][0, 0] == 0)
    assert torch.all(batch["uncertainty"] == 0)
    for name in ("O0_final", "O1_trajectory", "O2_trajectory_uncertainty"):
        model = _new_outcome(name, bundle, config, spec, torch.device("cpu"))
        assert _rates(model, name, batch).shape == (1, len(config["outcome"]["edges_days"]) - 1)
    dyn = _dynamics_records(world, bundle, "validation", "persistence", 0, 17, spec, torch.device("cpu"))
    assert dyn
    assert all(abs(row["drift_from_h1"]) < 1e-6 for row in dyn if row["horizon"] == 1)


def test_censoring_metrics_report_undefined_support():
    assert time_dependent_auc([30., 50.], [1, 0], [20., 80.], [1, 0], [.8, .2], 365.)["value"] is None
    assert integrated_brier([30., 50.], [1, 0], [20., 80.], [1, 0],
                            {365.: [.2, .8]}, 365.)["value"] is None


def test_versioned_manifest_migration_accepts_only_current_signature(tmp_path):
    source = RunArtifacts(tmp_path, "world", version="v2")
    source.root.mkdir(parents=True)
    source.path("models.pt").write_bytes(b"source model checkpoint")
    artifacts = RunArtifacts(tmp_path, "next_stage_v1", version="v3")
    config = {"paths": {"output_root": str(tmp_path)}}
    spec = {"base_run": "base", "world_run": "world", "run": "next_stage_v1"}
    base = {"data_signature": "base-data"}
    world_run = {"signature": "world-signature"}
    current, _, _ = _initialize(config, spec, base, world_run, artifacts)
    assert artifacts.root == tmp_path / "v3" / "next_stage_v1"

    historical = {**current, "signature": "historical-signature",
                  "implementation_sha256": "historical-implementation"}
    historical["layout_migration"] = {
        "kind": "versioned_paths_and_imports_only",
        "original_signature": historical["signature"],
        "original_implementation_sha256": historical["implementation_sha256"],
        "runtime_signature": current["signature"],
        "runtime_implementation_sha256": current["implementation_sha256"],
    }
    write_json(artifacts.path("run.json"), historical)
    reopened, _, _ = _initialize(config, spec, base, world_run, artifacts)
    assert reopened["signature"] == "historical-signature"
    with pytest.raises(NextStageError, match="different signature"):
        _initialize(config, {**spec, "run": "changed"}, base, world_run, artifacts)
