from __future__ import annotations

import torch

from cloop.v4.metrics import coverage_risk_curve, regression_calibration
from cloop.v4.next_experiment import (
    OUTCOMES,
    _batch,
    _dynamics_records,
    _new_outcome,
    _outcome_rows,
    load_experiment_config,
)
from cloop.v4.trajectory import (
    DualHeadOutcomeModel,
    TrajectoryEncoder,
    fixed_horizon_value_targets,
    pairwise_survival_ranking_loss,
)
from cloop.v1.world import EnsembleWorldModel, PersistenceDynamics


def _features():
    torch.manual_seed(5)
    states = torch.randn(3, 4, 5)
    actions = torch.randn(3, 4, 2)
    uncertainty = torch.rand(3, 4)
    intervals = torch.tensor([
        [0.0, 30.0, 0.0, 0.0],
        [0.0, 20.0, 40.0, 0.0],
        [0.0, 25.0, 50.0, 75.0],
    ])
    mask = torch.tensor([
        [True, True, False, False],
        [True, True, True, False],
        [True, True, True, True],
    ])
    clinical = torch.randn(3, 2)
    clinical_mask = torch.ones(3, 2)
    history = torch.randn(3, 3)
    return states, actions, uncertainty, intervals, mask, clinical, clinical_mask, history


def test_attention_pooling_is_padding_invariant_and_masks_weights():
    model = DualHeadOutcomeModel(
        5, 2, 2, 3, hidden_dim=16, nhead=4, layers=1,
        use_velocity=True, use_uncertainty=True,
    ).eval()
    features = list(_features())
    first = model(*features)
    states, actions, uncertainty, intervals, mask, *_ = features
    states[~mask] = 1000
    actions[~mask] = 1000
    uncertainty[~mask] = 1000
    intervals[~mask] = 1000
    second = model(*features)
    assert torch.allclose(first.rates, second.rates, atol=1e-6)
    assert torch.allclose(first.value, second.value, atol=1e-6)
    assert torch.all(first.attention_weights[~mask] == 0)
    assert torch.allclose(first.attention_weights.sum(1), torch.ones(3))


def test_uncertainty_gate_downweights_an_otherwise_identical_token():
    encoder = TrajectoryEncoder(
        3, 1, hidden_dim=8, nhead=2, layers=1,
        use_velocity=False, use_uncertainty=True,
    ).eval()
    states = torch.zeros(1, 3, 3)
    actions = torch.zeros(1, 3, 1)
    intervals = torch.zeros(1, 3)
    mask = torch.ones(1, 3, dtype=torch.bool)
    output = encoder(
        states, actions, torch.tensor([[0.0, 0.0, 10.0]]), intervals, mask,
    )
    assert output.attention_weights[0, 2] < output.attention_weights[0, 0]
    assert output.calibrated_error is not None
    assert output.uncertainty_stats.shape == (1, 3)


def test_dual_head_composite_loss_is_finite_and_differentiable():
    model = DualHeadOutcomeModel(
        5, 2, 2, 3, hidden_dim=16, nhead=4, layers=1,
        use_velocity=True, use_uncertainty=True,
    )
    features = _features()
    losses = model.loss(
        *features,
        days=torch.tensor([50.0, 500.0, 100.0]),
        events=torch.tensor([1, 0, 0]),
        latent_error=torch.rand(3, 4),
    )
    assert set(losses) == {
        "total", "survival_nll", "ranking", "value", "uncertainty_calibration",
    }
    assert all(torch.isfinite(value) for value in losses.values())
    losses["total"].backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_value_targets_exclude_early_censoring_and_ranking_uses_comparable_pairs():
    targets, valid = fixed_horizon_value_targets(
        torch.tensor([20.0, 200.0, 500.0]), torch.tensor([1, 0, 0]), 365.0,
    )
    assert targets.tolist() == [1.0, 0.0, 0.0]
    assert valid.tolist() == [True, False, True]
    good = pairwise_survival_ranking_loss(
        torch.tensor([0.9, 0.2]), torch.tensor([20.0, 500.0]), torch.tensor([1, 0]),
    )
    bad = pairwise_survival_ranking_loss(
        torch.tensor([0.2, 0.9]), torch.tensor([20.0, 500.0]), torch.tensor([1, 0]),
    )
    assert good < bad


def test_selective_prediction_metrics_are_explicit_diagnostics():
    curve = coverage_risk_curve([0.1, 0.2, 2.0, 3.0], [0.1, 0.2, 0.9, 1.0], [1.0, 0.5])
    assert curve[1]["coverage"] == 0.5
    assert curve[1]["risk"] < curve[0]["risk"]
    calibration = regression_calibration([0.1, 0.2, 0.8], [0.2, 0.1, 0.9], bins=2)
    assert calibration["n"] == 3
    assert calibration["mae"] is not None


def test_v4_rows_ablation_shapes_and_l2_drift(config, tiny):
    _, bundle = tiny
    spec = {
        "max_horizon": 3,
        "batch_size": 8,
        "hidden_dim": 16,
        "nhead": 4,
        "layers": 1,
        "score_days": 365.0,
    }
    world = EnsembleWorldModel([PersistenceDynamics()])
    rows = _outcome_rows(world, bundle, "validation", spec, torch.device("cpu"))
    assert rows
    batch = _batch(rows, [0], torch.device("cpu"))
    assert batch["state_mask"][0, 0]
    assert torch.all(batch["actions"][0, 0] == 0)
    assert torch.all(batch["intervals"][0, 0] == 0)
    assert torch.all(batch["latent_error"] >= 0)
    for name in OUTCOMES:
        model = _new_outcome(name, bundle, config, spec, torch.device("cpu"))
        if name != "O0_final":
            output = model(*(batch[key] for key in (
                "states", "actions", "uncertainty", "intervals", "state_mask",
                "clinical", "clinical_mask", "history",
            )))
            assert output.rates.shape == (1, len(config["outcome"]["edges_days"]) - 1)
            assert output.value.shape == (1,)
    dynamics = _dynamics_records(
        world, bundle, "validation", "baseline", 0, 17, spec, torch.device("cpu"),
    )
    assert dynamics
    assert all(abs(row["drift_from_h1"]) < 1e-6 for row in dynamics if row["horizon"] == 1)


def test_default_v4_config_is_valid():
    spec = load_experiment_config("configs/v4/next_experiment.yaml")
    assert spec["run"].startswith("next_experiment_")
    assert spec["coverage_levels"][0] == 1.0
