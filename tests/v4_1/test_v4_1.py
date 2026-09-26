from __future__ import annotations

import torch
from torch import nn

from cloop.v1.data import ActionCodec, history_dim
from cloop.v1.types import Action, PatientState
from cloop.v4.metrics import (
    outcome_uncertainty_calibration,
    survival_selective_prediction,
)
from cloop.v4.next_experiment import load_experiment_config
from cloop.v4.outcome import (
    TrajectoryOutcomeModel,
    pairwise_survival_ranking_loss,
)
from cloop.v4.planner import TrajectoryValueMPC
from cloop.v4.trajectory import TemporalEmbedding, TrajectoryEncoder


def _features(batch: int = 2):
    torch.manual_seed(41)
    states = torch.randn(batch, 3, 4)
    actions = torch.randn(batch, 3, 2)
    actions[:, 0] = 0
    uncertainty = torch.rand(batch, 3)
    intervals = torch.tensor([[0.0, 30.0, 60.0]]).expand(batch, -1)
    state_mask = torch.ones(batch, 3, dtype=torch.bool)
    clinical = torch.randn(batch, 1)
    clinical_mask = torch.ones(batch, 1)
    history = torch.randn(batch, 6)
    state_source = torch.tensor([[0, 1, 1]]).expand(batch, -1)
    return (
        states,
        actions,
        uncertainty,
        intervals,
        state_mask,
        clinical,
        clinical_mask,
        history,
        state_source,
    )


def test_temporal_source_encoder_and_endpoint_shortcut_are_present():
    embedding = TemporalEmbedding(6)
    intervals = torch.tensor([[0.0, 10.0, 20.0]])
    mask = torch.ones(1, 3, dtype=torch.bool)
    assert embedding(intervals, mask).shape == (1, 3, 6)

    encoder = TrajectoryEncoder(
        4, 2, hidden_dim=8, nhead=2, layers=1, use_velocity=True,
    )
    output = encoder(*_features(2)[:5], _features(2)[-1])
    assert output.embedding.shape == (2, 8)
    assert output.final_state.shape == (2, 4)
    assert hasattr(encoder, "state_source_embedding")


def test_value_is_derived_from_survival_and_no_value_head_exists():
    model = TrajectoryOutcomeModel(
        4, 2, 1, 6, hidden_dim=8, nhead=2, layers=1,
        edges_days=(0, 90, 365), value_horizon_days=365,
    ).eval()
    features = _features()
    output = model(*features)
    expected = 1.0 - model.survival(*features, days=365.0)
    assert not hasattr(model, "value_head")
    assert torch.allclose(output.value, expected)
    assert torch.allclose(model.value(*features), expected)
    assert output.final_state_embedding.shape == output.trajectory_embedding.shape


def test_member_trajectories_produce_outcome_risk_distribution():
    model = TrajectoryOutcomeModel(
        4, 2, 1, 6, hidden_dim=8, nhead=2, layers=1,
        edges_days=(0, 90, 365), value_horizon_days=365,
        use_uncertainty=True,
    ).eval()
    features = _features()
    states = features[0]
    member_states = torch.stack((states - 0.3, states, states + 0.4))
    distribution = model.outcome_distribution(
        member_states, *features[1:], days=365.0,
    )
    assert distribution.member_risk.shape == (3, 2)
    assert torch.allclose(
        distribution.mean_risk, distribution.member_risk.mean(0),
    )
    assert torch.all(distribution.risk_variance >= 0)

    losses = model.loss(
        *features,
        days=torch.tensor([40.0, 500.0]),
        events=torch.tensor([1, 0]),
        patient_ids=["p0", "p1"],
        member_states=member_states,
    )
    assert set(losses) == {
        "total", "survival_nll", "ranking", "uncertainty_auxiliary",
    }
    assert all(torch.isfinite(value) for value in losses.values())


def test_ranking_excludes_pairs_from_the_same_patient():
    risks = torch.tensor([0.8, 0.2])
    days = torch.tensor([20.0, 300.0])
    events = torch.tensor([1, 0])
    excluded = pairwise_survival_ranking_loss(
        risks, days, events, patient_ids=["p0", "p0"],
    )
    included = pairwise_survival_ranking_loss(
        risks, days, events, patient_ids=["p0", "p1"],
    )
    assert excluded == 0
    assert included > 0


def test_survival_uncertainty_metrics_use_brier_error():
    times = [20.0, 500.0, 100.0]
    events = [1, 0, 0]
    survival = [0.2, 0.9, 0.5]
    variance = [0.4, 0.1, 0.8]
    calibration = outcome_uncertainty_calibration(
        times, events, survival, variance, 365.0, bins=2,
    )
    curve = survival_selective_prediction(
        times, events, survival, variance, 365.0, [1.0, 0.5],
    )
    assert calibration["identifiable"] == 2
    assert calibration["uncertainty_definition"] == "ensemble outcome-risk variance"
    assert curve[0]["retained"] == 2
    assert curve[1]["retained"] == 1


class _ToyWorld(nn.Module):
    ensemble_size = 2

    def one_step_members(
        self,
        member_states,
        action,
        delta_days,
        clinical,
        clinical_mask,
        history,
    ):
        del delta_days, clinical, clinical_mask, history
        offsets = torch.tensor(
            [[-0.05], [0.05]],
            dtype=member_states.dtype,
            device=member_states.device,
        )
        return member_states + offsets + action.sum() * 0.01


def test_closed_loop_execution_replans_and_open_loop_does_not():
    actions = (
        Action("a0", (0,), ("a0",), 5),
        Action("a1", (1,), ("a1",), 4),
    )
    codec = ActionCodec(("a0", "a1"), actions, min_support=1)
    evaluator = TrajectoryOutcomeModel(
        3, 2, 1, history_dim(2), hidden_dim=8, nhead=2, layers=1,
        edges_days=(0, 90, 365), value_horizon_days=365,
        use_uncertainty=True,
    )
    planner = TrajectoryValueMPC(
        _ToyWorld(),
        evaluator,
        codec,
        planned_interval_days=30,
        horizon=2,
        beam_width=2,
        lambda_uncertainty=0.1,
    )
    state = PatientState(
        patient_key="p",
        timepoint_key="t0",
        observed_day=0.0,
        z=torch.zeros(3),
        clinical=torch.zeros(1),
        clinical_mask=torch.ones(1),
        history=torch.zeros(history_dim(2)),
        source="observed",
        state_version=0,
    )
    open_loop = planner.execute(
        state, stages=2, method="open_loop", candidates=actions,
    )
    closed_loop = planner.execute(
        state, stages=2, method="closed_loop", candidates=actions,
    )
    assert open_loop.replans == 0
    assert open_loop.replanning_rate == 0
    assert closed_loop.replans == 1
    assert closed_loop.replanning_rate == 1
    assert len(closed_loop.actions) == 2


def test_v4_1_config_has_isolated_output_and_no_value_loss_weight():
    spec = load_experiment_config("configs/v4_1/next_experiment.yaml")
    assert spec["run"].endswith("v4_1")
    assert "value_weight" not in spec
    assert spec["uncertainty_auxiliary_weight"] >= 0
