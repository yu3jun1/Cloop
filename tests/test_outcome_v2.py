from __future__ import annotations

import torch
from torch import nn

from cloop.outcome_diagnostics import TrainOnlyPCA, _eligible_latents
from cloop.outcome_v2 import (
    V2Dataset,
    _evaluate_op,
    _new_outcome,
    _train_outcome,
    _train_world,
    transition_representation,
)
from cloop.world import EnsembleWorldModel


class ShiftDynamics(nn.Module):
    def __init__(self, shift: float):
        super().__init__()
        self.shift = shift

    def forward(self, z, action, delta_days, clinical, clinical_mask, history):
        return z + self.shift


def test_matched_rows_use_target_label_history_and_factual_interval(tiny):
    _, bundle = tiny
    matched = V2Dataset(bundle, "train", transition=True)
    assert len(matched) > 0
    for row in matched:
        tr = bundle.trajectories[row["patient_id"]]
        source, target = row["source_index"], row["target_index"]
        assert target == source + 1
        assert bool(tr.action_known[source])
        assert torch.equal(row["time"], tr.survival_time[target])
        assert torch.equal(row["event"], tr.survival_event[target])
        assert torch.equal(row["history_target"], tr.histories[target])
        assert torch.equal(row["action"], tr.actions[source])
        assert torch.equal(row["delta_days"], tr.days[target] - tr.days[source])
        assert float(row["delta_days"]) > 0


def test_transition_residual_starts_at_clinical_baseline(tiny, config):
    _, bundle = tiny
    pca = TrainOnlyPCA.fit(_eligible_latents(bundle, "train"), 4)
    v2 = {"pca_dim": 4, "residual_hidden_dim": 6}
    base = _new_outcome("O0_matched", bundle, config, v2)
    model = _new_outcome("OT_observed", bundle, config, v2, base.state_dict())
    row = V2Dataset(bundle, "train", transition=True)[0]
    representation = transition_representation(
        pca, row["z_source"].unsqueeze(0), row["z_target"].unsqueeze(0)
    )
    assert representation.shape == (1, 12)
    assert torch.equal(representation[0, 8:], representation[0, 4:8] - representation[0, :4])
    clinical = row["clinical"].unsqueeze(0)
    mask = row["clinical_mask"].unsqueeze(0)
    history = row["history_target"].unsqueeze(0)
    assert torch.allclose(
        model.rates(representation, clinical, mask, history),
        base.rates(row["z_target"].unsqueeze(0), clinical, mask, history),
    )
    assert all(not parameter.requires_grad for parameter in model.clinical.parameters())


def test_op_uses_identical_matched_rows_and_reports_member_uncertainty(tiny, config):
    _, bundle = tiny
    pca = TrainOnlyPCA.fit(_eligible_latents(bundle, "train"), 4)
    v2 = {"pca_dim": 4, "residual_hidden_dim": 6}
    base = _new_outcome("O0_matched", bundle, config, v2)
    model = _new_outcome("OT_observed", bundle, config, v2, base.state_dict())
    model.requires_grad_(False)
    validation = V2Dataset(bundle, "validation", transition=True, first_only=True)
    training = V2Dataset(bundle, "train", transition=True, first_only=True)
    assert len(validation) > 0
    world = EnsembleWorldModel([ShiftDynamics(0.1), ShiftDynamics(-0.1)])
    summary, rows, diagnostics = _evaluate_op(
        model, world, validation, training, pca, config, torch.device("cpu")
    )
    assert summary["observations"] == len(validation)
    assert {(row["patient_id"], row["source_timepoint"], row["target_timepoint"]) for row in rows} == {
        (validation[i]["patient_id"], validation[i]["source_timepoint"], validation[i]["target_timepoint"])
        for i in range(len(validation))
    }
    assert all(row["disagreement"] > 0 for row in rows)
    assert diagnostics["ensemble_reliability"]["n"] == len(validation)

def test_fold_local_training_produces_finite_selected_checkpoints(tiny, config):
    _, bundle = tiny
    pca = TrainOnlyPCA.fit(_eligible_latents(bundle, "train"), 4)
    v2 = {"pca_dim": 4, "residual_hidden_dim": 6}
    train = V2Dataset(bundle, "train", transition=True)
    validation = V2Dataset(bundle, "validation", transition=True)
    base = _new_outcome("O0_matched", bundle, config, v2)
    state, best_epoch, history = _train_outcome(
        "O0_matched", base, train, validation, pca, config, 17, torch.device("cpu")
    )
    assert best_epoch == 1
    assert len(state) > 0
    assert torch.isfinite(torch.tensor(history[0]["val_nll"]))
    world_state, world_epoch, world_history = _train_world(
        bundle, config, "baseline", 17, 0, torch.device("cpu")
    )
    assert world_epoch == 1
    assert len(world_state) > 0
    assert torch.isfinite(torch.tensor(world_history[0]["val_long_mse"]))

def test_single_state_last_landmark_does_not_index_past_actions(tiny):
    _, bundle = tiny
    dataset = V2Dataset(bundle, "train", transition=False)
    terminal_refs = [(i, ref) for i, ref in enumerate(dataset.refs)
                     if ref[2] == bundle.trajectories[ref[0]].length - 1]
    assert terminal_refs
    for index, _ in terminal_refs:
        row = dataset[index]
        assert row["source_index"] == row["target_index"]
        assert row["action"].shape == (bundle.action_dim,)
        assert torch.count_nonzero(row["action"]) == 0
