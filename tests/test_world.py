from __future__ import annotations

import copy

import torch
from torch import nn

from cloop.data import DynamicsDataset, collate_dynamics, window_refs
from cloop.world import (
    EnsembleWorldModel,
    OneStepDynamics,
    PersistenceDynamics,
    ensemble_mean_and_disagreement,
    terminal_mse,
)


class RecordingMember(nn.Module):
    def __init__(self, increment: float):
        super().__init__()
        self.increment = increment
        self.inputs = []

    def forward(self, z, action, delta, clinical, mask, history):
        self.inputs.append(z.detach().clone())
        return z + self.increment


def _batch(bundle, mode="max_available"):
    refs = window_refs(bundle, "train", mode=mode, max_horizon=3)[:4]
    return collate_dynamics([DynamicsDataset(bundle, refs)[i] for i in range(len(refs))])


def test_k1_baseline_and_rrt_with_same_initialization_match(config, tiny):
    _, bundle = tiny
    batch = _batch(bundle, "one_step")
    first = OneStepDynamics.from_config(config, bundle.latent_dim, bundle.action_dim, bundle.clinical_dim, bundle.history_dim)
    second = copy.deepcopy(first)
    args = (batch["z0"], batch["actions"], batch["deltas"], batch["context"], batch["clinical_mask"], batch["history0"], batch["step_mask"])
    assert torch.equal(EnsembleWorldModel([first]).rollout(*args).states, EnsembleWorldModel([second]).rollout(*args).states)


def test_recursive_second_step_uses_prediction_not_true_middle(tiny):
    _, bundle = tiny
    batch = _batch(bundle)
    member = RecordingMember(2.0)
    EnsembleWorldModel([member]).rollout(
        batch["z0"], batch["actions"], batch["deltas"], batch["context"],
        batch["clinical_mask"], batch["history0"], batch["step_mask"]
    )
    active_second = batch["step_mask"][:, 1]
    assert torch.allclose(member.inputs[1], batch["z0"][active_second] + 2.0)


def test_terminal_gradient_reaches_first_step(config, tiny):
    _, bundle = tiny
    batch = _batch(bundle)
    model = OneStepDynamics.from_config(config, bundle.latent_dim, bundle.action_dim, bundle.clinical_dim, bundle.history_dim)
    rollout = EnsembleWorldModel([model]).rollout(
        batch["z0"], batch["actions"], batch["deltas"], batch["context"],
        batch["clinical_mask"], batch["history0"], batch["step_mask"]
    )
    terminal_mse(rollout, batch["horizons"], batch["target"]).backward()
    assert model.action_projection[0].weight.grad is not None
    assert model.action_projection[0].weight.grad.abs().sum() > 0


def test_padding_mask_does_not_advance_short_sequences(tiny):
    _, bundle = tiny
    batch = _batch(bundle)
    batch["step_mask"][0, 1:] = False
    member = RecordingMember(1.0)
    rollout = EnsembleWorldModel([member]).rollout(
        batch["z0"], batch["actions"], batch["deltas"], batch["context"],
        batch["clinical_mask"], batch["history0"], batch["step_mask"]
    )
    assert torch.allclose(rollout.states[0, 0, 1], rollout.states[0, 0, -1])


def test_disagreement_matches_manual_and_m1_is_zero():
    states = torch.tensor([[[0.0, 2.0]], [[2.0, 4.0]]])
    mean, disagreement = ensemble_mean_and_disagreement(states)
    assert torch.equal(mean, torch.tensor([[1.0, 3.0]]))
    assert torch.allclose(disagreement, torch.tensor([1.0]))
    assert torch.equal(ensemble_mean_and_disagreement(states[:1])[1], torch.zeros(1))


def test_members_keep_independent_recursive_states(tiny):
    _, bundle = tiny
    batch = _batch(bundle)
    one, two = RecordingMember(1.0), RecordingMember(3.0)
    rollout = EnsembleWorldModel([one, two]).rollout(
        batch["z0"], batch["actions"], batch["deltas"], batch["context"],
        batch["clinical_mask"], batch["history0"], batch["step_mask"]
    )
    active = batch["step_mask"][:, 1]
    assert torch.allclose(one.inputs[1], batch["z0"][active] + 1.0)
    assert torch.allclose(two.inputs[1], batch["z0"][active] + 3.0)
    assert not torch.allclose(rollout.states[0, :, -1], rollout.states[1, :, -1])


def test_persistence_reuses_start_without_target(tiny):
    _, bundle = tiny
    batch = _batch(bundle)
    rollout = EnsembleWorldModel([PersistenceDynamics()]).rollout(
        batch["z0"], batch["actions"], batch["deltas"], batch["context"],
        batch["clinical_mask"], batch["history0"], batch["step_mask"]
    )
    assert torch.allclose(rollout.states[0, :, -1], batch["z0"])

