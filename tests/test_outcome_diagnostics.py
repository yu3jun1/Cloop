from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from cloop.outcome import PiecewiseExponentialHead
from cloop.outcome_diagnostics import (
    DiagnosticOutcomeDataset,
    ResidualPCAOutcomeHead,
    TrainOnlyPCA,
    make_patient_derangement,
    make_stratified_patient_folds,
)


def test_train_only_pca_shape_and_state_roundtrip():
    generator = torch.Generator().manual_seed(4)
    x = torch.randn(20, 8, generator=generator)
    pca = TrainOnlyPCA.fit(x, 4)
    transformed = pca.transform(x)
    restored = TrainOnlyPCA.from_state_dict(pca.state_dict())
    assert transformed.shape == (20, 4)
    assert torch.allclose(transformed, restored.transform(x))
    assert 0 < float(pca.explained_variance_ratio.sum()) <= 1.0


def test_patient_derangement_has_no_self_and_is_deterministic():
    ids = ["p0", "p1", "p2", "p3"]
    first = make_patient_derangement(ids, 17)
    second = make_patient_derangement(ids, 17)
    assert first == second
    assert set(first) == set(ids)
    assert set(first.values()) == set(ids)
    assert all(pid != donor for pid, donor in first.items())


def test_stratified_folds_are_disjoint_cover():
    patients = []
    for index in range(12):
        patients.append(
            {
                "patient_id": f"p{index}",
                "labels": {
                    "valid": torch.tensor([True, True]),
                    "event": torch.tensor([1 if index < 6 else 0, 1 if index < 6 else 0]),
                },
            }
        )
    cache = {"patients": patients}
    folds = make_stratified_patient_folds(
        [row["patient_id"] for row in patients],
        cache,
        requested_folds=3,
        fallback_folds=3,
        seed=11,
    )
    flattened = [pid for fold in folds for pid in fold]
    assert len(folds) == 3
    assert len(flattened) == 12
    assert len(set(flattened)) == 12


def test_residual_alpha_zero_exactly_matches_loaded_e0_logits():
    latent_dim = 8
    clinical_dim = 3
    history_dim = 5
    hidden_dim = 7
    edges = [0, 90, 180, 365, 730]
    e0 = PiecewiseExponentialHead(
        latent_dim,
        clinical_dim,
        history_dim,
        hidden_dim=hidden_dim,
        edges_days=edges,
        clinical_only=True,
    )
    residual = ResidualPCAOutcomeHead(
        latent_dim,
        clinical_dim,
        history_dim,
        pca_dim=4,
        hidden_dim=hidden_dim,
        residual_hidden_dim=3,
        edges_days=edges,
    )
    residual.load_and_freeze_e0(e0.state_dict())
    batch = 6
    z = torch.randn(batch, latent_dim)
    pca_z = torch.randn(batch, 4)
    clinical = torch.randn(batch, clinical_dim)
    mask = torch.ones_like(clinical)
    history = torch.randn(batch, history_dim)
    e0_rates = e0.rates(z, clinical, mask, history)
    residual_rates = residual.rates(pca_z, clinical, mask, history)
    assert float(residual.alpha) == 0.0
    assert torch.allclose(e0_rates, residual_rates, atol=1e-7, rtol=1e-6)
    assert all(not parameter.requires_grad for parameter in residual.base_network.parameters())


def test_shuffled_dataset_uses_donor_patient(tiny):
    _, bundle = tiny
    ids = bundle.split_ids["train"]
    mapping = make_patient_derangement(ids, 23)
    dataset = DiagnosticOutcomeDataset(
        bundle,
        "train",
        first_only=True,
        donor_map=mapping,
    )
    row = dataset[0]
    donor = bundle.trajectories[row["donor_patient_id"]]
    expected_index = min(int(row["landmark_index"]), donor.length - 1)
    assert row["patient_id"] != row["donor_patient_id"]
    assert torch.equal(row["z"], donor.latents[expected_index])


def test_non_shuffled_dataset_can_be_collated(tiny):
    _, bundle = tiny
    dataset = DiagnosticOutcomeDataset(bundle, "train", first_only=True)
    batch = next(iter(DataLoader(dataset, batch_size=2)))
    assert len(batch["patient_id"]) == min(2, len(dataset))
    assert "donor_patient_id" not in batch
    assert "donor_landmark_index" not in batch
