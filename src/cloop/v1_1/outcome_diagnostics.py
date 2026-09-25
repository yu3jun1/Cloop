"""Standalone E0-E5 Outcome Head diagnostic experiments.

This module intentionally does not mutate or reuse the formal run's model artifacts.
It reads the prepared base run only to recover the audited data cache and the
registered patient split, then builds patient-level CV folds from the original
train+validation patients. The formal test patients are never evaluated here.

Run with:

    python -m cloop.v1_1.outcome_diagnostics \
      --config configs/v1/default.yaml \
      --paths configs/v1/server.yaml \
      --diag-config configs/v1_1/outcome_diagnostics.yaml

The six registered diagnostics are:

E0  current clinical-only baseline
E1  MRI-only raw 768-D latent
E2  current raw O1 (MRI + clinical + mask + history)
E3  shuffled-MRI negative control
E4  train-fold-only PCA64 + clinical + mask + history
E5  frozen E0 + additive PCA64 MRI residual, alpha initialised to zero
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import yaml
from torch import Tensor, nn
from torch.nn import functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from ..v1.artifacts import RunArtifacts, read_json, read_torch, write_json, write_torch
from ..v1.config import load_config
from ..v1.data import DataBundle, fit_preprocessing, load_bundle
from ..v1.metrics import mean_std, survival_summary
from ..v1.outcome import PiecewiseExponentialHead, event_bins, interval_exposure


EXPERIMENT_ORDER = ("E0", "E1", "E2", "E3", "E4", "E5")


class OutcomeDiagnosticError(RuntimeError):
    pass


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _sha256_json(value: Any) -> str:
    body = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(body.encode()).hexdigest()


def _resolve_device(value: str) -> torch.device:
    value = str(value)
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise OutcomeDiagnosticError("CUDA was requested but is not available")
    return device


def _merge_dict(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge_dict(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


DEFAULT_DIAGNOSTICS: dict[str, Any] = {
    "base_run": "brainiac_main_v1_provenance",
    "run": "outcome_diag_v1",
    "folds": 5,
    "fallback_folds": 3,
    "fold_seed": 1701,
    "seeds": [7, 17, 29],
    "experiments": list(EXPERIMENT_ORDER),
    "pca_dim": 64,
    "residual_hidden_dim": 32,
    "shuffle_seed": 3109,
    "save_models": True,
}


def load_diagnostic_config(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return copy.deepcopy(DEFAULT_DIAGNOSTICS)
    source = Path(path)
    if not source.is_file():
        raise OutcomeDiagnosticError(f"diagnostic config not found: {source}")
    raw = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise OutcomeDiagnosticError("diagnostic config must be a YAML mapping")
    raw = raw.get("diagnostics", raw)
    if not isinstance(raw, dict):
        raise OutcomeDiagnosticError("diagnostics section must be a mapping")
    result = _merge_dict(DEFAULT_DIAGNOSTICS, raw)
    experiments = [str(x).upper() for x in result["experiments"]]
    unknown = sorted(set(experiments) - set(EXPERIMENT_ORDER))
    if unknown:
        raise OutcomeDiagnosticError(f"unknown diagnostic experiments: {unknown}")
    result["experiments"] = [x for x in EXPERIMENT_ORDER if x in experiments]
    result["seeds"] = [int(x) for x in result["seeds"]]
    result["folds"] = int(result["folds"])
    result["fallback_folds"] = int(result["fallback_folds"])
    result["fold_seed"] = int(result["fold_seed"])
    result["shuffle_seed"] = int(result["shuffle_seed"])
    result["pca_dim"] = int(result["pca_dim"])
    result["residual_hidden_dim"] = int(result["residual_hidden_dim"])
    if result["pca_dim"] < 1:
        raise OutcomeDiagnosticError("pca_dim must be positive")
    return result


def _patient_has_eligible_label(row: dict[str, Any]) -> bool:
    valid = torch.as_tensor(row["labels"]["valid"]).bool()
    return bool(valid.any())


def _patient_event(row: dict[str, Any]) -> int:
    valid = torch.as_tensor(row["labels"]["valid"]).bool()
    events = torch.as_tensor(row["labels"]["event"]).long()
    return int(bool(((events == 1) & valid).any()))


def _hashed_order(ids: Iterable[str], seed: int) -> list[str]:
    return sorted(
        [str(x) for x in ids],
        key=lambda pid: hashlib.sha256(f"{seed}:{pid}".encode()).hexdigest(),
    )


def make_stratified_patient_folds(
    patient_ids: Sequence[str],
    cache: dict[str, Any],
    requested_folds: int = 5,
    fallback_folds: int = 3,
    seed: int = 1701,
) -> list[list[str]]:
    """Deterministic patient-level event/censoring stratification.

    Patients without any eligible survival landmark are excluded from the
    diagnostic CV cohort and should be reported separately by the caller.
    """
    by_id = {str(row["patient_id"]): row for row in cache["patients"]}
    eligible = [pid for pid in patient_ids if pid in by_id and _patient_has_eligible_label(by_id[pid])]
    events = [pid for pid in eligible if _patient_event(by_id[pid]) == 1]
    censored = [pid for pid in eligible if _patient_event(by_id[pid]) == 0]
    if len(eligible) < 4 or not events or not censored:
        raise OutcomeDiagnosticError(
            "diagnostic CV requires at least four eligible patients and both event/censored groups"
        )

    folds = int(requested_folds)
    if min(len(events), len(censored)) < folds:
        folds = min(int(fallback_folds), len(events), len(censored))
    if folds < 2:
        raise OutcomeDiagnosticError("insufficient event/censoring support for >=2 patient folds")

    groups = [
        _hashed_order(events, seed),
        _hashed_order(censored, seed + 1),
    ]
    result: list[list[str]] = [[] for _ in range(folds)]
    for group in groups:
        for index, pid in enumerate(group):
            result[index % folds].append(pid)
    for fold in result:
        fold.sort()
        if not fold:
            raise OutcomeDiagnosticError("stratified fold construction produced an empty fold")
    flattened = [pid for fold in result for pid in fold]
    if len(flattened) != len(set(flattened)) or set(flattened) != set(eligible):
        raise OutcomeDiagnosticError("folds are not a disjoint cover of eligible development patients")
    return result


def make_patient_derangement(patient_ids: Sequence[str], seed: int) -> dict[str, str]:
    """Return a deterministic no-self patient permutation."""
    ids = _hashed_order(patient_ids, seed)
    if len(ids) < 2:
        raise OutcomeDiagnosticError("MRI shuffling requires at least two patients")
    donors = ids[1:] + ids[:1]
    mapping = dict(zip(ids, donors))
    if any(pid == donor for pid, donor in mapping.items()):
        raise OutcomeDiagnosticError("shuffle mapping contains a self-donor")
    return mapping


@dataclass(frozen=True)
class TrainOnlyPCA:
    mean: Tensor
    components: Tensor
    explained_variance_ratio: Tensor
    sample_count: int

    @classmethod
    def fit(cls, values: Tensor, n_components: int) -> "TrainOnlyPCA":
        if values.ndim != 2 or values.shape[0] < 2:
            raise OutcomeDiagnosticError("PCA needs a [N,D] tensor with N>=2")
        n_components = int(n_components)
        max_rank = min(values.shape[0] - 1, values.shape[1])
        if n_components > max_rank:
            raise OutcomeDiagnosticError(
                f"PCA dimension {n_components} exceeds training-fold rank bound {max_rank}"
            )
        x = values.detach().cpu().double()
        mean = x.mean(0)
        centered = x - mean
        _, singular, vh = torch.linalg.svd(centered, full_matrices=False)
        total = (singular.square()).sum().clamp_min(1e-12)
        ratio = singular[:n_components].square() / total
        return cls(
            mean=mean.float(),
            components=vh[:n_components].float(),
            explained_variance_ratio=ratio.float(),
            sample_count=int(values.shape[0]),
        )

    @property
    def output_dim(self) -> int:
        return int(self.components.shape[0])

    def transform(self, values: Tensor) -> Tensor:
        mean = self.mean.to(values.device, values.dtype)
        components = self.components.to(values.device, values.dtype)
        return (values - mean) @ components.transpose(0, 1)

    def state_dict(self) -> dict[str, Any]:
        return {
            "mean": self.mean.cpu(),
            "components": self.components.cpu(),
            "explained_variance_ratio": self.explained_variance_ratio.cpu(),
            "sample_count": self.sample_count,
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "TrainOnlyPCA":
        return cls(
            mean=torch.as_tensor(state["mean"]).float(),
            components=torch.as_tensor(state["components"]).float(),
            explained_variance_ratio=torch.as_tensor(state["explained_variance_ratio"]).float(),
            sample_count=int(state["sample_count"]),
        )


class DiagnosticOutcomeDataset(Dataset[dict[str, Any]]):
    """Outcome landmarks with optional split-local patient-level MRI shuffling."""

    def __init__(
        self,
        bundle: DataBundle,
        split: str,
        *,
        first_only: bool,
        donor_map: dict[str, str] | None = None,
    ):
        self.bundle = bundle
        self.refs: list[tuple[str, int]] = []
        self.patient_counts: dict[str, int] = {}
        self.donor_map = donor_map
        if split not in bundle.split_ids:
            raise OutcomeDiagnosticError(f"unknown bundle split: {split}")
        for pid in bundle.split_ids[split]:
            tr = bundle.trajectories[pid]
            eligible = torch.where(tr.survival_valid)[0].tolist()
            if first_only and eligible:
                eligible = eligible[:1]
            self.patient_counts[pid] = len(eligible)
            self.refs.extend((pid, int(index)) for index in eligible)

    def __len__(self) -> int:
        return len(self.refs)

    def __getitem__(self, index: int) -> dict[str, Any]:
        pid, landmark = self.refs[index]
        tr = self.bundle.trajectories[pid]
        z = tr.latents[landmark]
        donor_pid = None
        donor_index = None
        if self.donor_map is not None:
            donor_pid = self.donor_map[pid]
            donor = self.bundle.trajectories[donor_pid]
            donor_index = min(landmark, donor.length - 1)
            z = donor.latents[donor_index]
        count = self.patient_counts[pid]
        if count <= 0:
            raise OutcomeDiagnosticError("dataset contains a patient without an eligible landmark")
        row: dict[str, Any] = {
            "z": z.float(),
            "clinical": tr.clinical.float(),
            "clinical_mask": tr.clinical_mask.float(),
            "history": tr.histories[landmark].float(),
            "time": tr.survival_time[landmark].float(),
            "event": tr.survival_event[landmark].long(),
            "weight": torch.tensor(1.0 / count, dtype=torch.float32),
            "patient_id": pid,
            "timepoint": tr.timepoint_ids[landmark],
            "landmark_index": landmark,
        }
        # ``default_collate`` cannot batch ``None`` values. Donor metadata is
        # only defined for the E3 shuffled-MRI dataset, so omit the optional
        # fields for every other experiment instead of returning ``None``.
        if donor_pid is not None and donor_index is not None:
            row["donor_patient_id"] = donor_pid
            row["donor_landmark_index"] = donor_index
        return row


def _nll_from_rates(
    rates: Tensor,
    edges: Tensor,
    times: Tensor,
    events: Tensor,
    weights: Tensor | None = None,
    reduction: str = "mean",
) -> Tensor:
    exposure = interval_exposure(times, edges)
    admin_event = events.float() * (times <= edges[-1]).float()
    bins = event_bins(times, edges)
    selected_rate = rates.gather(-1, bins.unsqueeze(-1)).squeeze(-1)
    losses = (rates * exposure).sum(-1) - admin_event * torch.log(selected_rate.clamp_min(1e-12))
    if weights is not None:
        losses = losses * weights
        if reduction == "mean":
            return losses.sum() / weights.sum().clamp_min(1e-12)
    if reduction == "none":
        return losses
    if reduction == "sum":
        return losses.sum()
    if reduction != "mean":
        raise OutcomeDiagnosticError(f"unknown reduction: {reduction}")
    return losses.mean()


class FeaturePiecewiseExponentialHead(nn.Module):
    """Piecewise-exponential head for an already assembled feature vector."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        edges_days: Sequence[float],
    ):
        super().__init__()
        edges = torch.tensor(edges_days, dtype=torch.float32)
        self.register_buffer("edges", edges)
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, len(edges) - 1),
        )

    def rates(self, features: Tensor) -> Tensor:
        return F.softplus(self.network(features)) / 365.0

    def survival(self, features: Tensor, times: Tensor | float) -> Tensor:
        rates = self.rates(features)
        query = torch.as_tensor(times, dtype=rates.dtype, device=rates.device)
        if query.ndim == 0:
            query = query.expand(rates.shape[:-1])
        query = torch.clamp(query, min=0, max=float(self.edges[-1]))
        left = self.edges[:-1].to(rates.device)
        widths = (self.edges[1:] - self.edges[:-1]).to(rates.device)
        exposure = torch.minimum(torch.clamp(query.unsqueeze(-1) - left, min=0), widths)
        return torch.exp(-(rates * exposure).sum(-1))

    def nll(
        self,
        features: Tensor,
        times: Tensor,
        events: Tensor,
        weights: Tensor | None = None,
        reduction: str = "mean",
    ) -> Tensor:
        return _nll_from_rates(self.rates(features), self.edges, times, events, weights, reduction)


class ResidualPCAOutcomeHead(nn.Module):
    """Exact current O0 network plus a small additive PCA-MRI residual.

    The clinical base uses the same zero-latent feature layout and hidden width
    as PiecewiseExponentialHead(clinical_only=True). This makes alpha=0 exactly
    reproduce the current O0 network once its weights are loaded.
    """

    def __init__(
        self,
        latent_dim: int,
        clinical_dim: int,
        history_dim: int,
        pca_dim: int,
        *,
        hidden_dim: int,
        residual_hidden_dim: int,
        edges_days: Sequence[float],
    ):
        super().__init__()
        edges = torch.tensor(edges_days, dtype=torch.float32)
        self.register_buffer("edges", edges)
        self.latent_dim = int(latent_dim)
        input_dim = latent_dim + clinical_dim * 2 + history_dim
        self.base_network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, len(edges) - 1),
        )
        self.mri_network = nn.Sequential(
            nn.LayerNorm(pca_dim),
            nn.Linear(pca_dim, residual_hidden_dim),
            nn.SiLU(),
            nn.Linear(residual_hidden_dim, len(edges) - 1),
        )
        self.alpha = nn.Parameter(torch.tensor(0.0))

    def load_and_freeze_e0(self, e0_state: dict[str, Tensor]) -> None:
        network_state = {
            key[len("network."):]: value
            for key, value in e0_state.items()
            if key.startswith("network.")
        }
        if not network_state:
            raise OutcomeDiagnosticError("E0 checkpoint has no network.* parameters")
        self.base_network.load_state_dict(network_state)
        for parameter in self.base_network.parameters():
            parameter.requires_grad_(False)

    def logits(
        self,
        pca_z: Tensor,
        clinical: Tensor,
        clinical_mask: Tensor,
        history: Tensor,
    ) -> Tensor:
        zero_z = torch.zeros(
            *clinical.shape[:-1],
            self.latent_dim,
            dtype=clinical.dtype,
            device=clinical.device,
        )
        base_features = torch.cat((zero_z, clinical, clinical_mask, history), -1)
        base = self.base_network(base_features)
        residual = self.mri_network(pca_z)
        return base + self.alpha * residual

    def rates(self, pca_z: Tensor, clinical: Tensor, clinical_mask: Tensor, history: Tensor) -> Tensor:
        return F.softplus(self.logits(pca_z, clinical, clinical_mask, history)) / 365.0

    def survival(
        self,
        pca_z: Tensor,
        clinical: Tensor,
        clinical_mask: Tensor,
        history: Tensor,
        times: Tensor | float,
    ) -> Tensor:
        rates = self.rates(pca_z, clinical, clinical_mask, history)
        query = torch.as_tensor(times, dtype=rates.dtype, device=rates.device)
        if query.ndim == 0:
            query = query.expand(rates.shape[:-1])
        query = torch.clamp(query, min=0, max=float(self.edges[-1]))
        left = self.edges[:-1].to(rates.device)
        widths = (self.edges[1:] - self.edges[:-1]).to(rates.device)
        exposure = torch.minimum(torch.clamp(query.unsqueeze(-1) - left, min=0), widths)
        return torch.exp(-(rates * exposure).sum(-1))

    def nll(
        self,
        pca_z: Tensor,
        clinical: Tensor,
        clinical_mask: Tensor,
        history: Tensor,
        times: Tensor,
        events: Tensor,
        weights: Tensor | None = None,
        reduction: str = "mean",
    ) -> Tensor:
        rates = self.rates(pca_z, clinical, clinical_mask, history)
        return _nll_from_rates(rates, self.edges, times, events, weights, reduction)


def _eligible_latents(bundle: DataBundle, split: str) -> Tensor:
    values = []
    for pid in bundle.split_ids[split]:
        tr = bundle.trajectories[pid]
        indices = torch.where(tr.survival_valid)[0]
        if len(indices):
            values.append(tr.latents[indices].float())
    if not values:
        raise OutcomeDiagnosticError(f"{split} has no eligible survival landmarks for PCA")
    return torch.cat(values, 0)


def _train_survival_labels(bundle: DataBundle) -> tuple[list[float], list[int]]:
    times: list[float] = []
    events: list[int] = []
    for pid in bundle.split_ids["train"]:
        tr = bundle.trajectories[pid]
        eligible = torch.where(tr.survival_valid)[0].tolist()
        if not eligible:
            continue
        index = int(eligible[0])
        raw_time = float(tr.survival_time[index])
        max_days = 730.0
        times.append(min(raw_time, max_days))
        events.append(int(tr.survival_event[index]) if raw_time <= max_days else 0)
    return times, events


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, Tensor) else value
        for key, value in batch.items()
    }


def _features_for_e4(batch: dict[str, Any], pca: TrainOnlyPCA) -> Tensor:
    return torch.cat(
        (
            pca.transform(batch["z"]),
            batch["clinical"],
            batch["clinical_mask"],
            batch["history"],
        ),
        -1,
    )


def _model_loss(
    experiment: str,
    model: nn.Module,
    batch: dict[str, Any],
    pca: TrainOnlyPCA | None,
    *,
    reduction: str = "mean",
) -> Tensor:
    weights = batch["weight"] if reduction == "mean" else None
    if experiment in {"E0", "E2", "E3"}:
        return model.nll(
            batch["z"],
            batch["clinical"],
            batch["clinical_mask"],
            batch["history"],
            batch["time"],
            batch["event"],
            weights=weights,
            reduction=reduction,
        )
    if experiment == "E1":
        return model.nll(
            batch["z"],
            batch["time"],
            batch["event"],
            weights=weights,
            reduction=reduction,
        )
    if experiment == "E4":
        if pca is None:
            raise OutcomeDiagnosticError("E4 requires PCA")
        return model.nll(
            _features_for_e4(batch, pca),
            batch["time"],
            batch["event"],
            weights=weights,
            reduction=reduction,
        )
    if experiment == "E5":
        if pca is None:
            raise OutcomeDiagnosticError("E5 requires PCA")
        return model.nll(
            pca.transform(batch["z"]),
            batch["clinical"],
            batch["clinical_mask"],
            batch["history"],
            batch["time"],
            batch["event"],
            weights=weights,
            reduction=reduction,
        )
    raise OutcomeDiagnosticError(f"unknown experiment: {experiment}")


def _model_survival(
    experiment: str,
    model: nn.Module,
    batch: dict[str, Any],
    pca: TrainOnlyPCA | None,
    horizon_days: float,
) -> Tensor:
    if experiment in {"E0", "E2", "E3"}:
        return model.survival(
            batch["z"],
            batch["clinical"],
            batch["clinical_mask"],
            batch["history"],
            horizon_days,
        )
    if experiment == "E1":
        return model.survival(batch["z"], horizon_days)
    if experiment == "E4":
        if pca is None:
            raise OutcomeDiagnosticError("E4 requires PCA")
        return model.survival(_features_for_e4(batch, pca), horizon_days)
    if experiment == "E5":
        if pca is None:
            raise OutcomeDiagnosticError("E5 requires PCA")
        return model.survival(
            pca.transform(batch["z"]),
            batch["clinical"],
            batch["clinical_mask"],
            batch["history"],
            horizon_days,
        )
    raise OutcomeDiagnosticError(f"unknown experiment: {experiment}")


def _new_model(
    experiment: str,
    bundle: DataBundle,
    config: dict[str, Any],
    diag: dict[str, Any],
    *,
    e0_state: dict[str, Tensor] | None = None,
) -> nn.Module:
    out = config["outcome"]
    hidden = int(out["hidden_dim"])
    edges = [float(x) for x in out["edges_days"]]
    if experiment == "E0":
        return PiecewiseExponentialHead(
            bundle.latent_dim,
            bundle.clinical_dim,
            bundle.history_dim,
            hidden_dim=hidden,
            edges_days=edges,
            clinical_only=True,
        )
    if experiment in {"E2", "E3"}:
        return PiecewiseExponentialHead(
            bundle.latent_dim,
            bundle.clinical_dim,
            bundle.history_dim,
            hidden_dim=hidden,
            edges_days=edges,
            clinical_only=False,
        )
    if experiment == "E1":
        return FeaturePiecewiseExponentialHead(bundle.latent_dim, hidden, edges)
    if experiment == "E4":
        input_dim = int(diag["pca_dim"]) + bundle.clinical_dim * 2 + bundle.history_dim
        return FeaturePiecewiseExponentialHead(input_dim, hidden, edges)
    if experiment == "E5":
        if e0_state is None:
            raise OutcomeDiagnosticError("E5 requires the matching E0 checkpoint")
        model = ResidualPCAOutcomeHead(
            bundle.latent_dim,
            bundle.clinical_dim,
            bundle.history_dim,
            int(diag["pca_dim"]),
            hidden_dim=hidden,
            residual_hidden_dim=int(diag["residual_hidden_dim"]),
            edges_days=edges,
        )
        model.load_and_freeze_e0(e0_state)
        return model
    raise OutcomeDiagnosticError(f"unknown experiment: {experiment}")


@torch.no_grad()
def _validation_nll(
    experiment: str,
    model: nn.Module,
    dataset: DiagnosticOutcomeDataset,
    pca: TrainOnlyPCA | None,
    batch_size: int,
    device: torch.device,
) -> float:
    if len(dataset) == 0:
        raise OutcomeDiagnosticError("validation fold has no eligible survival landmark")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    total = 0.0
    weight_total = 0.0
    model.eval()
    for raw in loader:
        batch = _move_batch(raw, device)
        losses = _model_loss(experiment, model, batch, pca, reduction="none")
        weights = batch["weight"]
        total += float((losses * weights).sum().cpu())
        weight_total += float(weights.sum().cpu())
    return total / max(weight_total, 1e-12)


def _train_one(
    experiment: str,
    model: nn.Module,
    train_dataset: DiagnosticOutcomeDataset,
    val_dataset: DiagnosticOutcomeDataset,
    pca: TrainOnlyPCA | None,
    config: dict[str, Any],
    seed: int,
    device: torch.device,
) -> tuple[dict[str, Tensor], int, list[dict[str, Any]]]:
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise OutcomeDiagnosticError("training and validation datasets must be non-empty")
    _seed_all(seed)
    model = model.to(device)
    parameters = [p for p in model.parameters() if p.requires_grad]
    if not parameters:
        raise OutcomeDiagnosticError(f"{experiment} has no trainable parameters")
    optimizer = AdamW(
        parameters,
        lr=float(config["outcome"]["lr"]),
        weight_decay=float(config["outcome"]["weight_decay"]),
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        train_dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=True,
        generator=generator,
        num_workers=int(config["training"]["num_workers"]),
    )
    best_state: dict[str, Tensor] = {}
    best_epoch = 0
    best_metric = float("inf")
    stale = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, int(config["outcome"]["max_epochs"]) + 1):
        model.train()
        total = 0.0
        weight_total = 0.0
        for raw in loader:
            batch = _move_batch(raw, device)
            optimizer.zero_grad(set_to_none=True)
            loss = _model_loss(experiment, model, batch, pca, reduction="mean")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                parameters, float(config["training"]["grad_clip"])
            )
            optimizer.step()
            weight = float(batch["weight"].sum().cpu())
            total += float(loss.detach().cpu()) * weight
            weight_total += weight
        val_nll = _validation_nll(
            experiment,
            model,
            val_dataset,
            pca,
            int(config["training"]["batch_size"]),
            device,
        )
        train_nll = total / max(weight_total, 1e-12)
        history.append(
            {
                "epoch": epoch,
                "train_survival_nll": train_nll,
                "val_survival_nll": val_nll,
            }
        )
        if val_nll < best_metric:
            best_metric = val_nll
            best_epoch = epoch
            stale = 0
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        else:
            stale += 1
        if stale >= int(config["outcome"]["patience"]):
            break
    if not best_state:
        raise OutcomeDiagnosticError(f"{experiment} produced no best checkpoint")
    return best_state, best_epoch, history


@torch.no_grad()
def _evaluate_primary(
    experiment: str,
    model: nn.Module,
    dataset: DiagnosticOutcomeDataset,
    pca: TrainOnlyPCA | None,
    bundle: DataBundle,
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    if len(dataset) == 0:
        return {"available": False, "reason": "no_eligible_primary_landmark", "n": 0}
    loader = DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
    )
    times: list[float] = []
    events: list[int] = []
    probabilities: list[float] = []
    losses: list[float] = []
    patients: list[str] = []
    model.eval()
    for raw in loader:
        batch = _move_batch(raw, device)
        survival = _model_survival(
            experiment,
            model,
            batch,
            pca,
            float(config["outcome"]["score_horizon_days"]),
        )
        nll = _model_loss(experiment, model, batch, pca, reduction="none")
        for i in range(len(survival)):
            raw_time = float(batch["time"][i].cpu())
            max_days = float(config["outcome"]["edges_days"][-1])
            times.append(min(raw_time, max_days))
            events.append(int(batch["event"][i].cpu()) if raw_time <= max_days else 0)
            probabilities.append(float(survival[i].cpu()))
            losses.append(float(nll[i].cpu()))
            patients.append(str(raw["patient_id"][i]))
    train_times, train_events = _train_survival_labels(bundle)
    result = survival_summary(
        times,
        events,
        probabilities,
        train_times,
        train_events,
        float(config["outcome"]["score_horizon_days"]),
    )
    result.update(
        available=True,
        survival_nll=float(np.mean(losses)),
        patients=len(set(patients)),
        observations=len(patients),
        landmark="first_eligible",
    )
    return result


def _extract_scalar(result: dict[str, Any], metric: str) -> float | None:
    if metric == "c_index":
        value = result.get("c_index", {}).get("value")
    elif metric == "brier365":
        value = result.get("ipcw_brier", {}).get("value")
    elif metric == "nll":
        value = result.get("survival_nll")
    else:
        raise OutcomeDiagnosticError(f"unknown summary metric: {metric}")
    if value is None or not math.isfinite(float(value)):
        return None
    return float(value)


def _generalization_diagnostics(
    history: list[dict[str, Any]],
    best_epoch: int,
) -> dict[str, Any]:
    best = next(row for row in history if int(row["epoch"]) == int(best_epoch))
    final = history[-1]
    return {
        "best_epoch": int(best_epoch),
        "train_nll_at_best": float(best["train_survival_nll"]),
        "val_nll_at_best": float(best["val_survival_nll"]),
        "final_train_nll": float(final["train_survival_nll"]),
        "final_val_nll": float(final["val_survival_nll"]),
        "generalization_gap_at_best": float(
            best["val_survival_nll"] - best["train_survival_nll"]
        ),
        "generalization_gap_final": float(
            final["val_survival_nll"] - final["train_survival_nll"]
        ),
    }


def _aggregate_tasks(tasks: dict[str, Any], experiments: Sequence[str]) -> dict[str, Any]:
    """Aggregate seed results within fold, then folds across the development cohort."""
    summary: dict[str, Any] = {}
    fold_values: dict[str, dict[str, dict[str, float]]] = {}
    for experiment in experiments:
        per_fold: dict[str, dict[str, list[float]]] = {}
        for key, value in tasks.items():
            if value.get("experiment") != experiment:
                continue
            fold_key = f"fold{int(value['fold'])}"
            bucket = per_fold.setdefault(
                fold_key,
                {"c_index": [], "brier365": [], "nll": [], "best_epoch": []},
            )
            primary = value["primary"]
            for metric in ("c_index", "brier365", "nll"):
                scalar = _extract_scalar(primary, metric)
                if scalar is not None:
                    bucket[metric].append(scalar)
            bucket["best_epoch"].append(float(value["training"]["best_epoch"]))
        fold_mean: dict[str, dict[str, float]] = {}
        for fold_key, values in per_fold.items():
            fold_mean[fold_key] = {
                metric: float(np.mean(metric_values))
                for metric, metric_values in values.items()
                if metric_values
            }
        fold_values[experiment] = fold_mean
        summary[experiment] = {
            metric: mean_std(
                [
                    values[metric]
                    for values in fold_mean.values()
                    if metric in values
                ]
            )
            for metric in ("c_index", "brier365", "nll", "best_epoch")
        }

    paired_vs_e0: dict[str, Any] = {}
    baseline = fold_values.get("E0", {})
    for experiment in experiments:
        if experiment == "E0":
            continue
        rows = []
        for fold_key in sorted(set(baseline) & set(fold_values.get(experiment, {}))):
            base = baseline[fold_key]
            candidate = fold_values[experiment][fold_key]
            if not all(metric in base and metric in candidate for metric in ("c_index", "brier365", "nll")):
                continue
            rows.append(
                {
                    "fold": fold_key,
                    "delta_c_index": candidate["c_index"] - base["c_index"],
                    "delta_brier365": candidate["brier365"] - base["brier365"],
                    "delta_nll": candidate["nll"] - base["nll"],
                }
            )
        paired_vs_e0[experiment] = {
            "folds": rows,
            "delta_c_index": mean_std([row["delta_c_index"] for row in rows]),
            "delta_brier365": mean_std([row["delta_brier365"] for row in rows]),
            "delta_nll": mean_std([row["delta_nll"] for row in rows]),
            "direction_counts": {
                "c_index_improved": sum(row["delta_c_index"] > 0 for row in rows),
                "brier365_improved": sum(row["delta_brier365"] < 0 for row in rows),
                "nll_improved": sum(row["delta_nll"] < 0 for row in rows),
                "folds": len(rows),
            },
        }
    return {"experiments": summary, "paired_vs_e0": paired_vs_e0}


def _load_base(
    config: dict[str, Any],
    diag: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], list[str], list[str]]:
    base_artifacts = RunArtifacts(config["paths"]["output_root"], str(diag["base_run"]))
    manifest = read_json(base_artifacts.path("run.json"))
    if not manifest:
        raise OutcomeDiagnosticError(f"base run not found: {diag['base_run']}")
    cache_path = Path(str(manifest.get("cache_path", "")))
    if not cache_path.is_file():
        raise OutcomeDiagnosticError(f"base run cache is missing: {cache_path}")
    cache = read_torch(cache_path, safe=True)
    if cache.get("data_signature") != manifest.get("data_signature"):
        raise OutcomeDiagnosticError("base run data signature no longer matches its cache")
    split = manifest.get("split") or {}
    dev_ids = [str(x) for x in split.get("train", []) + split.get("validation", [])]
    formal_test = [str(x) for x in split.get("test", [])]
    if set(dev_ids) & set(formal_test):
        raise OutcomeDiagnosticError("base split has train/validation overlap with formal test")
    if not dev_ids or not formal_test:
        raise OutcomeDiagnosticError("base run must contain train, validation, and formal test IDs")
    return manifest, cache, dev_ids, formal_test


def _build_fold_bundle(
    cache: dict[str, Any],
    config: dict[str, Any],
    train_ids: Sequence[str],
    val_ids: Sequence[str],
    formal_test_ids: Sequence[str],
) -> tuple[DataBundle, dict[str, Any]]:
    split = {
        "train": list(train_ids),
        "validation": list(val_ids),
        "test": list(formal_test_ids),
    }
    preprocessing = fit_preprocessing(cache, config, split)
    bundle = load_bundle(cache, config, split, preprocessing)
    return bundle, preprocessing


def run_diagnostics(
    config: dict[str, Any],
    diag: dict[str, Any],
    *,
    force: bool = False,
) -> Path:
    base_manifest, cache, dev_ids, formal_test_ids = _load_base(config, diag)
    by_id = {str(row["patient_id"]): row for row in cache["patients"]}
    eligible_dev = [pid for pid in dev_ids if pid in by_id and _patient_has_eligible_label(by_id[pid])]
    excluded_dev = sorted(set(dev_ids) - set(eligible_dev))
    folds = make_stratified_patient_folds(
        dev_ids,
        cache,
        requested_folds=int(diag["folds"]),
        fallback_folds=int(diag["fallback_folds"]),
        seed=int(diag["fold_seed"]),
    )
    effective_folds = len(folds)

    artifacts = RunArtifacts(config["paths"]["output_root"], str(diag["run"]), version="v1_1")
    if str(diag["run"]) == str(diag["base_run"]):
        raise OutcomeDiagnosticError("diagnostic run must differ from base formal run")
    artifacts.root.mkdir(parents=True, exist_ok=True)

    diagnostic_signature = _sha256_json(
        {
            "base_data_signature": base_manifest["data_signature"],
            "diagnostics": diag,
            "outcome": config["outcome"],
            "training": {
                key: config["training"][key]
                for key in ("batch_size", "num_workers", "grad_clip")
            },
        }
    )
    existing_manifest = read_json(artifacts.path("run.json"))
    if existing_manifest is not None:
        if existing_manifest.get("diagnostic_signature") != diagnostic_signature:
            raise OutcomeDiagnosticError(
                "existing diagnostic run has a different signature; use a new --run name"
            )
        manifest = existing_manifest
    else:
        manifest = {
            "schema_version": "cloop_outcome_diag_v1",
            "run_name": str(diag["run"]),
            "base_run": str(diag["base_run"]),
            "base_project_revision": base_manifest.get("project_revision"),
            "base_data_signature": base_manifest["data_signature"],
            "diagnostic_signature": diagnostic_signature,
            "formal_test_untouched": True,
            "formal_test_ids_sha256": _sha256_json(sorted(formal_test_ids)),
            "development_ids_sha256": _sha256_json(sorted(dev_ids)),
            "eligible_development_patients": len(eligible_dev),
            "excluded_without_survival_label": excluded_dev,
            "effective_folds": effective_folds,
            "folds": [
                {
                    "fold": index,
                    "validation_ids": fold,
                    "train_ids": sorted(set(eligible_dev) - set(fold)),
                }
                for index, fold in enumerate(folds)
            ],
            "diagnostics": copy.deepcopy(diag),
            "completed_tasks": [],
        }
        write_json(artifacts.path("run.json"), manifest)

    models = (
        read_torch(artifacts.path("models.pt"), safe=True)
        if artifacts.path("models.pt").exists()
        else {
            "schema_version": "cloop_outcome_diag_models_v1",
            "base_data_signature": base_manifest["data_signature"],
            "pca_by_fold": {},
            "tasks": {},
        }
    )
    metrics = (
        read_json(artifacts.path("metrics.json"))
        if artifacts.path("metrics.json").exists()
        else {
            "schema_version": "cloop_outcome_diag_metrics_v1",
            "tasks": {},
            "summary": {},
        }
    )

    device = _resolve_device(config["project"]["device"])
    experiments = list(diag["experiments"])
    seeds = [int(x) for x in diag["seeds"]]
    completed = set(str(x) for x in manifest.get("completed_tasks", []))

    for fold_index, val_ids in enumerate(folds):
        train_ids = sorted(set(eligible_dev) - set(val_ids))
        bundle, preprocessing = _build_fold_bundle(
            cache, config, train_ids, val_ids, formal_test_ids
        )

        pca: TrainOnlyPCA | None = None
        if any(experiment in {"E4", "E5"} for experiment in experiments):
            pca_key = f"fold{fold_index}"
            if pca_key in models["pca_by_fold"]:
                pca = TrainOnlyPCA.from_state_dict(models["pca_by_fold"][pca_key])
            else:
                pca = TrainOnlyPCA.fit(
                    _eligible_latents(bundle, "train"),
                    int(diag["pca_dim"]),
                )
                models["pca_by_fold"][pca_key] = {
                    **pca.state_dict(),
                    "fit_patient_ids_sha256": _sha256_json(sorted(train_ids)),
                    "fit_scope": "training_fold_eligible_landmarks_only",
                }
                write_torch(artifacts.path("models.pt"), models)

        train_donor = make_patient_derangement(
            train_ids, int(diag["shuffle_seed"]) + fold_index * 2
        )
        val_donor = make_patient_derangement(
            val_ids, int(diag["shuffle_seed"]) + fold_index * 2 + 1
        )

        for experiment in experiments:
            donor_train = train_donor if experiment == "E3" else None
            donor_val = val_donor if experiment == "E3" else None
            train_dataset = DiagnosticOutcomeDataset(
                bundle,
                "train",
                first_only=False,
                donor_map=donor_train,
            )
            val_trainlike = DiagnosticOutcomeDataset(
                bundle,
                "validation",
                first_only=False,
                donor_map=donor_val,
            )
            val_primary = DiagnosticOutcomeDataset(
                bundle,
                "validation",
                first_only=True,
                donor_map=donor_val,
            )
            for seed in seeds:
                task_key = f"{experiment}/fold{fold_index}/seed{seed}"
                if task_key in completed and not force:
                    continue
                e0_state = None
                if experiment == "E5":
                    e0_key = f"E0/fold{fold_index}/seed{seed}"
                    e0_entry = models["tasks"].get(e0_key)
                    if e0_entry is None:
                        raise OutcomeDiagnosticError(
                            f"E5 requires completed matching {e0_key}; run E0 first"
                        )
                    e0_state = e0_entry["state"]

                model = _new_model(
                    experiment,
                    bundle,
                    config,
                    diag,
                    e0_state=e0_state,
                )
                best_state, best_epoch, history = _train_one(
                    experiment,
                    model,
                    train_dataset,
                    val_trainlike,
                    pca,
                    config,
                    seed,
                    device,
                )
                model.load_state_dict(best_state)
                model = model.to(device).eval()
                primary = _evaluate_primary(
                    experiment,
                    model,
                    val_primary,
                    pca,
                    bundle,
                    config,
                    device,
                )
                training_diag = _generalization_diagnostics(history, best_epoch)
                extras: dict[str, Any] = {}
                if experiment == "E5":
                    extras["alpha_best"] = float(model.alpha.detach().cpu())
                    with torch.no_grad():
                        sample_loader = DataLoader(
                            val_primary,
                            batch_size=int(config["training"]["batch_size"]),
                            shuffle=False,
                        )
                        raw = next(iter(sample_loader), None)
                        if raw is not None:
                            batch = _move_batch(raw, device)
                            pca_z = pca.transform(batch["z"]) if pca is not None else None
                            if pca_z is not None:
                                zero_z = torch.zeros(
                                    len(batch["z"]),
                                    bundle.latent_dim,
                                    device=device,
                                )
                                base_features = torch.cat(
                                    (
                                        zero_z,
                                        batch["clinical"],
                                        batch["clinical_mask"],
                                        batch["history"],
                                    ),
                                    -1,
                                )
                                base_logits = model.base_network(base_features)
                                mri_logits = model.mri_network(pca_z)
                                extras["clinical_logit_norm"] = float(
                                    base_logits.norm(dim=-1).mean().cpu()
                                )
                                extras["mri_residual_logit_norm"] = float(
                                    mri_logits.norm(dim=-1).mean().cpu()
                                )

                models["tasks"][task_key] = {
                    "state": best_state,
                    "experiment": experiment,
                    "fold": fold_index,
                    "seed": seed,
                    "preprocessing_sha256": _sha256_json(
                        {
                            "latent_low_variance_count": int(
                                preprocessing["latent_normalizer"].get(
                                    "low_variance_count", 0
                                )
                            ),
                            "clinical_categories": preprocessing["clinical_codec"][
                                "categories"
                            ],
                            "action_vocab": preprocessing["action_codec"]["vocab"],
                        }
                    ),
                }
                metrics["tasks"][task_key] = {
                    "experiment": experiment,
                    "fold": fold_index,
                    "seed": seed,
                    "primary": primary,
                    "training": training_diag,
                    "extras": extras,
                    "shuffle": (
                        {
                            "scope": "split_local_patient_derangement",
                            "train_mapping_sha256": _sha256_json(train_donor),
                            "validation_mapping_sha256": _sha256_json(val_donor),
                        }
                        if experiment == "E3"
                        else None
                    ),
                    "pca": (
                        {
                            "dimension": pca.output_dim,
                            "sample_count": pca.sample_count,
                            "explained_variance_ratio_sum": float(
                                pca.explained_variance_ratio.sum()
                            ),
                            "fit_scope": "training_fold_eligible_landmarks_only",
                        }
                        if experiment in {"E4", "E5"} and pca is not None
                        else None
                    ),
                }
                completed.add(task_key)
                manifest["completed_tasks"] = sorted(completed)
                metrics["summary"] = _aggregate_tasks(metrics["tasks"], experiments)
                write_torch(artifacts.path("models.pt"), models)
                write_json(artifacts.path("metrics.json"), metrics)
                write_json(artifacts.path("run.json"), manifest)

    metrics["summary"] = _aggregate_tasks(metrics["tasks"], experiments)
    manifest["completed"] = all(
        f"{experiment}/fold{fold_index}/seed{seed}" in completed
        for experiment in experiments
        for fold_index in range(effective_folds)
        for seed in seeds
    )
    write_json(artifacts.path("metrics.json"), metrics)
    write_json(artifacts.path("run.json"), manifest)
    artifacts.assert_flat()
    return artifacts.root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m cloop.v1_1.outcome_diagnostics",
        description="E0-E5 patient-level CV diagnostics for the Cloop Outcome Head",
    )
    parser.add_argument("--config", default="configs/v1/default.yaml")
    parser.add_argument("--paths", default="configs/v1/server.yaml")
    parser.add_argument(
        "--diag-config",
        default="configs/v1_1/outcome_diagnostics.yaml",
        help="diagnostic-only YAML; not part of the formal main_v1 config",
    )
    parser.add_argument("--base-run", default=None)
    parser.add_argument("--run", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--folds", type=int, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=EXPERIMENT_ORDER,
        default=None,
    )
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_config(args.config, args.paths, device=args.device)
    diag = load_diagnostic_config(args.diag_config)
    if args.base_run is not None:
        diag["base_run"] = args.base_run
    if args.run is not None:
        diag["run"] = args.run
    if args.folds is not None:
        diag["folds"] = int(args.folds)
    if args.seeds is not None:
        diag["seeds"] = [int(x) for x in args.seeds]
    if args.experiments is not None:
        selected = {str(x).upper() for x in args.experiments}
        diag["experiments"] = [x for x in EXPERIMENT_ORDER if x in selected]
    output = run_diagnostics(config, diag, force=args.force)
    print(json.dumps({"status": "ok", "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
