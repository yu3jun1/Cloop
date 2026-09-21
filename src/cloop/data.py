"""Data import, temporal alignment, train-only codecs, and window construction."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from .artifacts import read_torch, write_torch
from .config import canonical_json
from .types import Action, FactualTarget, PatientState


class DataError(RuntimeError):
    pass


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(path: str | Path, patterns: tuple[str, ...] = ("*.npy", "*.npz", "*.json")) -> str:
    root = Path(path)
    digest = hashlib.sha256()
    if not root.exists():
        return "missing"
    files: list[Path] = []
    for pattern in patterns:
        files.extend(root.rglob(pattern))
    for item in sorted(set(files), key=lambda p: p.relative_to(root).as_posix()):
        digest.update(item.relative_to(root).as_posix().encode())
        digest.update(str(item.stat().st_size).encode())
        digest.update(sha256_file(item).encode())
    return digest.hexdigest()


def split_patients(
    patient_ids: Iterable[str],
    split_seed: int = 17,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> dict[str, list[str]]:
    """Order-independent patient split with deterministic small-cohort safeguards."""
    ids = sorted(
        set(str(p) for p in patient_ids),
        key=lambda p: hashlib.sha256(f"{split_seed}:{p}".encode()).hexdigest(),
    )
    n = len(ids)
    if n < 3:
        raise DataError("at least three patients are required for train/validation/test")
    n_train = max(1, int(math.floor(n * train_fraction)))
    n_val = max(1, int(math.floor(n * validation_fraction)))
    if n_train + n_val >= n:
        n_train = max(1, n - 2)
        n_val = 1
    return {
        "train": ids[:n_train],
        "validation": ids[n_train : n_train + n_val],
        "test": ids[n_train + n_val :],
    }


@dataclass(frozen=True)
class Normalizer:
    mean: Tensor
    std: Tensor
    low_variance_count: int = 0

    @classmethod
    def fit(cls, values: Tensor, min_std: float = 1e-6) -> "Normalizer":
        if values.ndim != 2 or values.shape[0] == 0:
            raise DataError("normalizer needs a non-empty [N,D] tensor")
        mean = values.float().mean(0)
        std_raw = values.float().std(0, unbiased=False)
        low = std_raw < min_std
        std = torch.where(low, torch.ones_like(std_raw), std_raw)
        return cls(mean=mean, std=std, low_variance_count=int(low.sum().item()))

    def transform(self, value: Tensor) -> Tensor:
        return (value.float() - self.mean.to(value.device)) / self.std.to(value.device)

    def inverse(self, value: Tensor) -> Tensor:
        return value * self.std.to(value.device) + self.mean.to(value.device)

    def state_dict(self) -> dict[str, Any]:
        return {
            "mean": self.mean.cpu(),
            "std": self.std.cpu(),
            "low_variance_count": self.low_variance_count,
        }

    @classmethod
    def from_state_dict(cls, value: dict[str, Any]) -> "Normalizer":
        return cls(value["mean"].float(), value["std"].float(), int(value.get("low_variance_count", 0)))


def _norm_term(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip().lower())
    return text


def _action_id(terms: Sequence[str], known_empty: bool = False) -> str:
    if known_empty:
        return "EMPTY"
    digest = hashlib.sha256("\x1f".join(sorted(terms)).encode()).hexdigest()[:12]
    return f"A_{digest}"


class ActionCodec:
    """Train-only token vocabulary and stable treatment-set catalog."""

    def __init__(
        self,
        vocab: Sequence[str],
        catalog: Sequence[Action] = (),
        *,
        min_support: int = 2,
    ):
        if not vocab:
            raise DataError("action vocabulary cannot be empty")
        self.vocab = tuple(vocab)
        self.index = {term: i for i, term in enumerate(self.vocab)}
        self.unk_index = self.index.get("<UNK>")
        self.catalog = tuple(catalog)
        self.by_id = {action.action_id: action for action in self.catalog}
        self.min_support = int(min_support)

    @property
    def dim(self) -> int:
        return len(self.vocab)

    @classmethod
    def fit(cls, action_sets: Iterable[tuple[str, ...]], min_support: int = 2) -> "ActionCodec":
        canonical_sets = [tuple(sorted(set(_norm_term(x) for x in terms if _norm_term(x)))) for terms in action_sets]
        vocab = ("<UNK>", *sorted({token for terms in canonical_sets for token in terms}))
        index = {term: i for i, term in enumerate(vocab)}
        counts = Counter(canonical_sets)
        actions: list[Action] = []
        for terms, count in counts.items():
            if count < min_support:
                continue
            empty = len(terms) == 0
            actions.append(
                Action(
                    action_id=_action_id(terms, empty),
                    token_ids=tuple(index[t] for t in terms),
                    display_terms=terms,
                    support_count=int(count),
                    known_empty=empty,
                )
            )
        actions.sort(key=lambda a: (-a.support_count, a.action_id))
        return cls(vocab, actions, min_support=min_support)

    def encode_terms(self, terms: Iterable[str]) -> Tensor:
        result = torch.zeros(self.dim, dtype=torch.float32)
        normalized = sorted(set(_norm_term(x) for x in terms if _norm_term(x)))
        for term in normalized:
            token = self.index.get(term, self.unk_index)
            if token is not None:
                result[token] = 1.0
        return result

    def action_from_terms(self, terms: Iterable[str], *, support_count: int = 0, known_empty: bool = False) -> Action:
        normalized = tuple(sorted(set(_norm_term(x) for x in terms if _norm_term(x))))
        token_ids = tuple(
            sorted(
                {
                    token
                    for term in normalized
                    if (token := self.index.get(term, self.unk_index)) is not None
                }
            )
        )
        return Action(
            action_id=_action_id(normalized, known_empty),
            token_ids=token_ids,
            display_terms=normalized,
            support_count=int(support_count),
            known_empty=known_empty,
        )

    def decode(self, action_id: str) -> Action:
        try:
            return self.by_id[action_id]
        except KeyError as exc:
            raise DataError(f"action id is outside the training catalog: {action_id}") from exc

    def state_dict(self) -> dict[str, Any]:
        return {
            "vocab": list(self.vocab),
            "min_support": self.min_support,
            "catalog": [
                {
                    "action_id": a.action_id,
                    "token_ids": list(a.token_ids),
                    "display_terms": list(a.display_terms),
                    "support_count": a.support_count,
                    "known_empty": a.known_empty,
                }
                for a in self.catalog
            ],
        }

    @classmethod
    def from_state_dict(cls, value: dict[str, Any]) -> "ActionCodec":
        catalog = [
            Action(
                action_id=row["action_id"],
                token_ids=tuple(int(x) for x in row["token_ids"]),
                display_terms=tuple(str(x) for x in row["display_terms"]),
                support_count=int(row["support_count"]),
                known_empty=bool(row.get("known_empty", False)),
            )
            for row in value.get("catalog", [])
        ]
        return cls(value["vocab"], catalog, min_support=int(value.get("min_support", 2)))


class ClinicalCodec:
    """Small auditable static-clinical codec; categories are fitted on train only."""

    categorical_fields = ("sex_at_birth", "who_grade", "idh", "mgmt")

    def __init__(self, age_mean: float, age_std: float, categories: dict[str, tuple[str, ...]]):
        self.age_mean = float(age_mean)
        self.age_std = float(age_std) if age_std > 1e-6 else 1.0
        self.categories = categories
        self._dim = 1 + sum(1 + len(categories[field]) for field in self.categorical_fields)

    @staticmethod
    def _raw_value(context: dict[str, Any], field: str) -> Any:
        if field in {"idh", "mgmt"}:
            genomics = context.get("genomics") or {}
            if isinstance(genomics, dict):
                for key, value in genomics.items():
                    if field in _norm_term(key):
                        return value
            return None
        return context.get(field)

    @classmethod
    def fit(cls, contexts: Iterable[dict[str, Any]]) -> "ClinicalCodec":
        rows = list(contexts)
        ages = []
        for row in rows:
            age = row.get("age_at_diagnosis_years")
            if isinstance(age, (int, float)) and math.isfinite(float(age)):
                ages.append(float(age))
        age_mean = float(np.mean(ages)) if ages else 0.0
        age_std = float(np.std(ages)) if ages else 1.0
        categories: dict[str, tuple[str, ...]] = {}
        for field in cls.categorical_fields:
            categories[field] = tuple(
                sorted(
                    {
                        _norm_term(cls._raw_value(row, field))
                        for row in rows
                        if cls._raw_value(row, field) not in (None, "")
                    }
                )
            )
        return cls(age_mean, age_std, categories)

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, context: dict[str, Any]) -> tuple[Tensor, Tensor]:
        values: list[float] = []
        masks: list[float] = []
        age = context.get("age_at_diagnosis_years")
        age_ok = isinstance(age, (int, float)) and math.isfinite(float(age))
        values.append((float(age) - self.age_mean) / self.age_std if age_ok else 0.0)
        masks.append(1.0 if age_ok else 0.0)
        for field in self.categorical_fields:
            raw = self._raw_value(context, field)
            known = raw not in (None, "")
            normalized = _norm_term(raw) if known else ""
            categories = self.categories[field]
            # First slot is train-unknown/missing; the explicit mask distinguishes them.
            one_hot = [0.0] * (1 + len(categories))
            one_hot[categories.index(normalized) + 1 if normalized in categories else 0] = 1.0
            values.extend(one_hot)
            masks.extend([1.0 if known else 0.0] * len(one_hot))
        return torch.tensor(values), torch.tensor(masks)

    def state_dict(self) -> dict[str, Any]:
        return {
            "age_mean": self.age_mean,
            "age_std": self.age_std,
            "categories": {k: list(v) for k, v in self.categories.items()},
        }

    @classmethod
    def from_state_dict(cls, value: dict[str, Any]) -> "ClinicalCodec":
        return cls(
            float(value["age_mean"]),
            float(value["age_std"]),
            {k: tuple(v) for k, v in value["categories"].items()},
        )


def history_dim(action_dim: int) -> int:
    return action_dim * 2 + 2


def empty_history(action_dim: int) -> Tensor:
    return torch.zeros(history_dim(action_dim), dtype=torch.float32)


def update_history(history: Tensor, action: Tensor, delta_days: Tensor | float) -> Tensor:
    """Parameter-free executed-history update used by data, rollout, and planner."""
    action_dim = action.shape[-1]
    last = action.float()
    cumulative = history[..., action_dim : 2 * action_dim] + action.float()
    count = history[..., 2 * action_dim : 2 * action_dim + 1] + 1.0
    delta = torch.as_tensor(delta_days, dtype=history.dtype, device=history.device)
    while delta.ndim < history.ndim:
        delta = delta.unsqueeze(-1)
    elapsed = history[..., 2 * action_dim + 1 :] + delta / 365.0
    return torch.cat((last, cumulative, count, elapsed), dim=-1)


def _finite_day(value: Any) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


MRI_DAY_SOURCES = (
    "observed",
    "imputed_interior",
    "imputed_leading",
    "imputed_trailing",
    "unknown",
)


def _mri_day_source(node: dict[str, Any]) -> str:
    """Return only auditable source labels; never infer observation from a numeric date."""
    value = node.get("mri_day_source")
    if value is None and isinstance(node.get("quality_flags"), dict):
        value = node["quality_flags"].get("mri_day_source")
    normalized = _norm_term(value).replace("-", "_").replace(" ", "_")
    aliases = {
        "interior": "imputed_interior",
        "interpolated": "imputed_interior",
        "imputed_interpolated": "imputed_interior",
        "leading": "imputed_leading",
        "trailing": "imputed_trailing",
    }
    normalized = aliases.get(normalized, normalized)
    return normalized if normalized in MRI_DAY_SOURCES else "unknown"


def _load_latent_provenance(config: dict[str, Any], latent_dir: Path) -> dict[str, Any]:
    manifest_path = Path(config["paths"].get("latent_provenance", ""))
    if not manifest_path.is_file():
        raise DataError(f"latent provenance manifest is required: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataError(f"cannot read latent provenance manifest: {exc}") from exc
    if not isinstance(manifest, dict):
        raise DataError("latent provenance manifest must be a JSON object")
    required = {
        "encoder",
        "frozen",
        "adapter",
        "checkpoint",
        "checkpoint_sha256",
        "output_dim",
        "extraction_protocol",
        "source_commit",
        "num_timepoints",
    }
    missing = sorted(required - set(manifest))
    if missing:
        raise DataError("latent provenance is missing fields: " + ", ".join(missing))
    if manifest["encoder"] != "brainiac" or manifest["encoder"] != config["data"]["encoder"]:
        raise DataError("latent provenance requires encoder=brainiac matching data.encoder")
    if manifest["frozen"] is not True:
        raise DataError("latent provenance requires frozen=true")
    if manifest["adapter"] is not None:
        raise DataError("latent provenance requires adapter=null")
    if int(manifest["output_dim"]) != 768:
        raise DataError("latent provenance requires output_dim=768")
    if not str(manifest["extraction_protocol"]).strip() or not str(manifest["source_commit"]).strip():
        raise DataError("latent provenance requires extraction_protocol and source_commit")
    checkpoint = Path(str(manifest["checkpoint"]))
    configured_checkpoint = Path(config["paths"].get("brainiac_checkpoint", ""))
    if checkpoint != configured_checkpoint:
        raise DataError("latent provenance checkpoint does not match paths.brainiac_checkpoint")
    if not checkpoint.is_file():
        raise DataError(f"BrainIAC checkpoint is missing: {checkpoint}")
    actual_checkpoint_hash = sha256_file(checkpoint)
    if actual_checkpoint_hash != str(manifest["checkpoint_sha256"]):
        raise DataError("BrainIAC checkpoint SHA-256 does not match latent provenance")
    actual_timepoints = len(_latent_lookup(latent_dir))
    if int(manifest["num_timepoints"]) != actual_timepoints:
        raise DataError(
            "latent provenance num_timepoints does not match latent directory: "
            f"{manifest['num_timepoints']} != {actual_timepoints}"
        )
    return {
        **manifest,
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
    }


def _event_fingerprint(event: dict[str, Any]) -> str:
    relevant = {
        key: event.get(key)
        for key in (
            "category",
            "agent",
            "start_day",
            "end_day",
            "day_of_insertion",
            "dose_gy",
            "fractions",
            "duration_unknown",
            "assigned_reason",
        )
        if key in event
    }
    return hashlib.sha256(canonical_json(relevant).encode()).hexdigest()


def collect_events(timeline: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten and exact-deduplicate event fragments while retaining provenance."""
    events: dict[str, dict[str, Any]] = {}
    for node_index, node in enumerate(timeline):
        actions = node.get("actions") or {}
        if not isinstance(actions, dict):
            continue
        for raw_category, rows in actions.items():
            if not isinstance(rows, list):
                continue
            category = _norm_term(raw_category).replace("_", " ")
            for raw in rows:
                if not isinstance(raw, dict):
                    continue
                event = {str(k): v for k, v in raw.items() if isinstance(v, (str, int, float, bool)) or v is None}
                event["category"] = category
                event["agent"] = _norm_term(raw.get("agent"))
                event["source_node_index"] = node_index
                event["source_timepoint"] = str(node.get("tp_id", node_index))
                fingerprint = str(raw.get("event_id") or _event_fingerprint(event))
                if fingerprint not in events:
                    event["fingerprint"] = fingerprint
                    events[fingerprint] = event
    return list(events.values())


def event_bounds(event: dict[str, Any]) -> tuple[float | None, float | None, bool]:
    point = _finite_day(event.get("day_of_insertion"))
    if point is not None:
        return point, point, True
    start = _finite_day(event.get("start_day"))
    end = _finite_day(event.get("end_day"))
    if start is not None and end is None and not event.get("duration_unknown"):
        # A dated one-off event is a point; duration_unknown is never promoted to exposure.
        return start, start, True
    if start is not None and end is not None and end >= start:
        return start, end, True
    return start, end, False


def event_terms(event: dict[str, Any]) -> tuple[str, ...]:
    terms = [f"category:{event['category']}"]
    if event.get("agent"):
        terms.append(f"agent:{event['agent']}")
    return tuple(terms)


def align_interval_actions(
    timeline: Sequence[dict[str, Any]], events: Sequence[dict[str, Any]]
) -> tuple[list[tuple[str, ...]], list[bool], dict[str, int]]:
    """Map explicitly dated events to (d_t,d_t+1]; undated destination events break continuity."""
    actions: list[tuple[str, ...]] = []
    known: list[bool] = []
    audit = Counter()
    for idx, (left, right) in enumerate(zip(timeline, timeline[1:])):
        d0, d1 = _finite_day(left.get("mri_day")), _finite_day(right.get("mri_day"))
        if d0 is None or d1 is None or d1 <= d0:
            actions.append(())
            known.append(False)
            audit["invalid_time_interval"] += 1
            continue
        terms: set[str] = set()
        interval_known = True
        for event in events:
            start, end, explicit = event_bounds(event)
            if explicit and start is not None and end is not None:
                if start == end:
                    overlaps = d0 < start <= d1
                else:
                    overlaps = max(start, d0) < min(end, d1)
                if overlaps:
                    terms.update(event_terms(event))
            elif str(event.get("source_timepoint")) == str(right.get("tp_id")):
                # Its location suggests this historical interval, but timing is not auditable.
                interval_known = False
                audit["uncertain_destination_event"] += 1
        actions.append(tuple(sorted(terms)))
        known.append(interval_known)
        audit["known_interval" if interval_known else "excluded_unknown_interval"] += 1
    return actions, known, dict(audit)


def _validate_timeline(raw_timeline: Any) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if not isinstance(raw_timeline, list):
        return [], {"timeline_not_list": 1}
    audit = Counter()
    rows = [dict(row) for row in raw_timeline if isinstance(row, dict)]
    missing = sum(_finite_day(row.get("mri_day")) is None for row in rows)
    if missing:
        audit["missing_mri_day"] += missing
        rows = [row for row in rows if _finite_day(row.get("mri_day")) is not None]
    original = [float(row["mri_day"]) for row in rows]
    rows.sort(key=lambda row: (float(row["mri_day"]), str(row.get("tp_id", ""))))
    if original != [float(row["mri_day"]) for row in rows]:
        audit["reordered_timeline"] += 1
    days = [float(row["mri_day"]) for row in rows]
    duplicates = len(days) - len(set(days))
    if duplicates:
        audit["duplicate_mri_day"] += duplicates
    return rows, dict(audit)


def _latent_lookup(latent_dir: Path) -> dict[tuple[str, int], Path]:
    result: dict[tuple[str, int], Path] = {}
    pattern = re.compile(r"^(?P<pid>.+)_Timepoint_(?P<tp>\d+)\.npy$")
    for path in latent_dir.glob("*.npy"):
        match = pattern.match(path.name)
        if match:
            result[(match.group("pid"), int(match.group("tp")))] = path
    return result


def _tp_number(tp_id: Any) -> int | None:
    match = re.search(r"(\d+)$", str(tp_id))
    return int(match.group(1)) if match else None


def _survival_label(node: dict[str, Any], exclude_shifted: bool) -> tuple[float, int, bool, str]:
    surv = node.get("survival") or {}
    if not isinstance(surv, dict):
        return float("nan"), -1, False, "missing_survival"
    rule = str(surv.get("censoring_rule") or "")
    if exclude_shifted and "death_shifted_to_L_plus_1" in rule:
        return float("nan"), -1, False, "death_shifted_to_L_plus_1"
    time = surv.get("survival_from_tp_days")
    event = surv.get("event_indicator")
    if not isinstance(time, (int, float)) or not math.isfinite(float(time)) or float(time) <= 0:
        return float("nan"), -1, False, "invalid_survival_time"
    if event not in (0, 1, False, True):
        return float("nan"), -1, False, "invalid_event"
    return float(time), int(event), True, rule or "unspecified"


def build_main_cache(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    paths = config["paths"]
    timeline_path = Path(paths["timeline"])
    latent_dir = Path(paths["latent_dir"])
    latent_provenance = _load_latent_provenance(config, latent_dir)
    try:
        raw = json.loads(timeline_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataError(f"cannot read clinical timeline: {exc}") from exc
    patients_raw = raw.get("patients") if isinstance(raw, dict) else None
    if not isinstance(patients_raw, dict):
        raise DataError("timeline must contain a patients mapping")
    latent_paths = _latent_lookup(latent_dir)
    patients: list[dict[str, Any]] = []
    audit = Counter()
    time_quality = Counter()
    warnings: list[str] = []
    for patient_id, patient in sorted(patients_raw.items()):
        if not isinstance(patient, dict):
            audit["invalid_patient"] += 1
            continue
        timeline, time_audit = _validate_timeline(patient.get("timeline"))
        audit.update(time_audit)
        if len(timeline) < 2:
            audit["fewer_than_two_valid_times"] += 1
            continue
        selected_nodes: list[dict[str, Any]] = []
        selected_latents: list[Tensor] = []
        for node in timeline:
            tp_num = _tp_number(node.get("tp_id"))
            latent_path = latent_paths.get((str(patient_id), tp_num or -1))
            if latent_path is None:
                audit["missing_latent"] += 1
                continue
            array = np.load(latent_path, allow_pickle=False)
            tensor = torch.from_numpy(np.asarray(array, dtype=np.float32)).reshape(-1)
            if not torch.isfinite(tensor).all():
                audit["nonfinite_latent"] += 1
                continue
            selected_nodes.append(node)
            selected_latents.append(tensor)
        if len(selected_nodes) < 2:
            audit["fewer_than_two_latents"] += 1
            continue
        dims = {int(z.numel()) for z in selected_latents}
        if len(dims) != 1:
            audit["latent_dimension_mismatch"] += 1
            continue
        events = collect_events(timeline)
        action_terms, action_known, action_audit = align_interval_actions(selected_nodes, events)
        audit.update(action_audit)
        mri_day_sources = [_mri_day_source(node) for node in selected_nodes]
        for source in mri_day_sources:
            time_quality[f"timepoint_{source}"] += 1
        survival_time: list[float] = []
        survival_event: list[int] = []
        survival_valid: list[bool] = []
        survival_rules: list[str] = []
        for node in selected_nodes:
            time, event, valid, reason = _survival_label(
                node, bool(config["data"]["exclude_shifted_death_labels"])
            )
            survival_time.append(time)
            survival_event.append(event)
            survival_valid.append(valid)
            survival_rules.append(reason)
            if not valid:
                audit[f"survival_{reason}"] += 1
        days = torch.tensor([float(node["mri_day"]) for node in selected_nodes], dtype=torch.float32)
        endpoint_groups: dict[tuple[int, str], list[int]] = {}
        for label_index, (event, valid, rule) in enumerate(
            zip(survival_event, survival_valid, survival_rules)
        ):
            if valid:
                endpoint_groups.setdefault((event, rule), []).append(label_index)
        inconsistent_indices: set[int] = set()
        for indices in endpoint_groups.values():
            endpoints = [float(days[i]) + survival_time[i] for i in indices]
            if len(endpoints) > 1 and max(endpoints) - min(endpoints) > 1.0:
                inconsistent_indices.update(indices)
        for label_index in inconsistent_indices:
            survival_time[label_index] = float("nan")
            survival_event[label_index] = -1
            survival_valid[label_index] = False
            survival_rules[label_index] = "inconsistent_absolute_endpoint"
        if inconsistent_indices:
            audit["survival_inconsistent_absolute_endpoint"] += len(inconsistent_indices)
        if bool((days[1:] <= days[:-1]).any()):
            # Duplicate days remain auditable and must never be silently altered.
            action_known = [False if days[i + 1] <= days[i] else flag for i, flag in enumerate(action_known)]
        for index in range(len(selected_nodes) - 1):
            time_quality["total_transitions"] += 1
            sources = mri_day_sources[index : index + 2]
            if any(source.startswith("imputed_") for source in sources):
                time_quality["transitions_involving_imputation"] += 1
            if "unknown" in sources:
                time_quality["transitions_involving_unknown_time"] += 1
        for horizon in (2, 3):
            for start in range(len(selected_nodes) - horizon):
                if not all(action_known[start : start + horizon]):
                    continue
                time_quality[f"h{horizon}_windows"] += 1
                sources = mri_day_sources[start : start + horizon + 1]
                if any(source.startswith("imputed_") for source in sources):
                    time_quality[f"h{horizon}_windows_involving_imputation"] += 1
                if "unknown" in sources:
                    time_quality[f"h{horizon}_windows_involving_unknown_time"] += 1
        source_counts = Counter(mri_day_sources)
        patient_time_quality = (
            next(iter(source_counts)) if len(source_counts) == 1 else "mixed"
        )
        patients.append(
            {
                "patient_id": str(patient_id),
                "timepoint_ids": [str(node.get("tp_id", i)) for i, node in enumerate(selected_nodes)],
                "mri_days": days,
                "mri_day_sources": mri_day_sources,
                "latents_raw": torch.stack(selected_latents),
                "clinical_raw": patient.get("context_static") if isinstance(patient.get("context_static"), dict) else {},
                "events": events,
                "action_terms": [list(x) for x in action_terms],
                "action_known": torch.tensor(action_known, dtype=torch.bool),
                "labels": {
                    "survival_time": torch.tensor(survival_time, dtype=torch.float32),
                    "event": torch.tensor(survival_event, dtype=torch.int64),
                    "valid": torch.tensor(survival_valid, dtype=torch.bool),
                    "censoring_rule": survival_rules,
                },
                "quality_flags": {
                    "time_quality": patient_time_quality,
                    "mri_day_source_counts": dict(source_counts),
                },
            }
        )
    if not patients:
        raise DataError("no patient has at least two valid timestamped latent observations")
    latent_dims = {int(p["latents_raw"].shape[1]) for p in patients}
    if len(latent_dims) != 1:
        raise DataError(f"inconsistent latent dimensions across patients: {sorted(latent_dims)}")
    latent_dim = next(iter(latent_dims))
    if latent_dim != int(latent_provenance["output_dim"]):
        raise DataError(
            "latent vectors do not match provenance output_dim: "
            f"{latent_dim} != {latent_provenance['output_dim']}"
        )
    time_quality_report = {
        "total_timepoints": sum(
            time_quality[f"timepoint_{source}"] for source in MRI_DAY_SOURCES
        ),
        "timepoint_sources": {
            source: time_quality[f"timepoint_{source}"] for source in MRI_DAY_SOURCES
        },
        "total_transitions": time_quality["total_transitions"],
        "transitions_involving_imputation": time_quality["transitions_involving_imputation"],
        "transitions_involving_unknown_time": time_quality["transitions_involving_unknown_time"],
        "windows": {
            f"H{horizon}": {
                "eligible": time_quality[f"h{horizon}_windows"],
                "involving_imputation": time_quality[
                    f"h{horizon}_windows_involving_imputation"
                ],
                "involving_unknown_time": time_quality[
                    f"h{horizon}_windows_involving_unknown_time"
                ],
            }
            for horizon in (2, 3)
        },
    }
    if time_quality_report["timepoint_sources"]["unknown"]:
        warnings.append(
            "MRI date provenance is absent for some timepoints; they are labeled unknown, not observed"
        )
    cache = {
        "schema_version": "cloop_data_v2",
        "source": {
            "timeline_sha256": sha256_file(timeline_path),
            "latent_manifest_sha256": sha256_tree(latent_dir, ("*.npy",)),
            "latent_provenance_sha256": latent_provenance["manifest_sha256"],
            "preparation_revision": "main_v1_alignment_r2",
        },
        "encoder": {
            "name": latent_provenance["encoder"],
            "frozen": latent_provenance["frozen"],
            "adapter": latent_provenance["adapter"],
            "checkpoint": latent_provenance["checkpoint"],
            "checkpoint_sha256": latent_provenance["checkpoint_sha256"],
            "latent_dim": latent_dim,
            "output_dim": latent_provenance["output_dim"],
            "extraction_protocol": latent_provenance["extraction_protocol"],
            "source_commit": latent_provenance["source_commit"],
            "num_timepoints": latent_provenance["num_timepoints"],
            "provenance_manifest_path": latent_provenance["manifest_path"],
            "provenance_manifest_sha256": latent_provenance["manifest_sha256"],
        },
        "protocol": "main_v1",
        "patients": patients,
        "audit": {
            "excluded_counts": dict(audit),
            "time_quality": time_quality_report,
            "warnings": warnings,
        },
    }
    return cache, cache["audit"]


def build_legacy_cache(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    root = Path(config["paths"]["legacy_trajectories"])
    metadata_path = root / "metadata.json"
    if not metadata_path.exists():
        raise DataError(f"legacy metadata missing: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    vocab = [str(x) for x in metadata.get("action_vocab", [])]
    patients: list[dict[str, Any]] = []
    audit = Counter()
    for row in metadata.get("trajectories", []):
        path = root / row["file"]
        try:
            with np.load(path, allow_pickle=False) as archive:
                latents = torch.from_numpy(archive["latents"].astype(np.float32))
                actions = torch.from_numpy(archive["actions"].astype(np.float32))
                deltas = torch.from_numpy(archive["delta_days"].astype(np.float32))
                timepoints = [str(x) for x in archive["timepoints"].tolist()]
                patient_id = str(archive["patient_id"].item())
        except (OSError, KeyError, ValueError) as exc:
            audit["invalid_legacy_trajectory"] += 1
            continue
        if len(latents) < 2 or len(actions) != len(latents) - 1 or bool((deltas <= 0).any()):
            audit["invalid_legacy_shapes_or_time"] += 1
            continue
        terms = [tuple(vocab[i] for i in torch.where(action > 0)[0].tolist()) for action in actions]
        days = torch.cat((torch.zeros(1), torch.cumsum(deltas, 0)))
        patients.append(
            {
                "patient_id": patient_id,
                "timepoint_ids": timepoints,
                "mri_days": days,
                "latents_raw": latents,
                "clinical_raw": {},
                "events": [],
                "action_terms": [list(x) for x in terms],
                "action_known": torch.ones(len(actions), dtype=torch.bool),
                "labels": {
                    "survival_time": torch.full((len(latents),), float("nan")),
                    "event": torch.full((len(latents),), -1, dtype=torch.int64),
                    "valid": torch.zeros(len(latents), dtype=torch.bool),
                    "censoring_rule": ["missing_legacy"] * len(latents),
                },
                "quality_flags": {"time_quality": "legacy_imputed"},
            }
        )
    if not patients:
        raise DataError("no usable legacy trajectory")
    cache = {
        "schema_version": "cloop_data_v1",
        "source": {
            "legacy_metadata_sha256": sha256_file(metadata_path),
            "preparation_revision": "legacy_stage1_import_r1",
        },
        "encoder": {"name": "brainiac", "frozen": True, "latent_dim": int(metadata["latent_dim"])},
        "protocol": "legacy_stage1",
        "legacy_action_vocab": vocab,
        "patients": patients,
        "audit": {
            "excluded_counts": dict(audit),
            "warnings": ["legacy_stage1 uses the historical all-data action vocabulary convention"],
        },
    }
    return cache, cache["audit"]


def cache_signature(cache: dict[str, Any]) -> str:
    compact = {
        "schema_version": cache["schema_version"],
        "source": cache["source"],
        "encoder": cache["encoder"],
        "protocol": cache["protocol"],
        "patients": [
            {
                "patient_id": row["patient_id"],
                "timepoint_ids": row["timepoint_ids"],
                "days": row["mri_days"].tolist(),
                "mri_day_sources": row.get(
                    "mri_day_sources", ["unknown"] * len(row["timepoint_ids"])
                ),
                "latent_shape": list(row["latents_raw"].shape),
                "action_terms": row["action_terms"],
                "action_known": row["action_known"].tolist(),
            }
            for row in cache["patients"]
        ],
    }
    return hashlib.sha256(canonical_json(compact).encode()).hexdigest()


def prepare_cache(config: dict[str, Any], cache_path: str | Path) -> tuple[dict[str, Any], bool]:
    cache_path = Path(cache_path)
    protocol = config["data"]["protocol"]
    builder = build_main_cache if protocol == "main_v1" else build_legacy_cache
    candidate, _ = builder(config)
    candidate["data_signature"] = cache_signature(candidate)
    if cache_path.exists():
        existing = read_torch(cache_path, safe=True)
        if existing.get("data_signature") == candidate["data_signature"]:
            return existing, True
        raise DataError(
            f"cache signature differs at {cache_path}; use a new protocol/cache name instead of overwriting"
        )
    write_torch(cache_path, candidate)
    return candidate, False


@dataclass
class Trajectory:
    patient_id: str
    timepoint_ids: tuple[str, ...]
    days: Tensor
    latents: Tensor
    clinical: Tensor
    clinical_mask: Tensor
    actions: Tensor
    action_objects: tuple[Action | None, ...]
    action_known: Tensor
    histories: Tensor
    survival_time: Tensor
    survival_event: Tensor
    survival_valid: Tensor
    censoring_rule: tuple[str, ...]

    @property
    def length(self) -> int:
        return int(self.latents.shape[0])


@dataclass
class DataBundle:
    trajectories: dict[str, Trajectory]
    split_ids: dict[str, list[str]]
    latent_normalizer: Normalizer
    clinical_codec: ClinicalCodec
    action_codec: ActionCodec
    data_signature: str
    planned_interval_days: float
    interval_support_days: tuple[float, float]
    audit: dict[str, Any]

    @property
    def latent_dim(self) -> int:
        return int(next(iter(self.trajectories.values())).latents.shape[-1])

    @property
    def clinical_dim(self) -> int:
        return self.clinical_codec.dim

    @property
    def action_dim(self) -> int:
        return self.action_codec.dim

    @property
    def history_dim(self) -> int:
        return history_dim(self.action_dim)


def fit_preprocessing(cache: dict[str, Any], config: dict[str, Any], split_ids: dict[str, list[str]]) -> dict[str, Any]:
    by_id = {row["patient_id"]: row for row in cache["patients"]}
    train = [by_id[pid] for pid in split_ids["train"]]
    latent_norm = Normalizer.fit(
        torch.cat([row["latents_raw"].float() for row in train], 0),
        float(config["data"]["min_latent_std"]),
    )
    clinical_codec = ClinicalCodec.fit(row["clinical_raw"] for row in train)
    if cache["protocol"] == "legacy_stage1":
        # Legacy regression preserves the historical projection width and vocabulary exactly.
        vocab = tuple(str(x) for x in cache.get("legacy_action_vocab", []))
        sets = [tuple(terms) for row in train for terms, ok in zip(row["action_terms"], row["action_known"]) if bool(ok)]
        counts = Counter(tuple(sorted(set(x))) for x in sets)
        idx = {x: i for i, x in enumerate(vocab)}
        catalog = [
            Action(_action_id(terms, not terms), tuple(idx[t] for t in terms), terms, count, not terms)
            for terms, count in counts.items()
            if count >= int(config["data"]["min_action_support"])
        ]
        catalog.sort(key=lambda action: (-action.support_count, action.action_id))
        action_codec = ActionCodec(vocab, catalog, min_support=int(config["data"]["min_action_support"]))
    else:
        train_sets = [
            tuple(terms)
            for row in train
            for terms, ok in zip(row["action_terms"], row["action_known"])
            if bool(ok)
        ]
        action_codec = ActionCodec.fit(train_sets, int(config["data"]["min_action_support"]))
    positive_deltas = []
    for row in train:
        deltas = row["mri_days"][1:] - row["mri_days"][:-1]
        positive_deltas.extend(float(x) for x in deltas[deltas > 0].tolist())
    if not positive_deltas:
        raise DataError("training split has no positive MRI interval")
    return {
        "latent_normalizer": latent_norm.state_dict(),
        "clinical_codec": clinical_codec.state_dict(),
        "action_codec": action_codec.state_dict(),
        "planned_interval_days": float(np.median(positive_deltas)),
        "interval_support_days": [float(min(positive_deltas)), float(max(positive_deltas))],
        "normalizer_source": "train_patients_only",
    }


def load_bundle(
    cache: dict[str, Any],
    config: dict[str, Any],
    split_ids: dict[str, list[str]] | None = None,
    preprocessing: dict[str, Any] | None = None,
) -> DataBundle:
    patient_ids = [row["patient_id"] for row in cache["patients"]]
    split_ids = split_ids or split_patients(
        patient_ids,
        int(config["data"]["split_seed"]),
        float(config["data"]["train_fraction"]),
        float(config["data"]["validation_fraction"]),
    )
    preprocessing = preprocessing or fit_preprocessing(cache, config, split_ids)
    latent_norm = Normalizer.from_state_dict(preprocessing["latent_normalizer"])
    clinical_codec = ClinicalCodec.from_state_dict(preprocessing["clinical_codec"])
    action_codec = ActionCodec.from_state_dict(preprocessing["action_codec"])
    trajectories: dict[str, Trajectory] = {}
    for row in cache["patients"]:
        clinical, clinical_mask = clinical_codec.encode(row["clinical_raw"])
        vectors = torch.stack([action_codec.encode_terms(x) for x in row["action_terms"]])
        objects = tuple(
            action_codec.action_from_terms(terms, known_empty=len(terms) == 0) if bool(ok) else None
            for terms, ok in zip(row["action_terms"], row["action_known"])
        )
        histories = [empty_history(action_codec.dim)]
        for action, delta, known in zip(
            vectors,
            row["mri_days"][1:] - row["mri_days"][:-1],
            row["action_known"],
        ):
            histories.append(
                update_history(histories[-1], action, delta) if bool(known) else histories[-1].clone()
            )
        labels = row["labels"]
        trajectories[row["patient_id"]] = Trajectory(
            patient_id=row["patient_id"],
            timepoint_ids=tuple(row["timepoint_ids"]),
            days=row["mri_days"].float(),
            latents=latent_norm.transform(row["latents_raw"]),
            clinical=clinical.float(),
            clinical_mask=clinical_mask.float(),
            actions=vectors.float(),
            action_objects=objects,
            action_known=row["action_known"].bool(),
            histories=torch.stack(histories),
            survival_time=labels["survival_time"].float(),
            survival_event=labels["event"].long(),
            survival_valid=labels["valid"].bool(),
            censoring_rule=tuple(labels["censoring_rule"]),
        )
    return DataBundle(
        trajectories=trajectories,
        split_ids=split_ids,
        latent_normalizer=latent_norm,
        clinical_codec=clinical_codec,
        action_codec=action_codec,
        data_signature=cache["data_signature"],
        planned_interval_days=float(preprocessing["planned_interval_days"]),
        interval_support_days=tuple(
            float(x)
            for x in preprocessing.get(
                "interval_support_days",
                [preprocessing["planned_interval_days"], preprocessing["planned_interval_days"]],
            )
        ),
        audit=cache.get("audit", {}),
    )


@dataclass(frozen=True)
class WindowRef:
    patient_id: str
    start: int
    horizon: int

    @property
    def target(self) -> int:
        return self.start + self.horizon


def window_refs(
    bundle: DataBundle,
    split: str,
    *,
    mode: str,
    max_horizon: int = 3,
    exact_horizon: int | None = None,
) -> list[WindowRef]:
    if split not in bundle.split_ids:
        raise DataError(f"unknown split: {split}")
    if mode not in {"one_step", "max_available", "all_exact"}:
        raise DataError(f"unknown window mode: {mode}")
    refs: list[WindowRef] = []
    for pid in bundle.split_ids[split]:
        trajectory = bundle.trajectories[pid]
        for start in range(trajectory.length - 1):
            contiguous = 0
            for offset in range(min(max_horizon, trajectory.length - 1 - start)):
                idx = start + offset
                delta = trajectory.days[idx + 1] - trajectory.days[idx]
                if not bool(trajectory.action_known[idx]) or float(delta) <= 0:
                    break
                contiguous += 1
            if mode == "one_step" and contiguous >= 1:
                refs.append(WindowRef(pid, start, 1))
            elif mode == "max_available" and contiguous >= 1:
                refs.append(WindowRef(pid, start, contiguous))
            elif mode == "all_exact" and exact_horizon is not None and contiguous >= exact_horizon:
                refs.append(WindowRef(pid, start, exact_horizon))
    return refs


class DynamicsDataset(Dataset[dict[str, Any]]):
    def __init__(self, bundle: DataBundle, refs: Sequence[WindowRef]):
        self.bundle = bundle
        self.refs = list(refs)

    def __len__(self) -> int:
        return len(self.refs)

    def __getitem__(self, index: int) -> dict[str, Any]:
        ref = self.refs[index]
        tr = self.bundle.trajectories[ref.patient_id]
        end = ref.target
        return {
            "z0": tr.latents[ref.start],
            "actions": tr.actions[ref.start:end],
            "deltas": tr.days[ref.start + 1 : end + 1] - tr.days[ref.start:end],
            "context": tr.clinical,
            "clinical_mask": tr.clinical_mask,
            "history0": tr.histories[ref.start],
            "target": tr.latents[end],
            "horizon": ref.horizon,
            "patient_id": ref.patient_id,
            "start_timepoint": tr.timepoint_ids[ref.start],
            "target_timepoint": tr.timepoint_ids[end],
            "target_index": end,
        }


def collate_dynamics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise DataError("cannot collate empty dynamics batch")
    batch = len(rows)
    max_k = max(int(row["horizon"]) for row in rows)
    action_dim = int(rows[0]["actions"].shape[-1])
    actions = torch.zeros(batch, max_k, action_dim)
    deltas = torch.zeros(batch, max_k)
    mask = torch.zeros(batch, max_k, dtype=torch.bool)
    for i, row in enumerate(rows):
        k = int(row["horizon"])
        actions[i, :k] = row["actions"]
        deltas[i, :k] = row["deltas"]
        mask[i, :k] = True
    return {
        "z0": torch.stack([row["z0"] for row in rows]),
        "actions": actions,
        "deltas": deltas,
        "step_mask": mask,
        "context": torch.stack([row["context"] for row in rows]),
        "clinical_mask": torch.stack([row["clinical_mask"] for row in rows]),
        "history0": torch.stack([row["history0"] for row in rows]),
        "target": torch.stack([row["target"] for row in rows]),
        "horizons": torch.tensor([row["horizon"] for row in rows], dtype=torch.long),
        "patient_id": [row["patient_id"] for row in rows],
        "start_timepoint": [row["start_timepoint"] for row in rows],
        "target_timepoint": [row["target_timepoint"] for row in rows],
        "target_index": torch.tensor([row["target_index"] for row in rows], dtype=torch.long),
    }


@dataclass(frozen=True)
class OutcomeRef:
    patient_id: str
    index: int


class OutcomeDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        bundle: DataBundle,
        split: str,
        *,
        first_only: bool = False,
        clinical_only: bool = False,
    ):
        self.bundle = bundle
        self.clinical_only = clinical_only
        self.refs: list[OutcomeRef] = []
        self.patient_counts: Counter[str] = Counter()
        for pid in bundle.split_ids[split]:
            tr = bundle.trajectories[pid]
            eligible = torch.where(tr.survival_valid)[0].tolist()
            if first_only and eligible:
                eligible = eligible[:1]
            self.refs.extend(OutcomeRef(pid, int(i)) for i in eligible)
            self.patient_counts[pid] = len(eligible)

    def __len__(self) -> int:
        return len(self.refs)

    def __getitem__(self, index: int) -> dict[str, Any]:
        ref = self.refs[index]
        tr = self.bundle.trajectories[ref.patient_id]
        z = torch.zeros_like(tr.latents[ref.index]) if self.clinical_only else tr.latents[ref.index]
        return {
            "z": z,
            "clinical": tr.clinical,
            "clinical_mask": tr.clinical_mask,
            "history": tr.histories[ref.index],
            "time": tr.survival_time[ref.index],
            "event": tr.survival_event[ref.index],
            "weight": torch.tensor(1.0 / self.patient_counts[ref.patient_id]),
            "patient_id": ref.patient_id,
            "timepoint": tr.timepoint_ids[ref.index],
            "index": ref.index,
        }


def collate_outcome(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    tensor_keys = ("z", "clinical", "clinical_mask", "history", "time", "event", "weight")
    result = {key: torch.stack([torch.as_tensor(row[key]) for row in rows]) for key in tensor_keys}
    result.update(
        patient_id=[row["patient_id"] for row in rows],
        timepoint=[row["timepoint"] for row in rows],
        index=torch.tensor([row["index"] for row in rows], dtype=torch.long),
    )
    return result


def patient_state(bundle: DataBundle, patient_id: str, index: int, state_version: int) -> PatientState:
    tr = bundle.trajectories[patient_id]
    return PatientState(
        patient_key=patient_id,
        timepoint_key=tr.timepoint_ids[index],
        observed_day=float(tr.days[index]),
        z=tr.latents[index].clone(),
        clinical=tr.clinical.clone(),
        clinical_mask=tr.clinical_mask.clone(),
        history=tr.histories[index].clone(),
        source="observed",
        state_version=state_version,
    )


def factual_target(bundle: DataBundle, patient_id: str, index: int) -> FactualTarget:
    tr = bundle.trajectories[patient_id]
    if index >= tr.length - 1 or not bool(tr.action_known[index]):
        raise DataError("no known factual transition at this index")
    target_index = index + 1
    valid_surv = bool(tr.survival_valid[target_index])
    return FactualTarget(
        next_z=tr.latents[target_index].clone(),
        actual_action=tr.action_objects[index],
        actual_delta_days=float(tr.days[target_index] - tr.days[index]),
        survival_time=float(tr.survival_time[target_index]) if valid_surv else None,
        event=int(tr.survival_event[target_index]) if valid_surv else None,
    )


class ObservedReplay:
    """Historical observation iterator; recommendations are deliberately not accepted by next_observation."""

    def __init__(self, bundle: DataBundle, patient_id: str):
        self.bundle = bundle
        self.patient_id = patient_id
        self.index = 0
        self.state_version = 0

    def next_observation(self) -> tuple[PatientState, FactualTarget] | None:
        trajectory = self.bundle.trajectories[self.patient_id]
        while self.index < trajectory.length - 1:
            current = self.index
            self.index += 1
            if not bool(trajectory.action_known[current]):
                self.state_version += 1
                continue
            state = patient_state(self.bundle, self.patient_id, current, self.state_version)
            target = factual_target(self.bundle, self.patient_id, current)
            self.state_version += 1
            return state, target
        return None


def make_tiny_cache(
    *,
    num_patients: int = 18,
    steps: int = 5,
    latent_dim: int = 8,
    seed: int = 17,
) -> dict[str, Any]:
    """Deterministic in-memory non-medical trajectories for smoke and tests."""
    generator = torch.Generator().manual_seed(seed)
    patients = []
    tokens = (("category:a0",), ("category:a1",), ("category:a2",))
    effects = torch.randn(3, latent_dim, generator=generator) * 0.15
    for p in range(num_patients):
        z = torch.randn(latent_dim, generator=generator) * 0.3
        latents = [z]
        action_terms = []
        days = [0.0]
        for t in range(steps - 1):
            action_index = (p + t) % 3
            action_terms.append(list(tokens[action_index]))
            z = 0.9 * z + effects[action_index] + 0.01 * torch.randn(latent_dim, generator=generator)
            latents.append(z)
            days.append(days[-1] + 90.0 + 5.0 * ((p + t) % 3))
        survival = torch.tensor([max(30.0, 800.0 - 20.0 * p - day) for day in days])
        patients.append(
            {
                "patient_id": f"SYN_{p:03d}",
                "timepoint_ids": [f"TP{i+1}" for i in range(steps)],
                "mri_days": torch.tensor(days),
                "latents_raw": torch.stack(latents),
                "clinical_raw": {
                    "age_at_diagnosis_years": 30 + p,
                    "sex_at_birth": "x" if p % 2 else "y",
                    "who_grade": str(2 + p % 3),
                    "genomics": {"idh": "a" if p % 2 else "b", "mgmt": "m" if p % 3 else "u"},
                },
                "events": [],
                "action_terms": action_terms,
                "action_known": torch.ones(steps - 1, dtype=torch.bool),
                "labels": {
                    "survival_time": survival,
                    "event": torch.tensor([int(p % 3 != 0)] * steps),
                    "valid": torch.ones(steps, dtype=torch.bool),
                    "censoring_rule": ["synthetic"] * steps,
                },
                "quality_flags": {"time_quality": "verified"},
            }
        )
    cache = {
        "schema_version": "cloop_data_v1",
        "source": {"kind": "deterministic_smoke", "seed": seed},
        "encoder": {"name": "synthetic", "frozen": True, "latent_dim": latent_dim},
        "protocol": "main_v1",
        "patients": patients,
        "audit": {"excluded_counts": {}, "warnings": ["non-medical smoke data"]},
    }
    cache["data_signature"] = cache_signature(cache)
    return cache


def iter_observed_replay(bundle: DataBundle, split: str) -> Iterator[tuple[PatientState, FactualTarget]]:
    """Yield only observed states; recommendation never drives the historical transition."""
    for pid in bundle.split_ids[split]:
        replay = ObservedReplay(bundle, pid)
        while (item := replay.next_observation()) is not None:
            yield item
