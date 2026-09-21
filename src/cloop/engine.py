"""Training, evaluation, replay, synthetic experiments, smoke, and reporting."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from torch import Tensor
from torch.optim import AdamW
from torch.utils.data import DataLoader

from . import __version__
from .artifacts import (
    ArtifactError,
    RunArtifacts,
    read_json,
    read_torch,
    upsert_jsonl,
    write_json,
    write_text,
    write_torch,
)
from .config import config_signature
from .data import (
    ActionCodec,
    DataBundle,
    DataError,
    DynamicsDataset,
    empty_history,
    OutcomeDataset,
    WindowRef,
    cache_signature,
    collate_dynamics,
    collate_outcome,
    fit_preprocessing,
    iter_observed_replay,
    load_bundle,
    make_tiny_cache,
    patient_state,
    prepare_cache,
    sha256_file,
    split_patients,
    update_history,
    window_refs,
)
from .metrics import (
    action_set_metrics,
    dynamics_paired_improvements,
    dynamics_summary,
    reliability_summary,
    survival_summary,
)
from .outcome import PiecewiseExponentialHead, SyntheticCostHead
from .planner import Planner
from .policy import CatalogPolicy, FakeProvider, LLMPolicy
from .synthetic import ToyEnv, ToyState, encode_state, env_cost, optimal_expected_value, sample_episodes
from .types import Action, PatientState
from .world import (
    EnsembleWorldModel,
    LegacyOneStepDynamics,
    OneStepDynamics,
    PersistenceDynamics,
    ensemble_mean_and_disagreement,
    terminal_mse,
    terminal_states,
    variant_spec,
)


class EngineError(RuntimeError):
    pass


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        return torch.device(
            "cuda" if torch.cuda.is_available() and visible not in {None, "", "-1"} else "cpu"
        )
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise EngineError("CUDA was requested but is unavailable")
    return device


def validate_device_visibility(config: dict[str, Any], device: torch.device) -> None:
    if device.type != "cuda":
        return
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible in {None, "", "-1"}:
        raise EngineError("CUDA use requires an explicit CUDA_VISIBLE_DEVICES selection")
    allowed = set(int(x) for x in config.get("runtime", {}).get("allowed_physical_gpus", []))
    tokens = [token.strip() for token in visible.split(",") if token.strip()]
    if allowed and all(token.isdigit() for token in tokens):
        selected = {int(token) for token in tokens}
        if not selected.issubset(allowed):
            raise EngineError(
                f"CUDA_VISIBLE_DEVICES={visible} is outside configured physical GPU allowlist {sorted(allowed)}"
            )


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _tuplify(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_tuplify(x) for x in value)
    return value


def capture_rng(generator: torch.Generator | None = None) -> dict[str, Any]:
    np_state = np.random.get_state()
    return {
        "python": list(random.getstate()),
        "numpy": {
            "name": np_state[0],
            "keys": torch.from_numpy(np_state[1].copy()),
            "position": int(np_state[2]),
            "has_gauss": int(np_state[3]),
            "cached_gaussian": float(np_state[4]),
        },
        "torch": torch.get_rng_state(),
        "cuda": [x.cpu() for x in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [],
        "data_generator": generator.get_state() if generator is not None else None,
    }


def restore_rng(state: dict[str, Any], generator: torch.Generator | None = None) -> None:
    random.setstate(_tuplify(state["python"]))
    np_state = state["numpy"]
    np.random.set_state(
        (
            np_state["name"],
            np_state["keys"].cpu().numpy().astype(np.uint32),
            int(np_state["position"]),
            int(np_state["has_gauss"]),
            float(np_state["cached_gaussian"]),
        )
    )
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and state.get("cuda"):
        torch.cuda.set_rng_state_all(state["cuda"])
    if generator is not None and state.get("data_generator") is not None:
        generator.set_state(state["data_generator"])


def _git_info(root: str | Path) -> dict[str, Any]:
    root = Path(root)
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"], cwd=root, check=True, capture_output=True, text=True
            ).stdout.strip()
        )
        return {"project_revision": revision, "project_dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        digest = hashlib.sha256()
        for path in sorted((root / "src").rglob("*.py")) if (root / "src").exists() else []:
            digest.update(path.relative_to(root).as_posix().encode())
            digest.update(path.read_bytes())
        return {"project_revision": "uncommitted", "project_dirty": True, "source_sha256": digest.hexdigest()}


def runtime_info(config: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        "cloop_version": __version__,
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
        "visible_cuda_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "configured_allowed_physical_gpus": config.get("runtime", {}).get("allowed_physical_gpus", []),
    }


def _new_run_manifest(config: dict[str, Any], run_name: str, device: torch.device) -> dict[str, Any]:
    manifest = {
        "schema_version": "cloop_run_v1",
        "run_name": run_name,
        "resolved_config": config,
        "config_signature": config_signature(config),
        "runtime": runtime_info(config, device),
        "references": {
            "my_mewm_commit": "b5e3b979e46e6ec254d72aac46fbc352b311bed6",
            "clarity_commit": "dadb82241a24f5ec5e4e4dc994e3116fd4a9da04",
        },
        "stage_states": {
            "prepared": False,
            "dynamics_done": False,
            "outcome_done": False,
            "evaluated": False,
            "protocol_frozen": False,
            "interrupted": False,
        },
        "models": {},
        "warnings": [],
        "test_revealed": False,
        "limitations": [
            "planning cost is an observational prognostic proxy, not a causal treatment effect",
            "historical replay evaluates agreement and behavior, not counterfactual benefit",
            "synthetic closed-loop results do not establish clinical efficacy",
        ],
        **_git_info(config["paths"]["project_root"]),
    }
    return manifest


def _manifest(artifacts: RunArtifacts) -> dict[str, Any]:
    result = read_json(artifacts.path("run.json"))
    if result is None:
        raise EngineError("run is not prepared; run `cloop ... prepare` first")
    return result


def _check_manifest_config(manifest: dict[str, Any], config: dict[str, Any]) -> None:
    if manifest.get("config_signature") != config_signature(config):
        raise EngineError("resolved configuration differs from this run; use a new run name")


def _load_models(artifacts: RunArtifacts, *, required: bool = False) -> dict[str, Any]:
    path = artifacts.path("models.pt")
    if path.exists():
        return read_torch(path, safe=True)
    if required:
        raise EngineError(f"models artifact is required but missing: {path}")
    return {
        "schema_version": "cloop_models_v1",
        "data_signature": None,
        "preprocessing": {},
        "dynamics": {},
        "outcome": {},
        "synthetic_cost": {},
    }


def _load_metrics(artifacts: RunArtifacts) -> dict[str, Any]:
    return read_json(
        artifacts.path("metrics.json"),
        {"training": {}, "dynamics": {}, "reliability": {}, "outcome": {}, "replay": {}, "synthetic": {}},
    )


def doctor(config: dict[str, Any], artifacts: RunArtifacts, run_name: str) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    required = {
        "timeline": "file",
        "timeline_alternative": "file",
        "latent_dir": "directory",
        "latent_provenance": "file",
        "legacy_trajectories": "directory",
        "source_project": "directory",
        "clarity_root": "directory",
        "mri_root": "directory",
        "brainiac_checkpoint": "file",
    }
    for key, kind in required.items():
        path = Path(config["paths"][key])
        exists = path.is_file() if kind == "file" else path.is_dir()
        checks[key] = {
            "path": str(path),
            "exists": exists,
            "readable": exists and os.access(path, os.R_OK),
            "required_for": (
                "main_v1"
                if key in {"timeline", "timeline_alternative", "latent_dir", "latent_provenance"}
                else "legacy_stage1" if key == "legacy_trajectories"
                else "provenance_or_future_raw_encoding"
            ),
        }
    python_path = Path(config.get("runtime", {}).get("python_reference", ""))
    checks["python_reference"] = {
        "path": str(python_path),
        "exists": python_path.is_file(),
        "executable": python_path.is_file() and os.access(python_path, os.X_OK),
    }
    primary, alternative = Path(config["paths"]["timeline"]), Path(config["paths"]["timeline_alternative"])
    if primary.is_file() and alternative.is_file():
        first, second = sha256_file(primary), sha256_file(alternative)
        checks["timeline_hashes"] = {"primary": first, "alternative": second, "equal": first == second}
        if first != second:
            checks["timeline_hashes"]["error"] = "timeline files differ; explicitly select one before prepare"
    source_paths = [
        Path(config["paths"][key]).resolve()
        for key in ("source_project", "clarity_root", "latent_dir", "legacy_trajectories", "mri_root")
        if config["paths"].get(key)
    ]
    destinations = [Path(config["paths"][key]).resolve() for key in ("cache_root", "output_root")]
    collisions = [
        str(dest)
        for dest in destinations
        if any(
            dest == source or dest.is_relative_to(source) or source.is_relative_to(dest)
            for source in source_paths
        )
    ]
    checks["destination_collisions"] = collisions
    checks["destinations"] = {}
    for key, destination in zip(("cache_root", "output_root"), destinations):
        existing_parent = destination if destination.exists() else next(
            (parent for parent in destination.parents if parent.exists()), destination.parent
        )
        checks["destinations"][key] = {
            "path": str(destination),
            "exists": destination.exists(),
            "writable_or_creatable": os.access(existing_parent, os.W_OK),
        }
    protocol = config["data"]["protocol"]
    needed = (
        ["timeline", "timeline_alternative", "latent_dir", "latent_provenance"]
        if protocol == "main_v1"
        else ["legacy_trajectories"]
    )
    ok = (
        all(checks[key]["readable"] for key in needed)
        and all(row["writable_or_creatable"] for row in checks["destinations"].values())
        and not collisions
    )
    if checks.get("timeline_hashes", {}).get("equal") is False and protocol == "main_v1":
        ok = False
    checks["ok"] = ok
    artifacts.root.mkdir(parents=True, exist_ok=True)
    manifest = read_json(artifacts.path("run.json")) or _new_run_manifest(config, run_name, resolve_device(config["project"]["device"]))
    _check_manifest_config(manifest, config)
    manifest["doctor"] = checks
    write_json(artifacts.path("run.json"), manifest)
    return checks


def prepare(config: dict[str, Any], artifacts: RunArtifacts, run_name: str) -> dict[str, Any]:
    device = resolve_device(config["project"]["device"])
    existing = read_json(artifacts.path("run.json"))
    manifest = existing or _new_run_manifest(config, run_name, device)
    _check_manifest_config(manifest, config)
    protocol = config["data"]["protocol"]
    cache_name = "brainiac_main_v2.pt" if protocol == "main_v1" else "brainiac_legacy.pt"
    cache_path = Path(config["paths"]["cache_root"]) / cache_name
    cache, reused = prepare_cache(config, cache_path)
    ids = [row["patient_id"] for row in cache["patients"]]
    split = split_patients(
        ids,
        int(config["data"]["split_seed"]),
        float(config["data"]["train_fraction"]),
        float(config["data"]["validation_fraction"]),
    )
    preprocessing = fit_preprocessing(cache, config, split)
    bundle = load_bundle(cache, config, split, preprocessing)
    window_counts = {
        split_name: {
            f"H{horizon}": len(
                window_refs(bundle, split_name, mode="all_exact", exact_horizon=horizon, max_horizon=horizon)
            )
            for horizon in (1, 2, 3)
        }
        for split_name in ("train", "validation", "test")
    }
    models = _load_models(artifacts)
    if models.get("data_signature") not in {None, cache["data_signature"]}:
        raise EngineError("models.pt belongs to a different data signature")
    models["data_signature"] = cache["data_signature"]
    models["preprocessing"] = preprocessing
    write_torch(artifacts.path("models.pt"), models)
    manifest.update(
        cache_path=str(cache_path),
        cache_reused=reused,
        data_signature=cache["data_signature"],
        data_provenance=cache["source"],
        encoder_provenance=cache["encoder"],
        latent_provenance_hash=cache["encoder"].get("provenance_manifest_sha256"),
        split=split,
        cohort={"patients": len(ids), "split_counts": {k: len(v) for k, v in split.items()}, "windows": window_counts},
        audit=cache.get("audit", {}),
        data_assumptions={
            "genomics_visibility": (
                "structured genomics are treated as baseline-known because per-test dates are not available; "
                "this assumption must be disclosed"
            ),
            "progression_input": "disabled unless a reliable occurrence date is audited",
            "time_quality": (
                "per-timepoint observed/imputed/unknown provenance is preserved and audited; "
                "TP numbering is never used as time"
            ),
        },
        action_vocab=list(bundle.action_codec.vocab),
        action_catalog=[
            {
                "action_id": a.action_id,
                "terms": list(a.display_terms),
                "support_count": a.support_count,
                "known_empty": a.known_empty,
            }
            for a in bundle.action_codec.catalog
        ],
        preprocessing={
            "normalizer_source": "train_patients_only",
            "latent_low_variance_count": bundle.latent_normalizer.low_variance_count,
            "clinical_dim": bundle.clinical_dim,
            "action_dim": bundle.action_dim,
            "history_dim": bundle.history_dim,
            "planned_interval_days": bundle.planned_interval_days,
            "interval_support_days": list(bundle.interval_support_days),
        },
    )
    manifest["stage_states"]["prepared"] = True
    manifest["stage_states"]["interrupted"] = False
    write_json(artifacts.path("run.json"), manifest)
    artifacts.assert_flat()
    return manifest


def _bundle_for_run(config: dict[str, Any], artifacts: RunArtifacts) -> tuple[dict[str, Any], DataBundle, dict[str, Any]]:
    manifest = _manifest(artifacts)
    _check_manifest_config(manifest, config)
    models = _load_models(artifacts, required=True)
    if not models.get("preprocessing"):
        raise EngineError("preprocessing is absent from models.pt")
    cache = read_torch(manifest["cache_path"], safe=True)
    if cache.get("data_signature") != manifest.get("data_signature"):
        raise EngineError("cache signature no longer matches the prepared run")
    bundle = load_bundle(cache, config, manifest["split"], models["preprocessing"])
    return manifest, bundle, models


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if isinstance(value, Tensor) else value for key, value in batch.items()}


def _new_world_member(
    config: dict[str, Any], bundle: DataBundle, device: torch.device
) -> torch.nn.Module:
    if config["data"]["protocol"] == "legacy_stage1":
        return LegacyOneStepDynamics.from_config(
            config, bundle.latent_dim, bundle.action_dim
        ).to(device)
    return OneStepDynamics.from_config(
        config, bundle.latent_dim, bundle.action_dim, bundle.clinical_dim, bundle.history_dim
    ).to(device)


@torch.no_grad()
def _world_validation_score(
    model: EnsembleWorldModel, bundle: DataBundle, split: str, config: dict[str, Any], device: torch.device
) -> tuple[float, dict[str, float | None]]:
    values: dict[int, float | None] = {}
    for horizon in (2, 3):
        refs = window_refs(bundle, split, mode="all_exact", exact_horizon=horizon, max_horizon=horizon)
        if not refs:
            values[horizon] = None
            continue
        loader = DataLoader(
            DynamicsDataset(bundle, refs),
            batch_size=int(config["training"]["batch_size"]),
            shuffle=False,
            collate_fn=collate_dynamics,
        )
        total, count = 0.0, 0
        model.eval()
        for raw in loader:
            batch = _to_device(raw, device)
            rollout = model.rollout(
                batch["z0"], batch["actions"], batch["deltas"], batch["context"],
                batch["clinical_mask"], batch["history0"], batch["step_mask"]
            )
            prediction = terminal_states(rollout, batch["horizons"]).mean(0)
            loss = ((prediction - batch["target"]) ** 2).mean(-1)
            total += float(loss.sum().cpu())
            count += len(loss)
        values[horizon] = total / count
    if values[2] is None or values[3] is None:
        raise EngineError("validation needs both H2 and H3 windows for the registered checkpoint metric")
    return (float(values[2] + values[3]) / 2, {f"mse@{k}": v for k, v in values.items()})


def _checkpoint_payload(
    *,
    task: str,
    key: str,
    member: int,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    best_state: dict[str, Tensor],
    best_epoch: int,
    best_metric: float,
    stale: int,
    history: list[dict[str, Any]],
    generator: torch.Generator,
    config: dict[str, Any],
    data_signature: str,
) -> dict[str, Any]:
    return {
        "schema_version": "cloop_last_v1",
        "task": task,
        "key": key,
        "member": member,
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_state": best_state,
        "best_epoch": best_epoch,
        "best_metric": best_metric,
        "stale": stale,
        "history": history,
        "rng": capture_rng(generator),
        "config_signature": config_signature(config),
        "data_signature": data_signature,
    }


def _train_world_member(
    config: dict[str, Any],
    bundle: DataBundle,
    variant: str,
    training_seed: int,
    member_index: int,
    device: torch.device,
    artifacts: RunArtifacts,
    *,
    resume: bool,
) -> tuple[dict[str, Tensor], int, list[dict[str, Any]]]:
    recursive, _ = variant_spec(variant, int(config["world"]["ensemble_size"]))
    mode = "max_available" if recursive else "one_step"
    refs = window_refs(bundle, "train", mode=mode, max_horizon=int(config["world"]["max_horizon"]))
    if not refs:
        raise EngineError(f"no training windows for {variant}")
    member_seed = training_seed * 1000 + member_index
    seed_all(member_seed)
    generator = torch.Generator().manual_seed(member_seed)
    model = _new_world_member(config, bundle, device)
    optimizer = AdamW(
        model.parameters(), lr=float(config["training"]["lr"]), weight_decay=float(config["training"]["weight_decay"])
    )
    loader = DataLoader(
        DynamicsDataset(bundle, refs), batch_size=int(config["training"]["batch_size"]), shuffle=True,
        generator=generator, num_workers=int(config["training"]["num_workers"]), collate_fn=collate_dynamics,
    )
    key = f"{variant}/{training_seed}"
    start_epoch, best_epoch, best_metric, stale = 1, 0, float("inf"), 0
    best_state: dict[str, Tensor] = {}
    history: list[dict[str, Any]] = []
    last_path = artifacts.path("last.pt")
    if resume and last_path.exists():
        saved = read_torch(last_path, safe=True)
        if (
            saved.get("task") == "dynamics"
            and saved.get("key") == key
            and int(saved.get("member", -1)) == member_index
        ):
            if saved.get("config_signature") != config_signature(config) or saved.get("data_signature") != bundle.data_signature:
                raise EngineError("resume checkpoint signature mismatch")
            model.load_state_dict(saved["model_state"])
            optimizer.load_state_dict(saved["optimizer_state"])
            best_state = saved["best_state"]
            best_epoch = int(saved["best_epoch"])
            best_metric = float(saved["best_metric"])
            stale = int(saved["stale"])
            history = list(saved.get("history", []))
            start_epoch = int(saved["epoch"]) + 1
            restore_rng(saved["rng"], generator)
    max_epochs = int(config["training"]["max_epochs"])
    try:
        for epoch in range(start_epoch, max_epochs + 1):
            epoch_started = time.perf_counter()
            model.train()
            total, count = 0.0, 0
            for raw in loader:
                batch = _to_device(raw, device)
                optimizer.zero_grad(set_to_none=True)
                rollout = EnsembleWorldModel([model]).rollout(
                    batch["z0"], batch["actions"], batch["deltas"], batch["context"],
                    batch["clinical_mask"], batch["history0"], batch["step_mask"]
                )
                loss = terminal_mse(rollout, batch["horizons"], batch["target"])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["training"]["grad_clip"]))
                optimizer.step()
                total += float(loss.detach().cpu()) * len(batch["z0"])
                count += len(batch["z0"])
            score, val_parts = _world_validation_score(
                EnsembleWorldModel([model]), bundle, "validation", config, device
            )
            row = {
                "epoch": epoch,
                "train_loss": total / count,
                "val_long_mse": score,
                "epoch_seconds": time.perf_counter() - epoch_started,
                **val_parts,
            }
            history.append(row)
            print(
                f"dynamics {key} member={member_index} epoch={epoch} "
                f"train={row['train_loss']:.6f} val_long={score:.6f}",
                flush=True,
            )
            if score < best_metric:
                best_metric, best_epoch, stale = score, epoch, 0
                best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            else:
                stale += 1
            write_torch(
                last_path,
                _checkpoint_payload(
                    task="dynamics", key=key, member=member_index, epoch=epoch, model=model,
                    optimizer=optimizer, best_state=best_state, best_epoch=best_epoch,
                    best_metric=best_metric, stale=stale, generator=generator,
                    history=history,
                    config=config, data_signature=bundle.data_signature,
                ),
            )
            if stale >= int(config["training"]["patience"]):
                break
    except BaseException:
        raise
    if not best_state:
        raise EngineError("world training produced no valid best state")
    return best_state, best_epoch, history


def train_dynamics(
    config: dict[str, Any],
    artifacts: RunArtifacts,
    variants: Sequence[str],
    seeds: Sequence[int],
    *,
    resume: bool = False,
    force_task: bool = False,
) -> None:
    manifest, bundle, models = _bundle_for_run(config, artifacts)
    if manifest.get("test_revealed"):
        raise EngineError("test results are already revealed; train changes require a new run")
    device = resolve_device(config["project"]["device"])
    validate_device_visibility(config, device)
    metrics = _load_metrics(artifacts)
    for variant in variants:
        _, member_count = variant_spec(variant, int(config["world"]["ensemble_size"]))
        for seed in seeds:
            key = f"{variant}/{seed}"
            existing = models["dynamics"].get(key)
            if existing and existing.get("complete") and not force_task:
                print(f"skip completed dynamics {key}")
                continue
            entry = existing or {"member_states": [], "best_epochs": [], "complete": False}
            if force_task:
                entry = {"member_states": [], "best_epochs": [], "complete": False}
                metrics["training"].pop(key, None)
                for section in ("dynamics", "reliability"):
                    for split_metrics in metrics.get(section, {}).values():
                        if isinstance(split_metrics, dict):
                            split_metrics.pop(key, None)
                            split_metrics.pop("paired_relative_improvement", None)
            histories = []
            for member in range(len(entry["member_states"]), member_count):
                state, best_epoch, history = _train_world_member(
                    config, bundle, variant, int(seed), member, device, artifacts, resume=resume
                )
                entry["member_states"].append(state)
                entry["best_epochs"].append(best_epoch)
                histories.append({"member": member, "best_epoch": best_epoch, "history": history})
                entry["complete"] = len(entry["member_states"]) == member_count
                entry["config"] = copy.deepcopy(config["world"])
                entry["architecture"] = (
                    "legacy_stage1_exact"
                    if config["data"]["protocol"] == "legacy_stage1"
                    else "cloop_main_v1"
                )
                entry["parameter_count_per_member"] = sum(
                    p.numel() for p in _new_world_member(config, bundle, torch.device("cpu")).parameters()
                )
                models["dynamics"][key] = entry
                write_torch(artifacts.path("models.pt"), models)
            metrics["training"][key] = {
                "members": histories,
                "best_epochs": entry["best_epochs"],
                "member_count": member_count,
            }
            write_json(artifacts.path("metrics.json"), metrics)
            manifest["models"][key] = {
                "complete": entry["complete"], "best_epochs": entry["best_epochs"], "member_count": member_count
            }
            write_json(artifacts.path("run.json"), manifest)
    manifest["stage_states"]["dynamics_done"] = all(
        models["dynamics"].get(f"{variant}/{seed}", {}).get("complete", False)
        for variant in variants for seed in seeds
    )
    manifest["stage_states"]["interrupted"] = False
    write_json(artifacts.path("run.json"), manifest)
    if artifacts.path("last.pt").exists():
        artifacts.path("last.pt").unlink()


def _world_from_entry(
    config: dict[str, Any], bundle: DataBundle, entry: dict[str, Any], device: torch.device
) -> EnsembleWorldModel:
    if not entry.get("complete"):
        raise EngineError("incomplete ensemble cannot be evaluated")
    members = []
    for state in entry["member_states"]:
        member = _new_world_member(config, bundle, device)
        member.load_state_dict(state)
        member.eval()
        members.append(member)
    return EnsembleWorldModel(members).to(device).eval()


@torch.no_grad()
def _evaluate_world(
    model: EnsembleWorldModel,
    bundle: DataBundle,
    split: str,
    variant: str,
    seed: int,
    config: dict[str, Any],
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    reliability: dict[str, Any] = {}
    for horizon in config["evaluation"]["horizons"]:
        refs = window_refs(bundle, split, mode="all_exact", exact_horizon=int(horizon), max_horizon=int(horizon))
        loader = DataLoader(
            DynamicsDataset(bundle, refs), batch_size=int(config["training"]["batch_size"]),
            shuffle=False, collate_fn=collate_dynamics,
        )
        horizon_errors: list[float] = []
        horizon_u: list[float] = []
        for raw in loader:
            batch = _to_device(raw, device)
            rollout = model.rollout(
                batch["z0"], batch["actions"], batch["deltas"], batch["context"],
                batch["clinical_mask"], batch["history0"], batch["step_mask"]
            )
            terminal = terminal_states(rollout, batch["horizons"])
            mean, disagreement = ensemble_mean_and_disagreement(terminal)
            per_sample = ((mean - batch["target"]) ** 2).mean(-1)
            cosine_similarity = torch.nn.functional.cosine_similarity(
                mean, batch["target"], dim=-1
            )
            for i in range(len(per_sample)):
                error = float(per_sample[i].cpu())
                u = float(disagreement[i].cpu())
                horizon_errors.append(error)
                horizon_u.append(u)
                rows.append(
                    {
                        "record_id": (
                            f"dyn:{split}:{variant}:{seed}:{raw['patient_id'][i]}:"
                            f"{raw['start_timepoint'][i]}:{raw['target_timepoint'][i]}:H{horizon}"
                        ),
                        "kind": "dynamics",
                        "split": split,
                        "variant": variant,
                        "seed": seed,
                        "patient_id": raw["patient_id"][i],
                        "start_timepoint": raw["start_timepoint"][i],
                        "target_timepoint": raw["target_timepoint"][i],
                        "horizon": int(horizon),
                        "mse": error,
                        "cosine_similarity": float(cosine_similarity[i].cpu()),
                        "disagreement": u if model.ensemble_size > 1 else None,
                        "conditioning": "factual_actions_and_times",
                    }
                )
        if model.ensemble_size > 1:
            reliability[f"H{horizon}"] = reliability_summary(
                horizon_errors, horizon_u, float(config["evaluation"]["selective_coverage"])
            )
            reliability[f"H{horizon}"]["uncertainty_q90"] = (
                float(np.quantile(horizon_u, config["planner"]["uncertainty_quantile"])) if horizon_u else None
            )
    return rows, dynamics_summary(rows), reliability


def evaluate_dynamics(
    config: dict[str, Any], artifacts: RunArtifacts, split: str, *, variants: Sequence[str] | None = None,
    seeds: Sequence[int] | None = None,
) -> None:
    manifest, bundle, models = _bundle_for_run(config, artifacts)
    if split == "test" and not manifest["stage_states"].get("protocol_frozen"):
        raise EngineError("freeze the protocol before revealing test metrics")
    variants = list(variants or config["training"]["variants"])
    seeds = list(seeds or config["training"]["seeds"])
    device = resolve_device(config["project"]["device"])
    validate_device_visibility(config, device)
    metrics = _load_metrics(artifacts)
    predictions: list[dict[str, Any]] = []
    summary_by_variant: dict[str, dict[str, float]] = defaultdict(dict)
    evaluation_variants = ["persistence", *variants]
    for variant in evaluation_variants:
        eval_seeds = [int(seeds[0])] if variant == "persistence" else [int(x) for x in seeds]
        for seed in eval_seeds:
            if variant == "persistence":
                model = EnsembleWorldModel([PersistenceDynamics()]).to(device)
                key = "persistence"
            else:
                entry = models["dynamics"].get(f"{variant}/{seed}")
                if entry is None:
                    continue
                model = _world_from_entry(config, bundle, entry, device)
                key = f"{variant}/{seed}"
            rows, summary, reliability = _evaluate_world(model, bundle, split, variant, seed, config, device)
            predictions.extend(rows)
            metrics["dynamics"].setdefault(split, {})[key] = summary
            if reliability:
                metrics["reliability"].setdefault(split, {})[key] = reliability
                if split == "validation" and not manifest["stage_states"].get("protocol_frozen"):
                    manifest.setdefault("uncertainty_scales", {})[key] = {
                        h: details.get("uncertainty_q90") for h, details in reliability.items()
                    }
            if summary.get("long_mse") is not None:
                summary_by_variant[variant][str(seed)] = float(summary["long_mse"])
    ri = dynamics_paired_improvements(summary_by_variant)
    metrics["dynamics"].setdefault(split, {})["paired_relative_improvement"] = ri
    if config["artifacts"]["save_predictions_jsonl"]:
        upsert_jsonl(artifacts.path("predictions.jsonl"), predictions)
    write_json(artifacts.path("metrics.json"), metrics)
    if split == "test":
        manifest["test_revealed"] = True
    manifest["stage_states"]["evaluated"] = True
    write_json(artifacts.path("run.json"), manifest)


@torch.no_grad()
def _outcome_validation_nll(
    model: PiecewiseExponentialHead,
    dataset: OutcomeDataset,
    config: dict[str, Any],
    device: torch.device,
) -> float:
    if len(dataset) == 0:
        raise EngineError("validation has no eligible survival label")
    loader = DataLoader(
        dataset, batch_size=int(config["training"]["batch_size"]), shuffle=False, collate_fn=collate_outcome
    )
    total, weight_total = 0.0, 0.0
    model.eval()
    for raw in loader:
        batch = _to_device(raw, device)
        losses = model.nll(
            batch["z"], batch["clinical"], batch["clinical_mask"], batch["history"],
            batch["time"], batch["event"], reduction="none"
        )
        total += float((losses * batch["weight"]).sum().cpu())
        weight_total += float(batch["weight"].sum().cpu())
    return total / weight_total


def _train_outcome_head(
    config: dict[str, Any],
    bundle: DataBundle,
    seed: int,
    clinical_only: bool,
    device: torch.device,
    artifacts: RunArtifacts,
    *,
    resume: bool,
) -> tuple[dict[str, Tensor], int, list[dict[str, Any]]]:
    seed_all(seed)
    generator = torch.Generator().manual_seed(seed)
    train_data = OutcomeDataset(bundle, "train", first_only=False, clinical_only=clinical_only)
    val_data = OutcomeDataset(bundle, "validation", first_only=False, clinical_only=clinical_only)
    if not train_data or not val_data:
        raise EngineError("outcome training requires eligible train and validation survival labels")
    model = PiecewiseExponentialHead.from_config(
        config, bundle.latent_dim, bundle.clinical_dim, bundle.history_dim, clinical_only=clinical_only
    ).to(device)
    optimizer = AdamW(
        model.parameters(), lr=float(config["outcome"]["lr"]), weight_decay=float(config["outcome"]["weight_decay"])
    )
    loader = DataLoader(
        train_data, batch_size=int(config["training"]["batch_size"]), shuffle=True,
        generator=generator, num_workers=int(config["training"]["num_workers"]), collate_fn=collate_outcome,
    )
    label = "clinical_only" if clinical_only else "state"
    key = f"{label}/{seed}"
    start_epoch, best_epoch, best_metric, stale = 1, 0, float("inf"), 0
    best_state: dict[str, Tensor] = {}
    history: list[dict[str, Any]] = []
    last_path = artifacts.path("last.pt")
    if resume and last_path.exists():
        saved = read_torch(last_path, safe=True)
        if saved.get("task") == "outcome" and saved.get("key") == key:
            if saved.get("config_signature") != config_signature(config) or saved.get("data_signature") != bundle.data_signature:
                raise EngineError("outcome resume checkpoint signature mismatch")
            model.load_state_dict(saved["model_state"])
            optimizer.load_state_dict(saved["optimizer_state"])
            best_state = saved["best_state"]
            best_epoch = int(saved["best_epoch"])
            best_metric = float(saved["best_metric"])
            stale = int(saved["stale"])
            history = list(saved.get("history", []))
            start_epoch = int(saved["epoch"]) + 1
            restore_rng(saved["rng"], generator)
    for epoch in range(start_epoch, int(config["outcome"]["max_epochs"]) + 1):
        epoch_started = time.perf_counter()
        model.train()
        total, weight_total = 0.0, 0.0
        for raw in loader:
            batch = _to_device(raw, device)
            optimizer.zero_grad(set_to_none=True)
            loss = model.nll(
                batch["z"], batch["clinical"], batch["clinical_mask"], batch["history"],
                batch["time"], batch["event"], weights=batch["weight"]
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["training"]["grad_clip"]))
            optimizer.step()
            total += float(loss.detach().cpu()) * float(batch["weight"].sum().cpu())
            weight_total += float(batch["weight"].sum().cpu())
        val_nll = _outcome_validation_nll(model, val_data, config, device)
        row = {
            "epoch": epoch,
            "train_survival_nll": total / weight_total,
            "val_survival_nll": val_nll,
            "epoch_seconds": time.perf_counter() - epoch_started,
        }
        history.append(row)
        print(
            f"outcome {key} epoch={epoch} train={row['train_survival_nll']:.6f} val={val_nll:.6f}",
            flush=True,
        )
        if val_nll < best_metric:
            best_metric, best_epoch, stale = val_nll, epoch, 0
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        else:
            stale += 1
        write_torch(
            last_path,
            _checkpoint_payload(
                task="outcome", key=key, member=0, epoch=epoch, model=model, optimizer=optimizer,
                best_state=best_state, best_epoch=best_epoch, best_metric=best_metric, stale=stale,
                history=history,
                generator=generator, config=config, data_signature=bundle.data_signature,
            ),
        )
        if stale >= int(config["outcome"]["patience"]):
            break
    if not best_state:
        raise EngineError("outcome training produced no best state")
    return best_state, best_epoch, history


def train_outcome(
    config: dict[str, Any],
    artifacts: RunArtifacts,
    seeds: Sequence[int],
    *,
    resume: bool = False,
    force_task: bool = False,
) -> None:
    manifest, bundle, models = _bundle_for_run(config, artifacts)
    if manifest.get("test_revealed"):
        raise EngineError("test results are already revealed; train changes require a new run")
    if not config["outcome"]["enabled"]:
        raise EngineError("outcome is disabled in configuration")
    device = resolve_device(config["project"]["device"])
    validate_device_visibility(config, device)
    metrics = _load_metrics(artifacts)
    for seed in seeds:
        for clinical_only in (True, False):
            name = "clinical_only" if clinical_only else "state"
            key = f"{name}/{seed}"
            if key in models["outcome"] and models["outcome"][key].get("complete") and not force_task:
                print(f"skip completed outcome {key}")
                continue
            if force_task:
                metrics["training"].pop(key, None)
                for split_metrics in metrics.get("outcome", {}).values():
                    if isinstance(split_metrics, dict):
                        for metric_key in list(split_metrics):
                            if metric_key in {f"O0/{seed}", f"O1/{seed}"} or f"/{seed}/" in metric_key:
                                split_metrics.pop(metric_key, None)
            state, best_epoch, history = _train_outcome_head(
                config, bundle, int(seed), clinical_only, device, artifacts, resume=resume
            )
            models["outcome"][key] = {
                "state": state,
                "best_epoch": best_epoch,
                "clinical_only": clinical_only,
                "config": copy.deepcopy(config["outcome"]),
                "complete": True,
            }
            metrics["training"][key] = {"best_epoch": best_epoch, "history": history}
            manifest["models"][key] = {"complete": True, "best_epoch": best_epoch}
            write_torch(artifacts.path("models.pt"), models)
            write_json(artifacts.path("metrics.json"), metrics)
            write_json(artifacts.path("run.json"), manifest)
    manifest["stage_states"]["outcome_done"] = all(
        models["outcome"].get(f"{name}/{seed}", {}).get("complete", False)
        for seed in seeds for name in ("clinical_only", "state")
    )
    manifest["stage_states"]["interrupted"] = False
    write_json(artifacts.path("run.json"), manifest)
    if artifacts.path("last.pt").exists():
        artifacts.path("last.pt").unlink()


def _outcome_from_entry(
    config: dict[str, Any], bundle: DataBundle, entry: dict[str, Any], device: torch.device
) -> PiecewiseExponentialHead:
    if not entry.get("complete"):
        raise EngineError("incomplete outcome model")
    model = PiecewiseExponentialHead.from_config(
        config,
        bundle.latent_dim,
        bundle.clinical_dim,
        bundle.history_dim,
        clinical_only=bool(entry["clinical_only"]),
    ).to(device)
    model.load_state_dict(entry["state"])
    return model.eval()


def _survival_training_labels(
    bundle: DataBundle,
    *,
    horizon: int | None = None,
    all_landmarks: bool = False,
    max_days: float = 730.0,
) -> tuple[list[float], list[int]]:
    times: list[float] = []
    events: list[int] = []
    if horizon is None:
        dataset = OutcomeDataset(bundle, "train", first_only=not all_landmarks)
        refs = [(ref.patient_id, ref.index) for ref in dataset.refs]
    else:
        windows = window_refs(bundle, "train", mode="all_exact", exact_horizon=horizon, max_horizon=horizon)
        seen: set[str] = set()
        refs = []
        for ref in windows:
            if ref.patient_id not in seen:
                refs.append((ref.patient_id, ref.target))
                seen.add(ref.patient_id)
    for pid, index in refs:
        tr = bundle.trajectories[pid]
        if bool(tr.survival_valid[index]):
            raw_time = float(tr.survival_time[index])
            times.append(min(raw_time, max_days))
            events.append(int(tr.survival_event[index]) if raw_time <= max_days else 0)
    return times, events


@torch.no_grad()
def _evaluate_observed_outcome(
    model: PiecewiseExponentialHead,
    bundle: DataBundle,
    split: str,
    config: dict[str, Any],
    device: torch.device,
    label: str,
    seed: int,
    *,
    first_only: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    dataset = OutcomeDataset(bundle, split, first_only=first_only, clinical_only=model.clinical_only)
    if not dataset:
        return {"available": False, "reason": "no_eligible_labels", "n": 0}, []
    loader = DataLoader(
        dataset, batch_size=int(config["training"]["batch_size"]), shuffle=False, collate_fn=collate_outcome
    )
    times: list[float] = []
    events: list[int] = []
    probabilities: list[float] = []
    nll: list[float] = []
    records = []
    for raw in loader:
        batch = _to_device(raw, device)
        survival = model.survival(
            batch["z"], batch["clinical"], batch["clinical_mask"], batch["history"],
            float(config["outcome"]["score_horizon_days"]),
        )
        losses = model.nll(
            batch["z"], batch["clinical"], batch["clinical_mask"], batch["history"],
            batch["time"], batch["event"], reduction="none",
        )
        for i in range(len(survival)):
            raw_time = float(batch["time"][i].cpu())
            max_days = float(config["outcome"]["edges_days"][-1])
            time_value = min(raw_time, max_days)
            event_value = int(batch["event"][i].cpu()) if raw_time <= max_days else 0
            probability = float(survival[i].cpu())
            times.append(time_value)
            events.append(event_value)
            probabilities.append(probability)
            nll.append(float(losses[i].cpu()))
            records.append(
                {
                    "record_id": (
                        f"outcome:{split}:{label}:{seed}:"
                        f"{'first' if first_only else 'all'}:{raw['patient_id'][i]}:{raw['timepoint'][i]}"
                    ),
                    "kind": "outcome",
                    "split": split,
                    "outcome_path": label,
                    "seed": seed,
                    "patient_id": raw["patient_id"][i],
                    "timepoint": raw["timepoint"][i],
                    "survival_probability_365": probability,
                    "survival_nll": nll[-1],
                }
            )
    train_times, train_events = _survival_training_labels(
        bundle,
        all_landmarks=not first_only,
        max_days=float(config["outcome"]["edges_days"][-1]),
    )
    result = survival_summary(
        times, events, probabilities, train_times, train_events,
        float(config["outcome"]["score_horizon_days"]),
    )
    result.update(
        available=True,
        survival_nll=float(np.mean(nll)),
        landmark="first_eligible" if first_only else "all_eligible_supplemental",
        patients=len(set(row["patient_id"] for row in records)),
        observations=len(records),
        repeated_observations=max(0, len(records) - len(set(row["patient_id"] for row in records))),
    )
    return result, records


@torch.no_grad()
def _evaluate_predicted_outcome(
    world: EnsembleWorldModel,
    outcome: PiecewiseExponentialHead,
    bundle: DataBundle,
    split: str,
    variant: str,
    seed: int,
    horizon: int,
    config: dict[str, Any],
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    eligible_refs = [
        ref
        for ref in window_refs(bundle, split, mode="all_exact", exact_horizon=horizon, max_horizon=horizon)
        if bool(bundle.trajectories[ref.patient_id].survival_valid[ref.target])
    ]
    seen_patients: set[str] = set()
    refs = []
    for ref in eligible_refs:
        if ref.patient_id not in seen_patients:
            refs.append(ref)
            seen_patients.add(ref.patient_id)
    if not refs:
        return {"available": False, "reason": "no_eligible_target_landmarks", "n": 0}, []
    loader = DataLoader(
        DynamicsDataset(bundle, refs), batch_size=int(config["training"]["batch_size"]),
        shuffle=False, collate_fn=collate_dynamics,
    )
    times: list[float] = []
    events: list[int] = []
    probabilities: list[float] = []
    nlls: list[float] = []
    records = []
    for raw in loader:
        batch = _to_device(raw, device)
        rollout = world.rollout(
            batch["z0"], batch["actions"], batch["deltas"], batch["context"],
            batch["clinical_mask"], batch["history0"], batch["step_mask"]
        )
        members = terminal_states(rollout, batch["horizons"])
        for i, (pid, target_index) in enumerate(zip(raw["patient_id"], raw["target_index"].tolist())):
            tr = bundle.trajectories[pid]
            member_z = members[:, i]
            count = member_z.shape[0]
            clinical = tr.clinical.to(device).expand(count, -1)
            clinical_mask = tr.clinical_mask.to(device).expand(count, -1)
            history = tr.histories[target_index].to(device).expand(count, -1)
            survival = outcome.survival(
                member_z, clinical, clinical_mask, history, float(config["outcome"]["score_horizon_days"])
            ).mean()
            label_time = tr.survival_time[target_index].to(device).expand(count)
            label_event = tr.survival_event[target_index].to(device).expand(count)
            member_nll = outcome.nll(
                member_z, clinical, clinical_mask, history, label_time, label_event, reduction="none"
            ).mean()
            raw_time = float(tr.survival_time[target_index])
            max_days = float(config["outcome"]["edges_days"][-1])
            times.append(min(raw_time, max_days))
            events.append(int(tr.survival_event[target_index]) if raw_time <= max_days else 0)
            probabilities.append(float(survival.cpu()))
            nlls.append(float(member_nll.cpu()))
            records.append(
                {
                    "record_id": (
                        f"outcome:{split}:O2-{variant}:{seed}:{pid}:"
                        f"{raw['target_timepoint'][i]}:H{horizon}"
                    ),
                    "kind": "outcome",
                    "split": split,
                    "outcome_path": f"O2-{variant}",
                    "seed": seed,
                    "patient_id": pid,
                    "timepoint": raw["target_timepoint"][i],
                    "horizon": horizon,
                    "survival_probability_365": probabilities[-1],
                    "survival_nll": nlls[-1],
                }
            )
    train_times, train_events = _survival_training_labels(
        bundle, horizon=horizon, max_days=float(config["outcome"]["edges_days"][-1])
    )
    summary = survival_summary(
        times, events, probabilities, train_times, train_events,
        float(config["outcome"]["score_horizon_days"]),
    )
    summary.update(
        available=True,
        survival_nll=float(np.mean(nlls)),
        horizon=horizon,
        patients=len(set(row["patient_id"] for row in records)),
        observations=len(records),
        landmark="first_eligible_target_per_patient",
    )
    return summary, records


def evaluate_outcome(
    config: dict[str, Any], artifacts: RunArtifacts, split: str, *, seeds: Sequence[int] | None = None
) -> None:
    manifest, bundle, models = _bundle_for_run(config, artifacts)
    if split == "test" and not manifest["stage_states"].get("protocol_frozen"):
        raise EngineError("freeze the protocol before revealing test metrics")
    seeds = list(seeds or config["training"]["seeds"])
    device = resolve_device(config["project"]["device"])
    validate_device_visibility(config, device)
    metrics = _load_metrics(artifacts)
    records: list[dict[str, Any]] = []
    for seed in seeds:
        outcome_models: dict[str, PiecewiseExponentialHead] = {}
        for label, stored in (("O0", "clinical_only"), ("O1", "state")):
            entry = models["outcome"].get(f"{stored}/{seed}")
            if entry is None:
                continue
            model = _outcome_from_entry(config, bundle, entry, device)
            outcome_models[label] = model
            summary, rows = _evaluate_observed_outcome(
                model, bundle, split, config, device, label, int(seed), first_only=True
            )
            supplemental, supplemental_rows = _evaluate_observed_outcome(
                model, bundle, split, config, device, label, int(seed), first_only=False
            )
            metrics["outcome"].setdefault(split, {})[f"{label}/{seed}"] = {
                "primary": summary,
                "all_landmarks_supplemental": supplemental,
            }
            records.extend(rows)
            records.extend(supplemental_rows)
        if "O1" not in outcome_models:
            continue
        for variant in config["training"]["variants"]:
            entry = models["dynamics"].get(f"{variant}/{seed}")
            if entry is None:
                continue
            world = _world_from_entry(config, bundle, entry, device)
            for horizon in config["evaluation"]["horizons"]:
                summary, rows = _evaluate_predicted_outcome(
                    world, outcome_models["O1"], bundle, split, variant, int(seed), int(horizon), config, device
                )
                metrics["outcome"].setdefault(split, {})[
                    f"O2-{variant}/{seed}/H{horizon}"
                ] = summary
                records.extend(rows)
    if config["artifacts"]["save_predictions_jsonl"]:
        upsert_jsonl(artifacts.path("predictions.jsonl"), records)
    write_json(artifacts.path("metrics.json"), metrics)
    if split == "test":
        manifest["test_revealed"] = True
    manifest["stage_states"]["evaluated"] = True
    write_json(artifacts.path("run.json"), manifest)


def _planner_for_method(
    method: str,
    seed: int,
    config: dict[str, Any],
    manifest: dict[str, Any],
    bundle: DataBundle,
    models: dict[str, Any],
    device: torch.device,
) -> Planner:
    if config["policy"]["kind"] != "catalog":
        raise EngineError(
            "the formal v1 run has no configured online provider adapter; use catalog policy or inject a tested provider"
        )
    policy = CatalogPolicy(bundle.action_codec, int(config["policy"]["max_candidates"]))
    common = {
        "policy": policy,
        "action_codec": bundle.action_codec,
        "planned_interval_days": bundle.planned_interval_days,
        "interval_support_days": bundle.interval_support_days,
        "horizon": int(config["planner"]["horizon"]),
        "beam_width": int(config["planner"]["beam_width"]),
        "uncertainty_clip": float(config["planner"]["uncertainty_clip"]),
        "discount_scale_days": float(config["planner"]["discount_scale_days"]),
        "max_candidates": int(config["policy"]["max_candidates"]),
        "min_action_support": int(config["data"]["min_action_support"]),
        "seed": seed,
    }
    if method == "frequency":
        return Planner(None, None, mode="frequency", lambda_uncertainty=0.0, **common)
    mapping = {
        "fixed_plan": ("rrt_ensemble", "mpc", 0.0),
        "greedy": ("rrt_ensemble", "greedy", 0.0),
        "mpc_ensemble": ("ensemble", "mpc", 0.0),
        "mpc_rrt_ensemble": ("rrt_ensemble", "mpc", 0.0),
        "mpc_rrt_ensemble_unc": (
            "rrt_ensemble", "mpc", float(config["planner"]["lambda_uncertainty"])
        ),
    }
    if method not in mapping:
        raise EngineError(f"unknown planning method: {method}")
    variant, mode, lambda_u = mapping[method]
    dynamics_entry = models["dynamics"].get(f"{variant}/{seed}")
    outcome_entry = models["outcome"].get(f"state/{seed}")
    if dynamics_entry is None or outcome_entry is None:
        raise EngineError(f"{method} requires complete {variant}/{seed} dynamics and state/{seed} outcome")
    world = _world_from_entry(config, bundle, dynamics_entry, device)
    outcome = _outcome_from_entry(config, bundle, outcome_entry, device)
    scales_raw = manifest.get("uncertainty_scales", {}).get(f"{variant}/{seed}", {})
    scales = {int(str(key).lstrip("H")): float(value) for key, value in scales_raw.items() if value is not None}
    return Planner(
        world,
        outcome,
        mode=mode,
        lambda_uncertainty=lambda_u,
        uncertainty_scales=scales,
        **common,
    )


def _decision_record(
    *,
    split: str,
    method: str,
    seed: int,
    state: PatientState,
    actual: Action | None,
    decision: Any,
    catalog_ids: set[str],
    candidate_ids: set[str],
) -> dict[str, Any]:
    recommended = decision.recommended_action
    match = action_set_metrics(
        actual.display_terms if actual is not None else (),
        recommended.display_terms if recommended is not None else (),
    )
    return {
        "record_id": f"replay:{split}:{method}:{seed}:{state.patient_key}:{state.timepoint_key}",
        "kind": "decision",
        "split": split,
        "planner": method,
        "seed": seed,
        "patient_id": state.patient_key,
        "timepoint": state.timepoint_key,
        "state_version": state.state_version,
        "recommended_action": recommended.action_id if recommended is not None else None,
        "recommended_terms": list(recommended.display_terms) if recommended is not None else [],
        "actual_action": actual.action_id if actual is not None else None,
        "actual_terms": list(actual.display_terms) if actual is not None else [],
        "status": decision.status,
        "reason_codes": list(decision.reason_codes),
        "imagined_plan": [action.action_id for action in decision.imagined_plan],
        "catalog_covers_actual": actual is not None and actual.action_id in catalog_ids,
        "candidate_contains_actual": actual is not None and actual.action_id in candidate_ids,
        "metrics": match,
        "diagnostics": decision.diagnostics,
        "score_components": decision.score_components,
        "evidence_level": "historical_action_agreement_only",
    }


def _replay_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0, "reason": "no_evaluable_transitions"}
    recommended = [row for row in rows if row["status"] == "recommend"]
    unconditional = {
        key: float(np.mean([row["metrics"][key] if row["status"] == "recommend" else 0.0 for row in rows]))
        for key in ("precision", "recall", "f1", "jaccard")
    }
    conditional = {
        key: float(np.mean([row["metrics"][key] for row in recommended])) if recommended else None
        for key in ("precision", "recall", "f1", "jaccard")
    }
    changed = 0
    comparisons = 0
    previous_by_patient: dict[str, str | None] = {}
    for row in rows:
        current = row["recommended_action"]
        pid = row["patient_id"]
        if pid in previous_by_patient:
            comparisons += 1
            changed += current != previous_by_patient[pid]
        previous_by_patient[pid] = current
    diagnostics = [row.get("diagnostics", {}) for row in rows]
    return {
        "n": len(rows),
        "recommendations": len(recommended),
        "recommendation_coverage": len(recommended) / len(rows),
        "abstain_rate": 1.0 - len(recommended) / len(rows),
        "catalog_coverage": float(np.mean([row["catalog_covers_actual"] for row in rows])),
        "candidate_recall": float(np.mean([row["candidate_contains_actual"] for row in rows])),
        "both_empty_fraction": float(np.mean([row["metrics"]["both_empty"] for row in rows])),
        "conditional_on_recommendation": conditional,
        "unconditional": unconditional,
        "replanning_action_change_rate": changed / comparisons if comparisons else None,
        "structural_rule_violation_rate": None,
        "structural_rule_violation_reason": "clinical_rules_disabled",
        "wall_time_ms_mean": float(np.mean([x.get("wall_time_ms", 0.0) for x in diagnostics])),
        "world_model_forwards": int(sum(x.get("world_model_forwards", 0) for x in diagnostics)),
        "beam_nodes": int(sum(x.get("beam_nodes", 0) for x in diagnostics)),
        "policy_calls": int(sum(x.get("policy_calls", 0) for x in diagnostics)),
        "interpretation": "agreement with recorded treatment; not causal treatment quality",
    }


def evaluate_replay(
    config: dict[str, Any], artifacts: RunArtifacts, split: str, *, seeds: Sequence[int] | None = None
) -> None:
    manifest, bundle, models = _bundle_for_run(config, artifacts)
    if split == "test" and not manifest["stage_states"].get("protocol_frozen"):
        raise EngineError("freeze the protocol before revealing test replay")
    seeds = list(seeds or config["training"]["seeds"])
    device = resolve_device(config["project"]["device"])
    validate_device_visibility(config, device)
    metrics = _load_metrics(artifacts)
    all_records: list[dict[str, Any]] = []
    catalog_ids = {action.action_id for action in bundle.action_codec.catalog}
    candidate_ids = {
        action.action_id
        for action in bundle.action_codec.catalog[: int(config["policy"]["max_candidates"])]
    }
    for seed in seeds:
        for method in config["evaluation"]["planning_methods"]:
            planner = _planner_for_method(method, int(seed), config, manifest, bundle, models, device)
            rows: list[dict[str, Any]] = []
            if method == "fixed_plan":
                for pid in bundle.split_ids[split]:
                    tr = bundle.trajectories[pid]
                    index = 0
                    version = 0
                    while index < tr.length - 1:
                        if not bool(tr.action_known[index]):
                            index += 1
                            version += 1
                            continue
                        start_state = patient_state(bundle, pid, index, version)
                        decision = planner.plan(start_state)
                        planned = decision.imagined_plan
                        block = min(int(config["planner"]["horizon"]), tr.length - 1 - index)
                        for offset in range(block):
                            actual_index = index + offset
                            if not bool(tr.action_known[actual_index]):
                                continue
                            observed = patient_state(bundle, pid, actual_index, version + offset)
                            if decision.status == "recommend" and offset < len(planned):
                                frozen_action = planned[offset]
                                frozen = copy.copy(decision)
                                object.__setattr__(frozen, "recommended_action", frozen_action)
                                object.__setattr__(frozen, "reason_codes", ("fixed_initial_plan",))
                                current_decision = frozen
                            else:
                                current_decision = decision
                            rows.append(
                                _decision_record(
                                    split=split, method=method, seed=int(seed), state=observed,
                                    actual=tr.action_objects[actual_index], decision=current_decision,
                                    catalog_ids=catalog_ids, candidate_ids=candidate_ids,
                                )
                            )
                        index += block
                        version += block
            else:
                for state, target in iter_observed_replay(bundle, split):
                    decision = planner.plan(state)
                    rows.append(
                        _decision_record(
                            split=split, method=method, seed=int(seed), state=state,
                            actual=target.actual_action, decision=decision,
                            catalog_ids=catalog_ids, candidate_ids=candidate_ids,
                        )
                    )
            metrics["replay"].setdefault(split, {})[f"{method}/{seed}"] = _replay_summary(rows)
            all_records.extend(rows)
    if config["artifacts"]["save_predictions_jsonl"]:
        upsert_jsonl(artifacts.path("predictions.jsonl"), all_records)
    write_json(artifacts.path("metrics.json"), metrics)
    if split == "test":
        manifest["test_revealed"] = True
    manifest["stage_states"]["evaluated"] = True
    write_json(artifacts.path("run.json"), manifest)


def _toy_cache_and_split(config: dict[str, Any], seed: int) -> tuple[dict[str, Any], dict[str, list[str]]]:
    counts = {
        "train": int(config["evaluation"]["synthetic_train_episodes"]),
        "validation": int(config["evaluation"]["synthetic_val_episodes"]),
        "test": int(config["evaluation"]["synthetic_test_episodes"]),
    }
    steps = int(config["evaluation"]["synthetic_episode_steps"])
    rng = np.random.default_rng(seed + 9000)
    patients = []
    split: dict[str, list[str]] = {name: [] for name in counts}
    offsets = {"train": 100, "validation": 200, "test": 300}
    for split_name, count in counts.items():
        episodes = sample_episodes(count, steps, seed + offsets[split_name])
        for episode_index, (initial, noise) in enumerate(episodes):
            pid = f"TOY_{split_name[:2].upper()}_{episode_index:05d}"
            split[split_name].append(pid)
            env = ToyEnv(initial, noise)
            states = [initial]
            terms = []
            for _ in range(steps):
                action_index = int(rng.integers(0, 3))
                terms.append([f"category:a{action_index}"])
                next_state, _, _, _ = env.step(action_index)
                states.append(next_state)
            patients.append(
                {
                    "patient_id": pid,
                    "timepoint_ids": [f"T{i}" for i in range(steps + 1)],
                    "mri_days": torch.arange(steps + 1, dtype=torch.float32) * 30.0,
                    "latents_raw": torch.stack([encode_state(state) for state in states]),
                    "clinical_raw": {
                        "age_at_diagnosis_years": 0.0,
                        "sex_at_birth": "synthetic",
                        "who_grade": str(initial.subtype),
                        "genomics": {},
                    },
                    "events": [],
                    "action_terms": terms,
                    "action_known": torch.ones(steps, dtype=torch.bool),
                    "labels": {
                        "survival_time": torch.full((steps + 1,), float("nan")),
                        "event": torch.full((steps + 1,), -1, dtype=torch.long),
                        "valid": torch.zeros(steps + 1, dtype=torch.bool),
                        "censoring_rule": ["not_clinical"] * (steps + 1),
                    },
                    "quality_flags": {"time_quality": "verified", "environment": "toy_v1"},
                }
            )
    cache = {
        "schema_version": "cloop_data_v1",
        "source": {"kind": "independent_toy_environment", "seed": seed},
        "encoder": {"name": "fixed_synthetic_map", "frozen": True, "latent_dim": 8},
        "protocol": "main_v1",
        "patients": patients,
        "audit": {"excluded_counts": {}, "warnings": ["synthetic actions have no medical semantics"]},
    }
    cache["data_signature"] = cache_signature(cache)
    return cache, split


def _toy_cost_targets(cache: dict[str, Any], bundle: DataBundle, split: str) -> tuple[Tensor, ...]:
    by_id = {row["patient_id"]: row for row in cache["patients"]}
    z_values, clinical, masks, histories, targets = [], [], [], [], []
    for pid in bundle.split_ids[split]:
        raw = by_id[pid]
        tr = bundle.trajectories[pid]
        for index, encoded in enumerate(raw["latents_raw"]):
            z_values.append(tr.latents[index])
            clinical.append(tr.clinical)
            masks.append(tr.clinical_mask)
            histories.append(tr.histories[index])
            burden_fraction, toxicity_fraction = float(encoded[0]), float(encoded[1])
            targets.append(burden_fraction + 0.4 * toxicity_fraction + float(toxicity_fraction == 1.0))
    return (
        torch.stack(z_values), torch.stack(clinical), torch.stack(masks), torch.stack(histories), torch.tensor(targets)
    )


def _train_toy_cost(
    config: dict[str, Any], cache: dict[str, Any], bundle: DataBundle, seed: int, device: torch.device
) -> tuple[dict[str, Tensor], list[dict[str, float]]]:
    seed_all(seed + 40000)
    model = SyntheticCostHead(bundle.latent_dim, bundle.clinical_dim, bundle.history_dim).to(device)
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    train = tuple(x.to(device) for x in _toy_cost_targets(cache, bundle, "train"))
    validation = tuple(x.to(device) for x in _toy_cost_targets(cache, bundle, "validation"))
    best_metric, best_state, stale = float("inf"), {}, 0
    history = []
    for epoch in range(1, int(config["outcome"]["max_epochs"]) + 1):
        epoch_started = time.perf_counter()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        prediction = model(*train[:4])
        loss = torch.nn.functional.mse_loss(prediction, train[4])
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val = float(torch.nn.functional.mse_loss(model(*validation[:4]), validation[4]).cpu())
        history.append(
            {
                "epoch": epoch,
                "train_mse": float(loss.detach().cpu()),
                "val_mse": val,
                "epoch_seconds": time.perf_counter() - epoch_started,
            }
        )
        if val < best_metric:
            best_metric, stale = val, 0
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        else:
            stale += 1
        if stale >= int(config["outcome"]["patience"]):
            break
    return best_state, history


def _toy_cost_from_state(
    state: dict[str, Tensor], bundle: DataBundle, device: torch.device
) -> SyntheticCostHead:
    model = SyntheticCostHead(bundle.latent_dim, bundle.clinical_dim, bundle.history_dim).to(device)
    model.load_state_dict(state)
    return model.eval()


def _toy_action_index(action: Action | None) -> int:
    if action is None:
        return 0
    for term in action.display_terms:
        if term.startswith("category:a") and term[-1] in "012":
            return int(term[-1])
    return 0


def _toy_planner(
    method: str,
    seed: int,
    config: dict[str, Any],
    bundle: DataBundle,
    models: dict[str, Any],
    cost_head: SyntheticCostHead,
    scales: dict[str, dict[int, float]],
    device: torch.device,
) -> Planner:
    policy = CatalogPolicy(bundle.action_codec, 3)
    common = dict(
        policy=policy,
        action_codec=bundle.action_codec,
        planned_interval_days=30.0,
        interval_support_days=bundle.interval_support_days,
        horizon=int(config["planner"]["horizon"]),
        beam_width=int(config["planner"]["beam_width"]),
        uncertainty_clip=float(config["planner"]["uncertainty_clip"]),
        discount_scale_days=float(config["planner"]["discount_scale_days"]),
        max_candidates=3,
        min_action_support=1,
        seed=seed,
    )
    if method == "frequency":
        return Planner(None, None, mode="frequency", **common)
    mapping = {
        "fixed_plan": ("rrt_ensemble", "mpc", 0.0),
        "greedy": ("rrt_ensemble", "greedy", 0.0),
        "mpc_ensemble": ("ensemble", "mpc", 0.0),
        "mpc_rrt_ensemble": ("rrt_ensemble", "mpc", 0.0),
        "mpc_rrt_ensemble_unc": (
            "rrt_ensemble", "mpc", float(config["planner"]["lambda_uncertainty"])
        ),
    }
    variant, mode, lambda_u = mapping[method]
    world = _world_from_entry(config, bundle, models["dynamics"][f"{variant}/{seed}"], device)
    return Planner(
        world, cost_head, mode=mode, lambda_uncertainty=lambda_u,
        uncertainty_scales=scales.get(variant, {}), **common
    )


def _evaluate_toy_method(
    method: str,
    seed: int,
    planner: Planner,
    bundle: DataBundle,
    episodes: Sequence[tuple[ToyState, tuple[int, ...]]],
    config: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = []
    costs, gaps, violations, abstains, budgets, recovery = [], [], 0, 0, [], []
    for episode_index, (initial, noise) in enumerate(episodes):
        env = ToyEnv(initial, noise)
        history = empty_history(bundle.action_dim)
        clinical, mask = bundle.clinical_codec.encode(
            {"age_at_diagnosis_years": 0.0, "sex_at_birth": "synthetic", "who_grade": str(initial.subtype), "genomics": {}}
        )
        total = 0.0
        after_perturbation = 0.0
        recovery_steps = 0
        fixed_plan: tuple[Action, ...] = ()
        fixed_offset = 0
        forwards = 0
        for step in range(len(noise)):
            observed = env.observe()
            z = bundle.latent_normalizer.transform(encode_state(observed))
            state = PatientState(
                patient_key=f"TOY_TEST_{episode_index}", timepoint_key=f"T{step}", observed_day=30.0 * step,
                z=z, clinical=clinical, clinical_mask=mask, history=history,
                source="observed", state_version=step,
            )
            if method == "fixed_plan" and (not fixed_plan or fixed_offset >= len(fixed_plan)):
                decision = planner.plan(state)
                fixed_plan = decision.imagined_plan
                fixed_offset = 0
            elif method == "fixed_plan":
                decision = copy.copy(last_decision)
                if fixed_offset < len(fixed_plan):
                    object.__setattr__(decision, "recommended_action", fixed_plan[fixed_offset])
                    object.__setattr__(decision, "reason_codes", ("fixed_initial_plan",))
            else:
                decision = planner.plan(state)
            last_decision = decision
            action_index = _toy_action_index(decision.recommended_action)
            if decision.status == "abstain":
                abstains += 1
            next_state, cost, _, info = env.step(action_index)
            total += cost
            if info["noise"] != 0:
                recovery_steps = 2
            if recovery_steps > 0:
                after_perturbation += cost
                recovery_steps -= 1
            action = decision.recommended_action
            vector = torch.zeros(bundle.action_dim)
            if action is not None and action.token_ids:
                vector[list(action.token_ids)] = 1.0
            history = update_history(history, vector, 30.0)
            forwards += int(decision.diagnostics.get("world_model_forwards", 0))
            if method == "fixed_plan":
                fixed_offset += 1
        oracle = optimal_expected_value(initial, len(noise))
        costs.append(total)
        gaps.append(total - oracle)
        recovery.append(after_perturbation)
        budgets.append(forwards)
        rows.append(
            {
                "record_id": f"synthetic:{method}:{seed}:episode:{episode_index}",
                "kind": "synthetic_episode",
                "planner": method,
                "seed": seed,
                "episode": episode_index,
                "cumulative_env_cost": total,
                "oracle_expected_value": oracle,
                "value_gap": total - oracle,
                "world_model_forwards": forwards,
            }
        )
    decisions = len(episodes) * len(episodes[0][1]) if episodes else 0
    return {
        "episodes": len(episodes),
        "mean_environment_cost": float(np.mean(costs)) if costs else None,
        "mean_expected_optimality_gap": float(np.mean(gaps)) if gaps else None,
        "constraint_violation_rate": violations / decisions if decisions else None,
        "abstain_rate": abstains / decisions if decisions else None,
        "mean_perturbation_recovery_cost": float(np.mean(recovery)) if recovery else None,
        "mean_world_model_forwards": float(np.mean(budgets)) if budgets else None,
        "interpretation": "independent non-medical synthetic environment only",
    }, rows


def run_synthetic_suite(
    config: dict[str, Any], artifacts: RunArtifacts, run_name: str, seeds: Sequence[int]
) -> None:
    device = resolve_device(config["project"]["device"])
    validate_device_visibility(config, device)
    data_seed = int(config["project"]["seed"])
    cache, split = _toy_cache_and_split(config, data_seed)
    preprocessing = fit_preprocessing(cache, config, split)
    # Synthetic planning admits all observed toy actions even when a tiny smoke cohort is used.
    preprocessing["action_codec"]["min_support"] = 1
    for row in preprocessing["action_codec"]["catalog"]:
        row["support_count"] = max(1, int(row["support_count"]))
    bundle = load_bundle(cache, config, split, preprocessing)
    artifacts.root.mkdir(parents=True, exist_ok=True)
    manifest = read_json(artifacts.path("run.json")) or _new_run_manifest(config, run_name, device)
    _check_manifest_config(manifest, config)
    if manifest.get("data_signature") not in {None, cache["data_signature"]}:
        raise EngineError("synthetic run already exists with a different generated-data signature")
    manifest.update(
        data_signature=cache["data_signature"],
        data_provenance=cache["source"],
        split=split,
        cohort={"split_counts": {key: len(value) for key, value in split.items()}},
        preprocessing={
            "normalizer_source": "synthetic_train_episodes_only",
            "planned_interval_days": 30.0,
            "action_dim": bundle.action_dim,
        },
    )
    manifest["stage_states"]["prepared"] = True
    models = _load_models(artifacts)
    models["data_signature"] = cache["data_signature"]
    models["preprocessing"] = preprocessing
    metrics = _load_metrics(artifacts)
    predictions: list[dict[str, Any]] = []
    for seed in seeds:
        for variant in ("ensemble", "rrt_ensemble"):
            _, member_count = variant_spec(variant, int(config["world"]["ensemble_size"]))
            key = f"{variant}/{seed}"
            entry = models["dynamics"].get(key) or {
                "member_states": [], "best_epochs": [], "complete": False
            }
            member_histories = []
            for member in range(len(entry["member_states"]), member_count):
                state, best_epoch, history = _train_world_member(
                    config, bundle, variant, int(seed), member, device, artifacts, resume=True
                )
                entry["member_states"].append(state)
                entry["best_epochs"].append(best_epoch)
                member_histories.append({"member": member, "best_epoch": best_epoch, "history": history})
                entry["complete"] = len(entry["member_states"]) == member_count
                entry["config"] = copy.deepcopy(config["world"])
                models["dynamics"][key] = entry
                write_torch(artifacts.path("models.pt"), models)
            metrics["training"][key] = {
                "members": member_histories, "best_epochs": entry["best_epochs"], "member_count": member_count
            }
        cost_key = f"toy_state_cost/{seed}"
        if cost_key not in models["synthetic_cost"]:
            cost_state, cost_history = _train_toy_cost(config, cache, bundle, int(seed), device)
            models["synthetic_cost"][cost_key] = {"state": cost_state, "complete": True}
            metrics["training"][cost_key] = {"history": cost_history}
            write_torch(artifacts.path("models.pt"), models)
        scales: dict[str, dict[int, float]] = {}
        for variant in ("ensemble", "rrt_ensemble"):
            world = _world_from_entry(config, bundle, models["dynamics"][f"{variant}/{seed}"], device)
            _, _, reliability = _evaluate_world(
                world, bundle, "validation", variant, int(seed), config, device
            )
            scales[variant] = {
                int(key.lstrip("H")): float(value["uncertainty_q90"])
                for key, value in reliability.items()
                if value.get("uncertainty_q90") is not None
            }
        cost_head = _toy_cost_from_state(models["synthetic_cost"][cost_key]["state"], bundle, device)
        episodes = sample_episodes(
            int(config["evaluation"]["synthetic_test_episodes"]),
            int(config["evaluation"]["synthetic_episode_steps"]),
            data_seed + 300,
        )
        for method in config["evaluation"]["planning_methods"]:
            planner = _toy_planner(method, int(seed), config, bundle, models, cost_head, scales, device)
            summary, rows = _evaluate_toy_method(method, int(seed), planner, bundle, episodes, config)
            metrics["synthetic"][f"{method}/{seed}"] = summary
            predictions.extend(rows)
    manifest["stage_states"].update(
        dynamics_done=True, outcome_done=True, evaluated=True, interrupted=False
    )
    manifest["synthetic_evidence_boundary"] = (
        "Actions A0/A1/A2 and results are non-medical; they validate feedback mechanics only."
    )
    write_torch(artifacts.path("models.pt"), models)
    write_json(artifacts.path("metrics.json"), metrics)
    if config["artifacts"]["save_predictions_jsonl"]:
        upsert_jsonl(artifacts.path("predictions.jsonl"), predictions)
    write_json(artifacts.path("run.json"), manifest)
    if artifacts.path("last.pt").exists():
        artifacts.path("last.pt").unlink()
    artifacts.assert_flat()


def freeze_protocol(config: dict[str, Any], artifacts: RunArtifacts) -> None:
    manifest = _manifest(artifacts)
    _check_manifest_config(manifest, config)
    if manifest.get("test_revealed"):
        raise EngineError("test has already been revealed; this run can no longer be newly frozen")
    if not manifest.get("uncertainty_scales"):
        raise EngineError("validation uncertainty scales are absent; evaluate dynamics validation first")
    manifest["stage_states"]["protocol_frozen"] = True
    manifest["frozen_protocol_signature"] = hashlib.sha256(
        json.dumps(
            {
                "config_signature": manifest["config_signature"],
                "data_signature": manifest["data_signature"],
                "uncertainty_scales": manifest["uncertainty_scales"],
                "models": manifest["models"],
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    write_json(artifacts.path("run.json"), manifest)


def _md_value(value: Any) -> str:
    if value is None:
        return "unavailable"
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (dict, list)):
        return f"`{json.dumps(value, ensure_ascii=False, sort_keys=True)}`"
    return str(value)


def generate_report(artifacts: RunArtifacts) -> Path:
    manifest = _manifest(artifacts)
    metrics = _load_metrics(artifacts)
    lines = [
        f"# CLARITY Loop report: {manifest['run_name']}",
        "",
        "This report separates factual forecasting, historical observed replay, and independent synthetic closed-loop evidence.",
        "",
        "## Run and data audit",
        "",
        f"- Protocol: `{manifest['resolved_config']['data']['protocol']}`",
        f"- Data signature: `{manifest.get('data_signature', 'unavailable')}`",
        f"- Project revision: `{manifest.get('project_revision', 'unavailable')}`",
        f"- Test revealed: `{manifest.get('test_revealed', False)}`",
        f"- Protocol frozen: `{manifest.get('stage_states', {}).get('protocol_frozen', False)}`",
        f"- Cohort: {_md_value(manifest.get('cohort'))}",
        f"- Audit: {_md_value(manifest.get('audit', {}))}",
        f"- Data assumptions: {_md_value(manifest.get('data_assumptions', {}))}",
        "",
        "## Completed experiment groups",
        "",
    ]
    for key, value in manifest.get("stage_states", {}).items():
        lines.append(f"- {key}: `{value}`")
    section_names = (
        ("Dynamics — factual actions and times", "dynamics"),
        ("Ensemble reliability", "reliability"),
        ("Outcome — observational landmark prediction", "outcome"),
        ("Observed replay — treatment agreement only", "replay"),
        ("Independent synthetic environment", "synthetic"),
    )
    for title, key in section_names:
        lines.extend(["", f"## {title}", ""])
        section = metrics.get(key, {})
        if not section:
            lines.append("Not completed.")
            continue
        lines.append("```json")
        lines.append(json.dumps(section, ensure_ascii=False, indent=2, sort_keys=True))
        lines.append("```")
    lines.extend(["", "## Resource and artifact summary", ""])
    for path in sorted(artifacts.root.iterdir()):
        lines.append(f"- `{path.name}`: {path.stat().st_size} bytes")
    lines.extend(["", "## Interpretation limits", ""])
    for limitation in manifest.get("limitations", []):
        lines.append(f"- {limitation}")
    lines.extend(
        [
            "- D1/D2 are internal controlled comparisons, not the full official CLARITY baseline.",
            "- Historical next MRI observations are never scored as outcomes of a different recommended action.",
            "- Synthetic environment performance must not be presented as patient survival benefit.",
            "",
            "## Incomplete items",
            "",
        ]
    )
    incomplete = [key for key, value in manifest.get("stage_states", {}).items() if not value and key != "interrupted"]
    lines.append("- " + ", ".join(incomplete) if incomplete else "None.")
    path = artifacts.path("report.md")
    write_text(path, "\n".join(lines) + "\n")
    return path


def evaluate_suite(
    config: dict[str, Any], artifacts: RunArtifacts, suite: str, split: str,
    *, seeds: Sequence[int] | None = None, variants: Sequence[str] | None = None
) -> None:
    if suite in {"dynamics", "all"}:
        evaluate_dynamics(config, artifacts, split, variants=variants, seeds=seeds)
    if suite in {"outcome", "all"}:
        evaluate_outcome(config, artifacts, split, seeds=seeds)
    if suite in {"replay", "all"}:
        evaluate_replay(config, artifacts, split, seeds=seeds)


def plan_case(
    config: dict[str, Any], artifacts: RunArtifacts, case_path: str | Path, seed: int
) -> dict[str, Any]:
    manifest, bundle, models = _bundle_for_run(config, artifacts)
    raw = json.loads(Path(case_path).read_text(encoding="utf-8"))
    forbidden = {"survival_time", "event", "next_z", "actual_action", "future", "target"}
    present = sorted(forbidden & set(raw))
    if present:
        raise EngineError("case contains forbidden future/label fields: " + ", ".join(present))
    if raw.get("latent_space") != "normalized":
        raise EngineError("case must explicitly declare latent_space=normalized")
    z = torch.tensor(raw["z"], dtype=torch.float32)
    clinical = torch.tensor(raw["clinical"], dtype=torch.float32)
    mask = torch.tensor(raw["clinical_mask"], dtype=torch.float32)
    history = torch.tensor(raw["history"], dtype=torch.float32)
    if z.numel() != bundle.latent_dim or clinical.numel() != bundle.clinical_dim:
        raise EngineError("case latent/clinical dimensions do not match the run protocol")
    if mask.shape != clinical.shape or history.numel() != bundle.history_dim:
        raise EngineError("case clinical mask/history dimensions do not match the run protocol")
    state = PatientState(
        patient_key=str(raw.get("patient_key", "external_case")),
        timepoint_key=str(raw.get("timepoint_key", "current")),
        observed_day=float(raw["observed_day"]),
        z=z,
        clinical=clinical,
        clinical_mask=mask,
        history=history,
        source="observed",
        state_version=int(raw["state_version"]),
    )
    device = resolve_device(config["project"]["device"])
    validate_device_visibility(config, device)
    planner = _planner_for_method(
        "mpc_rrt_ensemble_unc", seed, config, manifest, bundle, models,
        device,
    )
    decision = planner.plan(state)
    return {
        "status": decision.status,
        "reason_codes": list(decision.reason_codes),
        "recommended_action": (
            {
                "action_id": decision.recommended_action.action_id,
                "terms": list(decision.recommended_action.display_terms),
            }
            if decision.recommended_action is not None else None
        ),
        "imagined_plan": [
            {"action_id": action.action_id, "terms": list(action.display_terms)}
            for action in decision.imagined_plan
        ],
        "score_components": decision.score_components,
        "state_version": decision.state_version,
        "diagnostics": decision.diagnostics,
        "warning": "Research recommendation only; this command does not execute treatment and is not clinical advice.",
    }


def smoke(config: dict[str, Any]) -> dict[str, Any]:
    """No-data/no-network CPU contract test covering every module boundary."""
    local = copy.deepcopy(config)
    local["project"]["device"] = "cpu"
    local["world"].update(
        hidden_dim=24,
        action_embed_dim=8,
        time_embed_dim=4,
        clinical_embed_dim=6,
        history_embed_dim=6,
        ensemble_size=2,
    )
    local["training"].update(batch_size=8, max_epochs=1, patience=1)
    local["outcome"].update(hidden_dim=24, max_epochs=1, patience=1)
    local["data"]["min_action_support"] = 1
    seed_all(int(local["project"]["seed"]))
    cache = make_tiny_cache(num_patients=18, steps=5, latent_dim=8, seed=int(local["project"]["seed"]))
    split = split_patients(
        [row["patient_id"] for row in cache["patients"]],
        int(local["data"]["split_seed"]),
        float(local["data"]["train_fraction"]),
        float(local["data"]["validation_fraction"]),
    )
    preprocessing = fit_preprocessing(cache, local, split)
    bundle = load_bundle(cache, local, split, preprocessing)
    refs = window_refs(bundle, "train", mode="max_available", max_horizon=3)
    loader = DataLoader(DynamicsDataset(bundle, refs), batch_size=8, collate_fn=collate_dynamics)
    batch = next(iter(loader))
    first = _new_world_member(local, bundle, torch.device("cpu"))
    rollout = EnsembleWorldModel([first]).rollout(
        batch["z0"], batch["actions"], batch["deltas"], batch["context"],
        batch["clinical_mask"], batch["history0"], batch["step_mask"],
    )
    loss = terminal_mse(rollout, batch["horizons"], batch["target"])
    loss.backward()
    if not any(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()) for parameter in first.parameters()):
        raise EngineError("smoke RRT backward did not reach dynamics parameters")
    second = _new_world_member(local, bundle, torch.device("cpu"))
    ensemble = EnsembleWorldModel([first, second]).eval()
    with torch.no_grad():
        ensemble_rollout = ensemble.rollout(
            batch["z0"], batch["actions"], batch["deltas"], batch["context"],
            batch["clinical_mask"], batch["history0"], batch["step_mask"],
        )
    if ensemble_rollout.states.shape[0] != 2:
        raise EngineError("smoke ensemble contract failed")
    outcome_data = OutcomeDataset(bundle, "train")
    outcome_batch = collate_outcome([outcome_data[i] for i in range(min(8, len(outcome_data)))])
    outcome = PiecewiseExponentialHead.from_config(
        local, bundle.latent_dim, bundle.clinical_dim, bundle.history_dim
    )
    outcome_loss = outcome.nll(
        outcome_batch["z"], outcome_batch["clinical"], outcome_batch["clinical_mask"],
        outcome_batch["history"], outcome_batch["time"], outcome_batch["event"],
        weights=outcome_batch["weight"],
    )
    outcome_loss.backward()
    policy = CatalogPolicy(bundle.action_codec, 3)
    state = patient_state(bundle, bundle.split_ids["validation"][0], 0, 0)
    planner = Planner(
        ensemble, outcome.eval(), policy, bundle.action_codec,
        planned_interval_days=bundle.planned_interval_days, horizon=3, beam_width=2,
        lambda_uncertainty=0.05, uncertainty_scales={1: 1.0, 2: 1.0, 3: 1.0},
        max_candidates=3, min_action_support=1,
    )
    decision = planner.plan(state)
    if decision.status != "recommend" or len(decision.imagined_plan) != 3:
        raise EngineError("smoke MPC failed to produce a full horizon plan")
    fake = FakeProvider([action.action_id for action in policy.propose(planner._request(state, state.history)).candidates])
    llm_policy = LLMPolicy(
        bundle.action_codec, fake, model="fake", fallback=policy,
        allow_network=False, provider_is_fake=True,
    )
    llm_decision = Planner(
        ensemble, outcome.eval(), llm_policy, bundle.action_codec,
        planned_interval_days=bundle.planned_interval_days, horizon=1, beam_width=2,
        max_candidates=3, min_action_support=1,
    ).plan(state)
    if llm_decision.recommended_action != Planner(
        ensemble, outcome.eval(), policy, bundle.action_codec,
        planned_interval_days=bundle.planned_interval_days, horizon=1, beam_width=2,
        max_candidates=3, min_action_support=1,
    ).plan(state).recommended_action:
        raise EngineError("provider-neutral candidate contract failed")
    with tempfile.TemporaryDirectory(prefix="cloop-smoke-") as temp:
        artifacts = RunArtifacts(temp, "smoke")
        write_json(artifacts.path("run.json"), {"schema_version": "smoke", "ok": True})
        write_torch(artifacts.path("models.pt"), {"tensor": torch.arange(3)})
        write_torch(artifacts.path("last.pt"), {"rng": capture_rng(torch.Generator().manual_seed(1))})
        write_json(artifacts.path("metrics.json"), {"loss": float(loss.detach())})
        record = {"record_id": "same", "value": 1}
        upsert_jsonl(artifacts.path("predictions.jsonl"), [record, record])
        write_text(artifacts.path("report.md"), "# smoke\n")
        restored = read_torch(artifacts.path("models.pt"), safe=True)
        if not torch.equal(restored["tensor"], torch.arange(3)):
            raise EngineError("smoke artifact restore failed")
        if len(artifacts.path("predictions.jsonl").read_text().splitlines()) != 1:
            raise EngineError("smoke JSONL upsert is not idempotent")
        artifacts.assert_flat()
    toy_a = ToyEnv(ToyState(3, 1, 0), [0])
    toy_b = ToyEnv(ToyState(3, 1, 0), [0])
    if toy_a.step(0)[0] == toy_b.step(2)[0]:
        raise EngineError("synthetic environment did not respond to action")
    return {
        "status": "ok",
        "device": "cpu",
        "patients": len(bundle.trajectories),
        "rrt_loss": float(loss.detach()),
        "outcome_nll": float(outcome_loss.detach()),
        "ensemble_members": 2,
        "mpc_plan_length": len(decision.imagined_plan),
        "network_used": False,
    }


def mark_interrupted(artifacts: RunArtifacts) -> None:
    manifest = read_json(artifacts.path("run.json"))
    if manifest:
        manifest.setdefault("stage_states", {})["interrupted"] = True
        write_json(artifacts.path("run.json"), manifest)
