"""Strict configuration loading, protocol constraints, and stable signatures."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    pass


def _read_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"cannot read config {path}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise ConfigError(f"config root must be a mapping: {path}")
    return loaded


def _same_type(expected: Any, actual: Any) -> bool:
    if expected is None:
        return actual is None or isinstance(actual, (str, int, float, bool))
    if isinstance(expected, bool):
        return isinstance(actual, bool)
    if isinstance(expected, int) and not isinstance(expected, bool):
        return isinstance(actual, int) and not isinstance(actual, bool)
    if isinstance(expected, float):
        return isinstance(actual, (int, float)) and not isinstance(actual, bool)
    return isinstance(actual, type(expected))


def strict_merge(base: dict[str, Any], override: dict[str, Any], *, prefix: str = "") -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        dotted = f"{prefix}.{key}" if prefix else key
        if key not in result:
            raise ConfigError(f"unknown configuration field: {dotted}")
        if isinstance(result[key], dict):
            if not isinstance(value, dict):
                raise ConfigError(f"expected mapping for {dotted}")
            result[key] = strict_merge(result[key], value, prefix=dotted)
        elif not _same_type(result[key], value):
            raise ConfigError(
                f"type mismatch for {dotted}: expected {type(result[key]).__name__}, "
                f"got {type(value).__name__}"
            )
        else:
            result[key] = value
    return result


def _validate(cfg: dict[str, Any]) -> None:
    data = cfg["data"]
    world = cfg["world"]
    train = cfg["training"]
    outcome = cfg["outcome"]
    planner = cfg["planner"]
    if data["protocol"] not in {"main_v1", "legacy_stage1"}:
        raise ConfigError("data.protocol must be main_v1 or legacy_stage1")
    if data["action_alignment"] not in {"timestamps", "legacy_source"}:
        raise ConfigError("invalid data.action_alignment")
    expected_alignment = "timestamps" if data["protocol"] == "main_v1" else "legacy_source"
    if data["action_alignment"] != expected_alignment:
        raise ConfigError(
            f"{data['protocol']} requires data.action_alignment={expected_alignment}"
        )
    if data["unknown_interval_policy"] != "exclude_action_conditioned":
        raise ConfigError(
            "v1 only implements data.unknown_interval_policy=exclude_action_conditioned"
        )
    if (
        data["protocol"] == "main_v1"
        and data["require_mri_day_provenance"] is not True
    ):
        raise ConfigError(
            "main_v1 requires data.require_mri_day_provenance=true"
        )
    if abs(data["train_fraction"] + data["validation_fraction"] - 0.85) > 1e-8:
        raise ConfigError("train_fraction + validation_fraction must equal 0.85 (test is 0.15)")
    if not 1 <= int(world["max_horizon"]) <= 3 or not 1 <= int(planner["horizon"]) <= 3:
        raise ConfigError("world/planner horizon must be in [1,3]")
    if not world["terminal_only"] or world["teacher_forcing"]:
        raise ConfigError("v1 requires terminal_only=true and teacher_forcing=false")
    allowed_variants = {"baseline", "rrt", "ensemble", "rrt_ensemble"}
    if not set(train["variants"]).issubset(allowed_variants):
        raise ConfigError("unknown dynamics variant")
    edges = outcome["edges_days"]
    if len(edges) < 2 or edges[0] != 0 or any(b <= a for a, b in zip(edges, edges[1:])):
        raise ConfigError("outcome.edges_days must strictly increase from zero")
    if planner["lambda_uncertainty"] < 0 or planner["beam_width"] < 1:
        raise ConfigError("invalid planner settings")
    if planner["interval_source"] != "train_median":
        raise ConfigError("v1 only implements planner.interval_source=train_median")
    if planner["require_supported_actions"] is not True:
        raise ConfigError("v1 requires planner.require_supported_actions=true")
    if planner["abstain_when_no_valid_action"] is not True:
        raise ConfigError("v1 requires planner.abstain_when_no_valid_action=true")
    if cfg["policy"]["kind"] not in {"catalog", "llm"}:
        raise ConfigError("policy.kind must be catalog or llm")
    if cfg["policy"]["allow_network"] and cfg["policy"]["kind"] == "catalog":
        raise ConfigError("catalog policy cannot enable network access")
    if planner["toxicity_model"] != "disabled":
        raise ConfigError("v1 does not train a validated toxicity model; toxicity_model must be disabled")
    if planner["clinical_rules_path"] is not None:
        raise ConfigError("v1 refuses unvalidated clinical rule files; clinical_rules_path must be null")
    if data["protocol"] == "legacy_stage1":
        required = {
            "data.action_alignment": data["action_alignment"] == "legacy_source",
            "world.use_context": world["use_context"] is False,
            "world.use_history": world["use_history"] is False,
        }
        bad = [key for key, ok in required.items() if not ok]
        if bad:
            raise ConfigError("legacy_stage1 requires fixed settings: " + ", ".join(bad))


def load_config(
    config_path: str | Path,
    paths_path: str | Path | None = None,
    *,
    protocol: str | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    """Load the complete default schema, then strictly merge user and path files."""
    bundled = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"
    schema = _read_yaml(bundled)
    cfg = strict_merge(schema, _read_yaml(config_path))
    if paths_path is not None:
        paths_cfg = _read_yaml(paths_path)
        for section in paths_cfg:
            if section not in {"paths", "runtime"}:
                raise ConfigError(f"paths file may only contain paths/runtime, got {section}")
        path_schema = {
            "paths": {
                "project_root": "", "source_project": "", "clarity_root": "", "timeline": "",
                "timeline_alternative": "", "latent_dir": "", "legacy_trajectories": "",
                "latent_provenance": "", "mri_root": "", "brainiac_checkpoint": "",
                "cache_root": "", "output_root": "",
            },
            "runtime": {
                "python_reference": "", "allowed_physical_gpus": [], "preferred_physical_gpu": 0,
            },
        }
        cfg.update(strict_merge(path_schema, paths_cfg))
    else:
        cfg["paths"] = {
            "project_root": str(Path.cwd()),
            "source_project": "",
            "clarity_root": "",
            "timeline": "",
            "timeline_alternative": "",
            "latent_dir": "",
            "latent_provenance": "",
            "legacy_trajectories": "",
            "mri_root": "",
            "brainiac_checkpoint": "",
            "cache_root": str(Path.cwd() / ".cache"),
            "output_root": str(Path.cwd() / "outputs"),
        }
        cfg["runtime"] = {
            "python_reference": "",
            "allowed_physical_gpus": [],
            "preferred_physical_gpu": 0,
        }
    if protocol is not None:
        cfg["data"]["protocol"] = protocol
        if protocol == "legacy_stage1":
            cfg["data"]["action_alignment"] = "legacy_source"
            cfg["world"]["use_context"] = False
            cfg["world"]["use_history"] = False
    if device is not None:
        cfg["project"]["device"] = device
    _validate(cfg)
    return cfg


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def config_signature(config: dict[str, Any]) -> str:
    # api_key_env names are configuration, but secret values are never resolved here.
    return hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest()


def model_affecting_config(config: dict[str, Any]) -> dict[str, Any]:
    return {k: copy.deepcopy(v) for k, v in config.items() if k not in {"artifacts"}}
