"""Small, flat, atomic experiment artifacts."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable

import torch


class ArtifactError(RuntimeError):
    pass


def _atomic(path: Path, writer: Callable[[Path], None]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        writer(tmp)
        with tmp.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        try:
            dir_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError:
            pass
    finally:
        if tmp.exists():
            tmp.unlink()


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)

    def writer(tmp: Path) -> None:
        tmp.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    _atomic(path, writer)


def read_json(path: str | Path, default: Any = None) -> Any:
    path = Path(path)
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_text(path: str | Path, value: str) -> None:
    path = Path(path)
    _atomic(path, lambda tmp: tmp.write_text(value, encoding="utf-8"))


def write_torch(path: str | Path, value: Any) -> None:
    path = Path(path)
    _atomic(path, lambda tmp: torch.save(value, tmp))


def read_torch(path: str | Path, *, safe: bool = True, map_location: str = "cpu") -> Any:
    try:
        return torch.load(Path(path), map_location=map_location, weights_only=safe)
    except FileNotFoundError as exc:
        raise ArtifactError(f"artifact not found: {path}") from exc


def upsert_jsonl(path: str | Path, records: list[dict[str, Any]]) -> None:
    path = Path(path)
    indexed: dict[str, dict[str, Any]] = {}
    if path.exists():
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if "record_id" not in row:
                raise ArtifactError(f"missing record_id in {path}:{line_no}")
            indexed[str(row["record_id"])] = row
    for row in records:
        if "record_id" not in row:
            raise ArtifactError("every JSONL record needs record_id")
        indexed[str(row["record_id"])] = row
    body = "".join(
        json.dumps(indexed[key], ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
        for key in sorted(indexed)
    )
    write_text(path, body)


class RunArtifacts:
    allowed_names = {"run.json", "models.pt", "last.pt", "metrics.json", "predictions.jsonl", "report.md"}
    allowed_versions = {"v1", "v1_1", "v2", "v3"}

    def __init__(self, output_root: str | Path, run_name: str, *, version: str = "v1"):
        if not run_name or "/" in run_name or "\\" in run_name or run_name in {".", ".."}:
            raise ArtifactError("run name must be one safe path component")
        if version not in self.allowed_versions:
            raise ArtifactError(f"unknown artifact version: {version}")
        self.root = Path(output_root) / version / run_name

    def path(self, name: str) -> Path:
        if name not in self.allowed_names:
            raise ArtifactError(f"unsupported run artifact: {name}")
        return self.root / name

    def assert_flat(self) -> None:
        if not self.root.exists():
            return
        unexpected = [p.name for p in self.root.iterdir() if p.name not in self.allowed_names]
        directories = [p.name for p in self.root.iterdir() if p.is_dir()]
        if unexpected or directories:
            raise ArtifactError(f"run directory is not flat: {sorted(set(unexpected + directories))}")
