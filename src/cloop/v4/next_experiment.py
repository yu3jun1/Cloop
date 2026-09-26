"""Public entry point for the isolated Cloop v4.1 experiment pipeline."""
from __future__ import annotations

from typing import Any

from . import next_experiment_v4_1 as _implementation
from .next_experiment_v4_1 import (
    DYNAMICS_VARIANTS,
    FEATURES,
    OUTCOMES,
    PLANNING_METHODS,
    V4ExperimentError,
    load_experiment_config,
    main,
    run,
)


def __getattr__(name: str) -> Any:
    """Delegate private research helpers without duplicating their implementation."""
    try:
        return getattr(_implementation, name)
    except AttributeError as exc:
        raise AttributeError(name) from exc


__all__ = [
    "DYNAMICS_VARIANTS",
    "FEATURES",
    "OUTCOMES",
    "PLANNING_METHODS",
    "V4ExperimentError",
    "load_experiment_config",
    "main",
    "run",
]


if __name__ == "__main__":
    main()
