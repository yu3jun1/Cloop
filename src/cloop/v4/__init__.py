"""Cloop v4.1 calibrated trajectory and closed-loop planning experiments."""

from .outcome import TrajectoryOutcomeModel
from .trajectory import TemporalEmbedding, TrajectoryEncoder

# Historical import name retained without retaining the independent value head.
DualHeadOutcomeModel = TrajectoryOutcomeModel

__all__ = [
    "DualHeadOutcomeModel", "TemporalEmbedding", "TrajectoryEncoder",
    "TrajectoryOutcomeModel",
]
