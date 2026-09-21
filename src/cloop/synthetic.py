"""Independent action-responsive toy environment and exact finite-horizon oracle."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

import numpy as np
import torch
from torch import Tensor


SYNTHETIC_ACTIONS = ("A0", "A1", "A2")
NOISE_VALUES = (-1, 0, 1)
NOISE_PROBABILITIES = (0.1, 0.8, 0.1)


@dataclass(frozen=True)
class ToyState:
    burden: int
    toxicity: int
    subtype: int


def transition(state: ToyState, action: int, noise: int) -> ToyState:
    if action not in (0, 1, 2) or noise not in NOISE_VALUES:
        raise ValueError("invalid synthetic action/noise")
    effect = ((0, 1, 2), (0, 2, 1))[state.subtype][action]
    toxicity = (0, 1, 2)[action]
    return ToyState(
        burden=int(np.clip(state.burden + 1 - effect + noise, 0, 5)),
        toxicity=int(np.clip(state.toxicity + toxicity - 1, 0, 3)),
        subtype=state.subtype,
    )


def env_cost(state: ToyState) -> float:
    return state.burden / 5.0 + 0.4 * state.toxicity / 3.0 + (1.0 if state.toxicity == 3 else 0.0)


def encode_state(state: ToyState) -> Tensor:
    """Fixed injective 8D encoding, independent of the learned world model."""
    return torch.tensor(
        [
            state.burden / 5.0,
            state.toxicity / 3.0,
            float(state.subtype),
            (state.burden / 5.0) ** 2,
            (state.toxicity / 3.0) ** 2,
            state.burden * state.toxicity / 15.0,
            float(state.subtype == 0),
            1.0,
        ],
        dtype=torch.float32,
    )


class ToyEnv:
    def __init__(self, initial_state: ToyState, noise_stream: Sequence[int]):
        self.state = initial_state
        self.noise_stream = tuple(int(x) for x in noise_stream)
        self.step_index = 0

    def observe(self) -> ToyState:
        return self.state

    def step(self, action: int) -> tuple[ToyState, float, bool, dict[str, int]]:
        if self.step_index >= len(self.noise_stream):
            raise RuntimeError("episode already complete")
        noise = self.noise_stream[self.step_index]
        self.state = transition(self.state, action, noise)
        self.step_index += 1
        return (
            self.state,
            env_cost(self.state),
            self.step_index == len(self.noise_stream),
            {"noise": noise},
        )


def sample_episodes(count: int, steps: int, seed: int) -> list[tuple[ToyState, tuple[int, ...]]]:
    rng = np.random.default_rng(seed)
    episodes = []
    for _ in range(count):
        initial = ToyState(
            burden=int(rng.integers(0, 6)),
            toxicity=int(rng.integers(0, 4)),
            subtype=int(rng.integers(0, 2)),
        )
        noise = tuple(int(x) for x in rng.choice(NOISE_VALUES, size=steps, p=NOISE_PROBABILITIES))
        episodes.append((initial, noise))
    return episodes


@lru_cache(maxsize=None)
def optimal_expected_value(state: ToyState, steps: int) -> float:
    if steps <= 0:
        return 0.0
    action_values = []
    for action in range(3):
        value = 0.0
        for noise, probability in zip(NOISE_VALUES, NOISE_PROBABILITIES):
            next_state = transition(state, action, noise)
            value += probability * (env_cost(next_state) + optimal_expected_value(next_state, steps - 1))
        action_values.append(value)
    return min(action_values)


def optimal_action(state: ToyState, steps: int) -> int:
    values = []
    for action in range(3):
        expected = sum(
            probability
            * (env_cost(next_state := transition(state, action, noise)) + optimal_expected_value(next_state, steps - 1))
            for noise, probability in zip(NOISE_VALUES, NOISE_PROBABILITIES)
        )
        values.append(expected)
    return int(np.argmin(values))
