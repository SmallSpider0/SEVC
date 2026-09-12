"""Sliding-window exponentially weighted reputation from the prototype."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class ReputationSimulation:
    averages: Mapping[str, tuple[float, ...]]
    standard_deviations: Mapping[str, tuple[float, ...]]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def exponential_decay_reputation(
    outcomes: Sequence[int], decay: float, window_size: int
) -> tuple[float, ...]:
    if decay < 0 or window_size <= 0:
        raise ValueError("decay must be non-negative and window_size positive")
    history: list[int] = []
    scores: list[float] = []
    for outcome in outcomes:
        if outcome not in {0, 1}:
            raise ValueError("reputation outcomes must be binary")
        history.append(int(outcome))
        recent = history[-window_size:]
        weights = np.exp(-decay * np.arange(len(recent) - 1, -1, -1))
        scores.append(float(np.dot(recent, weights) / np.sum(weights)))
    return tuple(scores)


def simulate_historical_verifier_groups(
    *,
    epochs: int,
    decay: float,
    window_size: int,
    verifier_count: int,
    proportions: Sequence[float],
    group_size: int,
    attack_start: int,
    seed: int,
) -> ReputationSimulation:
    """Reproduce the seed-61 group simulation saved by the historical notebook."""

    if epochs <= 0 or verifier_count <= 0 or group_size <= 0:
        raise ValueError("epoch, verifier, and group counts must be positive")
    if len(proportions) != 3 or not np.isclose(sum(proportions), 1.0):
        raise ValueError("historical simulation requires three proportions summing to one")
    rng = np.random.RandomState(seed)
    type_names = ("honest", "Free-rider", "Malicious")
    verifiers: list[dict[str, object]] = []
    assigned = 0
    for index, (type_name, proportion) in enumerate(zip(type_names, proportions)):
        count = (
            verifier_count - assigned
            if index == len(type_names) - 1
            else int(verifier_count * proportion)
        )
        for _ in range(count):
            verifiers.append(
                {"type": type_name, "reputation": 0.5, "history": [], "active": True}
            )
        assigned += count

    averages = {name: [] for name in type_names}
    deviations = {name: [] for name in type_names}
    for epoch in range(epochs):
        rng.shuffle(verifiers)
        active = [item for item in verifiers if bool(item["active"])]
        groups = [active[index : index + group_size] for index in range(0, len(active), group_size)]
        for group in groups:
            for verifier in group:
                verifier_type = str(verifier["type"])
                if verifier_type == "honest":
                    outcome = 1 if rng.rand() > 0.01 else 0
                elif verifier_type == "Free-rider":
                    outcome = 0 if rng.rand() > 0.1 else 1
                elif epoch < attack_start:
                    outcome = 1 if rng.rand() > 0.01 else 0
                else:
                    outcome = 0
                history = verifier["history"]
                assert isinstance(history, list)
                history.append(outcome)

            for verifier in group:
                history = verifier["history"]
                assert isinstance(history, list)
                weights = np.exp(-decay * np.arange(1, len(history) + 1)[::-1])
                recent_history = history[-window_size:]
                recent_weights = weights[-window_size:]
                reputation = float(
                    np.dot(recent_history, recent_weights) / np.sum(recent_weights)
                )
                if epoch < window_size and reputation > 0.5:
                    reputation = 0.5
                verifier["reputation"] = reputation

        for type_name in type_names:
            values = [
                float(item["reputation"])
                for item in verifiers
                if item["type"] == type_name
            ]
            averages[type_name].append(float(np.mean(values)) if values else 0.0)
            deviations[type_name].append(float(np.std(values)) if values else 0.0)

    return ReputationSimulation(
        averages={key: tuple(values) for key, values in averages.items()},
        standard_deviations={key: tuple(values) for key, values in deviations.items()},
    )
