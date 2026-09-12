"""Poisson-binomial reliability plus modular committee formation variants."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import random
from typing import Callable, Sequence

from sevc.core.registry import Registry


@dataclass(frozen=True)
class Committee:
    member_indices: tuple[int, ...]
    reliability: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class CommitteeFormationRequest:
    probabilities: tuple[float, ...]
    threshold: float
    committee_count: int
    seed: int = 0


FORMATION_VARIANTS: Registry[
    Callable[[CommitteeFormationRequest], tuple[Committee, ...]]
] = Registry("committee formation variant")


def majority_success_probability(probabilities: Sequence[float]) -> float:
    values = tuple(float(value) for value in probabilities)
    if not values:
        return 0.0
    if any(not math.isfinite(value) or value < 0 or value > 1 for value in values):
        raise ValueError("probabilities must be finite values in [0, 1]")
    count = len(values)
    required = count // 2 + 1
    distribution = [0.0] * (count + 1)
    distribution[0] = 1.0
    for probability in values:
        for successes in range(count, 0, -1):
            distribution[successes] = (
                distribution[successes] * (1.0 - probability)
                + distribution[successes - 1] * probability
            )
        distribution[0] *= 1.0 - probability
    return float(sum(distribution[required:]))


def legacy_greedy_committees(
    probabilities: Sequence[float], threshold: float, committee_count: int
) -> tuple[Committee, ...]:
    """Preserve the historical notebook's index-order greedy behavior."""

    values = tuple(float(value) for value in probabilities)
    selected: list[Committee] = []
    used: set[int] = set()
    while len(selected) < committee_count:
        best_subset: list[int] | None = None
        best_probability_sum = -1.0
        for first in range(len(values)):
            if first in used:
                continue
            subset = [first]
            reliability = majority_success_probability([values[first]])
            if reliability >= threshold:
                best_subset = subset
                break
            for second in range(len(values)):
                if second in used or second == first:
                    continue
                subset.append(second)
                reliability = majority_success_probability([values[index] for index in subset])
                if reliability >= threshold:
                    probability_sum = sum(values[index] for index in subset)
                    if (
                        best_subset is None
                        or len(subset) < len(best_subset)
                        or (
                            len(subset) == len(best_subset)
                            and probability_sum > best_probability_sum
                        )
                    ):
                        best_subset = list(subset)
                        best_probability_sum = probability_sum
                subset.pop()
        if best_subset is None:
            break
        used.update(best_subset)
        selected.append(
            Committee(
                tuple(best_subset),
                majority_success_probability([values[index] for index in best_subset]),
            )
        )
    return tuple(selected)


def greedy_minimal_committees(
    probabilities: Sequence[float], threshold: float, committee_count: int
) -> tuple[Committee, ...]:
    values = tuple(float(value) for value in probabilities)
    remaining = sorted(range(len(values)), key=lambda index: (-values[index], index))
    committees: list[Committee] = []
    for _ in range(committee_count):
        members: list[int] = []
        while remaining:
            members.append(remaining.pop(0))
            reliability = majority_success_probability([values[index] for index in members])
            if reliability >= threshold:
                committees.append(Committee(tuple(members), reliability))
                break
        else:
            break
    return tuple(committees)


def genetic_committees(
    probabilities: Sequence[float],
    threshold: float,
    committee_count: int,
    *,
    seed: int,
    population_size: int = 80,
    generations: int = 80,
    mutation_probability: float = 0.05,
) -> tuple[Committee, ...]:
    """Dependency-free genetic search replacing the historical DEAP wrapper."""

    values = tuple(float(value) for value in probabilities)
    if committee_count <= 0 or population_size < 2 or generations <= 0:
        raise ValueError("invalid genetic search dimensions")
    rng = random.Random(seed)
    unassigned = committee_count

    def decode(individual: Sequence[int]) -> tuple[Committee, ...]:
        groups = [[] for _ in range(committee_count)]
        for index, assignment in enumerate(individual):
            if assignment < committee_count:
                groups[assignment].append(index)
        return tuple(
            Committee(
                tuple(group),
                majority_success_probability([values[index] for index in group]),
            )
            for group in groups
            if group
        )

    def fitness(individual: Sequence[int]) -> tuple[int, int, float]:
        committees = decode(individual)
        valid = sum(item.reliability > threshold for item in committees)
        size = sum(len(item.member_indices) for item in committees if item.reliability > threshold)
        reliability_sum = sum(item.reliability for item in committees)
        return valid, -size, reliability_sum

    population = [
        [rng.randint(0, unassigned) for _ in values] for _ in range(population_size)
    ]
    for _ in range(generations):
        population.sort(key=fitness, reverse=True)
        survivors = population[: max(2, population_size // 4)]
        offspring = [list(item) for item in survivors]
        while len(offspring) < population_size:
            left, right = rng.sample(survivors, 2)
            point = rng.randrange(1, len(values)) if len(values) > 1 else 1
            child = list(left[:point]) + list(right[point:])
            for index in range(len(child)):
                if rng.random() < mutation_probability:
                    child[index] = rng.randint(0, unassigned)
            offspring.append(child)
        population = offspring
    best = max(population, key=fitness)
    committees = tuple(
        item for item in decode(best) if item.reliability > threshold
    )
    return committees[:committee_count]


@FORMATION_VARIANTS.register("legacy-greedy")
def _legacy_variant(request: CommitteeFormationRequest) -> tuple[Committee, ...]:
    return legacy_greedy_committees(
        request.probabilities, request.threshold, request.committee_count
    )


@FORMATION_VARIANTS.register("greedy-minimal")
def _greedy_variant(request: CommitteeFormationRequest) -> tuple[Committee, ...]:
    return greedy_minimal_committees(
        request.probabilities, request.threshold, request.committee_count
    )


@FORMATION_VARIANTS.register("genetic")
def _genetic_variant(request: CommitteeFormationRequest) -> tuple[Committee, ...]:
    return genetic_committees(
        request.probabilities,
        request.threshold,
        request.committee_count,
        seed=request.seed,
    )


def form_committees(
    variant_key: str, request: CommitteeFormationRequest
) -> tuple[Committee, ...]:
    return FORMATION_VARIANTS.get(variant_key)(request)
