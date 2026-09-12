"""Historical contract semantics behind the canonical incentive interface."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Callable, Sequence

import numpy as np

from sevc.core.registry import Registry


@dataclass(frozen=True)
class LegacyContractItem:
    target_cost: float
    contribution: float
    reward: float

    def worker_utility(self, worker_cost: float) -> float:
        return self.reward - worker_cost * self.contribution

    def owner_utility(self, sigma_1: float, sigma_2: float) -> float:
        return sigma_1 * math.log1p(sigma_2 * self.contribution) - self.reward

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class ContractSelection:
    worker_cost: float
    item: LegacyContractItem


@dataclass(frozen=True)
class ContractDesignRequest:
    worker_types: tuple[float, ...]
    sigma_1: float
    sigma_2: float
    archived_pairs: tuple[tuple[float, float], ...] | None = None


CONTRACT_VARIANTS: Registry[
    Callable[[ContractDesignRequest], tuple[LegacyContractItem, ...]]
] = Registry("contract variant")


def _validate_types(worker_types: Sequence[float]) -> tuple[float, ...]:
    types = tuple(float(value) for value in worker_types)
    if not types or any(not math.isfinite(value) or value <= 0 for value in types):
        raise ValueError("worker types must be finite and positive")
    if any(left >= right for left, right in zip(types, types[1:])):
        raise ValueError("worker types must be strictly increasing")
    return types


def _best_integer_contribution(
    worker_cost: float, sigma_1: float, sigma_2: float
) -> float:
    if sigma_1 <= 0 or sigma_2 <= 0:
        raise ValueError("utility parameters must be positive")
    continuous = sigma_1 / worker_cost - 1.0 / sigma_2
    candidates = {1, max(1, math.floor(continuous)), max(1, math.ceil(continuous))}
    return float(
        max(
            candidates,
            key=lambda contribution: sigma_1
            * math.log1p(sigma_2 * contribution)
            - worker_cost * contribution,
        )
    )


def design_analytic_legacy_menu(
    worker_types: Sequence[float], sigma_1: float, sigma_2: float
) -> tuple[LegacyContractItem, ...]:
    """Reproduce the intended integer recurrence without the ECOS_BB drift."""

    types = _validate_types(worker_types)
    contributions = tuple(
        _best_integer_contribution(worker_type, sigma_1, sigma_2)
        for worker_type in types
    )
    rewards = [0.0] * len(types)
    rewards[-1] = types[-1] * contributions[-1]
    for index in range(len(types) - 2, -1, -1):
        rewards[index] = (
            rewards[index + 1]
            - types[index] * contributions[index + 1]
            + types[index] * contributions[index]
        )
    return tuple(
        LegacyContractItem(worker_type, contribution, reward)
        for worker_type, contribution, reward in reversed(
            tuple(zip(types, contributions, rewards))
        )
    )


def archived_legacy_menu(
    worker_types: Sequence[float], archived_pairs: Sequence[Sequence[float]]
) -> tuple[LegacyContractItem, ...]:
    """Create the frozen ECOS_BB compatibility variant from verified output."""

    types = _validate_types(worker_types)
    pairs = tuple(tuple(float(value) for value in pair) for pair in archived_pairs)
    if len(pairs) != len(types) or any(len(pair) != 2 for pair in pairs):
        raise ValueError("archived menu must contain one contribution/reward pair per type")
    return tuple(
        LegacyContractItem(target_cost, pair[0], pair[1])
        for target_cost, pair in zip(reversed(types), pairs)
    )


@CONTRACT_VARIANTS.register("analytic-integer")
def _analytic_variant(request: ContractDesignRequest) -> tuple[LegacyContractItem, ...]:
    return design_analytic_legacy_menu(
        request.worker_types, request.sigma_1, request.sigma_2
    )


@CONTRACT_VARIANTS.register("archived-ecos-bb")
def _archived_variant(request: ContractDesignRequest) -> tuple[LegacyContractItem, ...]:
    if request.archived_pairs is None:
        raise ValueError("archived-ecos-bb requires verified archived_pairs")
    return archived_legacy_menu(request.worker_types, request.archived_pairs)


def design_contract_variant(
    variant_key: str, request: ContractDesignRequest
) -> tuple[LegacyContractItem, ...]:
    return CONTRACT_VARIANTS.get(variant_key)(request)


def select_contracts(
    worker_costs: Sequence[float],
    menu_descending_cost: Sequence[LegacyContractItem],
    *,
    method: str = "best",
) -> tuple[ContractSelection, ...]:
    menu = tuple(menu_descending_cost)
    if not menu:
        raise ValueError("contract menu cannot be empty")
    selections: list[ContractSelection] = []
    for worker_cost in worker_costs:
        cost = float(worker_cost)
        if method == "best":
            item = max(menu, key=lambda candidate: candidate.worker_utility(cost))
        elif method == "uniform":
            item = menu[0]
        else:
            raise ValueError(f"unsupported selection method: {method}")
        selections.append(ContractSelection(cost, item))
    return tuple(selections)


def evaluate_selections(
    selections: Sequence[ContractSelection], sigma_1: float, sigma_2: float
) -> dict[str, object]:
    worker_utilities = tuple(
        selection.item.worker_utility(selection.worker_cost)
        for selection in selections
    )
    owner_utilities = tuple(
        selection.item.owner_utility(sigma_1, sigma_2)
        for selection in selections
    )
    return {
        "worker_utilities": worker_utilities,
        "owner_utilities": owner_utilities,
        "worker_utility_total": float(sum(worker_utilities)),
        "owner_utility_total": float(sum(owner_utilities)),
    }


def actual_utilities(
    selections: Sequence[ContractSelection],
    actual_contributions: Sequence[float],
    actual_rewards: Sequence[float],
    sigma_1: float,
    sigma_2: float,
) -> tuple[float, tuple[float, ...]]:
    if not (
        len(selections) == len(actual_contributions) == len(actual_rewards)
    ):
        raise ValueError("selection, contribution, and reward lengths must match")
    owner_total = 0.0
    worker_utilities: list[float] = []
    for selection, contribution, reward in zip(
        selections, actual_contributions, actual_rewards
    ):
        owner_total += sigma_1 * math.log1p(sigma_2 * float(contribution)) - float(
            reward
        )
        contracted_cost = selection.worker_cost * selection.item.contribution
        worker_utilities.append(float(reward) - contracted_cost)
    return owner_total, tuple(worker_utilities)


def sample_worker_costs(
    worker_types: Sequence[float],
    proportions: Sequence[float],
    worker_count: int,
    standard_deviation: float,
    seed: int,
) -> tuple[float, ...]:
    types = _validate_types(worker_types)
    probs = tuple(float(value) for value in proportions)
    if len(types) != len(probs) or not math.isclose(sum(probs), 1.0, abs_tol=1e-12):
        raise ValueError("proportions must align with types and sum to one")
    rng = np.random.RandomState(seed)
    samples: list[float] = []
    assigned = 0
    for index, (worker_type, proportion) in enumerate(zip(types, probs)):
        count = (
            worker_count - assigned
            if index == len(types) - 1
            else int(worker_count * proportion)
        )
        accepted: list[float] = []
        while len(accepted) < count:
            draws = rng.normal(worker_type, standard_deviation, count - len(accepted))
            accepted.extend(float(value) for value in draws if value >= 0)
        samples.extend(accepted)
        assigned += count
    return tuple(sorted(samples))
