"""Selected canonical scientific routines; deployment wrappers are omitted."""
from __future__ import annotations


import hashlib

from typing import Any, Mapping, Sequence

import numpy as np

from sevc.incentives import ContractDesignRequest, LegacyContractItem, design_contract_variant, select_contracts

CHANGE_ID = "experiment-trainer-population-network-scale-v1"


def derive_scale_seed(base_seed: int, namespace: str) -> int:
    material = f"{CHANGE_ID}|{int(base_seed)}|{namespace}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % 2147483647


def balanced_worker_costs(
    worker_types: Sequence[float],
    trainer_count: int,
    *,
    standard_deviation: float,
    seed: int,
) -> tuple[float, ...]:
    """Assign trainer i to layer i mod M, then draw one non-negative cost."""

    types = tuple(float(value) for value in worker_types)
    if not types or trainer_count <= 0 or standard_deviation <= 0:
        raise ValueError("invalid balanced worker-cost inputs")
    rng = np.random.RandomState(seed)
    costs: list[float] = []
    for trainer_id in range(trainer_count):
        center = types[trainer_id % len(types)]
        value = -1.0
        while value < 0:
            value = float(rng.normal(center, standard_deviation))
        costs.append(value)
    return tuple(costs)


def trainer_contract_selections(
    config: Mapping[str, Any], trainer_count: int, seed: int
) -> tuple[tuple[Any, ...], tuple[float, ...]]:
    """Build the one frozen contract selection used by execution and projection."""

    contract = config["contract"]
    request = ContractDesignRequest(
        worker_types=tuple(float(value) for value in contract["worker_types"]),
        sigma_1=float(contract["sigma_1"]),
        sigma_2=float(contract["sigma_2"]),
    )
    menu = design_contract_variant(str(contract["variant_key"]), request)
    reserve = float(contract.get("common_reward_reserve", 0.0))
    if reserve:
        menu = tuple(
            LegacyContractItem(
                target_cost=item.target_cost,
                contribution=item.contribution,
                reward=item.reward + reserve,
            )
            for item in menu
        )
    costs = balanced_worker_costs(
        contract["worker_types"],
        trainer_count,
        standard_deviation=float(contract["worker_cost_standard_deviation"]),
        seed=derive_scale_seed(seed, "worker-costs"),
    )
    selections = select_contracts(
        costs, menu, method=str(contract["selection_method"])
    )
    return selections, tuple(
        float(selection.item.contribution) for selection in selections
    )


