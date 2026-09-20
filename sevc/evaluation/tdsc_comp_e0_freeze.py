"""Selected canonical scientific routines; deployment wrappers are omitted."""
from __future__ import annotations


from collections import Counter

import hashlib

import json

import math

import random

from typing import Any, Sequence

import numpy as np

FUTURE_E2_CHANGE_ID = "experiment-tdsc-comparative-robustness-v1"


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _compact_int_sha256(values: Sequence[int]) -> str:
    return hashlib.sha256(
        json.dumps(
            [int(value) for value in values], separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def _derived_seed(base_seed: int, namespace: str) -> int:
    material = f"{FUTURE_E2_CHANGE_ID}|{int(base_seed)}|{namespace}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % 2147483647


def _target_capacities(weights: Sequence[float], total: int) -> tuple[int, ...]:
    normalized = np.asarray([float(value) for value in weights], dtype=np.float64)
    if normalized.size == 0 or np.any(~np.isfinite(normalized)) or np.any(normalized <= 0):
        raise ValueError("invalid contribution weights")
    normalized /= float(normalized.sum())
    raw = normalized * int(total)
    sizes = np.floor(raw).astype(np.int64)
    remainder = int(total) - int(sizes.sum())
    if remainder:
        order = np.argsort(-(raw - sizes), kind="stable")
        sizes[order[:remainder]] += 1
    return tuple(int(value) for value in sizes)


def _root_indices(
    labels: Sequence[int],
    formal_seed: int,
    *,
    class_count: int = 10,
    per_class: int = 100,
) -> tuple[int, ...]:
    class_buckets: list[list[int]] = [[] for _ in range(int(class_count))]
    for index, value in enumerate(labels):
        label = int(value)
        if 0 <= label < int(class_count):
            class_buckets[label].append(index)
    chosen: list[int] = []
    for label in range(int(class_count)):
        candidates = list(class_buckets[label])
        random.Random(
            _derived_seed(formal_seed, f"trusted-root|class={label}")
        ).shuffle(candidates)
        if len(candidates) < int(per_class):
            raise ValueError(f"class {label} cannot supply trusted-root quota {per_class}")
        chosen.extend(candidates[: int(per_class)])
    return tuple(sorted(chosen))


def _dirichlet_counts(
    class_sizes: Sequence[int],
    capacities: Sequence[int],
    *,
    alpha: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    trainer_count = len(capacities)
    rng = np.random.RandomState(seed)
    probabilities = np.vstack(
        [rng.dirichlet(np.full(trainer_count, float(alpha))) for _ in class_sizes]
    )
    counts = np.zeros((len(class_sizes), trainer_count), dtype=np.int64)
    for class_id, class_size in enumerate(class_sizes):
        raw = probabilities[class_id] * int(class_size)
        row = np.floor(raw).astype(np.int64)
        remainder = int(class_size) - int(row.sum())
        if remainder:
            order = np.argsort(-(raw - row), kind="stable")
            row[order[:remainder]] += 1
        counts[class_id] = row

    capacities_array = np.asarray(capacities, dtype=np.int64)
    delta = counts.sum(axis=0) - capacities_array
    while np.any(delta != 0):
        donors = np.where(delta > 0)[0]
        receivers = np.where(delta < 0)[0]
        candidates: list[tuple[float, int, int, int]] = []
        for class_id in range(len(class_sizes)):
            for donor in donors:
                if counts[class_id, donor] <= 0:
                    continue
                for receiver in receivers:
                    cost = -math.log(max(float(probabilities[class_id, receiver]), 1e-300))
                    cost += math.log(max(float(probabilities[class_id, donor]), 1e-300))
                    candidates.append((cost, class_id, int(donor), int(receiver)))
        if not candidates:
            raise RuntimeError("Dirichlet capacity rebalance has no feasible move")
        _, class_id, donor, receiver = min(candidates)
        amount = min(
            int(delta[donor]),
            int(-delta[receiver]),
            int(counts[class_id, donor]),
        )
        counts[class_id, donor] -= amount
        counts[class_id, receiver] += amount
        delta[donor] -= amount
        delta[receiver] += amount
    if not np.array_equal(counts.sum(axis=0), capacities_array):
        raise AssertionError("Dirichlet partition does not match target capacities")
    if not np.array_equal(counts.sum(axis=1), np.asarray(class_sizes, dtype=np.int64)):
        raise AssertionError("Dirichlet partition does not conserve class counts")
    return counts, probabilities


def _summarize_partitions(
    partitions: Sequence[Sequence[int]],
    labels: Sequence[int],
    pool: Sequence[int],
    *,
    class_count: int = 10,
) -> tuple[list[dict[str, Any]], str]:
    flattened = [int(value) for partition in partitions for value in partition]
    if len(flattened) != len(pool) or len(set(flattened)) != len(pool):
        raise AssertionError("partition duplication or omission")
    if set(flattened) != set(int(value) for value in pool):
        raise AssertionError("partition union does not equal the frozen trainer pool")
    rows: list[dict[str, Any]] = []
    for trainer_id, partition in enumerate(partitions):
        values = [int(value) for value in partition]
        histogram = Counter(int(labels[index]) for index in values)
        rows.append(
            {
                "trainer_id": trainer_id,
                "size": len(values),
                "class_histogram": {
                    str(label): int(histogram.get(label, 0))
                    for label in range(int(class_count))
                },
                "index_sha256": _compact_int_sha256(values),
            }
        )
    aggregate = hashlib.sha256(_canonical_json_bytes(rows)).hexdigest()
    return rows, aggregate


def _e2_partition_indices(
    labels: Sequence[int],
    pool: Sequence[int],
    capacities: Sequence[int],
    *,
    formal_seed: int,
    trainer_count: int,
    mode: str,
    alpha: float | None,
    class_count: int = 10,
) -> tuple[tuple[list[int], ...], str | None]:
    """Return the exact arrays behind the E0 partition identity lock."""

    partition_seed = _derived_seed(formal_seed, f"partition|N={trainer_count}|mode={mode}")
    if alpha is None:
        permuted = np.random.RandomState(partition_seed).permutation(
            np.asarray(pool, dtype=np.int64)
        )
        boundaries = np.cumsum(np.asarray(capacities, dtype=np.int64))[:-1]
        partitions = tuple(
            np.asarray(chunk, dtype=np.int64).tolist()
            for chunk in np.split(permuted, boundaries)
        )
        probability_hash = None
    else:
        class_pools: list[list[int]] = [
            [] for _ in range(int(class_count))
        ]
        for index in pool:
            class_pools[int(labels[int(index)])].append(int(index))
        counts, probabilities = _dirichlet_counts(
            [len(values) for values in class_pools],
            capacities,
            alpha=alpha,
            seed=partition_seed,
        )
        trainer_values: list[list[int]] = [[] for _ in range(trainer_count)]
        for class_id, class_pool in enumerate(class_pools):
            shuffled = np.random.RandomState(
                _derived_seed(formal_seed, f"{mode}|N={trainer_count}|class={class_id}|indices")
            ).permutation(np.asarray(class_pool, dtype=np.int64))
            boundaries = np.cumsum(counts[class_id])[:-1]
            for trainer_id, chunk in enumerate(np.split(shuffled, boundaries)):
                trainer_values[trainer_id].extend(int(value) for value in chunk)
        partitions_list: list[list[int]] = []
        for trainer_id, values in enumerate(trainer_values):
            shuffled = np.random.RandomState(
                _derived_seed(formal_seed, f"{mode}|N={trainer_count}|trainer={trainer_id}|final")
            ).permutation(np.asarray(values, dtype=np.int64))
            partitions_list.append([int(value) for value in shuffled])
        partitions = tuple(partitions_list)
        probability_hash = hashlib.sha256(
            probabilities.astype("<f8", copy=False).tobytes(order="C")
        ).hexdigest()
    return partitions, probability_hash


