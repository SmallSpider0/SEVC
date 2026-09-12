"""Runtime materialization and full verification of the frozen COMP-E2 partitions."""

from __future__ import annotations

from collections import Counter
import base64
import json
from typing import Any, Mapping, Sequence

import numpy as np
import zlib

from sevc.evaluation.tdsc_comp_e0_freeze import (
    _compact_int_sha256,
    _e2_partition_indices,
    _root_indices,
    _summarize_partitions,
    _target_capacities,
)


def _validate_cifar10_labels(labels: Sequence[int], lock: Mapping[str, Any]) -> None:
    if len(labels) != 50000 or Counter(int(value) for value in labels) != Counter(
        {label: 5000 for label in range(10)}
    ):
        raise ValueError("CIFAR-10 label identity/cardinality drift")
    if _compact_int_sha256(labels) != str(lock["dataset"]["label_vector_sha256"]):
        raise ValueError("CIFAR-10 label-vector SHA-256 drift")


def materialize_locked_e2_partitions(
    labels: Sequence[int],
    lock: Mapping[str, Any],
    *,
    probability_compatibility: Mapping[str, Any] | None = None,
) -> dict[tuple[int, int, str], tuple[tuple[int, ...], ...]]:
    _validate_cifar10_labels(labels, lock)
    root_rows = {int(row["formal_seed"]): row for row in lock["trusted_root_sets"]}
    results: dict[tuple[int, int, str], tuple[tuple[int, ...], ...]] = {}
    for seed, root_row in root_rows.items():
        root = _root_indices(labels, seed)
        pool = tuple(index for index in range(len(labels)) if index not in set(root))
        if (
            len(root) != 1000
            or _compact_int_sha256(root) != str(root_row["index_sha256"])
            or _compact_int_sha256(pool) != str(root_row["trainer_pool_index_sha256"])
        ):
            raise ValueError(f"trusted-root identity drift for seed {seed}")
    for row in lock["partition_sets"]:
        seed = int(row["formal_seed"])
        trainer_count = int(row["trainer_count"])
        mode = str(row["mode"])
        root = _root_indices(labels, seed)
        pool = tuple(index for index in range(len(labels)) if index not in set(root))
        partitions, probability_hash = _e2_partition_indices(
            labels,
            pool,
            tuple(int(value) for value in row["target_capacities"]),
            formal_seed=seed,
            trainer_count=trainer_count,
            mode=mode,
            alpha=None if row["dirichlet_alpha"] is None else float(row["dirichlet_alpha"]),
        )
        summaries, aggregate = _summarize_partitions(partitions, labels, pool)
        if summaries != row["trainer_partitions"]:
            raise ValueError(f"trainer partition identity drift: {(seed, trainer_count, mode)}")
        if aggregate != str(row["partition_set_sha256"]):
            raise ValueError(f"partition-set SHA-256 drift: {(seed, trainer_count, mode)}")
        if probability_hash != row["dirichlet_probability_matrix_sha256"]:
            compatibility = probability_compatibility or {}
            key = f"{seed}|{trainer_count}|{mode}"
            compatible = (
                str(compatibility.get("environment", {}).get("numpy")) == np.__version__
                and compatibility.get("frozen_before_formal_units") is True
                and compatibility.get("scientific_partition_identity_changed") is False
                and compatibility.get("allowed_probability_hashes", {}).get(key)
                == probability_hash
            )
            if not compatible:
                raise ValueError(
                    f"Dirichlet probability identity drift: {(seed, trainer_count, mode)}"
                )
        results[(seed, trainer_count, mode)] = tuple(
            tuple(int(value) for value in values) for values in partitions
        )
    if len(results) != 15 or int(lock["partition_set_count"]) != 15:
        raise ValueError("COMP-E2 partition-set cardinality drift")
    return results


def materialize_locked_e2_partitions_for_dataset(
    labels: Sequence[int],
    lock: Mapping[str, Any],
    *,
    dataset_key: str,
    probability_compatibility: Mapping[str, Any] | None = None,
    development_probe: bool = False,
    requested_keys: set[tuple[int, int, str]] | None = None,
) -> dict[tuple[int, int, str], tuple[tuple[int, ...], ...]]:
    """Materialize one dataset from the prospective three-dataset E2 lock.

    The archived CIFAR-10 E0 lock remains an independently verified upstream
    input.  This lock records the exact runtime identities for the added MNIST
    and CIFAR-100 confirmations (and mirrors CIFAR-10 for one uniform runner).
    """

    datasets = lock.get("datasets", {})
    if dataset_key not in datasets:
        raise ValueError(f"dataset {dataset_key!r} absent from three-dataset lock")
    dataset_lock = datasets[dataset_key]
    class_count = int(dataset_lock["class_count"])
    per_class = int(dataset_lock["trusted_root_per_class"])
    expected_histogram = Counter(
        {int(key): int(value) for key, value in dataset_lock["label_histogram"].items()}
    )
    if len(labels) != int(dataset_lock["train_cardinality"]):
        raise ValueError(f"{dataset_key} training cardinality drift")
    if Counter(int(value) for value in labels) != expected_histogram:
        raise ValueError(f"{dataset_key} label histogram drift")
    if _compact_int_sha256(labels) != str(dataset_lock["label_vector_sha256"]):
        raise ValueError(f"{dataset_key} label-vector SHA-256 drift")

    available_rows = {
        (int(row["formal_seed"]), int(row["trainer_count"]), str(row["mode"])): row
        for row in dataset_lock["partition_sets"]
    }
    selected_keys = set(available_rows) if requested_keys is None else set(requested_keys)
    if not selected_keys or not selected_keys.issubset(available_rows):
        raise ValueError(f"{dataset_key} requested partition identity drift")
    needed_seeds = {key[0] for key in selected_keys}
    roots = {
        int(row["formal_seed"]): row
        for row in dataset_lock["trusted_root_sets"]
        if int(row["formal_seed"]) in needed_seeds
    }
    results: dict[tuple[int, int, str], tuple[tuple[int, ...], ...]] = {}
    root_cache: dict[int, tuple[tuple[int, ...], tuple[int, ...]]] = {}
    for seed, root_row in roots.items():
        root = _root_indices(
            labels, seed, class_count=class_count, per_class=per_class
        )
        root_set = set(root)
        pool = tuple(index for index in range(len(labels)) if index not in root_set)
        if (
            len(root) != class_count * per_class
            or _compact_int_sha256(root) != str(root_row["index_sha256"])
            or _compact_int_sha256(pool) != str(root_row["trainer_pool_index_sha256"])
        ):
            raise ValueError(f"{dataset_key} trusted-root identity drift for seed {seed}")
        root_cache[seed] = (root, pool)

    for key in sorted(selected_keys):
        row = available_rows[key]
        seed = int(row["formal_seed"])
        trainer_count = int(row["trainer_count"])
        mode = str(row["mode"])
        _, pool = root_cache[seed]
        if row.get("partition_indices_codec") == "canonical-json-zlib-base64-v1":
            decoded = json.loads(
                zlib.decompress(
                    base64.b64decode(str(row["partition_indices_zlib_base64"]))
                ).decode("utf-8")
            )
            partitions = tuple(
                [int(value) for value in partition] for partition in decoded
            )
            probability_hash = row["dirichlet_probability_matrix_sha256"]
        else:
            partitions, probability_hash = _e2_partition_indices(
                labels,
                pool,
                tuple(int(value) for value in row["target_capacities"]),
                formal_seed=seed,
                trainer_count=trainer_count,
                mode=mode,
                alpha=None if row["dirichlet_alpha"] is None else float(row["dirichlet_alpha"]),
                class_count=class_count,
            )
        summaries, aggregate = _summarize_partitions(
            partitions, labels, pool, class_count=class_count
        )
        if summaries != row["trainer_partitions"]:
            raise ValueError(
                f"{dataset_key} trainer partition identity drift: {(seed, trainer_count, mode)}"
            )
        if aggregate != str(row["partition_set_sha256"]):
            raise ValueError(
                f"{dataset_key} partition-set SHA-256 drift: {(seed, trainer_count, mode)}"
            )
        if probability_hash != row["dirichlet_probability_matrix_sha256"]:
            compatibility = probability_compatibility or {}
            field = (
                "development_probe_probability_hashes"
                if development_probe
                else "allowed_probability_hashes"
            )
            key = f"{seed}|{trainer_count}|{mode}"
            compatible = (
                str(compatibility.get("environment", {}).get("numpy")) == np.__version__
                and compatibility.get("frozen_before_formal_run_02_units") is True
                and compatibility.get("scientific_partition_identity_changed") is False
                and compatibility.get(field, {}).get(dataset_key, {}).get(key)
                == probability_hash
            )
            if not compatible:
                raise ValueError(
                    f"{dataset_key} Dirichlet probability identity drift: "
                    f"{(seed, trainer_count, mode)}"
                )
        results[(seed, trainer_count, mode)] = tuple(
            tuple(int(value) for value in values) for values in partitions
        )
    if len(results) != len(selected_keys):
        raise ValueError(f"{dataset_key} E2 partition-set cardinality drift")
    return results


def materialize_prospective_e2_partition(
    labels: Sequence[int],
    e2_lock: Mapping[str, Any],
    *,
    seed: int,
    trainer_count: int,
    mode: str,
    target_capacities: Sequence[int],
    expected_identity: Mapping[str, Any],
) -> tuple[tuple[tuple[int, ...], ...], dict[str, Any]]:
    """Materialize a preregistered non-formal seed with the frozen E2 algorithm.

    The E2 lock freezes the dataset identity and allowed partition algorithms, while
    expected_identity prospectively freezes the exact derived identities.
    """

    _validate_cifar10_labels(labels, e2_lock)
    candidates = {
        (
            int(row["trainer_count"]),
            str(row["mode"]),
            None if row["dirichlet_alpha"] is None else float(row["dirichlet_alpha"]),
        )
        for row in e2_lock["partition_sets"]
        if int(row["trainer_count"]) == int(trainer_count) and str(row["mode"]) == str(mode)
    }
    if len(candidates) != 1:
        raise ValueError("prospective partition algorithm is absent or ambiguous in E2 lock")
    _, _, alpha = next(iter(candidates))
    capacities = tuple(int(value) for value in target_capacities)
    if len(capacities) != int(trainer_count) or any(value <= 0 for value in capacities):
        raise ValueError("invalid prospective target capacities")
    root = _root_indices(labels, int(seed))
    root_set = set(root)
    pool = tuple(index for index in range(len(labels)) if index not in root_set)
    if sum(capacities) != len(pool):
        raise ValueError("prospective target capacities do not cover the trainer pool")
    partitions, probability_hash = _e2_partition_indices(
        labels,
        pool,
        capacities,
        formal_seed=int(seed),
        trainer_count=int(trainer_count),
        mode=str(mode),
        alpha=alpha,
    )
    summaries, aggregate = _summarize_partitions(partitions, labels, pool)
    identity = {
        "seed": int(seed),
        "trainer_count": int(trainer_count),
        "mode": str(mode),
        "dirichlet_alpha": alpha,
        "target_capacities": list(capacities),
        "trusted_root_index_sha256": _compact_int_sha256(root),
        "trainer_pool_index_sha256": _compact_int_sha256(pool),
        "trainer_partitions": summaries,
        "partition_set_sha256": aggregate,
        "dirichlet_probability_matrix_sha256": probability_hash,
    }
    if identity != dict(expected_identity):
        raise ValueError(
            f"prospective partition identity drift: {(int(seed), int(trainer_count), str(mode))}"
        )
    return (
        tuple(tuple(int(value) for value in values) for values in partitions),
        identity,
    )


def prospective_target_capacities(
    contribution_weights: Sequence[float], total: int
) -> tuple[int, ...]:
    """Expose the frozen E2 capacity apportionment for prospective diagnostic seeds."""

    return _target_capacities(contribution_weights, int(total))


class IndexedSubset:
    """A deterministic subset that preserves the original dataset index."""

    def __init__(self, dataset: Any, indices: Sequence[int]) -> None:
        self.dataset = dataset
        self.indices = tuple(int(value) for value in indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int) -> tuple[Any, Any, int]:
        index = self.indices[position]
        data, target = self.dataset[index]
        return data, target, index


__all__ = [
    "IndexedSubset",
    "materialize_locked_e2_partitions",
    "materialize_locked_e2_partitions_for_dataset",
    "materialize_prospective_e2_partition",
    "prospective_target_capacities",
]
