"""Deterministic weighted partitions shared by every SEVC experiment."""

from __future__ import annotations

import hashlib
import json
import random
from typing import Any, Sequence

import numpy as np

from .datasets import DatasetBundle


def _normalized_weights(weights: Sequence[float]) -> tuple[float, ...]:
    values = tuple(float(value) for value in weights)
    if not values or any(not np.isfinite(value) or value <= 0 for value in values):
        raise ValueError("partition weights must be finite and positive")
    total = sum(values)
    return tuple(value / total for value in values)


def partition_indices(
    sample_count: int, weights: Sequence[float], seed: int
) -> tuple[np.ndarray, ...]:
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    normalized = _normalized_weights(weights)
    rng = np.random.RandomState(seed)
    permutation = rng.permutation(sample_count)
    raw_sizes = np.asarray(normalized) * sample_count
    sizes = np.floor(raw_sizes).astype(int)
    remainder = sample_count - int(sizes.sum())
    if remainder:
        order = np.argsort(-(raw_sizes - sizes), kind="stable")
        sizes[order[:remainder]] += 1
    boundaries = np.cumsum(sizes)
    return tuple(np.asarray(chunk, dtype=np.int64) for chunk in np.split(permutation, boundaries[:-1]))


def partition_dataset(
    dataset: Any,
    weights: Sequence[float],
    *,
    batch_size: int,
    seed: int,
    shuffle_batches: bool = False,
    drop_last: bool = False,
) -> tuple[Any, ...]:
    from torch.utils.data import DataLoader, Subset

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    partitions = partition_indices(len(dataset), weights, seed)
    return tuple(
        DataLoader(
            Subset(dataset, indices.tolist()),
            batch_size=batch_size,
            shuffle=shuffle_batches,
            drop_last=drop_last,
            num_workers=0,
        )
        for indices in partitions
    )


def partition_index_pool(
    dataset_size: int,
    weights: Sequence[float],
    seed: int,
    *,
    sample_limit: int | None = None,
) -> tuple[np.ndarray, ...]:
    """Partition one deterministic fixed-total sample pool without duplication."""

    if dataset_size <= 0:
        raise ValueError("dataset_size must be positive")
    if sample_limit is None:
        pool_size = dataset_size
    else:
        if sample_limit <= 0:
            raise ValueError("sample_limit must be positive")
        pool_size = min(dataset_size, int(sample_limit))
    rng = np.random.RandomState(seed)
    pool = rng.permutation(dataset_size)[:pool_size]
    normalized = _normalized_weights(weights)
    raw_sizes = np.asarray(normalized) * pool_size
    sizes = np.floor(raw_sizes).astype(int)
    remainder = pool_size - int(sizes.sum())
    if remainder:
        order = np.argsort(-(raw_sizes - sizes), kind="stable")
        sizes[order[:remainder]] += 1
    boundaries = np.cumsum(sizes)
    return tuple(
        np.asarray(chunk, dtype=np.int64)
        for chunk in np.split(pool, boundaries[:-1])
    )


def loaders_from_partitions(
    dataset: Any,
    partitions: Sequence[Sequence[int]],
    *,
    batch_size: int,
    drop_last: bool = False,
) -> tuple[Any, ...]:
    from torch.utils.data import DataLoader, Subset

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return tuple(
        DataLoader(
            Subset(dataset, [int(value) for value in indices]),
            batch_size=batch_size,
            shuffle=False,
            drop_last=drop_last,
            num_workers=0,
        )
        for indices in partitions
    )


def dataset_targets(dataset: Any) -> tuple[int, ...]:
    values = getattr(dataset, "targets", getattr(dataset, "labels", None))
    if values is None:
        raise TypeError("dataset does not expose targets or labels")
    if hasattr(values, "tolist"):
        values = values.tolist()
    return tuple(int(value) for value in values)


def _namespace_seed(material: str) -> int:
    return int.from_bytes(hashlib.sha256(material.encode("utf-8")).digest()[:8], "big") % 2147483647


def stratified_subset_indices(
    dataset: Any,
    *,
    per_class: int,
    seed_material: str,
) -> tuple[int, ...]:
    """Choose a class-balanced deterministic subset and return sorted indices."""

    if per_class <= 0:
        raise ValueError("per_class must be positive")
    labels = dataset_targets(dataset)
    classes = sorted(set(labels))
    chosen: list[int] = []
    for label in classes:
        candidates = [index for index, value in enumerate(labels) if value == label]
        if len(candidates) < per_class:
            raise ValueError(f"class {label} has fewer than {per_class} samples")
        random.Random(_namespace_seed(f"{seed_material}|label={label}")).shuffle(
            candidates
        )
        chosen.extend(candidates[:per_class])
    return tuple(sorted(chosen))


def balanced_subset_indices(
    dataset: Any,
    *,
    sample_count: int,
    seed_material: str,
) -> tuple[int, ...]:
    """Choose an approximately class-balanced deterministic subset of exact size."""

    labels = dataset_targets(dataset)
    classes = sorted(set(labels))
    if sample_count <= 0 or sample_count > len(labels):
        raise ValueError("sample_count is outside dataset cardinality")
    if sample_count == len(labels):
        return tuple(range(len(labels)))
    quotient, remainder = divmod(sample_count, len(classes))
    selected: list[int] = []
    for ordinal, label in enumerate(classes):
        take = quotient + (1 if ordinal < remainder else 0)
        if take == 0:
            continue
        candidates = [index for index, value in enumerate(labels) if value == label]
        random.Random(_namespace_seed(f"{seed_material}|label={label}")).shuffle(
            candidates
        )
        selected.extend(candidates[:take])
    return tuple(sorted(selected))


def indices_sha256(indices: Sequence[int]) -> str:
    encoded = json.dumps(
        [int(value) for value in indices], separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def subset_loader(dataset: Any, indices: Sequence[int], *, batch_size: int) -> Any:
    from torch.utils.data import DataLoader, Subset

    return DataLoader(
        Subset(dataset, [int(value) for value in indices]),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )


def make_synthetic_bundle(
    *, seed: int, sample_count: int = 64, class_count: int = 10
) -> DatasetBundle:
    import torch
    from torch.utils.data import TensorDataset

    if sample_count < 8 or class_count < 2:
        raise ValueError("synthetic fixture requires at least 8 samples and 2 classes")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    features = torch.randn(sample_count, 1, 28, 28, generator=generator)
    projection = features[:, 0, :4, :4].sum(dim=(1, 2))
    normalized = (projection - projection.min()) / (
        projection.max() - projection.min() + 1e-12
    )
    labels = torch.clamp((normalized * class_count).long(), max=class_count - 1)
    split = max(4, int(sample_count * 0.75))
    return DatasetBundle(
        train=TensorDataset(features[:split], labels[:split]),
        test=TensorDataset(features[split:], labels[split:]),
        num_classes=class_count,
        input_shape=(1, 28, 28),
        dataset_key="synthetic-mnist-shape",
    )
