"""Canonical statistics for verifier-incentive evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import NormalDist
from typing import Mapping, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class BinomialInterval:
    numerator: int
    denominator: int
    point_estimate: float
    lower: float
    upper: float
    confidence: float


@dataclass(frozen=True)
class PairedBootstrapInterval:
    denominator: int
    point_estimate: float
    lower: float
    upper: float
    confidence: float
    seed: int
    resamples: int


@dataclass(frozen=True)
class GroupedAUCInterval:
    point_estimate: float
    lower: float
    upper: float
    confidence: float
    group_count: int
    bootstrap_resamples: int
    seed: int
    train_groups: tuple[str, ...]
    test_groups: tuple[str, ...]


@dataclass(frozen=True)
class ClusterBootstrapInterval:
    numerator: int
    denominator: int
    point_estimate: float
    lower: float
    upper: float
    confidence: float
    cluster_count: int
    seed: int
    resamples: int


def wilson_interval(
    successes: int,
    total: int,
    *,
    confidence: float = 0.95,
) -> BinomialInterval:
    if total <= 0:
        raise ValueError("binomial denominator must be positive")
    if successes < 0 or successes > total:
        raise ValueError("binomial numerator must lie in [0, denominator]")
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie in (0, 1)")
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    estimate = successes / total
    denominator = 1.0 + z * z / total
    center = (estimate + z * z / (2.0 * total)) / denominator
    half_width = (
        z
        * math.sqrt(
            estimate * (1.0 - estimate) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return BinomialInterval(
        numerator=int(successes),
        denominator=int(total),
        point_estimate=float(estimate),
        lower=float(max(0.0, center - half_width)),
        upper=float(min(1.0, center + half_width)),
        confidence=float(confidence),
    )


def paired_bootstrap_interval(
    baseline: Mapping[str, float],
    candidate: Mapping[str, float],
    *,
    seed: int,
    resamples: int,
    confidence: float = 0.95,
) -> PairedBootstrapInterval:
    baseline_keys = set(baseline)
    candidate_keys = set(candidate)
    if not baseline_keys or baseline_keys != candidate_keys:
        raise ValueError("paired scenario keys must be equal and non-empty")
    if resamples <= 0 or seed < 0:
        raise ValueError("bootstrap seed and resamples are invalid")
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie in (0, 1)")
    keys = tuple(sorted(baseline_keys))
    differences = np.asarray(
        [float(candidate[key]) - float(baseline[key]) for key in keys],
        dtype=float,
    )
    if np.any(~np.isfinite(differences)):
        raise ValueError("paired values must be finite")
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(keys), size=(resamples, len(keys)))
    estimates = differences[sampled].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(estimates, [tail, 1.0 - tail])
    point = float(np.mean(differences))
    return PairedBootstrapInterval(
        denominator=len(keys),
        point_estimate=point,
        lower=float(min(lower, point)),
        upper=float(max(upper, point)),
        confidence=float(confidence),
        seed=int(seed),
        resamples=int(resamples),
    )


def clustered_binary_interval(
    outcomes: Sequence[bool | int],
    clusters: Sequence[str],
    *,
    seed: int,
    resamples: int,
    confidence: float = 0.95,
) -> ClusterBootstrapInterval:
    """Cluster-resampled interval for correlated verifier-block outcomes."""

    values = np.asarray(tuple(int(bool(value)) for value in outcomes), dtype=int)
    group_array = np.asarray(tuple(str(value) for value in clusters), dtype=object)
    if len(values) == 0 or len(values) != len(group_array):
        raise ValueError("clustered binary inputs must have equal non-zero rows")
    if resamples <= 0 or seed < 0 or not 0 < confidence < 1:
        raise ValueError("invalid clustered binary interval settings")
    unique = tuple(sorted(set(group_array.tolist())))
    if len(unique) < 2:
        raise ValueError("clustered binary interval requires at least two clusters")
    successes = np.asarray(
        [int(values[group_array == group].sum()) for group in unique], dtype=float
    )
    totals = np.asarray(
        [int((group_array == group).sum()) for group in unique], dtype=float
    )
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(unique), size=(resamples, len(unique)))
    sampled_successes = successes[sampled].sum(axis=1)
    sampled_totals = totals[sampled].sum(axis=1)
    estimates = sampled_successes / sampled_totals
    point = float(values.mean())
    tail = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(estimates, [tail, 1.0 - tail])
    return ClusterBootstrapInterval(
        numerator=int(values.sum()),
        denominator=len(values),
        point_estimate=point,
        lower=float(min(lower, point)),
        upper=float(max(upper, point)),
        confidence=float(confidence),
        cluster_count=len(unique),
        seed=int(seed),
        resamples=int(resamples),
    )


def clustered_paired_bootstrap_interval(
    baseline: Mapping[str, float],
    candidate: Mapping[str, float],
    row_clusters: Mapping[str, str],
    *,
    seed: int,
    resamples: int,
    confidence: float = 0.95,
) -> PairedBootstrapInterval:
    """Pair rows first, then bootstrap means over independent block clusters."""

    keys = set(baseline)
    if not keys or keys != set(candidate) or keys != set(row_clusters):
        raise ValueError("clustered paired inputs must share equal non-empty keys")
    by_cluster: dict[str, list[float]] = {}
    for key in sorted(keys):
        difference = float(candidate[key]) - float(baseline[key])
        if not math.isfinite(difference):
            raise ValueError("paired differences must be finite")
        by_cluster.setdefault(str(row_clusters[key]), []).append(difference)
    if len(by_cluster) < 2:
        raise ValueError("clustered paired bootstrap requires at least two clusters")
    cluster_values = {
        cluster: float(np.mean(values))
        for cluster, values in sorted(by_cluster.items())
    }
    zeros = {cluster: 0.0 for cluster in cluster_values}
    return paired_bootstrap_interval(
        zeros,
        cluster_values,
        seed=seed,
        resamples=resamples,
        confidence=confidence,
    )


def grouped_auc_interval(
    features: Sequence[Sequence[float]],
    labels: Sequence[int],
    groups: Sequence[str],
    *,
    seed: int,
    bootstrap_resamples: int,
    confidence: float = 0.95,
) -> GroupedAUCInterval:
    x = np.asarray(features, dtype=float)
    y = np.asarray(labels, dtype=int)
    group_array = np.asarray(tuple(str(value) for value in groups), dtype=object)
    if x.ndim != 2 or len(x) == 0 or len(x) != len(y) or len(y) != len(group_array):
        raise ValueError("grouped AUC inputs must have equal non-zero rows")
    if np.any(~np.isfinite(x)) or set(np.unique(y)) != {0, 1}:
        raise ValueError("grouped AUC requires finite features and both labels")
    unique_groups = tuple(sorted(set(group_array.tolist())))
    if len(unique_groups) < 2:
        raise ValueError("grouped AUC requires at least two groups")
    if bootstrap_resamples <= 0 or seed < 0 or not 0 < confidence < 1:
        raise ValueError("invalid grouped AUC interval settings")
    n_splits = min(5, len(unique_groups))
    splitter = GroupKFold(n_splits=n_splits)
    scores = np.full(len(y), np.nan, dtype=float)
    first_train_groups: tuple[str, ...] = ()
    first_test_groups: tuple[str, ...] = ()
    for fold, (train_index, test_index) in enumerate(
        splitter.split(x, y, group_array)
    ):
        if set(np.unique(y[train_index])) != {0, 1}:
            raise ValueError("each grouped-CV training fold must contain both labels")
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(random_state=seed, solver="liblinear"),
        )
        model.fit(x[train_index], y[train_index])
        scores[test_index] = model.predict_proba(x[test_index])[:, 1]
        if fold == 0:
            first_train_groups = tuple(sorted(set(group_array[train_index].tolist())))
            first_test_groups = tuple(sorted(set(group_array[test_index].tolist())))
    if np.any(~np.isfinite(scores)):
        raise RuntimeError("grouped-CV did not score every row")
    point = float(roc_auc_score(y, scores))
    rng = np.random.default_rng(seed)
    estimates: list[float] = []
    group_indices = {
        group: np.flatnonzero(group_array == group) for group in unique_groups
    }
    for _ in range(bootstrap_resamples):
        sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        sampled_indices = np.concatenate(
            [group_indices[str(group)] for group in sampled_groups]
        )
        sampled_labels = y[sampled_indices]
        if len(np.unique(sampled_labels)) < 2:
            continue
        estimates.append(float(roc_auc_score(sampled_labels, scores[sampled_indices])))
    if not estimates:
        raise ValueError("grouped bootstrap produced no two-class resamples")
    tail = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(np.asarray(estimates), [tail, 1.0 - tail])
    return GroupedAUCInterval(
        point_estimate=point,
        lower=float(min(lower, point)),
        upper=float(max(upper, point)),
        confidence=float(confidence),
        group_count=len(unique_groups),
        bootstrap_resamples=bootstrap_resamples,
        seed=seed,
        train_groups=first_train_groups,
        test_groups=first_test_groups,
    )
