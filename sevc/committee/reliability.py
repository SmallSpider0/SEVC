"""Canonical pre-task reliability predictors for capacity-constrained committees."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import statistics
from typing import Mapping, Sequence


@dataclass(frozen=True)
class BetaResponsePrior:
    mean: float
    icc: float
    alpha: float
    beta: float
    fitting_cluster_count: int
    fitting_member_count: int


def fit_beta_response_prior(
    response_clusters: Sequence[Sequence[bool]],
) -> BetaResponsePrior:
    """Fit the exchangeable Bernoulli prior from an independent fitting split."""

    if len(response_clusters) < 2:
        raise ValueError("response prior requires at least two fitting clusters")
    sizes = {len(cluster) for cluster in response_clusters}
    if len(sizes) != 1 or next(iter(sizes)) <= 1:
        raise ValueError("fitting clusters must share a member count above one")
    member_count = next(iter(sizes))
    rates = [sum(bool(value) for value in cluster) / member_count for cluster in response_clusters]
    mean = min(1.0 - 1e-9, max(1e-9, float(statistics.mean(rates))))
    observed_variance = float(statistics.variance(rates))
    bernoulli_variance = mean * (1.0 - mean)
    raw_icc = (
        observed_variance / bernoulli_variance - 1.0 / member_count
    ) / (1.0 - 1.0 / member_count)
    icc = min(0.95, max(1e-6, raw_icc))
    concentration = (1.0 - icc) / icc
    return BetaResponsePrior(
        mean=mean,
        icc=icc,
        alpha=mean * concentration,
        beta=(1.0 - mean) * concentration,
        fitting_cluster_count=len(response_clusters),
        fitting_member_count=member_count,
    )


def beta_prior(mean: float, icc: float) -> BetaResponsePrior:
    if not 0.0 <= mean <= 1.0 or not 0.0 <= icc < 1.0:
        raise ValueError("beta prior parameters are outside their domains")
    clipped = min(1.0 - 1e-9, max(1e-9, float(mean)))
    effective_icc = max(1e-9, float(icc))
    concentration = (1.0 - effective_icc) / effective_icc
    return BetaResponsePrior(
        mean=clipped,
        icc=float(icc),
        alpha=clipped * concentration,
        beta=(1.0 - clipped) * concentration,
        fitting_cluster_count=0,
        fitting_member_count=0,
    )


def _binomial_tail(size: int, threshold: int, probability: float) -> float:
    if threshold <= 0:
        return 1.0
    if threshold > size:
        return 0.0
    return float(
        sum(
            math.comb(size, count)
            * probability**count
            * (1.0 - probability) ** (size - count)
            for count in range(threshold, size + 1)
        )
    )


def _beta_binomial_tail(
    size: int,
    threshold: int,
    alpha: float,
    beta: float,
) -> float:
    if threshold <= 0:
        return 1.0
    if threshold > size:
        return 0.0
    if size == 0:
        return 0.0
    if alpha <= 0.0 or beta <= 0.0:
        raise ValueError("beta-binomial parameters must be positive")
    log_normalizer = math.lgamma(alpha) + math.lgamma(beta) - math.lgamma(alpha + beta)
    total = 0.0
    for count in range(threshold, size + 1):
        log_probability = (
            math.lgamma(size + 1)
            - math.lgamma(count + 1)
            - math.lgamma(size - count + 1)
            + math.lgamma(count + alpha)
            + math.lgamma(size - count + beta)
            - math.lgamma(size + alpha + beta)
            - log_normalizer
        )
        total += math.exp(log_probability)
    return float(min(1.0, max(0.0, total)))


def capacity_independent_probability(
    roster: Sequence[int],
    usage: Sequence[int],
    *,
    capacity: int,
    threshold: int,
    probability: float,
) -> float:
    eligible = [member for member in roster if int(usage[member]) < capacity]
    return _binomial_tail(len(eligible), threshold, min(1.0, max(0.0, probability)))


def posterior_completion_probability(
    roster: Sequence[int],
    usage: Sequence[int],
    observed_responses: Mapping[int, bool],
    prior: BetaResponsePrior,
    *,
    capacity: int,
    threshold: int,
    enforce_capacity: bool,
) -> float:
    eligible = [
        int(member)
        for member in roster
        if not enforce_capacity or int(usage[int(member)]) < capacity
    ]
    known_successes = sum(
        bool(observed_responses[member])
        for member in eligible
        if member in observed_responses
    )
    unknown_count = sum(member not in observed_responses for member in eligible)
    required = threshold - known_successes
    unique_observations = tuple(bool(value) for _, value in sorted(observed_responses.items()))
    alpha = prior.alpha + sum(unique_observations)
    beta = prior.beta + len(unique_observations) - sum(unique_observations)
    if prior.icc <= 1e-6:
        probability = alpha / (alpha + beta)
        result = _binomial_tail(unknown_count, required, probability)
    else:
        result = _beta_binomial_tail(unknown_count, required, alpha, beta)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError("posterior completion probability is invalid")
    return float(result)


def prior_payload(prior: BetaResponsePrior) -> dict[str, float | int]:
    return asdict(prior)


__all__ = [
    "BetaResponsePrior",
    "beta_prior",
    "capacity_independent_probability",
    "fit_beta_response_prior",
    "posterior_completion_probability",
    "prior_payload",
]
