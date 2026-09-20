"""Public pre-lock reliability certificates for bounded paid rosters.

This is a replaceable policy component inside the existing verifier-incentive
assembly.  It owns no runner, behavior generator, settlement, gate, metric, or
reporter.  Every decision is made from public offers and public primary
committees before a behavior map or sealed truth is released.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

from sevc.committee import majority_success_probability
from sevc.core.artifacts import canonical_json_text, sha256_text
from sevc.core.registry import Registry
from sevc.incentives.verifier_protocol import (
    CommitteeAssignment,
    PublicVerifierOffer,
    ReserveAvailabilityTerms,
    ReserveRoster,
)
from sevc.incentives.verifier_reserve_recovery import (
    build_registered_reserve_roster,
)


FIXED_M20_BASELINE_POLICY_KEY = "fixed-m20-baseline-v1"
PUBLIC_PREFIX_HYPERGEOM_MIN_POLICY_KEY = "public-prefix-hypergeom-min-v1"
PUBLIC_PREFIX_HYPERGEOM_BUFFER10_POLICY_KEY = (
    "public-prefix-hypergeom-buffer10-v1"
)
PUBLIC_PREFIX_HYPERGEOM_BUDGET_FEASIBLE_MIN_POLICY_KEY = (
    "public-prefix-hypergeom-budget-feasible-min-v1"
)
MULTI_JOB_CAPACITY_CERTIFICATE_POLICY_KEY = (
    "public-multi-job-capacity-certified-min-v1"
)
ALL_RESPONSE_CAPACITY_CERTIFICATE_POLICY_KEY = "public-all-response-capacity-certified-min-v1"


@dataclass(frozen=True)
class RosterCertificationPolicy:
    """Frozen public certificate parameters for one candidate."""

    policy_key: str
    allowed_caps: tuple[int, ...]
    minimum_certificate_probability: float
    reliability_margin: float
    budget_feasible: bool = False
    all_response_sets: bool = False

    def __post_init__(self) -> None:
        if not self.policy_key or self.policy_key != self.policy_key.strip().lower():
            raise ValueError("roster certification policy key must be normalized")
        if (
            not self.allowed_caps
            or tuple(sorted(set(self.allowed_caps))) != self.allowed_caps
            or any(value <= 0 for value in self.allowed_caps)
        ):
            raise ValueError("allowed paid-roster caps must be positive and increasing")
        if not 0.0 < float(self.minimum_certificate_probability) <= 1.0:
            raise ValueError("certificate probability must lie in (0, 1]")
        if not math.isfinite(float(self.reliability_margin)) or (
            self.reliability_margin < 0.0
        ):
            raise ValueError("certificate reliability margin must be non-negative")


ROSTER_CERTIFICATION_POLICIES: Registry[RosterCertificationPolicy] = Registry(
    "verifier paid-roster certification policy"
)
ROSTER_CERTIFICATION_POLICIES.add(
    ALL_RESPONSE_CAPACITY_CERTIFICATE_POLICY_KEY,
    RosterCertificationPolicy(
        policy_key=ALL_RESPONSE_CAPACITY_CERTIFICATE_POLICY_KEY,
        allowed_caps=(9, 20, 24, 28, 32, 33, 36, 40),
        minimum_certificate_probability=0.925,
        reliability_margin=0.0,
        budget_feasible=True,
        all_response_sets=True,
    ),
)
ROSTER_CERTIFICATION_POLICIES.add(
    FIXED_M20_BASELINE_POLICY_KEY,
    RosterCertificationPolicy(
        policy_key=FIXED_M20_BASELINE_POLICY_KEY,
        allowed_caps=(20,),
        minimum_certificate_probability=0.925,
        reliability_margin=0.0,
    ),
)
ROSTER_CERTIFICATION_POLICIES.add(
    MULTI_JOB_CAPACITY_CERTIFICATE_POLICY_KEY,
    RosterCertificationPolicy(
        policy_key=MULTI_JOB_CAPACITY_CERTIFICATE_POLICY_KEY,
        allowed_caps=(20, 24, 28, 32, 33, 36, 40),
        minimum_certificate_probability=0.925,
        reliability_margin=0.0,
        budget_feasible=True,
    ),
)
ROSTER_CERTIFICATION_POLICIES.add(
    PUBLIC_PREFIX_HYPERGEOM_BUDGET_FEASIBLE_MIN_POLICY_KEY,
    RosterCertificationPolicy(
        policy_key=PUBLIC_PREFIX_HYPERGEOM_BUDGET_FEASIBLE_MIN_POLICY_KEY,
        allowed_caps=(20, 24),
        minimum_certificate_probability=0.925,
        reliability_margin=0.0,
        budget_feasible=True,
    ),
)
ROSTER_CERTIFICATION_POLICIES.add(
    PUBLIC_PREFIX_HYPERGEOM_MIN_POLICY_KEY,
    RosterCertificationPolicy(
        policy_key=PUBLIC_PREFIX_HYPERGEOM_MIN_POLICY_KEY,
        allowed_caps=(20, 24),
        minimum_certificate_probability=0.925,
        reliability_margin=0.0,
    ),
)
ROSTER_CERTIFICATION_POLICIES.add(
    PUBLIC_PREFIX_HYPERGEOM_BUFFER10_POLICY_KEY,
    RosterCertificationPolicy(
        policy_key=PUBLIC_PREFIX_HYPERGEOM_BUFFER10_POLICY_KEY,
        allowed_caps=(20, 24, 33),
        minimum_certificate_probability=0.925,
        reliability_margin=0.001,
    ),
)


def hypergeometric_upper_tail(
    *, population: int, responsive: int, draws: int, threshold: int
) -> float:
    """Return P[X >= threshold] for an exact finite-population draw."""

    n = int(population)
    good = int(responsive)
    sample = int(draws)
    required = int(threshold)
    if n <= 0 or not 0 <= good <= n or not 0 <= sample <= n:
        raise ValueError("invalid hypergeometric population")
    if required <= 0:
        return 1.0
    if required > min(good, sample):
        return 0.0
    denominator = math.comb(n, sample)
    numerator = sum(
        math.comb(good, value) * math.comb(n - good, sample - value)
        for value in range(required, min(good, sample) + 1)
        if 0 <= sample - value <= n - good
    )
    return numerator / denominator


def _prefix_certificate_witness(
    *,
    ordered_reputations: Sequence[float],
    population: int,
    responsive: int,
    reliability_target: float,
    reliability_margin: float,
) -> dict[str, Any]:
    """Find the strongest public sufficient prefix/survivor witness."""

    target_with_margin = float(reliability_target) + float(reliability_margin)
    candidates: list[dict[str, Any]] = []
    for prefix_size in range(1, len(ordered_reputations) + 1):
        prefix = tuple(float(value) for value in ordered_reputations[:prefix_size])
        for survivor_count in range(1, prefix_size + 1, 2):
            # If at least q members of this prefix respond, the first q
            # responsive entries are component-wise no worse than the q
            # least-reputable entries in the prefix.  This makes the majority
            # probability below a public sufficient lower bound.
            worst_reputations = tuple(sorted(prefix)[:survivor_count])
            reliability = float(majority_success_probability(worst_reputations))
            if reliability + 1e-12 < target_with_margin:
                continue
            probability = hypergeometric_upper_tail(
                population=population,
                responsive=responsive,
                draws=prefix_size,
                threshold=survivor_count,
            )
            candidates.append(
                {
                    "prefix_size": prefix_size,
                    "survivor_count": survivor_count,
                    "worst_survivor_reputations": list(worst_reputations),
                    "majority_reliability": reliability,
                    "certificate_probability": probability,
                }
            )
    if not candidates:
        return {
            "prefix_size": None,
            "survivor_count": None,
            "worst_survivor_reputations": [],
            "majority_reliability": 0.0,
            "certificate_probability": 0.0,
        }
    return max(
        candidates,
        key=lambda item: (
            float(item["certificate_probability"]),
            float(item["majority_reliability"]),
            -int(item["prefix_size"]),
            -int(item["survivor_count"]),
        ),
    )


def _budget_feasible_prefix_certificate_witness(
    *,
    ordered_reputations: Sequence[float],
    population: int,
    responsive: int,
    reliability_target: float,
    reliability_margin: float,
    roster_cap: int,
    service_fee: float,
    availability_retainer: float,
    sentinel_generation_cost_per_assignment: float,
    adjudication_reserve_per_assignment: float,
    hard_cap_expenditure: float,
) -> dict[str, Any]:
    """Find the strongest witness satisfying reliability and peak exposure."""

    target_with_margin = float(reliability_target) + float(reliability_margin)
    candidates: list[dict[str, Any]] = []
    for prefix_size in range(1, len(ordered_reputations) + 1):
        prefix = tuple(float(value) for value in ordered_reputations[:prefix_size])
        for survivor_count in range(1, prefix_size + 1, 2):
            worst_reputations = tuple(sorted(prefix)[:survivor_count])
            reliability = float(majority_success_probability(worst_reputations))
            if reliability + 1e-12 < target_with_margin:
                continue
            probability = hypergeometric_upper_tail(
                population=population,
                responsive=responsive,
                draws=prefix_size,
                threshold=survivor_count,
            )
            # Recovery and its hard cap are evaluated independently for each
            # trainer job.  A verifier-level offer may cover every job in the
            # scale level, but each job owns a separate activation/settlement
            # ledger.  Multiplying this bound by the number of jobs would
            # therefore mix independent ledgers and triple-count (or worse)
            # the same posted-price activation.
            peak_exposure = (
                (int(roster_cap) - prefix_size) * float(availability_retainer)
                + prefix_size
                * (
                    float(sentinel_generation_cost_per_assignment)
                    + float(adjudication_reserve_per_assignment)
                )
                + survivor_count * float(service_fee)
            )
            if peak_exposure > float(hard_cap_expenditure) + 1e-12:
                continue
            candidates.append(
                {
                    "prefix_size": prefix_size,
                    "survivor_count": survivor_count,
                    "worst_survivor_reputations": list(worst_reputations),
                    "majority_reliability": reliability,
                    "certificate_probability": probability,
                    "peak_exposure_bound_per_job": peak_exposure,
                    "peak_exposure_bound": peak_exposure,
                    "hard_cap_expenditure": float(hard_cap_expenditure),
                    "budget_margin": (
                        float(hard_cap_expenditure) - peak_exposure
                    ),
                    "budget_feasible": True,
                }
            )
    if not candidates:
        return {
            "prefix_size": None,
            "survivor_count": None,
            "worst_survivor_reputations": [],
            "majority_reliability": 0.0,
            "certificate_probability": 0.0,
            "peak_exposure_bound_per_job": None,
            "peak_exposure_bound": None,
            "hard_cap_expenditure": float(hard_cap_expenditure),
            "budget_margin": None,
            "budget_feasible": False,
        }
    return max(
        candidates,
        key=lambda item: (
            float(item["certificate_probability"]),
            float(item["majority_reliability"]),
            -int(item["prefix_size"]),
            -int(item["survivor_count"]),
        ),
    )


def _public_offer_payload(
    offers: Sequence[PublicVerifierOffer],
) -> list[dict[str, Any]]:
    return [
        {
            "verifier_id": item.verifier_id,
            "reputation": float(item.reputation),
            "capacity": int(item.capacity),
            "accepts_offer": bool(item.accepts_offer),
            "conflict_job_ids": list(item.conflict_job_ids),
        }
        for item in offers
    ]


def resolve_registered_certified_roster(
    *,
    policy_key: str,
    implementation_key: str,
    scenario_id: str,
    parameter_set_id: str,
    offers: Sequence[PublicVerifierOffer],
    primary_committees: Sequence[CommitteeAssignment],
    parameters: Mapping[str, Any],
    order_seed: str,
    availability_terms: ReserveAvailabilityTerms,
    reliability_target: float,
    frozen_lazy_fraction: float,
    frozen_dropout_fraction: float,
    sentinel_generation_cost_per_assignment: float | None = None,
    adjudication_reserve_per_assignment: float | None = None,
    committed_expenditure_multiplier_cap: float | None = None,
) -> tuple[ReserveRoster, dict[str, Any]]:
    """Choose the minimum certified cap and return its already-locked roster."""

    policy = ROSTER_CERTIFICATION_POLICIES.get(policy_key)
    hard_ceiling = int(parameters["reserve_cap"])
    if max(policy.allowed_caps) != hard_ceiling:
        raise ValueError("certificate policy/candidate hard ceiling mismatch")
    if not scenario_id or not parameter_set_id or not order_seed:
        raise ValueError("certificate roster identity is incomplete")
    if not 0.0 <= float(frozen_lazy_fraction) <= 1.0 or not 0.0 <= float(
        frozen_dropout_fraction
    ) <= 1.0:
        raise ValueError("frozen threat fractions must lie in [0, 1]")
    public_offers = tuple(offers)
    if len({item.verifier_id for item in public_offers}) != len(public_offers):
        raise ValueError("certificate public offers must be unique")
    eligible = tuple(
        item for item in public_offers if item.accepts_offer and item.capacity > 0
    )
    if not eligible:
        raise ValueError("certificate requires public eligible offers")
    service_fee_tiers = {float(item.service_fee) for item in eligible}
    if len(service_fee_tiers) != 1:
        raise ValueError("certificate requires the frozen single posted-price tier")
    population = len(eligible)
    lazy_count = round(population * float(frozen_lazy_fraction))
    dropout_count = round(population * float(frozen_dropout_fraction))
    if lazy_count + dropout_count > population:
        raise ValueError("frozen certificate threat exceeds the public pool")
    responsive = population - lazy_count - dropout_count
    target = float(reliability_target)
    if not 0.5 < target <= 1.0:
        raise ValueError("certificate reliability target must lie in (0.5, 1]")
    if policy.budget_feasible and (
        sentinel_generation_cost_per_assignment is None
        or adjudication_reserve_per_assignment is None
        or committed_expenditure_multiplier_cap is None
    ):
        raise ValueError("budget-feasible certificate requires frozen public costs")
    service_fee = float(next(iter(service_fee_tiers)))
    primary_assignment_counts = tuple(
        len(item.verifier_ids) for item in primary_committees
    )
    if not primary_assignment_counts or any(
        value <= 0 for value in primary_assignment_counts
    ):
        raise ValueError(
            "budget certificate requires nonempty primary committees"
        )
    # The certificate is shared across every frozen reliability target.  Use
    # the minimum single-verifier no-recovery denominator so its public peak
    # witness remains valid for the cheapest target as well as larger primary
    # committees.  Runtime enforcement still uses each job's actual frozen
    # no-recovery expenditure and is therefore never weaker than this bound.
    no_recovery_owner_expenditure = (
        service_fee
        + float(sentinel_generation_cost_per_assignment or 0.0)
        + float(adjudication_reserve_per_assignment or 0.0)
    )
    hard_cap_expenditure = (
        no_recovery_owner_expenditure
        * float(committed_expenditure_multiplier_cap or 0.0)
    )

    public_input = {
        "schema_version": "sevc-public-roster-certificate-input-v1",
        "policy_key": policy.policy_key,
        "implementation_key": implementation_key,
        "parameter_set_id": parameter_set_id,
        "scenario_id": scenario_id,
        "allowed_caps": list(policy.allowed_caps),
        "minimum_certificate_probability": float(
            policy.minimum_certificate_probability
        ),
        "reliability_target": target,
        "reliability_margin": float(policy.reliability_margin),
        "frozen_lazy_fraction": float(frozen_lazy_fraction),
        "frozen_dropout_fraction": float(frozen_dropout_fraction),
        "lazy_count": lazy_count,
        "dropout_count": dropout_count,
        "responsive_count": responsive,
        "block_level_correlated_abstention": True,
        "budget_feasible_required": bool(policy.budget_feasible),
        "service_fee": service_fee,
        "availability_retainer": float(availability_terms.availability_retainer),
        "sentinel_generation_cost_per_assignment": (
            sentinel_generation_cost_per_assignment
        ),
        "adjudication_reserve_per_assignment": adjudication_reserve_per_assignment,
        "committed_expenditure_multiplier_cap": (
            committed_expenditure_multiplier_cap
        ),
        "no_recovery_owner_expenditure": no_recovery_owner_expenditure,
        "hard_cap_expenditure": hard_cap_expenditure,
        "primary_assignment_counts_per_job": list(primary_assignment_counts),
        "budget_denominator_rule": (
            "minimum-single-verifier-no-recovery-expenditure"
        ),
        "budget_ledger_scope": "per-job-verifier-level",
        "eligible_public_offers": _public_offer_payload(eligible),
        "primary_committees": [
            {
                "job_id": item.job_id,
                "verifier_ids": list(item.verifier_ids),
            }
            for item in primary_committees
        ],
        "order_seed_sha256": sha256_text(order_seed),
    }
    public_input_sha256 = sha256_text(canonical_json_text(public_input))
    evaluated: list[dict[str, Any]] = []
    rosters: dict[int, ReserveRoster] = {}
    for cap in policy.allowed_caps:
        effective_parameters = {**dict(parameters), "reserve_cap": int(cap)}
        roster = build_registered_reserve_roster(
            implementation_key=implementation_key,
            scenario_id=scenario_id,
            parameter_set_id=parameter_set_id,
            offers=public_offers,
            primary_committees=primary_committees,
            parameters=effective_parameters,
            order_seed=order_seed,
            availability_terms=availability_terms,
        )
        rosters[int(cap)] = roster
        if policy.budget_feasible:
            witness = _budget_feasible_prefix_certificate_witness(
                ordered_reputations=[item.reputation for item in roster.entries],
                population=population,
                responsive=responsive,
                reliability_target=target,
                reliability_margin=float(policy.reliability_margin),
                roster_cap=int(cap),
                service_fee=service_fee,
                availability_retainer=float(
                    availability_terms.availability_retainer
                ),
                sentinel_generation_cost_per_assignment=float(
                    sentinel_generation_cost_per_assignment
                ),
                adjudication_reserve_per_assignment=float(
                    adjudication_reserve_per_assignment
                ),
                hard_cap_expenditure=hard_cap_expenditure,
            )
        else:
            witness = _prefix_certificate_witness(
                ordered_reputations=[item.reputation for item in roster.entries],
                population=population,
                responsive=responsive,
                reliability_target=target,
                reliability_margin=float(policy.reliability_margin),
            )
        full_paid_roster = len(roster.entries) == min(population, int(cap))
        probability = float(witness["certificate_probability"])
        passed = bool(
            full_paid_roster
            and probability + 1e-12
            >= float(policy.minimum_certificate_probability)
        )
        evaluated.append(
            {
                "cap": int(cap),
                "full_paid_roster": full_paid_roster,
                "ordered_entries": [
                    [item.verifier_id, float(item.reputation)]
                    for item in roster.entries
                ],
                "roster_sha256": roster.roster_sha256,
                "witness": witness,
                "certificate_probability_margin": probability
                - float(policy.minimum_certificate_probability),
                "certificate_reliability_margin": float(
                    witness["majority_reliability"]
                )
                - target,
                "passed": passed,
            }
        )
    required_row = next((item for item in evaluated if item["passed"]), None)
    required_m = int(required_row["cap"]) if required_row is not None else None
    chosen_m = required_m if required_m is not None else int(policy.allowed_caps[-1])
    chosen_row = next(item for item in evaluated if int(item["cap"]) == chosen_m)
    chosen_roster = rosters[chosen_m]
    certificate_record = {
        "schema_version": "sevc-public-roster-reliability-certificate-v1",
        "policy_key": policy.policy_key,
        "public_only": True,
        "pre_behavior_truth_lock": True,
        "block_level_correlated_abstention": True,
        "public_input": public_input,
        "public_input_sha256": public_input_sha256,
        "candidate_hard_ceiling_m": hard_ceiling,
        "allowed_caps": list(policy.allowed_caps),
        "certificate_required_m": required_m,
        "chosen_m": chosen_m,
        "certificate_confidence": float(
            policy.minimum_certificate_probability
        ),
        "certificate_margin": float(policy.reliability_margin),
        "certificate_probability": float(
            chosen_row["witness"]["certificate_probability"]
        ),
        "certificate_probability_margin": float(
            chosen_row["certificate_probability_margin"]
        ),
        "certificate_reliability": float(
            chosen_row["witness"]["majority_reliability"]
        ),
        "certificate_reliability_margin": float(
            chosen_row["certificate_reliability_margin"]
        ),
        "certificate_witness_prefix_size": chosen_row["witness"][
            "prefix_size"
        ],
        "certificate_witness_survivor_count": chosen_row["witness"][
            "survivor_count"
        ],
        "frozen_lazy_fraction": float(frozen_lazy_fraction),
        "frozen_dropout_fraction": float(frozen_dropout_fraction),
        "responsive_count": responsive,
        "evaluated_caps": evaluated,
        "chosen_roster_sha256": chosen_roster.roster_sha256,
        "certificate_passed": required_row is not None,
        "budget_feasible_required": bool(policy.budget_feasible),
        "certificate_budget_feasible": bool(
            chosen_row["witness"].get("budget_feasible", not policy.budget_feasible)
        ),
        "certificate_peak_exposure_bound": chosen_row["witness"].get(
            "peak_exposure_bound"
        ),
        "certificate_hard_cap_expenditure": chosen_row["witness"].get(
            "hard_cap_expenditure"
        ),
        "certificate_budget_margin": chosen_row["witness"].get("budget_margin"),
    }
    certificate_record["certificate_lock_sha256"] = sha256_text(
        canonical_json_text(certificate_record)
    )
    return chosen_roster, certificate_record


def _responsive_roster_lower_bound(
    *,
    population: int,
    responsive_population: int,
    roster_size: int,
    minimum_probability: float,
) -> tuple[int, float]:
    """Return the largest exact hypergeometric lower confidence bound."""

    passing = tuple(
        (value, hypergeometric_upper_tail(
            population=population,
            responsive=responsive_population,
            draws=roster_size,
            threshold=value,
        ))
        for value in range(roster_size + 1)
    )
    eligible = tuple(
        item for item in passing if item[1] + 1e-12 >= minimum_probability
    )
    return max(eligible, key=lambda item: item[0])


def _multi_job_hall_witness(
    *,
    jobs: Sequence[Any],
    requirements: Mapping[str, int],
    responsive_entries: Sequence[Any],
) -> dict[str, Any]:
    """Enumerate every Hall/b-matching job subset exactly."""

    from itertools import combinations

    rows: list[dict[str, Any]] = []
    ordered_jobs = tuple(sorted(jobs, key=lambda item: item.job_id))
    for subset_size in range(1, len(ordered_jobs) + 1):
        for subset in combinations(ordered_jobs, subset_size):
            job_ids = tuple(item.job_id for item in subset)
            supply = sum(
                min(
                    int(entry.capacity),
                    sum(
                        job_id not in set(entry.conflict_job_ids)
                        for job_id in job_ids
                    ),
                )
                for entry in responsive_entries
            )
            demand = sum(int(requirements[job_id]) for job_id in job_ids)
            rows.append(
                {
                    "job_ids": list(job_ids),
                    "supply": int(supply),
                    "demand": int(demand),
                    "margin": int(supply - demand),
                }
            )
    minimum_margin = min((int(row["margin"]) for row in rows), default=0)
    tight = next(
        (row for row in rows if int(row["margin"]) == minimum_margin),
        None,
    )
    return {
        "all_job_subsets_enumerated": True,
        "subset_count": len(rows),
        "minimum_margin": minimum_margin,
        "tight_subset": tight,
        "subsets": rows,
        "passed": minimum_margin >= 0,
    }


def all_response_capacity_witness(
    *, jobs: Sequence[Any], requirements: Mapping[str, int],
    entries: Sequence[Any], responsive_count: int,
) -> dict[str, Any]:
    """Exact Hall test for every responsive subset of a fixed cardinality.

    For each job subset B, each member contributes min(capacity, neighbors in
    B). Its smallest L contributions give the minimum over all L-member U.
    The minimizing U may differ between B; no single sorted U is sufficient.
    This certifies a static assignment, not an online scheduling policy.
    """
    from itertools import combinations

    if not 0 <= responsive_count <= len(entries):
        raise ValueError("responsive count is outside the roster")
    if len({e.verifier_id for e in entries}) != len(entries):
        raise ValueError("roster identities must be unique")
    job_ids = tuple(sorted(j.job_id for j in jobs))
    if not job_ids or len(set(job_ids)) != len(job_ids) or set(requirements) != set(job_ids):
        raise ValueError("job identities and requirements must match")
    if any(int(v) != v or v < 0 for v in requirements.values()):
        raise ValueError("job demands must be nonnegative integers")
    if any(int(e.capacity) != e.capacity or e.capacity < 0 for e in entries):
        raise ValueError("capacities must be nonnegative integers")
    rows = []
    for count in range(1, len(job_ids) + 1):
        for subset in combinations(job_ids, count):
            contributions = sorted(
                (min(int(e.capacity), sum(j not in e.conflict_job_ids for j in subset)), e.verifier_id)
                for e in entries
            )[:responsive_count]
            supply = sum(value for value, _ in contributions)
            demand = sum(int(requirements[j]) for j in subset)
            rows.append({"job_ids": list(subset), "supply": supply, "demand": demand,
                         "margin": supply - demand,
                         "worst_responsive_ids": [v for _, v in contributions]})
    tight = min(rows, key=lambda r: r["margin"])
    return {"all_job_subsets_enumerated": True, "all_response_sets_covered": True,
            "response_domain": "every subset of exactly L roster members",
            "responsive_count": responsive_count,
            "response_set_count": math.comb(len(entries), responsive_count),
            "subset_count": len(rows), "minimum_margin": tight["margin"],
            "tight_subset": tight, "subsets": rows, "passed": tight["margin"] >= 0,
            "scope": "static-capacity-feasibility; executed scheduling checked separately"}


def _minimum_capacity_reliability_witness(
    *, ordered_reputations: Sequence[float], reliability_target: float
) -> dict[str, Any]:
    """Return the minimum odd survivor demand valid for any roster subset."""

    ordered = tuple(sorted(float(value) for value in ordered_reputations))
    for survivor_count in range(1, len(ordered) + 1, 2):
        worst = ordered[:survivor_count]
        reliability = float(majority_success_probability(worst))
        if reliability + 1e-12 >= float(reliability_target):
            return {
                "survivor_count": survivor_count,
                "worst_survivor_reputations": list(worst),
                "majority_reliability": reliability,
                "reliability_target": float(reliability_target),
                "margin": reliability - float(reliability_target),
                "passed": True,
            }
    return {
        "survivor_count": None,
        "worst_survivor_reputations": list(ordered),
        "majority_reliability": 0.0,
        "reliability_target": float(reliability_target),
        "margin": -float(reliability_target),
        "passed": False,
    }


def certify_registered_multi_job_capacity(
    *,
    policy_key: str,
    implementation_key: str,
    scenario_id: str,
    parameter_set_id: str,
    offers: Sequence[PublicVerifierOffer],
    primary_committees: Sequence[CommitteeAssignment],
    jobs: Sequence[Any],
    parameters: Mapping[str, Any],
    order_seed: str,
    availability_terms: ReserveAvailabilityTerms,
    m_candidates: Sequence[int],
    minimum_certificate_probability: float,
    frozen_lazy_fraction: float,
    frozen_dropout_fraction: float,
    sentinel_generation_cost_per_assignment: float,
    adjudication_reserve_per_assignment: float,
    fixed_roster: ReserveRoster | None = None,
) -> tuple[ReserveRoster, dict[str, Any]]:
    """Choose the smallest public reliability/capacity/cost-certified roster.

    Only offer messages, job requirements, primary identities, frozen threat
    bounds, and posted SARE terms enter this function.  Behavior, truth,
    reports, private costs, and observed development outcomes are absent from
    both the API and the canonical certificate payload.
    """

    policy = ROSTER_CERTIFICATION_POLICIES.get(policy_key)
    if policy.policy_key not in {MULTI_JOB_CAPACITY_CERTIFICATE_POLICY_KEY,
                               ALL_RESPONSE_CAPACITY_CERTIFICATE_POLICY_KEY}:
        raise ValueError("multi-job certificate requires its registered policy")
    probability = float(minimum_certificate_probability)
    if not 0.0 < probability <= 1.0:
        raise ValueError("minimum certificate probability must lie in (0, 1]")
    offer_rows = tuple(offers)
    job_rows = tuple(sorted(jobs, key=lambda item: item.job_id))
    if not offer_rows or not job_rows:
        raise ValueError("multi-job certificate requires offers and jobs")
    eligible = tuple(
        item for item in offer_rows if item.accepts_offer and item.capacity > 0
    )
    population = len(eligible)
    hard_ceiling = min(population, int(parameters["reserve_cap"]))
    candidates = tuple(
        sorted(
            {
                min(population, int(value))
                for value in m_candidates
                if int(value) > 0 and min(population, int(value)) <= hard_ceiling
            }
        )
    )
    if not candidates or candidates[-1] != hard_ceiling:
        raise ValueError("multi-job candidate scan must include the hard ceiling")
    lazy_count = round(population * float(frozen_lazy_fraction))
    dropout_count = round(population * float(frozen_dropout_fraction))
    removal_count = int(parameters["removal_budget"])
    unavailable_count = lazy_count + dropout_count + removal_count
    if unavailable_count >= population:
        responsive_population = 0
    else:
        responsive_population = population - unavailable_count
    service_fees = {float(item.service_fee) for item in eligible}
    if len(service_fees) != 1:
        raise ValueError("multi-job certificate requires one public price tier")
    service_fee = next(iter(service_fees))
    primary_counts = {
        item.job_id: len(item.verifier_ids) for item in primary_committees
    }
    evaluated: list[dict[str, Any]] = []
    rosters: dict[int, ReserveRoster] = {}
    for m in candidates:
        roster = fixed_roster or build_registered_reserve_roster(
            implementation_key=implementation_key,
            scenario_id=scenario_id,
            parameter_set_id=parameter_set_id,
            offers=offer_rows,
            primary_committees=primary_committees,
            parameters={**dict(parameters), "reserve_cap": int(m)},
            order_seed=order_seed,
            availability_terms=availability_terms,
        )
        if fixed_roster is not None:
            if (tuple(m_candidates) != (len(roster.entries),)
                    or roster.primary_committees != tuple(primary_committees)
                    or {e.verifier_id for e in roster.entries} != {o.verifier_id for o in offer_rows}):
                raise ValueError("fixed raw roster identity differs from certificate inputs")
        rosters[m] = roster
        roster_size = len(roster.entries)
        responsive_bound, responsive_probability = _responsive_roster_lower_bound(
            population=population,
            responsive_population=responsive_population,
            roster_size=roster_size,
            minimum_probability=probability,
        )
        requirements: dict[str, int] = {}
        reliability_witnesses: dict[str, dict[str, Any]] = {}
        for job in job_rows:
            witness = _minimum_capacity_reliability_witness(
                ordered_reputations=[item.reputation for item in roster.entries],
                reliability_target=(
                    float(job.reliability_target)
                    + float(policy.reliability_margin)
                ),
            )
            requirement = witness.get("survivor_count")
            requirements[job.job_id] = (
                roster_size + 1 if requirement is None else int(requirement)
            )
            reliability_witnesses[job.job_id] = witness
        conservative_responsive = tuple(
            sorted(
                roster.entries,
                key=lambda entry: (
                    int(entry.capacity)
                    * sum(
                        job.job_id not in set(entry.conflict_job_ids)
                        for job in job_rows
                    ),
                    int(entry.capacity),
                    float(entry.reputation),
                    entry.verifier_id,
                ),
            )[:responsive_bound]
        )
        hall = _multi_job_hall_witness(
            jobs=job_rows,
            requirements=requirements,
            responsive_entries=conservative_responsive,
        )
        total_supply = sum(int(item.capacity) for item in conservative_responsive)
        if policy.all_response_sets:
            hall = all_response_capacity_witness(
                jobs=job_rows, requirements=requirements, entries=roster.entries,
                responsive_count=responsive_bound,
            )
            total_supply = sum(sorted(int(e.capacity) for e in roster.entries)[:responsive_bound])
        total_demand = sum(requirements.values())
        total_capacity_margin = total_supply - total_demand - 1
        maximum_requirement = max(requirements.values(), default=0)
        reliability_passed = all(
            int(value) <= roster_size for value in requirements.values()
        )
        responsive_passed = responsive_bound >= maximum_requirement
        largest_requirement = max(requirements.values(), default=0)
        primary_denominator = min(primary_counts.values(), default=1) * (
            service_fee
            + float(sentinel_generation_cost_per_assignment)
            + float(adjudication_reserve_per_assignment)
        )
        peak_exposure_bound = (
            max(0, roster_size - largest_requirement)
            * float(availability_terms.availability_retainer)
            + largest_requirement
            * (
                service_fee
                + float(sentinel_generation_cost_per_assignment)
                + float(adjudication_reserve_per_assignment)
            )
        )
        hard_cap = float(parameters["committed_expenditure_multiplier_cap"]) * (
            primary_denominator
        )
        cost_margin = hard_cap - peak_exposure_bound
        owner_multiplier = 1.0 + (float(roster_size) - 1.5) / 180.0
        cost_passed = cost_margin >= -1e-12 and owner_multiplier <= 1.25 + 1e-12
        passed = bool(
            roster_size == m
            and reliability_passed
            and responsive_passed
            and total_capacity_margin >= 0
            and hall["passed"]
            and cost_passed
        )
        evaluated.append(
            {
                "m": m,
                "full_paid_roster": roster_size == m,
                "roster_sha256": roster.roster_sha256,
                "responsive_lower_bound": responsive_bound,
                "responsive_probability": responsive_probability,
                "job_survivor_requirements": [
                    [key, value] for key, value in sorted(requirements.items())
                ],
                "reliability_witnesses": reliability_witnesses,
                "conservative_responsive_entries": [
                    {
                        "verifier_id": item.verifier_id,
                        "capacity": int(item.capacity),
                        "reputation": float(item.reputation),
                        "conflict_job_ids": list(item.conflict_job_ids),
                    }
                    for item in conservative_responsive
                ],
                "total_responsive_capacity": total_supply,
                "total_capacity_demand_plus_one": total_demand + 1,
                "total_capacity_margin": total_capacity_margin,
                "hall_witness": hall,
                "peak_exposure_bound": peak_exposure_bound,
                "hard_cap_expenditure": hard_cap,
                "cost_margin": cost_margin,
                "honest_owner_multiplier": owner_multiplier,
                "passed": passed,
            }
        )
    chosen = next((item for item in evaluated if item["passed"]), None)
    chosen_m = int(chosen["m"]) if chosen is not None else hard_ceiling
    record = {
        "schema_version": ("sevc-public-all-response-capacity-certificate-v1"
                           if policy.all_response_sets else "sevc-public-multi-job-capacity-certificate-v1"),
        "policy_key": policy.policy_key,
        "implementation_key": implementation_key,
        "scenario_id": scenario_id,
        "parameter_set_id": parameter_set_id,
        "public_only": True,
        "pre_behavior_truth_lock": True,
        "forbidden_input_field_count": 0,
        "minimum_certificate_probability": probability,
        "frozen_lazy_fraction": float(frozen_lazy_fraction),
        "frozen_dropout_fraction": float(frozen_dropout_fraction),
        "frozen_removal_budget": removal_count,
        "responsive_population": responsive_population,
        "m_candidates": list(candidates),
        "chosen_m": chosen_m,
        "certificate_passed": chosen is not None,
        "status": "CERTIFIED" if chosen is not None else "INFEASIBLE",
        "evaluated_candidates": evaluated,
        "hall_witness": (
            chosen["hall_witness"]
            if chosen is not None
            else evaluated[-1]["hall_witness"]
        ),
        "cost_margin": (
            chosen["cost_margin"] if chosen is not None else evaluated[-1]["cost_margin"]
        ),
    }
    record["certificate_lock_sha256"] = sha256_text(canonical_json_text(record))
    return rosters[chosen_m], record


__all__ = [
    "FIXED_M20_BASELINE_POLICY_KEY",
    "PUBLIC_PREFIX_HYPERGEOM_BUFFER10_POLICY_KEY",
    "PUBLIC_PREFIX_HYPERGEOM_BUDGET_FEASIBLE_MIN_POLICY_KEY",
    "PUBLIC_PREFIX_HYPERGEOM_MIN_POLICY_KEY",
    "MULTI_JOB_CAPACITY_CERTIFICATE_POLICY_KEY",
    "ROSTER_CERTIFICATION_POLICIES",
    "RosterCertificationPolicy",
    "hypergeometric_upper_tail",
    "certify_registered_multi_job_capacity",
    "resolve_registered_certified_roster",
]
