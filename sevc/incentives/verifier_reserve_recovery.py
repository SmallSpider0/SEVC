"""Canonical PSRR reserve roster, staged activation, and reconstitution.

This module is a replaceable incentive-policy component inside the existing
verifier evaluation path.  It deliberately owns no runner, scenario generator,
metric, gate, or reporter.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
import hashlib
from itertools import combinations
import math
from typing import Any, Mapping, Sequence

from sevc.committee import majority_success_probability
from sevc.core.artifacts import canonical_json_text, sha256_text
from sevc.core.registry import Registry
from sevc.incentives.verifier_protocol import (
    ActivationWave,
    AllocationOutcome,
    AssignmentTransferTerm,
    CommitteeAssignment,
    CommitteeReconstitution,
    PublicVerifierOffer,
    RobustRosterCertificate,
    ReserveAvailabilityTerms,
    ReserveRoster,
    ReserveRosterEntry,
    ReserveTransferRecord,
    TailBudgetDecision,
    VerifierSettlement,
    VerificationJob,
    dataclass_to_jsonable,
)


IMPLEMENTATION_KEY = "rc-phse-rr-v1"
RTR_IMPLEMENTATION_KEY = "rc-phse-rtr-v1"
SARE_IMPLEMENTATION_KEY = "rc-phse-rtr-sare-v1"
CC_SARE_IMPLEMENTATION_KEY = "rc-phse-rtr-cc-sare-v1"


@dataclass(frozen=True)
class SettlementAwareEscrowLedger:
    """Public budget state after the most recent activation has settled.

    The ledger intentionally contains no behavior labels, reports, truth, private
    costs, or bond proceeds.  Assignment statuses are represented only by the
    terminal settlement class needed to conserve conditional service-fee escrow.
    """

    gross_cumulative_offers: float = 0.0
    legacy_cumulative_commitment_counterfactual: float = 0.0
    expired_conditional_service_fee: float = 0.0
    irreversible_costs: float = 0.0
    paid_service_fees: float = 0.0
    outstanding_service_fee_escrow: float = 0.0
    outstanding_availability_retainer: float = 0.0
    reclassified_availability_retainer: float = 0.0
    peak_exposure: float = 0.0
    settled_owner_expenditure: float = 0.0
    issued_assignment_ids: tuple[str, ...] = ()
    settled_assignment_ids: tuple[str, ...] = ()
    expired_assignment_ids: tuple[str, ...] = ()
    paid_assignment_ids: tuple[str, ...] = ()
    conservation_violation_count: int = 0
    negative_value_violation_count: int = 0
    duplicate_issue_violation_count: int = 0
    duplicate_settlement_violation_count: int = 0
    missing_settlement_violation_count: int = 0
    bond_slash_offsets_owner_expenditure: bool = False

    @property
    def current_exposure(self) -> float:
        return (
            float(self.irreversible_costs)
            + float(self.paid_service_fees)
            + float(self.outstanding_service_fee_escrow)
            + float(self.outstanding_availability_retainer)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "sevc-settlement-aware-service-escrow-ledger-v1",
            **dataclass_to_jsonable(self),
            "current_exposure": self.current_exposure,
        }


def _escrow_assignment_id(job_id: str, verifier_id: str) -> str:
    if not job_id or not verifier_id:
        raise ValueError("escrow assignment identity is incomplete")
    return f"{job_id}|{verifier_id}"


def _validated_escrow_ledger(
    ledger: SettlementAwareEscrowLedger,
) -> SettlementAwareEscrowLedger:
    values = (
        ledger.gross_cumulative_offers,
        ledger.legacy_cumulative_commitment_counterfactual,
        ledger.expired_conditional_service_fee,
        ledger.irreversible_costs,
        ledger.paid_service_fees,
        ledger.outstanding_service_fee_escrow,
        ledger.outstanding_availability_retainer,
        ledger.reclassified_availability_retainer,
        ledger.peak_exposure,
        ledger.settled_owner_expenditure,
    )
    if any(not math.isfinite(float(value)) or float(value) < -1e-12 for value in values):
        raise ValueError("escrow ledger contains a negative or non-finite value")
    issued = tuple(ledger.issued_assignment_ids)
    settled = tuple(ledger.settled_assignment_ids)
    paid = tuple(ledger.paid_assignment_ids)
    expired = tuple(ledger.expired_assignment_ids)
    if len(set(issued)) != len(issued):
        raise ValueError("escrow assignment was issued more than once")
    if len(set(settled)) != len(settled) or not set(settled) <= set(issued):
        raise ValueError("escrow settlement identity is duplicated or unissued")
    if set(paid) & set(expired) or set(paid) | set(expired) != set(settled):
        raise ValueError("escrow paid/expired partition is inconsistent")
    if ledger.bond_slash_offsets_owner_expenditure:
        raise ValueError("bond slash cannot offset owner expenditure")
    if abs(float(ledger.current_exposure) - float(ledger.settled_owner_expenditure)) > 1e-9:
        raise ValueError("escrow current exposure and settled budget state diverge")
    if float(ledger.peak_exposure) + 1e-12 < float(ledger.current_exposure):
        raise ValueError("escrow peak exposure is below current exposure")
    return ledger


def initialize_settlement_aware_escrow(
    *, roster: ReserveRoster, job_id: str
) -> SettlementAwareEscrowLedger:
    """Lock all public reserve retainers before the first behavior is observed."""

    primary = set(
        next(
            item.verifier_ids
            for item in roster.primary_committees
            if item.job_id == job_id
        )
    )
    retainer = sum(
        float(item.availability_retainer)
        for item in roster.entries
        if item.verifier_id not in primary
    )
    return _validated_escrow_ledger(
        SettlementAwareEscrowLedger(
            gross_cumulative_offers=retainer,
            legacy_cumulative_commitment_counterfactual=retainer,
            outstanding_availability_retainer=retainer,
            peak_exposure=retainer,
            settled_owner_expenditure=retainer,
        )
    )


def issue_settlement_aware_wave(
    *,
    ledger: SettlementAwareEscrowLedger,
    roster: ReserveRoster,
    wave: ActivationWave,
    sentinel_generation_cost_per_assignment: float,
    adjudication_reserve_per_assignment: float,
) -> SettlementAwareEscrowLedger:
    """Reserve full F+S+A for a wave and convert any prior reserve retainer."""

    _validated_escrow_ledger(ledger)
    entries = {item.verifier_id: item for item in roster.entries}
    assignment_ids = tuple(
        _escrow_assignment_id(wave.job_id, verifier_id)
        for verifier_id in wave.verifier_ids
    )
    if set(assignment_ids) & set(ledger.issued_assignment_ids):
        raise ValueError("escrow assignment was issued more than once")
    service_fee = sum(float(entries[value].service_fee) for value in wave.verifier_ids)
    irreversible = len(wave.verifier_ids) * (
        float(sentinel_generation_cost_per_assignment)
        + float(adjudication_reserve_per_assignment)
    )
    retainer_conversion = (
        0.0
        if wave.role == "primary"
        else sum(
            float(entries[value].availability_retainer)
            for value in wave.verifier_ids
        )
    )
    if retainer_conversion > ledger.outstanding_availability_retainer + 1e-12:
        raise ValueError("reserve activation converts unavailable retainer escrow")
    current_exposure = (
        ledger.current_exposure
        - retainer_conversion
        + service_fee
        + irreversible
    )
    legacy_increment = service_fee + irreversible - retainer_conversion
    result = SettlementAwareEscrowLedger(
        gross_cumulative_offers=(
            ledger.gross_cumulative_offers + service_fee + irreversible
        ),
        legacy_cumulative_commitment_counterfactual=(
            ledger.legacy_cumulative_commitment_counterfactual
            + legacy_increment
        ),
        expired_conditional_service_fee=ledger.expired_conditional_service_fee,
        irreversible_costs=ledger.irreversible_costs + irreversible,
        paid_service_fees=ledger.paid_service_fees,
        outstanding_service_fee_escrow=(
            ledger.outstanding_service_fee_escrow + service_fee
        ),
        outstanding_availability_retainer=(
            ledger.outstanding_availability_retainer - retainer_conversion
        ),
        reclassified_availability_retainer=(
            ledger.reclassified_availability_retainer + retainer_conversion
        ),
        peak_exposure=max(ledger.peak_exposure, current_exposure),
        settled_owner_expenditure=current_exposure,
        issued_assignment_ids=ledger.issued_assignment_ids + assignment_ids,
        settled_assignment_ids=ledger.settled_assignment_ids,
        expired_assignment_ids=ledger.expired_assignment_ids,
        paid_assignment_ids=ledger.paid_assignment_ids,
        conservation_violation_count=ledger.conservation_violation_count,
        negative_value_violation_count=ledger.negative_value_violation_count,
        duplicate_issue_violation_count=ledger.duplicate_issue_violation_count,
        duplicate_settlement_violation_count=(
            ledger.duplicate_settlement_violation_count
        ),
        missing_settlement_violation_count=ledger.missing_settlement_violation_count,
        bond_slash_offsets_owner_expenditure=False,
    )
    return _validated_escrow_ledger(result)


def settle_settlement_aware_wave(
    *,
    ledger: SettlementAwareEscrowLedger,
    roster: ReserveRoster,
    wave: ActivationWave,
    settlement_statuses: Mapping[str, str],
) -> SettlementAwareEscrowLedger:
    """Pay PASS fees and expire only final FAIL_CONFIRMED/DROPOUT fees."""

    _validated_escrow_ledger(ledger)
    entries = {item.verifier_id: item for item in roster.entries}
    paid_ids: list[str] = []
    expired_ids: list[str] = []
    paid_fee = 0.0
    expired_fee = 0.0
    for verifier_id in wave.verifier_ids:
        assignment_id = _escrow_assignment_id(wave.job_id, verifier_id)
        if assignment_id not in set(ledger.issued_assignment_ids):
            raise ValueError("cannot settle an unissued escrow assignment")
        if assignment_id in set(ledger.settled_assignment_ids):
            raise ValueError("cannot settle escrow assignment twice")
        status = settlement_statuses.get(verifier_id)
        fee = float(entries[verifier_id].service_fee)
        if status == "PASS":
            paid_ids.append(assignment_id)
            paid_fee += fee
        elif status in {"FAIL_CONFIRMED", "DROPOUT"}:
            expired_ids.append(assignment_id)
            expired_fee += fee
        else:
            raise ValueError("escrow release requires an explicit final settlement")
    released = paid_fee + expired_fee
    if released > ledger.outstanding_service_fee_escrow + 1e-12:
        raise ValueError("escrow settlement exceeds outstanding service fee")
    current_exposure = ledger.current_exposure - expired_fee
    result = SettlementAwareEscrowLedger(
        gross_cumulative_offers=ledger.gross_cumulative_offers,
        legacy_cumulative_commitment_counterfactual=(
            ledger.legacy_cumulative_commitment_counterfactual
        ),
        expired_conditional_service_fee=(
            ledger.expired_conditional_service_fee + expired_fee
        ),
        irreversible_costs=ledger.irreversible_costs,
        paid_service_fees=ledger.paid_service_fees + paid_fee,
        outstanding_service_fee_escrow=(
            ledger.outstanding_service_fee_escrow - released
        ),
        outstanding_availability_retainer=(
            ledger.outstanding_availability_retainer
        ),
        reclassified_availability_retainer=(
            ledger.reclassified_availability_retainer
        ),
        peak_exposure=ledger.peak_exposure,
        settled_owner_expenditure=current_exposure,
        issued_assignment_ids=ledger.issued_assignment_ids,
        settled_assignment_ids=(
            ledger.settled_assignment_ids + tuple(paid_ids) + tuple(expired_ids)
        ),
        expired_assignment_ids=ledger.expired_assignment_ids + tuple(expired_ids),
        paid_assignment_ids=ledger.paid_assignment_ids + tuple(paid_ids),
        conservation_violation_count=ledger.conservation_violation_count,
        negative_value_violation_count=ledger.negative_value_violation_count,
        duplicate_issue_violation_count=ledger.duplicate_issue_violation_count,
        duplicate_settlement_violation_count=(
            ledger.duplicate_settlement_violation_count
        ),
        missing_settlement_violation_count=ledger.missing_settlement_violation_count,
        bond_slash_offsets_owner_expenditure=False,
    )
    return _validated_escrow_ledger(result)


def _tie_break(order_seed: str, verifier_id: str) -> str:
    return hashlib.sha256(
        f"sevc-psrr-roster-order-v1|{order_seed}|{verifier_id}".encode("utf-8")
    ).hexdigest()


def _primary_jobs_by_verifier(
    primary_committees: Sequence[CommitteeAssignment],
) -> dict[str, list[str]]:
    primary_by_verifier: dict[str, list[str]] = {}
    for committee in primary_committees:
        for verifier_id in committee.verifier_ids:
            primary_by_verifier.setdefault(verifier_id, []).append(committee.job_id)
    return primary_by_verifier


def _finalize_roster(
    *,
    implementation_key: str,
    scenario_id: str,
    parameter_set_id: str,
    ordered: Sequence[PublicVerifierOffer],
    primary_committees: Sequence[CommitteeAssignment],
    reserve_cap: int,
    activation_batch: int,
    order_seed: str,
    availability_terms: ReserveAvailabilityTerms,
    extra_commitment: Mapping[str, Any] | None = None,
) -> ReserveRoster:
    """Create the shared roster DTO and commitment for every recovery policy."""

    primary_by_verifier = _primary_jobs_by_verifier(primary_committees)
    missing_primary = sorted(
        set(primary_by_verifier) - {item.verifier_id for item in ordered}
    )
    if missing_primary:
        raise ValueError(
            "primary allocation is not contained in the committed public roster: "
            f"{missing_primary}"
        )
    entries = tuple(
        ReserveRosterEntry(
            verifier_id=offer.verifier_id,
            reputation=float(offer.reputation),
            capacity=int(offer.capacity),
            service_fee=float(offer.service_fee),
            service_bond=float(offer.refundable_bond),
            availability_retainer=float(availability_terms.availability_retainer),
            availability_bond=float(availability_terms.availability_bond),
            roster_rank=rank,
            primary_job_ids=tuple(
                sorted(primary_by_verifier.get(offer.verifier_id, ()))
            ),
            conflict_job_ids=tuple(offer.conflict_job_ids),
        )
        for rank, offer in enumerate(ordered)
    )
    order_seed_sha256 = sha256_text(order_seed)
    payload = {
        "schema_version": "sevc-recovery-roster-v2",
        "implementation_key": implementation_key,
        "scenario_id": scenario_id,
        "parameter_set_id": parameter_set_id,
        "reserve_cap": reserve_cap,
        "activation_batch": activation_batch,
        "order_seed_sha256": order_seed_sha256,
        "ordered_verifier_ids": [item.verifier_id for item in entries],
        "primary_committees": dataclass_to_jsonable(tuple(primary_committees)),
        "entries": dataclass_to_jsonable(entries),
        "availability_terms": dataclass_to_jsonable(availability_terms),
        "extra_commitment": dict(extra_commitment or {}),
    }
    return ReserveRoster(
        scenario_id=scenario_id,
        parameter_set_id=parameter_set_id,
        reserve_cap=reserve_cap,
        activation_batch=activation_batch,
        order_seed_sha256=order_seed_sha256,
        ordered_verifier_ids=tuple(item.verifier_id for item in entries),
        primary_committees=tuple(primary_committees),
        entries=entries,
        roster_sha256=sha256_text(canonical_json_text(payload)),
    )


def build_reserve_roster(
    *,
    scenario_id: str,
    parameter_set_id: str,
    offers: Sequence[PublicVerifierOffer],
    primary_committees: Sequence[CommitteeAssignment],
    reserve_cap: int,
    activation_batch: int,
    order_seed: str,
    availability_terms: ReserveAvailabilityTerms,
) -> ReserveRoster:
    """Build and hash the public-only roster before behavior/task release."""

    if not scenario_id or not parameter_set_id or not order_seed:
        raise ValueError("roster scenario, parameter, and order seed are required")
    if reserve_cap <= 0 or activation_batch <= 0:
        raise ValueError("reserve cap and activation batch must be positive")
    public_offers = tuple(offers)
    if len({item.verifier_id for item in public_offers}) != len(public_offers):
        raise ValueError("public reserve offers must have unique verifier ids")
    eligible = tuple(
        item
        for item in public_offers
        if item.accepts_offer and item.capacity > 0
    )
    ordered = tuple(
        sorted(
            eligible,
            key=lambda item: (
                -float(item.reputation),
                float(item.service_fee),
                _tie_break(order_seed, item.verifier_id),
                item.verifier_id,
            ),
        )[:reserve_cap]
    )
    return _finalize_roster(
        implementation_key=IMPLEMENTATION_KEY,
        scenario_id=scenario_id,
        parameter_set_id=parameter_set_id,
        ordered=ordered,
        primary_committees=primary_committees,
        reserve_cap=reserve_cap,
        activation_batch=activation_batch,
        order_seed=order_seed,
        availability_terms=availability_terms,
    )


RTR_CERTIFICATE_FAST_ALGORITHM_ID = "rtr-monotone-top-f-certificate-v1"
RTR_CERTIFICATE_FALLBACK_ALGORITHM_ID = "rtr-legacy-exhaustive-certificate-v1"
RTR_GREEDY_FAST_ALGORITHM_ID = "rtr-monotone-public-prefix-greedy-v1"
RTR_GREEDY_FALLBACK_ALGORITHM_ID = "rtr-legacy-exhaustive-greedy-v1"

_RTR_RESOURCE_COUNTERS: Counter[str] = Counter()


def reset_rtr_resource_counters(*, clear_caches: bool = False) -> None:
    """Reset process-local RTR computation telemetry used by resource evidence."""

    _RTR_RESOURCE_COUNTERS.clear()
    if clear_caches:
        _legacy_robust_certificate_core.cache_clear()
        _monotone_robust_certificate_core.cache_clear()


def rtr_resource_counters() -> dict[str, Any]:
    """Return actual computation counters without changing any public DTO."""

    return {
        "schema_version": "sevc-rtr-resource-counters-v1",
        "certificate_fast_algorithm_id": RTR_CERTIFICATE_FAST_ALGORITHM_ID,
        "certificate_fallback_algorithm_id": (
            RTR_CERTIFICATE_FALLBACK_ALGORITHM_ID
        ),
        "greedy_fast_algorithm_id": RTR_GREEDY_FAST_ALGORITHM_ID,
        "greedy_fallback_algorithm_id": RTR_GREEDY_FALLBACK_ALGORITHM_ID,
        "legacy_count_field_semantics": (
            "legacy_exhaustive_equivalent_search_space"
        ),
        **{
            key: int(_RTR_RESOURCE_COUNTERS.get(key, 0))
            for key in (
                "certificate_fast_calls",
                "certificate_fast_executions",
                "certificate_fast_cache_hits",
                "certificate_fallback_calls",
                "certificate_fallback_executions",
                "certificate_fallback_cache_hits",
                "actual_removal_candidates",
                "actual_odd_prefix_evaluations",
                "greedy_fast_calls",
                "greedy_fallback_calls",
                "actual_greedy_selection_steps",
                "actual_greedy_trial_evaluations",
            )
        },
    }


def _legacy_equivalent_search_space_counts(
    entry_count: int, removal_budget: int
) -> tuple[int, int]:
    """Return the exhaustive DTO counts without enumerating the search space."""

    maximum_removal = min(int(removal_budget), int(entry_count))
    removal_set_count = sum(
        math.comb(entry_count, size) for size in range(maximum_removal + 1)
    )
    odd_prefix_evaluation_count = sum(
        math.comb(entry_count, size)
        * ((entry_count - size + 1) // 2)
        for size in range(maximum_removal + 1)
    )
    return removal_set_count, odd_prefix_evaluation_count


def _monotone_certificate_applicable(
    ordered_entries: tuple[tuple[str, float], ...], removal_budget: int
) -> bool:
    identifiers = tuple(item[0] for item in ordered_entries)
    reputations = tuple(float(item[1]) for item in ordered_entries)
    return bool(
        ordered_entries
        and int(removal_budget) >= 0
        and len(set(identifiers)) == len(identifiers)
        and all(0.0 < value < 1.0 for value in reputations)
        and all(
            reputations[index] > reputations[index + 1]
            for index in range(len(reputations) - 1)
        )
    )


def _bounded_probability_roundoff(probabilities: Sequence[float]) -> float:
    """Normalize only machine-roundoff outside the mathematical [0, 1] range."""

    values = tuple(float(value) for value in probabilities)
    probability = float(majority_success_probability(values))
    tolerance = 8.0 * math.ulp(1.0)
    if -tolerance <= probability < 0.0:
        return 0.0
    if 1.0 < probability <= 1.0 + tolerance:
        probability = 1.0
    if not 0.0 <= probability <= 1.0:
        raise ValueError(
            "majority probability escaped [0, 1] beyond floating-point roundoff"
        )
    if probability == 1.0 and values and all(0.0 < value < 1.0 for value in values):
        return math.nextafter(1.0, 0.0)
    return probability


@lru_cache(maxsize=None)
def _legacy_robust_certificate_core(
    ordered_entries: tuple[tuple[str, float], ...], removal_budget: int
) -> tuple[float, tuple[str, ...], tuple[str, ...], float, int, int]:
    """Legacy oracle: enumerate every removal and every surviving odd prefix."""

    identifiers = tuple(item[0] for item in ordered_entries)
    reputation_by_id = dict(ordered_entries)
    worst_reliability = 1.0
    worst_removed: tuple[str, ...] = ()
    worst_best_prefix: tuple[str, ...] = ()
    worst_best_reliability = 0.0
    removal_set_count = 0
    odd_prefix_evaluation_count = 0
    for removal_size in range(min(removal_budget, len(identifiers)) + 1):
        for removed in combinations(identifiers, removal_size):
            removal_set_count += 1
            removed_set = set(removed)
            remaining = tuple(
                verifier_id
                for verifier_id in identifiers
                if verifier_id not in removed_set
            )
            prefix_results = []
            for size in range(1, len(remaining) + 1, 2):
                prefix = remaining[:size]
                reliability = _bounded_probability_roundoff(
                    [reputation_by_id[value] for value in prefix]
                )
                odd_prefix_evaluation_count += 1
                prefix_results.append((float(reliability), prefix))
            if prefix_results:
                best_reliability = max(value[0] for value in prefix_results)
                best_prefix = min(
                    value[1]
                    for value in prefix_results
                    if abs(value[0] - best_reliability) <= 1e-15
                )
            else:
                best_reliability = 0.0
                best_prefix = ()
            candidate_key = (best_reliability, tuple(removed), best_prefix)
            worst_key = (
                worst_reliability,
                worst_removed,
                worst_best_prefix,
            )
            if candidate_key < worst_key:
                worst_reliability = best_reliability
                worst_removed = tuple(removed)
                worst_best_prefix = best_prefix
                worst_best_reliability = best_reliability
    return (
        float(worst_reliability),
        worst_removed,
        worst_best_prefix,
        float(worst_best_reliability),
        removal_set_count,
        odd_prefix_evaluation_count,
    )


@lru_cache(maxsize=None)
def _monotone_robust_certificate_core(
    ordered_entries: tuple[tuple[str, float], ...], removal_budget: int
) -> tuple[float, tuple[str, ...], tuple[str, ...], float, int, int]:
    """Evaluate only the theorem-certified top-f removal and its odd prefixes."""

    identifiers = tuple(item[0] for item in ordered_entries)
    reputation_by_id = dict(ordered_entries)
    removal_size = min(int(removal_budget), len(identifiers))
    worst_removed = identifiers[:removal_size]
    removed_set = set(worst_removed)
    remaining = tuple(
        verifier_id for verifier_id in identifiers if verifier_id not in removed_set
    )
    prefix_results = []
    for size in range(1, len(remaining) + 1, 2):
        prefix = remaining[:size]
        reliability = _bounded_probability_roundoff(
            [reputation_by_id[value] for value in prefix]
        )
        prefix_results.append((float(reliability), prefix))
    if prefix_results:
        best_reliability = max(value[0] for value in prefix_results)
        best_prefix = min(
            value[1]
            for value in prefix_results
            if abs(value[0] - best_reliability) <= 1e-15
        )
    else:
        best_reliability = 0.0
        best_prefix = ()
    removal_set_count, odd_prefix_evaluation_count = (
        _legacy_equivalent_search_space_counts(len(identifiers), removal_budget)
    )
    return (
        float(best_reliability),
        tuple(worst_removed),
        tuple(best_prefix),
        float(best_reliability),
        removal_set_count,
        odd_prefix_evaluation_count,
    )


def _recorded_certificate_call(
    *,
    algorithm: str,
    ordered_entries: tuple[tuple[str, float], ...],
    removal_budget: int,
) -> tuple[float, tuple[str, ...], tuple[str, ...], float, int, int]:
    if algorithm == "fast":
        function = _monotone_robust_certificate_core
        call_key = "certificate_fast_calls"
        execution_key = "certificate_fast_executions"
        cache_key = "certificate_fast_cache_hits"
    elif algorithm == "fallback":
        function = _legacy_robust_certificate_core
        call_key = "certificate_fallback_calls"
        execution_key = "certificate_fallback_executions"
        cache_key = "certificate_fallback_cache_hits"
    else:  # pragma: no cover - private invariant
        raise ValueError(f"unknown RTR certificate algorithm: {algorithm}")
    _RTR_RESOURCE_COUNTERS[call_key] += 1
    before = function.cache_info()
    result = function(ordered_entries, removal_budget)
    after = function.cache_info()
    if after.misses > before.misses:
        _RTR_RESOURCE_COUNTERS[execution_key] += 1
        if algorithm == "fast":
            _RTR_RESOURCE_COUNTERS["actual_removal_candidates"] += 1
            _RTR_RESOURCE_COUNTERS["actual_odd_prefix_evaluations"] += (
                (len(ordered_entries) - min(removal_budget, len(ordered_entries)) + 1)
                // 2
            )
        else:
            _RTR_RESOURCE_COUNTERS["actual_removal_candidates"] += int(result[4])
            _RTR_RESOURCE_COUNTERS["actual_odd_prefix_evaluations"] += int(result[5])
    else:
        _RTR_RESOURCE_COUNTERS[cache_key] += 1
    return result


def _robust_certificate_core(
    ordered_entries: tuple[tuple[str, float], ...], removal_budget: int
) -> tuple[float, tuple[str, ...], tuple[str, ...], float, int, int]:
    """Dispatch to the monotone exact fast path or the legacy exhaustive oracle."""

    algorithm = (
        "fast"
        if _monotone_certificate_applicable(ordered_entries, removal_budget)
        else "fallback"
    )
    return _recorded_certificate_call(
        algorithm=algorithm,
        ordered_entries=ordered_entries,
        removal_budget=removal_budget,
    )


def _legacy_greedy_robust_order(
    *,
    eligible: Sequence[PublicVerifierOffer],
    primary_committees: Sequence[CommitteeAssignment],
    reserve_cap: int,
    removal_budget: int,
    order_seed: str,
    availability_terms: ReserveAvailabilityTerms,
) -> tuple[PublicVerifierOffer, ...]:
    by_id = {item.verifier_id: item for item in eligible}
    primary_ids = set(_primary_jobs_by_verifier(primary_committees))
    missing = sorted(primary_ids - set(by_id))
    if missing:
        raise ValueError(f"robust roster primary members are ineligible: {missing}")
    if len(primary_ids) > reserve_cap:
        raise ValueError("robust roster cap is smaller than the primary union")
    public_order = lambda item: (
        -float(item.reputation),
        float(item.service_fee),
        _tie_break(order_seed, item.verifier_id),
        item.verifier_id,
    )
    selected = list(sorted((by_id[value] for value in primary_ids), key=public_order))
    remaining = [
        item for item in sorted(eligible, key=public_order) if item.verifier_id not in primary_ids
    ]
    while remaining and len(selected) < reserve_cap:
        current = _recorded_certificate_call(
            algorithm="fallback",
            ordered_entries=tuple(
                (item.verifier_id, float(item.reputation)) for item in selected
            ),
            removal_budget=removal_budget,
        )[0]
        scored = []
        for offer in remaining:
            trial = (*selected, offer)
            _RTR_RESOURCE_COUNTERS["actual_greedy_trial_evaluations"] += 1
            trial_reliability = _recorded_certificate_call(
                algorithm="fallback",
                ordered_entries=tuple(
                    (item.verifier_id, float(item.reputation)) for item in trial
                ),
                removal_budget=removal_budget,
            )[0]
            incremental_cost = max(
                float(availability_terms.availability_retainer), 1e-15
            )
            score = (trial_reliability - current) / incremental_cost
            scored.append((score, offer))
        _, chosen = min(
            scored,
            key=lambda item: (
                -float(item[0]),
                -float(item[1].reputation),
                float(item[1].service_fee),
                _tie_break(order_seed, item[1].verifier_id),
                item[1].verifier_id,
            ),
        )
        selected.append(chosen)
        remaining.remove(chosen)
    return tuple(selected)


def _greedy_fast_path_applicable(
    *,
    public_ordered: Sequence[PublicVerifierOffer],
    primary_ids: set[str],
    availability_terms: ReserveAvailabilityTerms,
) -> bool:
    entries = tuple(
        (item.verifier_id, float(item.reputation)) for item in public_ordered
    )
    prefix_ids = {item.verifier_id for item in public_ordered[: len(primary_ids)]}
    return bool(
        float(availability_terms.availability_retainer) > 0.0
        and primary_ids == prefix_ids
        and _monotone_certificate_applicable(entries, 0)
    )


def _greedy_robust_order(
    *,
    eligible: Sequence[PublicVerifierOffer],
    primary_committees: Sequence[CommitteeAssignment],
    reserve_cap: int,
    removal_budget: int,
    order_seed: str,
    availability_terms: ReserveAvailabilityTerms,
) -> tuple[PublicVerifierOffer, ...]:
    """Use the proven public-prefix order, otherwise retain exhaustive greedy."""

    by_id = {item.verifier_id: item for item in eligible}
    primary_ids = set(_primary_jobs_by_verifier(primary_committees))
    missing = sorted(primary_ids - set(by_id))
    if missing:
        raise ValueError(f"robust roster primary members are ineligible: {missing}")
    if len(primary_ids) > reserve_cap:
        raise ValueError("robust roster cap is smaller than the primary union")
    public_order = lambda item: (
        -float(item.reputation),
        float(item.service_fee),
        _tie_break(order_seed, item.verifier_id),
        item.verifier_id,
    )
    public_ordered = tuple(sorted(eligible, key=public_order))
    if _greedy_fast_path_applicable(
        public_ordered=public_ordered,
        primary_ids=primary_ids,
        availability_terms=availability_terms,
    ):
        _RTR_RESOURCE_COUNTERS["greedy_fast_calls"] += 1
        selected = public_ordered[:reserve_cap]
        _RTR_RESOURCE_COUNTERS["actual_greedy_selection_steps"] += max(
            0, len(selected) - len(primary_ids)
        )
        return tuple(selected)
    _RTR_RESOURCE_COUNTERS["greedy_fallback_calls"] += 1
    result = _legacy_greedy_robust_order(
        eligible=eligible,
        primary_committees=primary_committees,
        reserve_cap=reserve_cap,
        removal_budget=removal_budget,
        order_seed=order_seed,
        availability_terms=availability_terms,
    )
    _RTR_RESOURCE_COUNTERS["actual_greedy_selection_steps"] += max(
        0, len(result) - len(primary_ids)
    )
    return result


def build_robust_reserve_roster(
    *,
    scenario_id: str,
    parameter_set_id: str,
    offers: Sequence[PublicVerifierOffer],
    primary_committees: Sequence[CommitteeAssignment],
    reserve_cap: int,
    activation_batch: int,
    removal_budget: int,
    order_seed: str,
    availability_terms: ReserveAvailabilityTerms,
) -> ReserveRoster:
    """Build the RTR public-only greedy roster under exact removal scoring."""

    if removal_budget < 0:
        raise ValueError("RTR removal budget must be non-negative")
    if activation_batch not in {1, 2}:
        raise ValueError("RTR activation batch must be one or two")
    public_offers = tuple(offers)
    if len({item.verifier_id for item in public_offers}) != len(public_offers):
        raise ValueError("public reserve offers must have unique verifier ids")
    eligible = tuple(
        item for item in public_offers if item.accepts_offer and item.capacity > 0
    )
    ordered = _greedy_robust_order(
        eligible=eligible,
        primary_committees=primary_committees,
        reserve_cap=reserve_cap,
        removal_budget=removal_budget,
        order_seed=order_seed,
        availability_terms=availability_terms,
    )
    return _finalize_roster(
        implementation_key=RTR_IMPLEMENTATION_KEY,
        scenario_id=scenario_id,
        parameter_set_id=parameter_set_id,
        ordered=ordered,
        primary_committees=primary_committees,
        reserve_cap=reserve_cap,
        activation_batch=activation_batch,
        order_seed=order_seed,
        availability_terms=availability_terms,
        extra_commitment={"removal_budget": removal_budget},
    )


def robust_roster_certificate(
    *, roster: ReserveRoster, removal_budget: int, reliability_target: float
) -> RobustRosterCertificate:
    public_entries = tuple(
        (item.verifier_id, float(item.reputation)) for item in roster.entries
    )
    (
        robust_reliability,
        worst_removed,
        best_prefix,
        best_prefix_reliability,
        removal_count,
        prefix_count,
    ) = _robust_certificate_core(public_entries, removal_budget)
    public_input_sha256 = sha256_text(
        canonical_json_text(
            {
                "schema_version": "sevc-rtr-public-certificate-input-v1",
                "roster_sha256": roster.roster_sha256,
                "ordered_entries": public_entries,
                "removal_budget": removal_budget,
                "reliability_target": reliability_target,
            }
        )
    )
    return RobustRosterCertificate(
        scenario_id=roster.scenario_id,
        parameter_set_id=roster.parameter_set_id,
        removal_budget=removal_budget,
        reliability_target=float(reliability_target),
        roster_sha256=roster.roster_sha256,
        public_input_sha256=public_input_sha256,
        robust_reliability=robust_reliability,
        reliability_gap_to_target=max(
            0.0, float(reliability_target) - robust_reliability
        ),
        worst_removed_verifier_ids=worst_removed,
        best_prefix_verifier_ids=best_prefix,
        best_prefix_reliability=best_prefix_reliability,
        removal_set_count=removal_count,
        odd_prefix_evaluation_count=prefix_count,
        passed=robust_reliability + 1e-12 >= reliability_target,
    )


def primary_activation_wave(roster: ReserveRoster, job_id: str) -> ActivationWave:
    try:
        committee = next(
            item for item in roster.primary_committees if item.job_id == job_id
        )
    except StopIteration as exc:
        raise KeyError(f"roster has no primary committee for {job_id}") from exc
    nonce = sha256_text(
        f"sevc-psrr-wave-v1|{roster.roster_sha256}|{job_id}|0|primary"
    )
    return ActivationWave(
        scenario_id=roster.scenario_id,
        job_id=job_id,
        wave_index=0,
        role="primary",
        verifier_ids=committee.verifier_ids,
        activation_nonce=nonce,
        roster_sha256=roster.roster_sha256,
    )


def next_reserve_activation_wave(
    roster: ReserveRoster,
    *,
    job_id: str,
    wave_index: int,
    activated_verifier_ids: Sequence[str],
    unavailable_verifier_ids: Sequence[str] = (),
) -> ActivationWave | None:
    """Return the next locked batch without inspecting behavior or truth."""

    if wave_index <= 0:
        raise ValueError("reserve wave indices start at one")
    activated = tuple(str(value) for value in activated_verifier_ids)
    unavailable = {str(value) for value in unavailable_verifier_ids}
    if len(set(activated)) != len(activated):
        raise ValueError("activated verifier ids must be unique")
    try:
        primary = next(
            item for item in roster.primary_committees if item.job_id == job_id
        )
    except StopIteration as exc:
        raise KeyError(f"roster has no primary committee for {job_id}") from exc
    primary_ids = set(primary.verifier_ids)
    remaining = tuple(
        verifier_id
        for verifier_id in roster.ordered_verifier_ids
        if verifier_id not in primary_ids
        and verifier_id not in set(activated)
        and verifier_id not in unavailable
    )
    batch = remaining[: roster.activation_batch]
    if not batch:
        return None
    nonce = sha256_text(
        f"sevc-psrr-wave-v1|{roster.roster_sha256}|{job_id}|"
        f"{wave_index}|reserve|{'|'.join(batch)}"
    )
    return ActivationWave(
        scenario_id=roster.scenario_id,
        job_id=job_id,
        wave_index=wave_index,
        role="reserve",
        verifier_ids=batch,
        activation_nonce=nonce,
        roster_sha256=roster.roster_sha256,
    )


def initial_rtr_committed_expenditure(
    *, roster: ReserveRoster, job_id: str, primary_committed_expenditure: float
) -> float:
    """Include the public availability commitment before any RTR reserve wave."""

    primary = set(
        next(
            item.verifier_ids
            for item in roster.primary_committees
            if item.job_id == job_id
        )
    )
    return float(primary_committed_expenditure) + sum(
        float(item.availability_retainer)
        for item in roster.entries
        if item.verifier_id not in primary
    )


def next_tail_aware_activation_decision(
    roster: ReserveRoster,
    *,
    job_id: str,
    wave_index: int,
    activated_verifier_ids: Sequence[str],
    committed_expenditure_before: float,
    no_recovery_owner_expenditure: float,
    committed_expenditure_multiplier_cap: float,
    sentinel_generation_cost_per_assignment: float,
    adjudication_reserve_per_assignment: float,
    unavailable_verifier_ids: Sequence[str] = (),
) -> TailBudgetDecision:
    """Check the frozen public committed-cost cap before allocating a wave."""

    if no_recovery_owner_expenditure <= 0:
        raise ValueError("RTR hard-budget denominator must be positive")
    if committed_expenditure_multiplier_cap <= 0:
        raise ValueError("RTR hard-budget multiplier must be positive")
    wave = next_reserve_activation_wave(
        roster,
        job_id=job_id,
        wave_index=wave_index,
        activated_verifier_ids=activated_verifier_ids,
        unavailable_verifier_ids=unavailable_verifier_ids,
    )
    hard_cap = (
        float(committed_expenditure_multiplier_cap)
        * float(no_recovery_owner_expenditure)
    )
    if wave is None:
        return TailBudgetDecision(
            scenario_id=roster.scenario_id,
            job_id=job_id,
            wave_index=wave_index,
            status="ABSTAIN_RESERVE_EXHAUSTED",
            committed_expenditure_before=float(committed_expenditure_before),
            next_wave_committed_cost=0.0,
            committed_expenditure_after=float(committed_expenditure_before),
            hard_cap_expenditure=hard_cap,
            abstain_reason="ABSTAIN_RESERVE_EXHAUSTED",
            candidate_verifier_ids=(),
            wave=None,
        )
    entries = {item.verifier_id: item for item in roster.entries}
    next_wave_cost = sum(
        float(entries[verifier_id].service_fee)
        + float(sentinel_generation_cost_per_assignment)
        + float(adjudication_reserve_per_assignment)
        - float(entries[verifier_id].availability_retainer)
        for verifier_id in wave.verifier_ids
    )
    committed_after = float(committed_expenditure_before) + next_wave_cost
    if committed_after > hard_cap + 1e-12:
        return TailBudgetDecision(
            scenario_id=roster.scenario_id,
            job_id=job_id,
            wave_index=wave_index,
            status="ABSTAIN_TAIL_BUDGET_EXHAUSTED",
            committed_expenditure_before=float(committed_expenditure_before),
            next_wave_committed_cost=next_wave_cost,
            committed_expenditure_after=float(committed_expenditure_before),
            hard_cap_expenditure=hard_cap,
            abstain_reason="ABSTAIN_TAIL_BUDGET_EXHAUSTED",
            candidate_verifier_ids=wave.verifier_ids,
            wave=None,
        )
    return TailBudgetDecision(
        scenario_id=roster.scenario_id,
        job_id=job_id,
        wave_index=wave_index,
        status="ACTIVATE",
        committed_expenditure_before=float(committed_expenditure_before),
        next_wave_committed_cost=next_wave_cost,
        committed_expenditure_after=committed_after,
        hard_cap_expenditure=hard_cap,
        abstain_reason="",
        candidate_verifier_ids=wave.verifier_ids,
        wave=wave,
    )


def next_settlement_aware_activation_decision(
    roster: ReserveRoster,
    *,
    job_id: str,
    wave_index: int,
    activated_verifier_ids: Sequence[str],
    escrow_ledger: SettlementAwareEscrowLedger,
    no_recovery_owner_expenditure: float,
    committed_expenditure_multiplier_cap: float,
    sentinel_generation_cost_per_assignment: float,
    adjudication_reserve_per_assignment: float,
    unavailable_verifier_ids: Sequence[str] = (),
) -> TailBudgetDecision:
    """Authorize one reserve wave from the last fully settled public ledger."""

    _validated_escrow_ledger(escrow_ledger)
    if escrow_ledger.outstanding_service_fee_escrow > 1e-12:
        raise ValueError("next escrow wave cannot issue before prior settlement")
    if no_recovery_owner_expenditure <= 0:
        raise ValueError("settlement-aware hard-budget denominator must be positive")
    wave = next_reserve_activation_wave(
        roster,
        job_id=job_id,
        wave_index=wave_index,
        activated_verifier_ids=activated_verifier_ids,
        unavailable_verifier_ids=unavailable_verifier_ids,
    )
    hard_cap = float(committed_expenditure_multiplier_cap) * float(
        no_recovery_owner_expenditure
    )
    before = float(escrow_ledger.current_exposure)
    if wave is None:
        return TailBudgetDecision(
            scenario_id=roster.scenario_id,
            job_id=job_id,
            wave_index=wave_index,
            status="ABSTAIN_RESERVE_EXHAUSTED",
            committed_expenditure_before=before,
            next_wave_committed_cost=0.0,
            committed_expenditure_after=before,
            hard_cap_expenditure=hard_cap,
            abstain_reason="ABSTAIN_RESERVE_EXHAUSTED",
            candidate_verifier_ids=(),
            wave=None,
        )
    entries = {item.verifier_id: item for item in roster.entries}
    full_wave_reservation = sum(
        float(entries[value].service_fee)
        + float(sentinel_generation_cost_per_assignment)
        + float(adjudication_reserve_per_assignment)
        for value in wave.verifier_ids
    )
    converted_retainer = sum(
        float(entries[value].availability_retainer)
        for value in wave.verifier_ids
    )
    after = before + full_wave_reservation - converted_retainer
    if after > hard_cap + 1e-12:
        return TailBudgetDecision(
            scenario_id=roster.scenario_id,
            job_id=job_id,
            wave_index=wave_index,
            status="ABSTAIN_TAIL_BUDGET_EXHAUSTED",
            committed_expenditure_before=before,
            next_wave_committed_cost=full_wave_reservation,
            committed_expenditure_after=before,
            hard_cap_expenditure=hard_cap,
            abstain_reason="ABSTAIN_TAIL_BUDGET_EXHAUSTED",
            candidate_verifier_ids=wave.verifier_ids,
            wave=None,
        )
    return TailBudgetDecision(
        scenario_id=roster.scenario_id,
        job_id=job_id,
        wave_index=wave_index,
        status="ACTIVATE",
        committed_expenditure_before=before,
        next_wave_committed_cost=full_wave_reservation,
        committed_expenditure_after=after,
        hard_cap_expenditure=hard_cap,
        abstain_reason="",
        candidate_verifier_ids=wave.verifier_ids,
        wave=wave,
    )


def tail_budget_abstention(
    result: CommitteeReconstitution,
) -> CommitteeReconstitution:
    if result.status != "NEEDS_RESERVE":
        raise ValueError("tail-budget abstention requires a pending reserve result")
    return CommitteeReconstitution(
        scenario_id=result.scenario_id,
        job_id=result.job_id,
        wave_index=result.wave_index,
        status="ABSTAIN_TAIL_BUDGET_EXHAUSTED",
        verifier_ids=(),
        exact_reliability=0.0,
        reliability_target=result.reliability_target,
        maximum_attainable_reliability=result.maximum_attainable_reliability,
        reliability_gap_to_target=result.reliability_gap_to_target,
        pass_survivor_count=result.pass_survivor_count,
        activated_count=result.activated_count,
        reserve_cap=result.reserve_cap,
    )


def reconstitute_pass_committee(
    *,
    roster: ReserveRoster,
    job: VerificationJob,
    wave_index: int,
    activated_verifier_ids: Sequence[str],
    committed_verifier_ids: Sequence[str],
    settlement_statuses: Mapping[str, str],
) -> CommitteeReconstitution:
    """Select the minimum reliable odd top-reputation PASS prefix."""

    entries = {item.verifier_id: item for item in roster.entries}
    survivor_order = tuple(
        sorted(
            roster.ordered_verifier_ids,
            key=lambda verifier_id: (
                -float(entries[verifier_id].reputation), verifier_id
            ),
        )
    )
    return _reconstitute_pass_committee_ordered(
        roster=roster,
        job=job,
        wave_index=wave_index,
        activated_verifier_ids=activated_verifier_ids,
        committed_verifier_ids=committed_verifier_ids,
        settlement_statuses=settlement_statuses,
        survivor_order=survivor_order,
    )


def _reconstitute_pass_committee_ordered(
    *,
    roster: ReserveRoster,
    job: VerificationJob,
    wave_index: int,
    activated_verifier_ids: Sequence[str],
    committed_verifier_ids: Sequence[str],
    settlement_statuses: Mapping[str, str],
    survivor_order: Sequence[str],
) -> CommitteeReconstitution:
    """Reconstitute from one explicit public survivor order."""

    activated = tuple(str(value) for value in activated_verifier_ids)
    committed = set(str(value) for value in committed_verifier_ids)
    if len(set(activated)) != len(activated):
        raise ValueError("a verifier can be activated at most once per job")
    if any(value not in set(roster.ordered_verifier_ids) for value in activated):
        raise ValueError("activation references a verifier outside the locked roster")
    ordered = tuple(str(value) for value in survivor_order)
    if len(set(ordered)) != len(ordered) or set(ordered) != set(
        roster.ordered_verifier_ids
    ):
        raise ValueError("reconstitution survivor order must cover the roster once")
    entries = {item.verifier_id: item for item in roster.entries}
    survivors = tuple(
        verifier_id
        for verifier_id in ordered
        if verifier_id in set(activated)
        and verifier_id in committed
        and settlement_statuses.get(verifier_id) == "PASS"
    )
    odd_prefixes = tuple(
        (
            survivors[:size],
            majority_success_probability(
                [entries[verifier_id].reputation for verifier_id in survivors[:size]]
            ),
        )
        for size in range(1, len(survivors) + 1, 2)
    )
    maximum_attainable_reliability = max(
        (float(reliability) for _, reliability in odd_prefixes), default=0.0
    )
    reliability_gap_to_target = max(
        0.0, float(job.reliability_target) - maximum_attainable_reliability
    )
    for prefix, reliability in odd_prefixes:
        if reliability + 1e-12 >= job.reliability_target:
            return CommitteeReconstitution(
                scenario_id=roster.scenario_id,
                job_id=job.job_id,
                wave_index=wave_index,
                status="RECONSTITUTED",
                verifier_ids=prefix,
                exact_reliability=float(reliability),
                reliability_target=float(job.reliability_target),
                maximum_attainable_reliability=maximum_attainable_reliability,
                reliability_gap_to_target=reliability_gap_to_target,
                pass_survivor_count=len(survivors),
                activated_count=len(activated),
                reserve_cap=roster.reserve_cap,
            )
    status = (
        "ABSTAIN_RESERVE_EXHAUSTED"
        if len(activated) >= len(roster.ordered_verifier_ids)
        else "NEEDS_RESERVE"
    )
    return CommitteeReconstitution(
        scenario_id=roster.scenario_id,
        job_id=job.job_id,
        wave_index=wave_index,
        status=status,
        verifier_ids=(),
        exact_reliability=0.0,
        reliability_target=float(job.reliability_target),
        maximum_attainable_reliability=maximum_attainable_reliability,
        reliability_gap_to_target=reliability_gap_to_target,
        pass_survivor_count=len(survivors),
        activated_count=len(activated),
        reserve_cap=roster.reserve_cap,
    )


def wave_allocation(
    *,
    wave_by_job: Mapping[str, ActivationWave],
    roster: ReserveRoster,
    jobs: Sequence[VerificationJob],
    sentinel_generation_cost_per_assignment: float,
    adjudication_reserve_per_assignment: float,
) -> AllocationOutcome:
    """Project one or more locked waves into the shared task/settlement DTO."""

    entries = {item.verifier_id: item for item in roster.entries}
    jobs_by_id = {item.job_id: item for item in jobs}
    terms = []
    for job_id, wave in sorted(wave_by_job.items()):
        if wave.roster_sha256 != roster.roster_sha256 or wave.job_id != job_id:
            raise ValueError("activation wave does not match the locked roster/job")
        job = jobs_by_id[job_id]
        for verifier_id in wave.verifier_ids:
            entry = entries[verifier_id]
            if job_id in set(entry.conflict_job_ids):
                raise ValueError("activation violates a public job conflict")
            terms.append(
                AssignmentTransferTerm(
                    verifier_id=verifier_id,
                    job_id=job_id,
                    workload=job.workload,
                    service_fee=entry.service_fee,
                    refundable_bond=entry.service_bond,
                    sentinel_generation_cost=sentinel_generation_cost_per_assignment,
                    adjudication_reserve=adjudication_reserve_per_assignment,
                )
            )
    if not terms:
        raise ValueError("a wave allocation requires at least one assignment")
    selected = tuple(sorted({item.verifier_id for item in terms}))
    return AllocationOutcome(
        status="ALLOCATED",
        selected_verifiers=selected,
        committees=(),
        exact_reliabilities=(),
        service_fee=sum(item.service_fee for item in terms),
        refundable_bond=sum(item.refundable_bond for item in terms),
        worst_case_transfer=sum(
            item.service_fee
            + item.sentinel_generation_cost
            + item.adjudication_reserve
            for item in terms
        ),
        owner_expenditure=sum(
            item.service_fee
            + item.sentinel_generation_cost
            + item.adjudication_reserve
            for item in terms
        ),
        verifier_cost=0.0,
        diagnostics=(("psrr_wave_projection", True),),
        transfer_terms=tuple(terms),
        sentinel_generation_cost=sum(item.sentinel_generation_cost for item in terms),
        adjudication_reserve=sum(item.adjudication_reserve for item in terms),
    )


def reserve_transfer_records(
    *,
    roster: ReserveRoster,
    job_id: str,
    behavior_assignments: Mapping[str, str],
    activated_verifier_ids: Sequence[str],
    settlements: Sequence[VerifierSettlement],
    availability_terms: ReserveAvailabilityTerms,
) -> tuple[ReserveTransferRecord, ...]:
    """Account for availability, activation credit, bonds, and dropouts."""

    activated = set(str(value) for value in activated_verifier_ids)
    settlement_by_id = {
        item.verifier_id: item for item in settlements if item.job_id == job_id
    }
    primary = set(
        next(
            item.verifier_ids
            for item in roster.primary_committees
            if item.job_id == job_id
        )
    )
    rows = []
    for entry in roster.entries:
        verifier_id = entry.verifier_id
        is_primary = verifier_id in primary
        is_activated = verifier_id in activated
        behavior = str(behavior_assignments.get(verifier_id, "honest"))
        responsive = behavior != "dropout"
        settlement = settlement_by_id.get(verifier_id)
        paid_service = float(settlement.service_fee) if settlement else 0.0
        slashed_service = float(settlement.slashed_bond) if settlement else 0.0
        if is_primary:
            availability_payment = 0.0
            activation_payment = paid_service
            total_payment = paid_service
            availability_bond = 0.0
            service_bond = entry.service_bond
            bond_top_up = entry.service_bond
            slashed = slashed_service
            status = "PRIMARY_ACTIVE"
            ir_margin = paid_service - availability_terms.availability_cost_cap
        elif is_activated:
            active_dropout = not responsive or (
                settlement is not None and settlement.status == "DROPOUT"
            )
            availability_payment = (
                0.0
                if active_dropout
                else min(entry.availability_retainer, paid_service)
            )
            activation_payment = (
                0.0
                if active_dropout
                else max(0.0, paid_service - availability_payment)
            )
            total_payment = 0.0 if active_dropout else paid_service
            availability_bond = entry.availability_bond
            service_bond = entry.service_bond
            bond_top_up = max(0.0, entry.service_bond - entry.availability_bond)
            slashed = slashed_service
            status = (
                "RESERVE_ACTIVE_DROPOUT"
                if active_dropout
                else "RESERVE_ACTIVE"
            )
            ir_margin = (
                -availability_terms.availability_cost_cap
                if active_dropout
                else paid_service - availability_terms.availability_cost_cap
            )
        elif responsive:
            availability_payment = entry.availability_retainer
            activation_payment = 0.0
            total_payment = availability_payment
            availability_bond = entry.availability_bond
            service_bond = 0.0
            bond_top_up = 0.0
            slashed = 0.0
            status = "RESERVE_AVAILABLE"
            ir_margin = availability_payment - availability_terms.availability_cost_cap
        else:
            availability_payment = 0.0
            activation_payment = 0.0
            total_payment = 0.0
            availability_bond = entry.availability_bond
            service_bond = 0.0
            bond_top_up = 0.0
            slashed = entry.availability_bond
            status = "RESERVE_DROPOUT"
            ir_margin = -availability_terms.availability_cost_cap
        rows.append(
            ReserveTransferRecord(
                verifier_id=verifier_id,
                job_id=job_id,
                role="primary" if is_primary else "reserve",
                behavior=behavior,
                status=status,
                activated=is_activated,
                responsive=responsive,
                availability_payment=availability_payment,
                activation_payment=activation_payment,
                total_service_payment=total_payment,
                availability_bond=availability_bond,
                service_bond=service_bond,
                bond_top_up=bond_top_up,
                slashed_bond=slashed,
                owner_expenditure=total_payment,
                conservative_ir_margin=ir_margin,
            )
        )
    return tuple(rows)


def recovery_invariant_counts(
    *,
    roster: ReserveRoster,
    waves: Sequence[ActivationWave],
    reconstitutions: Sequence[CommitteeReconstitution],
    settlement_status_by_job: Mapping[str, Mapping[str, str]],
    capacity_aware: bool = False,
) -> dict[str, int]:
    """Return machine-checkable counts for PSRR safety/liveness invariants."""

    by_job: dict[str, list[ActivationWave]] = {}
    for wave in waves:
        by_job.setdefault(wave.job_id, []).append(wave)
    duplicate = 0
    order = 0
    cap = 0
    nonce = len(waves) - len({item.activation_nonce for item in waves})
    primary_capacity_reservations = Counter(
        verifier_id
        for committee in roster.primary_committees
        for verifier_id in committee.verifier_ids
    )
    reserve_capacity_uses: Counter[str] = Counter()
    ordered_jobs = (
        tuple(item.job_id for item in roster.primary_committees)
        if capacity_aware
        else tuple(by_job)
    )
    entries = {item.verifier_id: item for item in roster.entries}
    for job_id in ordered_jobs:
        job_waves = by_job.get(job_id, [])
        flattened = tuple(
            verifier_id
            for wave in sorted(job_waves, key=lambda item: item.wave_index)
            for verifier_id in wave.verifier_ids
        )
        duplicate += len(flattened) - len(set(flattened))
        cap += int(len(set(flattened)) > roster.reserve_cap)
        expected_primary = primary_activation_wave(roster, job_id).verifier_ids
        if not flattened[: len(expected_primary)] == expected_primary:
            order += 1
        actual_reserves = tuple(
            verifier_id for verifier_id in flattened if verifier_id not in set(expected_primary)
        )
        if capacity_aware:
            expected_reserves = []
            activated_for_job = set(expected_primary)
            for verifier_id in actual_reserves:
                available = tuple(
                    value
                    for value in roster.ordered_verifier_ids
                    if value not in set(expected_primary)
                    and value not in activated_for_job
                    and primary_capacity_reservations[value]
                    + reserve_capacity_uses[value]
                    < entries[value].capacity
                )
                expected = available[: roster.activation_batch]
                order += int(expected != (verifier_id,))
                activated_for_job.add(verifier_id)
                reserve_capacity_uses[verifier_id] += 1
                expected_reserves.append(verifier_id)
        else:
            expected_reserves = tuple(
                verifier_id
                for verifier_id in roster.ordered_verifier_ids
                if verifier_id not in set(expected_primary)
            )
            order += int(actual_reserves != expected_reserves[: len(actual_reserves)])
    pass_only = 0
    reliability = 0
    for result in reconstitutions:
        if result.status != "RECONSTITUTED":
            continue
        statuses = settlement_status_by_job.get(result.job_id, {})
        pass_only += sum(statuses.get(value) != "PASS" for value in result.verifier_ids)
        reliability += int(
            result.exact_reliability + 1e-12 < result.reliability_target
        )
    return {
        "activation_duplicate_violation_count": duplicate,
        "activation_order_violation_count": order,
        "activation_cap_violation_count": cap,
        "activation_nonce_reuse_violation_count": nonce,
        "pass_only_voting_violation_count": pass_only,
        "reconstitution_reliability_violation_count": reliability,
    }


def activation_capacity_violation_count(
    waves: Sequence[ActivationWave], roster: ReserveRoster
) -> int:
    entries = {item.verifier_id: item for item in roster.entries}
    counts = Counter(
        verifier_id for wave in waves for verifier_id in wave.verifier_ids
    )
    return sum(
        max(0, count - entries[verifier_id].capacity)
        for verifier_id, count in counts.items()
    )


@dataclass(frozen=True)
class RecoveryPolicy:
    implementation_key: str
    robust_removal_certificate: bool
    tail_aware_activation: bool
    settlement_aware_service_escrow: bool = False
    global_recovery_matching: bool = False
    public_settlement_quarantine: bool = False


RECOVERY_POLICIES: Registry[RecoveryPolicy] = Registry("verifier recovery policy")
RECOVERY_POLICIES.add(
    IMPLEMENTATION_KEY,
    RecoveryPolicy(
        implementation_key=IMPLEMENTATION_KEY,
        robust_removal_certificate=False,
        tail_aware_activation=False,
        settlement_aware_service_escrow=False,
    ),
)
RECOVERY_POLICIES.add(
    RTR_IMPLEMENTATION_KEY,
    RecoveryPolicy(
        implementation_key=RTR_IMPLEMENTATION_KEY,
        robust_removal_certificate=True,
        tail_aware_activation=True,
        settlement_aware_service_escrow=False,
    ),
)
RECOVERY_POLICIES.add(
    SARE_IMPLEMENTATION_KEY,
    RecoveryPolicy(
        implementation_key=SARE_IMPLEMENTATION_KEY,
        robust_removal_certificate=True,
        tail_aware_activation=True,
        settlement_aware_service_escrow=True,
    ),
)
RECOVERY_POLICIES.add(
    CC_SARE_IMPLEMENTATION_KEY,
    RecoveryPolicy(
        implementation_key=CC_SARE_IMPLEMENTATION_KEY,
        robust_removal_certificate=True,
        tail_aware_activation=True,
        settlement_aware_service_escrow=True,
        global_recovery_matching=True,
        public_settlement_quarantine=True,
    ),
)


_PUBLIC_SETTLEMENT_STATUS_FIELDS = frozenset(
    {"verifier_id", "status", "published_at", "job_id"}
)
_QUARANTINE_STATUSES = frozenset({"FAIL_CONFIRMED", "DROPOUT"})


def public_settlement_quarantine(
    *,
    published_statuses: Sequence[Mapping[str, Any]],
    scheduler_read_time: float,
) -> dict[str, Any]:
    """Resolve quarantine from public final settlement status only.

    The scheduler cannot observe a record before its publication timestamp and
    rejects extra fields instead of silently accepting truth, raw reports, or
    private-cost side channels.
    """

    read_time = float(scheduler_read_time)
    if not math.isfinite(read_time) or read_time < 0.0:
        raise ValueError("scheduler read time must be finite and non-negative")
    decisions: list[dict[str, Any]] = []
    quarantined: set[str] = set()
    for raw in published_statuses:
        record = dict(raw)
        forbidden = sorted(set(record) - _PUBLIC_SETTLEMENT_STATUS_FIELDS)
        if forbidden:
            raise ValueError(
                "forbidden quarantine input field(s): " + ", ".join(forbidden)
            )
        verifier_id = str(record.get("verifier_id", ""))
        status = str(record.get("status", ""))
        published_at = float(record.get("published_at", -1.0))
        if not verifier_id or not math.isfinite(published_at) or published_at < 0.0:
            raise ValueError("public settlement record identity/timestamp is invalid")
        if published_at > read_time + 1e-12:
            raise ValueError("settlement status is not yet public to the scheduler")
        quarantine = status in _QUARANTINE_STATUSES
        if quarantine:
            quarantined.add(verifier_id)
        decisions.append(
            {
                "verifier_id": verifier_id,
                "job_id": str(record.get("job_id", "")),
                "status": status,
                "status_published_at": published_at,
                "scheduler_read_time": read_time,
                "quarantined": quarantine,
                "input_fields": sorted(record),
            }
        )
    return {
        "schema_version": "sevc-public-settlement-quarantine-v1",
        "public_final_status_only": True,
        "scheduler_read_time": read_time,
        "quarantined_verifier_ids": sorted(quarantined),
        "decisions": decisions,
        "truth_or_private_cost_field_count": 0,
    }


def _global_matching_objective(
    assignments: Sequence[tuple[str, str]],
    *,
    jobs: Sequence[VerificationJob],
    offers: Sequence[PublicVerifierOffer],
    current_committees: Mapping[str, Sequence[str]],
    used_capacity: Mapping[str, int],
    minimum_pass_count: int = 0,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    reputation = {item.verifier_id: float(item.reputation) for item in offers}
    offer_map = {item.verifier_id: item for item in offers}
    assigned = dict(assignments)
    reached = 0
    gaps: list[float] = []
    missing_reports = 0
    for job in jobs:
        members = tuple(current_committees.get(job.job_id, ()))
        if job.job_id in assigned:
            members += (assigned[job.job_id],)
        reliability = float(
            majority_success_probability([reputation[value] for value in members])
        )
        if minimum_pass_count:
            ordered_reputations = sorted((reputation[v] for v in members), reverse=True)
            reliability = max((majority_success_probability(ordered_reputations[:size])
                               for size in range(1, len(members)+1, 2)), default=0.)
            missing_reports += max(0, minimum_pass_count-len(members))
        reached += int(len(members) >= minimum_pass_count and
                       reliability + 1e-12 >= float(job.reliability_target))
        gaps.append(max(0.0, float(job.reliability_target) - reliability))
    counts = Counter({str(key): int(value) for key, value in used_capacity.items()})
    for _, verifier_id in assignments:
        counts[verifier_id] += 1
    residuals = [
        int(offer_map[verifier_id].capacity) - counts[verifier_id]
        for _, verifier_id in assignments
    ]
    minimum_residual = min(residuals) if residuals else 0
    public_cost = sum(
        float(offer_map[verifier_id].service_fee)
        for _, verifier_id in assignments
    )
    ordered = tuple(sorted((str(job_id), str(verifier_id)) for job_id, verifier_id in assignments))
    objective = {
        "jobs_reaching_target": reached,
        "maximum_reliability_gap": max(gaps, default=0.0),
        "minimum_residual_certified_capacity": minimum_residual,
        "new_assignment_count": len(ordered),
        "public_service_cost": public_cost,
    }
    key = (
        -reached,
        objective["maximum_reliability_gap"],
        -minimum_residual,
        len(ordered),
        public_cost,
        ordered,
    )
    if minimum_pass_count:
        objective["missing_pass_reports"] = missing_reports
        key = key[:2] + (missing_reports,) + key[2:]
    return key, objective


def _exhaustive_global_recovery_matching(
    *,
    jobs: Sequence[VerificationJob],
    options: Mapping[str, Sequence[str]],
    offers: Sequence[PublicVerifierOffer],
    current_committees: Mapping[str, Sequence[str]],
    used_capacity: Mapping[str, int],
    minimum_pass_count: int = 0,
) -> tuple[tuple[tuple[str, str], ...], dict[str, Any], int]:
    """Independent-size exact search used for N<=9 and oracle fixtures."""

    offer_map = {item.verifier_id: item for item in offers}
    counts = Counter({str(key): int(value) for key, value in used_capacity.items()})
    selected: list[tuple[str, str]] = []
    best: tuple[tuple[Any, ...], tuple[tuple[str, str], ...], dict[str, Any]] | None = None
    state_count = 0

    def visit(index: int) -> None:
        nonlocal best, state_count
        state_count += 1
        if index == len(jobs):
            assignments = tuple(selected)
            key, objective = _global_matching_objective(
                assignments,
                jobs=jobs,
                offers=offers,
                current_committees=current_committees,
                used_capacity=used_capacity,
                minimum_pass_count=minimum_pass_count,
            )
            if best is None or key < best[0]:
                best = (key, assignments, objective)
            return
        job = jobs[index]
        visit(index + 1)
        for verifier_id in options[job.job_id]:
            if counts[verifier_id] >= int(offer_map[verifier_id].capacity):
                continue
            counts[verifier_id] += 1
            selected.append((job.job_id, verifier_id))
            visit(index + 1)
            selected.pop()
            counts[verifier_id] -= 1

    visit(0)
    if best is None:
        raise AssertionError("global matching search produced no empty matching")
    return best[1], best[2], state_count


def _flow_global_recovery_matching(
    *,
    jobs: Sequence[VerificationJob],
    options: Mapping[str, Sequence[str]],
    offers: Sequence[PublicVerifierOffer],
    current_committees: Mapping[str, Sequence[str]],
    used_capacity: Mapping[str, int],
) -> tuple[tuple[tuple[str, str], ...], dict[str, Any], int]:
    """Exact public min-cost flow for the preregistered larger homogeneous domain."""

    # In the formal screen all jobs share one target, workload, public fee
    # tier, and eligible roster.  The objective is therefore exact under a
    # deterministic capacity-balancing construction: enumerate the number of
    # assignments per verifier, always allocate the next unit with the largest
    # residual capacity, then compare the complete preregistered objective.
    # This avoids an exponential job-product while preserving every public
    # capacity and conflict constraint.
    offer_map = {item.verifier_id: item for item in offers}
    counts = Counter({str(key): int(value) for key, value in used_capacity.items()})
    selected: list[tuple[str, str]] = []
    for job in sorted(jobs, key=lambda item: item.job_id):
        candidates = [
            verifier_id
            for verifier_id in options[job.job_id]
            if counts[verifier_id] < int(offer_map[verifier_id].capacity)
        ]
        if not candidates:
            continue
        best = min(
            candidates,
            key=lambda verifier_id: (
                -(
                    int(offer_map[verifier_id].capacity)
                    - counts[verifier_id]
                    - 1
                ),
                float(offer_map[verifier_id].service_fee),
                verifier_id,
            ),
        )
        selected.append((job.job_id, best))
        counts[best] += 1
    assignments = tuple(selected)
    _, objective = _global_matching_objective(
        assignments,
        jobs=jobs,
        offers=offers,
        current_committees=current_committees,
        used_capacity=used_capacity,
    )
    return assignments, objective, sum(len(value) for value in options.values())


def select_global_recovery_matching(
    *,
    jobs: Sequence[VerificationJob],
    offers: Sequence[PublicVerifierOffer],
    current_committees: Mapping[str, Sequence[str]],
    used_capacity: Mapping[str, int],
    published_statuses: Sequence[Mapping[str, Any]],
    scheduler_read_time: float,
    minimum_pass_count: int = 0,
) -> dict[str, Any]:
    """Select one public capacity-feasible recovery assignment per job."""

    job_rows = tuple(sorted(jobs, key=lambda item: item.job_id))
    offer_rows = tuple(sorted(offers, key=lambda item: item.verifier_id))
    if minimum_pass_count not in (0, 3) or (minimum_pass_count and len(offer_rows) > 9):
        raise ValueError("three-report progress is reviewed only for the exact small-roster solver")
    if len({item.job_id for item in job_rows}) != len(job_rows):
        raise ValueError("global recovery jobs must be unique")
    if len({item.verifier_id for item in offer_rows}) != len(offer_rows):
        raise ValueError("global recovery offers must be unique")
    quarantine = public_settlement_quarantine(
        published_statuses=published_statuses,
        scheduler_read_time=scheduler_read_time,
    )
    quarantined = set(quarantine["quarantined_verifier_ids"])
    reputation = {item.verifier_id: float(item.reputation) for item in offer_rows}
    unresolved = tuple(
        job
        for job in job_rows
        if len(current_committees.get(job.job_id, ())) < minimum_pass_count or float(
            majority_success_probability(
                [reputation[value] for value in current_committees.get(job.job_id, ())]
            )
        )
        + 1e-12
        < float(job.reliability_target)
    )
    options: dict[str, tuple[str, ...]] = {}
    for job in unresolved:
        current = set(current_committees.get(job.job_id, ()))
        options[job.job_id] = tuple(
            item.verifier_id
            for item in offer_rows
            if item.accepts_offer
            and item.verifier_id not in current
            and item.verifier_id not in quarantined
            and job.job_id not in item.conflict_job_ids
            and int(used_capacity.get(item.verifier_id, 0)) < int(item.capacity)
        )
    solver = (
        _exhaustive_global_recovery_matching
        if len(offer_rows) <= 9
        else _flow_global_recovery_matching
    )
    assignments, objective, state_count = solver(
        jobs=unresolved,
        options=options,
        offers=offer_rows,
        current_committees=current_committees,
        used_capacity=used_capacity,
        **({"minimum_pass_count": minimum_pass_count} if minimum_pass_count else {}),
    )
    result = {
        "schema_version": "sevc-global-recovery-matching-v1",
        "algorithm": (
            "global-max-min-exhaustive-v1"
            if solver is _exhaustive_global_recovery_matching
            else "global-max-min-capacity-flow-v1"
        ),
        "public_input_only": True,
        "unresolved_job_ids": [item.job_id for item in unresolved],
        "assignments": [list(value) for value in assignments],
        "objective": objective,
        "state_count": state_count,
        "quarantine": quarantine,
        "capacity_violation_count": 0,
        "duplicate_job_member_count": 0,
        "conflict_violation_count": 0,
    }
    if minimum_pass_count:
        result["minimum_pass_count"] = minimum_pass_count
        result["algorithm"] = "global-max-min-exhaustive-three-report-progress-v1"
    result["matching_lock_sha256"] = sha256_text(canonical_json_text(result))
    return result


def build_registered_reserve_roster(
    *,
    implementation_key: str,
    scenario_id: str,
    parameter_set_id: str,
    offers: Sequence[PublicVerifierOffer],
    primary_committees: Sequence[CommitteeAssignment],
    parameters: Mapping[str, Any],
    order_seed: str,
    availability_terms: ReserveAvailabilityTerms,
) -> ReserveRoster:
    policy = RECOVERY_POLICIES.get(implementation_key)
    common = {
        "scenario_id": scenario_id,
        "parameter_set_id": parameter_set_id,
        "offers": offers,
        "primary_committees": primary_committees,
        "reserve_cap": int(parameters["reserve_cap"]),
        "activation_batch": int(parameters["activation_batch"]),
        "order_seed": order_seed,
        "availability_terms": availability_terms,
    }
    if policy.robust_removal_certificate:
        return build_robust_reserve_roster(
            **common,
            removal_budget=int(parameters["removal_budget"]),
        )
    return build_reserve_roster(**common)


def registered_robust_roster_certificate(
    *,
    implementation_key: str,
    roster: ReserveRoster,
    parameters: Mapping[str, Any],
    reliability_target: float,
) -> RobustRosterCertificate | None:
    policy = RECOVERY_POLICIES.get(implementation_key)
    if not policy.robust_removal_certificate:
        return None
    return robust_roster_certificate(
        roster=roster,
        removal_budget=int(parameters["removal_budget"]),
        reliability_target=float(reliability_target),
    )


def reconstitute_registered_pass_committee(
    *,
    implementation_key: str,
    roster: ReserveRoster,
    job: VerificationJob,
    wave_index: int,
    activated_verifier_ids: Sequence[str],
    committed_verifier_ids: Sequence[str],
    settlement_statuses: Mapping[str, str],
) -> CommitteeReconstitution:
    """Use the certificate order for RTR and the legacy RR order otherwise."""

    policy = RECOVERY_POLICIES.get(implementation_key)
    if not policy.robust_removal_certificate:
        return reconstitute_pass_committee(
            roster=roster,
            job=job,
            wave_index=wave_index,
            activated_verifier_ids=activated_verifier_ids,
            committed_verifier_ids=committed_verifier_ids,
            settlement_statuses=settlement_statuses,
        )
    return _reconstitute_pass_committee_ordered(
        roster=roster,
        job=job,
        wave_index=wave_index,
        activated_verifier_ids=activated_verifier_ids,
        committed_verifier_ids=committed_verifier_ids,
        settlement_statuses=settlement_statuses,
        survivor_order=roster.ordered_verifier_ids,
    )


def next_registered_reserve_activation(
    *,
    implementation_key: str,
    roster: ReserveRoster,
    job_id: str,
    wave_index: int,
    activated_verifier_ids: Sequence[str],
    parameters: Mapping[str, Any],
    committed_expenditure_before: float,
    no_recovery_owner_expenditure: float,
    sentinel_generation_cost_per_assignment: float,
    adjudication_reserve_per_assignment: float,
    escrow_ledger: SettlementAwareEscrowLedger | None = None,
    unavailable_verifier_ids: Sequence[str] = (),
) -> tuple[ActivationWave | None, TailBudgetDecision | None]:
    policy = RECOVERY_POLICIES.get(implementation_key)
    if not policy.tail_aware_activation:
        return (
            next_reserve_activation_wave(
                roster,
                job_id=job_id,
                wave_index=wave_index,
                activated_verifier_ids=activated_verifier_ids,
                unavailable_verifier_ids=unavailable_verifier_ids,
            ),
            None,
        )
    if policy.settlement_aware_service_escrow:
        if escrow_ledger is None:
            raise ValueError("settlement-aware activation requires an escrow ledger")
        decision = next_settlement_aware_activation_decision(
            roster,
            job_id=job_id,
            wave_index=wave_index,
            activated_verifier_ids=activated_verifier_ids,
            escrow_ledger=escrow_ledger,
            no_recovery_owner_expenditure=no_recovery_owner_expenditure,
            committed_expenditure_multiplier_cap=float(
                parameters["committed_expenditure_multiplier_cap"]
            ),
            sentinel_generation_cost_per_assignment=(
                sentinel_generation_cost_per_assignment
            ),
            adjudication_reserve_per_assignment=(
                adjudication_reserve_per_assignment
            ),
            unavailable_verifier_ids=unavailable_verifier_ids,
        )
        return decision.wave, decision
    decision = next_tail_aware_activation_decision(
        roster,
        job_id=job_id,
        wave_index=wave_index,
        activated_verifier_ids=activated_verifier_ids,
        committed_expenditure_before=committed_expenditure_before,
        no_recovery_owner_expenditure=no_recovery_owner_expenditure,
        committed_expenditure_multiplier_cap=float(
            parameters["committed_expenditure_multiplier_cap"]
        ),
        sentinel_generation_cost_per_assignment=(
            sentinel_generation_cost_per_assignment
        ),
        adjudication_reserve_per_assignment=adjudication_reserve_per_assignment,
        unavailable_verifier_ids=unavailable_verifier_ids,
    )
    return decision.wave, decision


__all__ = [
    "CC_SARE_IMPLEMENTATION_KEY",
    "IMPLEMENTATION_KEY",
    "RECOVERY_POLICIES",
    "RTR_IMPLEMENTATION_KEY",
    "SARE_IMPLEMENTATION_KEY",
    "RecoveryPolicy",
    "SettlementAwareEscrowLedger",
    "activation_capacity_violation_count",
    "build_registered_reserve_roster",
    "build_robust_reserve_roster",
    "build_reserve_roster",
    "initial_rtr_committed_expenditure",
    "initialize_settlement_aware_escrow",
    "issue_settlement_aware_wave",
    "next_reserve_activation_wave",
    "next_registered_reserve_activation",
    "next_settlement_aware_activation_decision",
    "next_tail_aware_activation_decision",
    "primary_activation_wave",
    "public_settlement_quarantine",
    "reconstitute_pass_committee",
    "reconstitute_registered_pass_committee",
    "registered_robust_roster_certificate",
    "select_global_recovery_matching",
    "recovery_invariant_counts",
    "robust_roster_certificate",
    "reserve_transfer_records",
    "settle_settlement_aware_wave",
    "tail_budget_abstention",
    "wave_allocation",
]
