"""Canonical contracts for verifier-incentive variants and policies.

The dataclasses in this module deliberately separate verifier-visible inputs
from evaluator-only adjudication material.  Candidate implementations compose
registered allocation, task, and settlement policies; they do not own runners,
metrics, gates, or reporters.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from sevc.core.registry import Registry


ALLOCATION_STATUSES = {
    "ALLOCATED",
    "ABSTAIN_INSUFFICIENT_SUPPLY",
    "INFEASIBLE_RELIABILITY_OR_BUDGET",
}
SETTLEMENT_STATUSES = {
    "PASS",
    "FAIL_CONFIRMED",
    "TECHNICAL_FAILURE",
    "DROPOUT",
}
RECONSTITUTION_STATUSES = {
    "RECONSTITUTED",
    "NEEDS_RESERVE",
    "ABSTAIN_RESERVE_EXHAUSTED",
    "ABSTAIN_TAIL_BUDGET_EXHAUSTED",
}


def _finite_nonnegative(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def dataclass_to_jsonable(value: Any) -> Any:
    """Convert protocol values to deterministic JSON-compatible objects."""

    if is_dataclass(value):
        return {
            field.name: dataclass_to_jsonable(getattr(value, field.name))
            for field in fields(value)
            if not field.name.startswith("_")
        }
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {
            str(key): dataclass_to_jsonable(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list, set)):
        return [dataclass_to_jsonable(item) for item in value]
    return value


@dataclass(frozen=True)
class VerifierType:
    verifier_id: str
    unit_cost: float
    reputation: float
    capacity: int
    accepts_offer: bool = True
    conflict_job_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.verifier_id:
            raise ValueError("verifier_id must be non-empty")
        _finite_nonnegative(self.unit_cost, "unit_cost")
        if not math.isfinite(float(self.reputation)) or not 0 <= self.reputation <= 1:
            raise ValueError("reputation must lie in [0, 1]")
        if self.capacity < 0:
            raise ValueError("capacity must be non-negative")
        if len(set(self.conflict_job_ids)) != len(self.conflict_job_ids):
            raise ValueError("conflict_job_ids must be unique")


@dataclass(frozen=True)
class PublicVerifierProfile:
    """Public allocation attributes, deliberately excluding verifier cost."""

    verifier_id: str
    reputation: float
    capacity: int
    conflict_job_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.verifier_id:
            raise ValueError("verifier_id must be non-empty")
        if not math.isfinite(float(self.reputation)) or not 0 <= self.reputation <= 1:
            raise ValueError("reputation must lie in [0, 1]")
        if self.capacity < 0:
            raise ValueError("capacity must be non-negative")
        if len(set(self.conflict_job_ids)) != len(self.conflict_job_ids):
            raise ValueError("conflict_job_ids must be unique")


@dataclass(frozen=True)
class SealedVerifierEconomics:
    """Evaluator-only economics used to simulate a binary offer response."""

    verifier_id: str
    honest_cost: float
    liquidity_cost: float
    behavior_costs: tuple[tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        if not self.verifier_id:
            raise ValueError("verifier_id must be non-empty")
        _finite_nonnegative(self.honest_cost, "honest_cost")
        _finite_nonnegative(self.liquidity_cost, "liquidity_cost")
        if len({str(key) for key, _ in self.behavior_costs}) != len(
            self.behavior_costs
        ):
            raise ValueError("behavior cost keys must be unique")
        for _, value in self.behavior_costs:
            _finite_nonnegative(value, "behavior_cost")


@dataclass(frozen=True)
class PostedPriceTerms:
    workload_tier: str
    service_fee: float
    refundable_bond: float
    honest_false_fail_upper: float

    def __post_init__(self) -> None:
        if not self.workload_tier:
            raise ValueError("workload_tier must be non-empty")
        _finite_nonnegative(self.service_fee, "service_fee")
        _finite_nonnegative(self.refundable_bond, "refundable_bond")
        if not 0 <= float(self.honest_false_fail_upper) <= 1:
            raise ValueError("honest_false_fail_upper must lie in [0, 1]")


@dataclass(frozen=True)
class PublicVerifierOffer:
    """The complete public allocation message after accept/reject response."""

    verifier_id: str
    reputation: float
    capacity: int
    accepts_offer: bool
    workload_tier: str
    service_fee: float
    refundable_bond: float
    conflict_job_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        PublicVerifierProfile(
            verifier_id=self.verifier_id,
            reputation=self.reputation,
            capacity=self.capacity,
            conflict_job_ids=self.conflict_job_ids,
        )
        if not self.workload_tier:
            raise ValueError("workload_tier must be non-empty")
        _finite_nonnegative(self.service_fee, "service_fee")
        _finite_nonnegative(self.refundable_bond, "refundable_bond")


@dataclass(frozen=True)
class VerificationJob:
    job_id: str
    workload: int
    reliability_target: float

    def __post_init__(self) -> None:
        if not self.job_id or self.workload <= 0:
            raise ValueError("job_id and positive workload are required")
        if not math.isfinite(float(self.reliability_target)) or not (
            0.5 < self.reliability_target <= 1.0
        ):
            raise ValueError("reliability_target must lie in (0.5, 1]")


@dataclass(frozen=True)
class MarketRequest:
    scenario_id: str
    verifiers: tuple[VerifierType, ...]
    jobs: tuple[VerificationJob, ...]
    budget: float
    reference_cost: float
    evaluator_reserve: float = 0.0

    def __post_init__(self) -> None:
        if not self.scenario_id or not self.verifiers or not self.jobs:
            raise ValueError("market request requires scenario, verifiers, and jobs")
        if len({item.verifier_id for item in self.verifiers}) != len(self.verifiers):
            raise ValueError("verifier ids must be unique")
        if len({item.job_id for item in self.jobs}) != len(self.jobs):
            raise ValueError("job ids must be unique")
        _finite_nonnegative(self.budget, "budget")
        if not math.isfinite(float(self.reference_cost)) or self.reference_cost <= 0:
            raise ValueError("reference_cost must be finite and positive")
        _finite_nonnegative(self.evaluator_reserve, "evaluator_reserve")


@dataclass(frozen=True)
class RCPHSEMarketRequest:
    """Public-only market input for RC-PHSE exact joint assignment."""

    scenario_id: str
    offers: tuple[PublicVerifierOffer, ...]
    jobs: tuple[VerificationJob, ...]
    budget: float
    sentinel_generation_cost_per_assignment: float
    adjudication_reserve_per_assignment: float

    def __post_init__(self) -> None:
        if not self.scenario_id or not self.offers or not self.jobs:
            raise ValueError("RC-PHSE market requires scenario, offers, and jobs")
        if len({item.verifier_id for item in self.offers}) != len(self.offers):
            raise ValueError("offer verifier ids must be unique")
        if len({item.job_id for item in self.jobs}) != len(self.jobs):
            raise ValueError("job ids must be unique")
        _finite_nonnegative(self.budget, "budget")
        _finite_nonnegative(
            self.sentinel_generation_cost_per_assignment,
            "sentinel_generation_cost_per_assignment",
        )
        _finite_nonnegative(
            self.adjudication_reserve_per_assignment,
            "adjudication_reserve_per_assignment",
        )


@dataclass(frozen=True)
class CommitteeAssignment:
    job_id: str
    verifier_ids: tuple[str, ...]
    exact_reliability: float

    def __post_init__(self) -> None:
        if not self.job_id or not self.verifier_ids or len(self.verifier_ids) % 2 == 0:
            raise ValueError("committee must be non-empty and odd")
        if len(set(self.verifier_ids)) != len(self.verifier_ids):
            raise ValueError("committee verifier ids must be unique")
        if not 0 <= self.exact_reliability <= 1:
            raise ValueError("exact_reliability must lie in [0, 1]")


@dataclass(frozen=True)
class AssignmentTransferTerm:
    verifier_id: str
    job_id: str
    workload: int
    service_fee: float
    refundable_bond: float
    sentinel_generation_cost: float
    adjudication_reserve: float

    def __post_init__(self) -> None:
        if not self.verifier_id or not self.job_id or self.workload <= 0:
            raise ValueError("assignment transfer identity/workload is invalid")
        for name in (
            "service_fee",
            "refundable_bond",
            "sentinel_generation_cost",
            "adjudication_reserve",
        ):
            _finite_nonnegative(getattr(self, name), name)


@dataclass(frozen=True)
class AllocationOutcome:
    status: str
    selected_verifiers: tuple[str, ...]
    committees: tuple[CommitteeAssignment, ...]
    exact_reliabilities: tuple[float, ...]
    service_fee: float
    refundable_bond: float
    worst_case_transfer: float
    owner_expenditure: float
    verifier_cost: float
    diagnostics: tuple[tuple[str, Any], ...] = ()
    transfer_terms: tuple[AssignmentTransferTerm, ...] = ()
    sentinel_generation_cost: float = 0.0
    adjudication_reserve: float = 0.0

    def __post_init__(self) -> None:
        if self.status not in ALLOCATION_STATUSES:
            raise ValueError(f"unknown allocation status: {self.status}")
        if len(set(self.selected_verifiers)) != len(self.selected_verifiers):
            raise ValueError("selected_verifiers must be unique")
        if len(self.committees) != len(self.exact_reliabilities):
            raise ValueError("committee and reliability counts must match")
        if tuple(item.exact_reliability for item in self.committees) != tuple(
            self.exact_reliabilities
        ):
            raise ValueError("exact reliability fields disagree")
        for name in (
            "service_fee",
            "refundable_bond",
            "worst_case_transfer",
            "owner_expenditure",
            "verifier_cost",
            "sentinel_generation_cost",
            "adjudication_reserve",
        ):
            _finite_nonnegative(getattr(self, name), name)
        identities = tuple(
            (item.verifier_id, item.job_id) for item in self.transfer_terms
        )
        if len(set(identities)) != len(identities):
            raise ValueError("assignment transfer identities must be unique")


@dataclass(frozen=True)
class ReserveAvailabilityTerms:
    """Public fixed-price terms for one PSRR reserve opportunity."""

    c_max: float
    availability_cost_cap: float
    availability_retainer: float
    availability_bond: float
    active_service_fee: float
    active_service_bond: float

    def __post_init__(self) -> None:
        for name in (
            "c_max",
            "availability_cost_cap",
            "availability_retainer",
            "availability_bond",
            "active_service_fee",
            "active_service_bond",
        ):
            _finite_nonnegative(getattr(self, name), name)
        if self.c_max <= 0:
            raise ValueError("c_max must be positive")
        if self.availability_retainer > self.active_service_fee + 1e-12:
            raise ValueError("availability retainer must be credited within service fee")
        if self.availability_bond > self.active_service_bond + 1e-12:
            raise ValueError("availability bond cannot exceed active service bond")


@dataclass(frozen=True)
class ReserveRosterEntry:
    """One public roster entry; no sealed cost or behavior field is allowed."""

    verifier_id: str
    reputation: float
    capacity: int
    service_fee: float
    service_bond: float
    availability_retainer: float
    availability_bond: float
    roster_rank: int
    primary_job_ids: tuple[str, ...] = ()
    conflict_job_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        PublicVerifierProfile(
            verifier_id=self.verifier_id,
            reputation=self.reputation,
            capacity=self.capacity,
            conflict_job_ids=self.conflict_job_ids,
        )
        for name in (
            "service_fee",
            "service_bond",
            "availability_retainer",
            "availability_bond",
        ):
            _finite_nonnegative(getattr(self, name), name)
        if self.roster_rank < 0:
            raise ValueError("roster_rank must be non-negative")
        if len(set(self.primary_job_ids)) != len(self.primary_job_ids):
            raise ValueError("primary_job_ids must be unique")


@dataclass(frozen=True)
class ReserveRoster:
    """Precommitted public primary/reserve roster for one scenario."""

    scenario_id: str
    parameter_set_id: str
    reserve_cap: int
    activation_batch: int
    order_seed_sha256: str
    ordered_verifier_ids: tuple[str, ...]
    primary_committees: tuple[CommitteeAssignment, ...]
    entries: tuple[ReserveRosterEntry, ...]
    roster_sha256: str

    def __post_init__(self) -> None:
        if not self.scenario_id or not self.parameter_set_id:
            raise ValueError("reserve roster identity must be complete")
        if self.reserve_cap <= 0 or self.activation_batch <= 0:
            raise ValueError("reserve cap and activation batch must be positive")
        if len(self.ordered_verifier_ids) > self.reserve_cap:
            raise ValueError("roster exceeds its committed cap")
        if len(set(self.ordered_verifier_ids)) != len(self.ordered_verifier_ids):
            raise ValueError("roster verifier ids must be unique")
        if tuple(item.verifier_id for item in self.entries) != self.ordered_verifier_ids:
            raise ValueError("roster entry order does not match committed ids")
        if tuple(item.roster_rank for item in self.entries) != tuple(
            range(len(self.entries))
        ):
            raise ValueError("roster ranks must be contiguous")
        if len({item.job_id for item in self.primary_committees}) != len(
            self.primary_committees
        ):
            raise ValueError("primary committee job ids must be unique")
        if any(
            verifier_id not in set(self.ordered_verifier_ids)
            for committee in self.primary_committees
            for verifier_id in committee.verifier_ids
        ):
            raise ValueError("primary committee member is absent from the roster")
        if len(self.order_seed_sha256) != 64 or len(self.roster_sha256) != 64:
            raise ValueError("roster commitments must be SHA-256 hex digests")


@dataclass(frozen=True)
class RobustRosterCertificate:
    """Exact public-only bounded-removal certificate for one locked roster."""

    scenario_id: str
    parameter_set_id: str
    removal_budget: int
    reliability_target: float
    roster_sha256: str
    public_input_sha256: str
    robust_reliability: float
    reliability_gap_to_target: float
    worst_removed_verifier_ids: tuple[str, ...]
    best_prefix_verifier_ids: tuple[str, ...]
    best_prefix_reliability: float
    removal_set_count: int
    odd_prefix_evaluation_count: int
    passed: bool

    def __post_init__(self) -> None:
        if not self.scenario_id or not self.parameter_set_id:
            raise ValueError("robust certificate identity must be complete")
        if self.removal_budget < 0:
            raise ValueError("robust certificate removal budget must be non-negative")
        if not 0.5 < self.reliability_target <= 1.0:
            raise ValueError("robust certificate target must lie in (0.5, 1]")
        if len(self.roster_sha256) != 64 or len(self.public_input_sha256) != 64:
            raise ValueError("robust certificate hashes must be SHA-256 digests")
        for name in ("robust_reliability", "best_prefix_reliability"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")
        expected_gap = max(0.0, self.reliability_target - self.robust_reliability)
        if abs(self.reliability_gap_to_target - expected_gap) > 1e-12:
            raise ValueError("robust certificate target gap is inconsistent")
        if self.removal_set_count <= 0 or self.odd_prefix_evaluation_count < 0:
            raise ValueError("robust certificate enumeration counts are invalid")
        if len(set(self.worst_removed_verifier_ids)) != len(
            self.worst_removed_verifier_ids
        ):
            raise ValueError("robust certificate removal set must be unique")
        if len(self.worst_removed_verifier_ids) > self.removal_budget:
            raise ValueError("robust certificate removal set exceeds its budget")
        if self.best_prefix_verifier_ids and len(self.best_prefix_verifier_ids) % 2 == 0:
            raise ValueError("robust certificate best prefix must be odd")
        if self.passed != (
            self.robust_reliability + 1e-12 >= self.reliability_target
        ):
            raise ValueError("robust certificate pass flag is inconsistent")


@dataclass(frozen=True)
class ActivationWave:
    scenario_id: str
    job_id: str
    wave_index: int
    role: str
    verifier_ids: tuple[str, ...]
    activation_nonce: str
    roster_sha256: str

    def __post_init__(self) -> None:
        if not self.scenario_id or not self.job_id or not self.activation_nonce:
            raise ValueError("activation wave identity must be complete")
        if self.wave_index < 0 or self.role not in {"primary", "reserve"}:
            raise ValueError("activation wave index/role is invalid")
        if not self.verifier_ids or len(set(self.verifier_ids)) != len(
            self.verifier_ids
        ):
            raise ValueError("activation wave verifier ids must be non-empty and unique")
        if len(self.roster_sha256) != 64:
            raise ValueError("activation wave requires a roster SHA-256")


@dataclass(frozen=True)
class TailBudgetDecision:
    """Public pre-wave committed-expenditure decision for RTR activation."""

    scenario_id: str
    job_id: str
    wave_index: int
    status: str
    committed_expenditure_before: float
    next_wave_committed_cost: float
    committed_expenditure_after: float
    hard_cap_expenditure: float
    abstain_reason: str
    candidate_verifier_ids: tuple[str, ...] = ()
    wave: ActivationWave | None = None

    def __post_init__(self) -> None:
        if not self.scenario_id or not self.job_id or self.wave_index <= 0:
            raise ValueError("tail-budget decision identity is invalid")
        if self.status not in {
            "ACTIVATE",
            "ABSTAIN_TAIL_BUDGET_EXHAUSTED",
            "ABSTAIN_RESERVE_EXHAUSTED",
        }:
            raise ValueError("tail-budget decision status is invalid")
        for name in (
            "committed_expenditure_before",
            "next_wave_committed_cost",
            "committed_expenditure_after",
            "hard_cap_expenditure",
        ):
            _finite_nonnegative(getattr(self, name), name)
        if self.status == "ACTIVATE":
            if self.wave is None or self.abstain_reason:
                raise ValueError("activation decision requires a wave and no abstain reason")
            if self.candidate_verifier_ids != self.wave.verifier_ids:
                raise ValueError("activation decision candidate ids drift from its wave")
            if self.committed_expenditure_after > self.hard_cap_expenditure + 1e-12:
                raise ValueError("activation decision exceeds the hard cap")
        else:
            if self.wave is not None or self.abstain_reason != self.status:
                raise ValueError("abstention decision cannot allocate a wave")
        if len(set(self.candidate_verifier_ids)) != len(
            self.candidate_verifier_ids
        ):
            raise ValueError("tail-budget candidate verifier ids must be unique")


@dataclass(frozen=True)
class CommitteeReconstitution:
    scenario_id: str
    job_id: str
    wave_index: int
    status: str
    verifier_ids: tuple[str, ...]
    exact_reliability: float
    reliability_target: float
    maximum_attainable_reliability: float
    reliability_gap_to_target: float
    pass_survivor_count: int
    activated_count: int
    reserve_cap: int

    def __post_init__(self) -> None:
        if not self.scenario_id or not self.job_id:
            raise ValueError("reconstitution identity must be complete")
        if self.wave_index < 0 or self.status not in RECONSTITUTION_STATUSES:
            raise ValueError("reconstitution wave/status is invalid")
        if not 0 <= self.exact_reliability <= 1:
            raise ValueError("reconstitution reliability must lie in [0, 1]")
        if not 0.5 < self.reliability_target <= 1:
            raise ValueError("reconstitution target must lie in (0.5, 1]")
        if not 0 <= self.maximum_attainable_reliability <= 1:
            raise ValueError(
                "maximum attainable reconstitution reliability must lie in [0, 1]"
            )
        expected_gap = max(
            0.0, self.reliability_target - self.maximum_attainable_reliability
        )
        if (
            not math.isfinite(self.reliability_gap_to_target)
            or self.reliability_gap_to_target < 0
            or abs(self.reliability_gap_to_target - expected_gap) > 1e-12
        ):
            raise ValueError("reconstitution reliability gap is inconsistent")
        if min(self.pass_survivor_count, self.activated_count, self.reserve_cap) < 0:
            raise ValueError("reconstitution counts must be non-negative")
        if self.activated_count > self.reserve_cap:
            raise ValueError("reconstitution activated count exceeds the cap")
        if self.status == "RECONSTITUTED":
            if not self.verifier_ids or len(self.verifier_ids) % 2 == 0:
                raise ValueError("reconstituted committee must be non-empty and odd")
            if self.exact_reliability + 1e-12 < self.reliability_target:
                raise ValueError("reconstituted committee misses the locked target")
        elif self.verifier_ids:
            raise ValueError("non-final reconstitution cannot expose a voting committee")


@dataclass(frozen=True)
class ReserveTransferRecord:
    verifier_id: str
    job_id: str
    role: str
    behavior: str
    status: str
    activated: bool
    responsive: bool
    availability_payment: float
    activation_payment: float
    total_service_payment: float
    availability_bond: float
    service_bond: float
    bond_top_up: float
    slashed_bond: float
    owner_expenditure: float
    conservative_ir_margin: float

    def __post_init__(self) -> None:
        if not self.verifier_id or not self.job_id or not self.behavior:
            raise ValueError("reserve transfer identity must be complete")
        if self.role not in {"primary", "reserve"}:
            raise ValueError("reserve transfer role is invalid")
        if self.status not in {
            "PRIMARY_ACTIVE",
            "RESERVE_ACTIVE",
            "RESERVE_ACTIVE_DROPOUT",
            "RESERVE_AVAILABLE",
            "RESERVE_DROPOUT",
        }:
            raise ValueError("reserve transfer status is invalid")
        for name in (
            "availability_payment",
            "activation_payment",
            "total_service_payment",
            "availability_bond",
            "service_bond",
            "bond_top_up",
            "slashed_bond",
            "owner_expenditure",
        ):
            _finite_nonnegative(getattr(self, name), name)
        if not math.isfinite(float(self.conservative_ir_margin)):
            raise ValueError("reserve IR margin must be finite")
        if abs(
            self.availability_payment
            + self.activation_payment
            - self.total_service_payment
        ) > 1e-12:
            raise ValueError("availability credit and activation payment do not conserve")


@dataclass(frozen=True)
class PublicReplaySegment:
    segment_id: str
    payload_digest: str
    public_features: tuple[float, ...] = ()
    public_nonce: str = ""
    serialized_size: int = 0
    tensor_count: int = 0
    checkpoint_count: int = 0
    batch_count: int = 0

    def __post_init__(self) -> None:
        if not self.segment_id or not self.payload_digest:
            raise ValueError("segment_id and payload_digest must be non-empty")
        if any(not math.isfinite(float(value)) for value in self.public_features):
            raise ValueError("public features must be finite")
        if any(
            int(value) < 0
            for value in (
                self.serialized_size,
                self.tensor_count,
                self.checkpoint_count,
                self.batch_count,
            )
        ):
            raise ValueError("public replay counts/sizes must be non-negative")


@dataclass(frozen=True)
class PublicTaskBundle:
    scenario_id: str
    verifier_id: str
    segments: tuple[PublicReplaySegment, ...]
    job_id: str = ""
    bundle_nonce: str = ""

    def __post_init__(self) -> None:
        if not self.scenario_id or not self.verifier_id or not self.segments:
            raise ValueError("public task bundle fields must be non-empty")
        ids = tuple(item.segment_id for item in self.segments)
        if len(set(ids)) != len(ids):
            raise ValueError("public task bundle segment ids must be unique")


@dataclass(frozen=True)
class SealedSegmentTruth:
    segment_id: str
    expected_verdict: bool
    is_sentinel: bool
    technical_failure: bool = False
    source_segment_id: str = ""
    job_id: str = ""
    proof_sha256: str = ""


@dataclass(frozen=True)
class SealedEvaluationTruth:
    scenario_id: str
    segments: tuple[SealedSegmentTruth, ...]

    def __post_init__(self) -> None:
        if not self.scenario_id or not self.segments:
            raise ValueError("evaluation bundle requires scenario and segments")
        ids = tuple(item.segment_id for item in self.segments)
        if len(set(ids)) != len(ids):
            raise ValueError("evaluation segment ids must be unique")


def report_commitment(
    scenario_id: str,
    verifier_id: str,
    ordered_segment_ids: Sequence[str],
    verdicts: Sequence[bool | None],
    nonce: str,
    job_id: str = "",
) -> str:
    payload = {
        "domain": (
            "sevc-verifier-assignment-report-v2"
            if job_id
            else "sevc-verifier-report-v1"
        ),
        "scenario_id": scenario_id,
        "verifier_id": verifier_id,
        "ordered_segment_ids": list(ordered_segment_ids),
        "verdicts": list(verdicts),
        "nonce": nonce,
    }
    if job_id:
        payload["job_id"] = job_id
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CommittedVerifierReport:
    scenario_id: str
    verifier_id: str
    ordered_segment_ids: tuple[str, ...]
    verdicts: tuple[bool | None, ...]
    nonce: str
    commitment: str
    committed: bool
    revealed: bool
    job_id: str = ""

    def __post_init__(self) -> None:
        if not self.scenario_id or not self.verifier_id or not self.nonce:
            raise ValueError("report identity and nonce must be non-empty")
        if not self.ordered_segment_ids or len(self.ordered_segment_ids) != len(
            self.verdicts
        ):
            raise ValueError("report segment ids and verdicts must have equal length")
        if len(set(self.ordered_segment_ids)) != len(self.ordered_segment_ids):
            raise ValueError("report segment ids must be unique")
        expected = report_commitment(
            self.scenario_id,
            self.verifier_id,
            self.ordered_segment_ids,
            self.verdicts,
            self.nonce,
            self.job_id,
        )
        if self.commitment != expected:
            raise ValueError("report commitment does not match its payload")
        if self.revealed and not self.committed:
            raise ValueError("an uncommitted report cannot be revealed")

    @classmethod
    def create(
        cls,
        *,
        scenario_id: str,
        verifier_id: str,
        ordered_segment_ids: Sequence[str],
        verdicts: Sequence[bool | None],
        nonce: str,
        job_id: str = "",
    ) -> "CommittedVerifierReport":
        ids = tuple(str(value) for value in ordered_segment_ids)
        values = tuple(verdicts)
        committed = any(value is not None for value in values)
        return cls(
            scenario_id=scenario_id,
            verifier_id=verifier_id,
            ordered_segment_ids=ids,
            verdicts=values,
            nonce=nonce,
            commitment=report_commitment(
                scenario_id, verifier_id, ids, values, nonce, job_id
            ),
            committed=committed,
            revealed=committed,
            job_id=job_id,
        )


@dataclass(frozen=True)
class VerifierSettlement:
    verifier_id: str
    status: str
    service_fee: float
    refundable_bond: float
    slashed_bond: float
    owner_expenditure: float
    verifier_cost: float
    effort_fraction: float
    committed: bool
    revealed: bool
    accepted_report: bool
    technical_failure: bool
    abstained: bool
    diagnostics: tuple[tuple[str, Any], ...] = ()
    job_id: str = ""

    def __post_init__(self) -> None:
        if self.status not in SETTLEMENT_STATUSES:
            raise ValueError(f"unknown settlement status: {self.status}")
        for name in (
            "service_fee",
            "refundable_bond",
            "slashed_bond",
            "owner_expenditure",
            "verifier_cost",
            "effort_fraction",
        ):
            _finite_nonnegative(getattr(self, name), name)


@dataclass(frozen=True)
class SettlementRequest:
    scenario_id: str
    allocation: AllocationOutcome
    reports: tuple[CommittedVerifierReport, ...]
    evaluation_bundle: SealedEvaluationTruth
    verifier_costs: tuple[tuple[str, float], ...]
    effort_fractions: tuple[tuple[str, float], ...]
    assignment_costs: tuple[tuple[str, str, float], ...] = ()
    assignment_effort_fractions: tuple[tuple[str, str, float], ...] = ()
    assignment_replay_seconds: tuple[tuple[str, str, float], ...] = ()
    commit_window_closed: bool = False
    adjudication_records: tuple[AdjudicationRecord, ...] = ()

    def __post_init__(self) -> None:
        if self.scenario_id != self.evaluation_bundle.scenario_id:
            raise ValueError("settlement and evaluation scenario ids must match")
        report_ids = tuple((item.verifier_id, item.job_id) for item in self.reports)
        if len(set(report_ids)) != len(report_ids):
            raise ValueError("settlement report assignment ids must be unique")


@dataclass(frozen=True)
class SettlementOutcome:
    status: str
    service_fee: float
    refundable_bond: float
    slashed_bond: float
    owner_expenditure: float
    verifier_cost: float
    effort_fraction: float
    committed: int
    revealed: int
    accepted_report: int
    technical_failure: int
    abstained: int
    rows: tuple[VerifierSettlement, ...]
    diagnostics: tuple[tuple[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if self.status not in SETTLEMENT_STATUSES:
            raise ValueError(f"unknown settlement outcome status: {self.status}")
        for name in (
            "service_fee",
            "refundable_bond",
            "slashed_bond",
            "owner_expenditure",
            "verifier_cost",
            "effort_fraction",
        ):
            _finite_nonnegative(getattr(self, name), name)
        for name in (
            "committed",
            "revealed",
            "accepted_report",
            "technical_failure",
            "abstained",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} count must be non-negative")


@dataclass(frozen=True)
class AdjudicationRecord:
    scenario_id: str
    verifier_id: str
    job_id: str
    initial_status: str
    final_status: str
    first_pass_mismatch_count: int
    rerun_count: int
    confirmed_mismatch_count: int
    technical_failure_count: int
    rerun_cache_keys: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.scenario_id or not self.verifier_id or not self.job_id:
            raise ValueError("adjudication identity must be complete")
        if self.final_status not in SETTLEMENT_STATUSES:
            raise ValueError("adjudication final status is invalid")
        for name in (
            "first_pass_mismatch_count",
            "rerun_count",
            "confirmed_mismatch_count",
            "technical_failure_count",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")


@dataclass(frozen=True)
class TrainerJobTruth:
    scenario_id: str
    job_id: str
    trainer_id: str
    valid_submission: bool

    def __post_init__(self) -> None:
        if not self.scenario_id or not self.job_id or not self.trainer_id:
            raise ValueError("trainer/job truth identity must be complete")


@dataclass(frozen=True)
class TrainerArbitrationResult:
    scenario_id: str
    job_id: str
    variant_id: str
    ground_truth_valid: bool
    predicted_valid: bool | None
    correct: bool
    covered: bool
    abstained: bool
    surviving_verifier_ids: tuple[str, ...]
    exact_reliability: float
    true_positive: int
    true_negative: int
    false_positive: int
    false_negative: int

    def __post_init__(self) -> None:
        if not self.scenario_id or not self.job_id or not self.variant_id:
            raise ValueError("arbitration identity must be complete")
        if self.abstained != (self.predicted_valid is None):
            raise ValueError("abstention and predicted verdict disagree")
        if self.covered == self.abstained:
            raise ValueError("covered and abstained must be complements")
        if not 0 <= self.exact_reliability <= 1:
            raise ValueError("arbitration reliability must lie in [0, 1]")
        confusion = (
            self.true_positive,
            self.true_negative,
            self.false_positive,
            self.false_negative,
        )
        if any(value not in {0, 1} for value in confusion) or sum(confusion) > 1:
            raise ValueError("arbitration confusion indicators are invalid")


@dataclass(frozen=True)
class VariantEvaluationRow:
    protocol_version: str
    scenario_id: str
    variant_id: str
    implementation_key: str
    parameter_set_id: str
    seed: int
    denominator: int
    status: str
    selected_verifiers: tuple[str, ...]
    committees: tuple[CommitteeAssignment, ...]
    exact_reliabilities: tuple[float, ...]
    service_fee: float
    refundable_bond: float
    slashed_bond: float
    owner_expenditure: float
    verifier_cost: float
    effort_fraction: float
    committed: int
    revealed: int
    accepted_report: int
    technical_failure: int
    abstained: int
    diagnostics: tuple[tuple[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if not self.protocol_version or not self.scenario_id or "@" not in self.variant_id:
            raise ValueError("evaluation row identities are incomplete")
        if self.variant_id != f"{self.implementation_key}@{self.parameter_set_id}":
            raise ValueError("evaluation row variant identity is inconsistent")
        if self.seed < 0 or self.denominator < 0:
            raise ValueError("evaluation row seed/denominator are invalid")
        for name in (
            "service_fee",
            "refundable_bond",
            "slashed_bond",
            "owner_expenditure",
            "verifier_cost",
            "effort_fraction",
        ):
            _finite_nonnegative(getattr(self, name), name)


@dataclass(frozen=True)
class VariantCapabilities:
    exact_reliability: bool
    hidden_task_integrity: bool
    hidden_effort_incentive: bool
    escrow_settlement: bool
    explicit_abstention: bool
    legacy_compatibility: bool


_VALUE_TYPES: dict[str, type] = {
    "bool": bool,
    "float": float,
    "int": int,
    "str": str,
}


@dataclass(frozen=True)
class ParameterRule:
    name: str
    value_type: str
    default: Any
    required: bool = True
    minimum: float | None = None
    maximum: float | None = None
    choices: tuple[Any, ...] = ()

    def __post_init__(self) -> None:
        if not self.name or self.value_type not in _VALUE_TYPES:
            raise ValueError("invalid parameter rule")
        self.validate(self.default)

    def validate(self, value: Any) -> Any:
        expected = _VALUE_TYPES[self.value_type]
        if expected is float:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{self.name} must be a float")
            normalized: Any = float(value)
            if not math.isfinite(normalized):
                raise ValueError(f"{self.name} must be finite")
        elif expected is int:
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{self.name} must be an int")
            normalized = int(value)
        elif not isinstance(value, expected):
            raise ValueError(f"{self.name} must be a {self.value_type}")
        else:
            normalized = value
        if self.minimum is not None and normalized < self.minimum:
            raise ValueError(f"{self.name} must be >= {self.minimum}")
        if self.maximum is not None and normalized > self.maximum:
            raise ValueError(f"{self.name} must be <= {self.maximum}")
        if self.choices and normalized not in self.choices:
            raise ValueError(f"{self.name} must be one of {self.choices}")
        return normalized


@dataclass(frozen=True)
class VerifierIncentiveVariant:
    implementation_key: str
    schema_version: str
    allocation_policy_key: str
    task_policy_key: str
    settlement_policy_key: str
    capabilities: VariantCapabilities
    parameter_schema: tuple[ParameterRule, ...]

    def __post_init__(self) -> None:
        keys = (
            self.implementation_key,
            self.allocation_policy_key,
            self.task_policy_key,
            self.settlement_policy_key,
        )
        if any(not key or key != key.strip().lower() for key in keys):
            raise ValueError("variant and policy keys must be normalized")
        names = tuple(rule.name for rule in self.parameter_schema)
        if len(set(names)) != len(names):
            raise ValueError("variant parameter names must be unique")

    def variant_id(self, parameter_set_id: str) -> str:
        normalized = parameter_set_id.strip().lower()
        if not normalized or normalized != parameter_set_id or "@" in normalized:
            raise ValueError("parameter_set_id must be a normalized key")
        return f"{self.implementation_key}@{normalized}"

    def default_parameter_values(self) -> dict[str, Any]:
        return {rule.name: rule.default for rule in self.parameter_schema}

    def validate_parameters(self, parameters: Mapping[str, Any]) -> dict[str, Any]:
        rules = {rule.name: rule for rule in self.parameter_schema}
        unknown = sorted(set(parameters) - set(rules))
        if unknown:
            raise ValueError(f"unknown parameter(s): {', '.join(unknown)}")
        missing = sorted(
            rule.name
            for rule in self.parameter_schema
            if rule.required and rule.name not in parameters
        )
        if missing:
            raise ValueError(f"missing required parameter(s): {', '.join(missing)}")
        result = {
            name: rules[name].validate(value)
            for name, value in sorted(parameters.items())
        }
        if "failure_threshold" in result and "sentinel_count" in result:
            if result["failure_threshold"] > result["sentinel_count"]:
                raise ValueError("failure_threshold cannot exceed sentinel_count")
        return result

    def to_dict(self) -> dict[str, Any]:
        return dataclass_to_jsonable(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "VerifierIncentiveVariant":
        capabilities = VariantCapabilities(**dict(payload["capabilities"]))
        rules = tuple(
            ParameterRule(
                **{
                    **dict(item),
                    "choices": tuple(dict(item).get("choices", ())),
                }
            )
            for item in payload["parameter_schema"]
        )
        return cls(
            implementation_key=str(payload["implementation_key"]),
            schema_version=str(payload["schema_version"]),
            allocation_policy_key=str(payload["allocation_policy_key"]),
            task_policy_key=str(payload["task_policy_key"]),
            settlement_policy_key=str(payload["settlement_policy_key"]),
            capabilities=capabilities,
            parameter_schema=rules,
        )


class AllocationPolicy(Protocol):
    def __call__(
        self,
        request: MarketRequest | RCPHSEMarketRequest,
        parameters: Mapping[str, Any],
    ) -> AllocationOutcome: ...


class TaskPolicy(Protocol):
    def __call__(
        self,
        public_view: Any,
        evaluation_bundle: SealedEvaluationTruth,
        allocation: AllocationOutcome,
        parameters: Mapping[str, Any],
    ) -> tuple[PublicTaskBundle, ...]: ...


class SettlementPolicy(Protocol):
    def __call__(
        self, request: SettlementRequest, parameters: Mapping[str, Any]
    ) -> SettlementOutcome: ...


ALLOCATION_POLICIES: Registry[AllocationPolicy] = Registry(
    "verifier allocation policy"
)
TASK_POLICIES: Registry[TaskPolicy] = Registry("verifier task policy")
SETTLEMENT_POLICIES: Registry[SettlementPolicy] = Registry(
    "verifier settlement policy"
)
VERIFIER_INCENTIVE_VARIANTS: Registry[VerifierIncentiveVariant] = Registry(
    "verifier incentive variant"
)


def variant_identity(implementation_key: str, parameter_set_id: str) -> str:
    return VERIFIER_INCENTIVE_VARIANTS.get(implementation_key).variant_id(
        parameter_set_id
    )
