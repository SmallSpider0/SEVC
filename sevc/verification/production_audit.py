"""Commit-before-challenge production-audit primitives.

The owner selects production transitions only after both the trainer trajectory
and verifier report commitments exist.  The public task representation carries
opaque aliases and commitments; audit membership is never serialized into the
pre-challenge verifier view.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

from dataclasses import asdict, replace
import secrets
import time
from typing import Callable

from sevc.core.registry import Registry
from sevc.incentives.verifier_protocol import (
    CommittedVerifierReport, SealedSegmentTruth, SealedEvaluationTruth,
)
from sevc.verification.paid_replay_service import (
    OwnerProbeReferences, OwnerProductionReferences, identity, settle_service,
)
from sevc.verification.verifier_task_policies import settle_threshold_assignment


PROTOCOL_STATES = (
    "TERMS_PUBLISHED",
    "TRAINER_ALLOCATED",
    "TRAJECTORY_COMMITTED",
    "REPLAY_ASSIGNED",
    "REPORT_COMMITTED",
    "OWNER_RANDOMNESS_REVEALED",
    "PRODUCTION_AUDIT_SELECTED",
    "REPORT_REVEALED",
    "SETTLED_OR_DEFERRED",
)


def canonical_commitment(domain: str, payload: Any) -> str:
    """Return a domain-separated SHA-256 commitment to a JSON value."""

    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(domain.encode("utf-8") + b"\0" + encoded).hexdigest()


def select_audit_indices(
    *,
    transition_count: int,
    audit_count: int,
    trainer_commitment: str,
    report_commitment: str,
    owner_nonce: str,
) -> tuple[int, ...]:
    """Select a deterministic sample without replacement after commitments."""

    if transition_count <= 0 or not 0 <= audit_count <= transition_count:
        raise ValueError("invalid audit dimensions")
    if not trainer_commitment or not report_commitment or not owner_nonce:
        raise ValueError("both commitments and fresh owner randomness are required")
    ranked = []
    for index in range(transition_count):
        material = (
            f"sevc-production-audit-v1|{trainer_commitment}|{report_commitment}|"
            f"{owner_nonce}|{index}"
        ).encode("utf-8")
        ranked.append((hashlib.sha256(material).digest(), index))
    return tuple(sorted(index for _, index in sorted(ranked)[:audit_count]))


@dataclass
class ProductionAuditTranscript:
    """Small executable state machine enforcing the candidate timing order."""

    state_index: int = -1
    trainer_commitment: str = ""
    report_commitment: str = ""
    owner_nonce: str = ""
    audit_indices: tuple[int, ...] = ()

    @property
    def state(self) -> str:
        return "INITIAL" if self.state_index < 0 else PROTOCOL_STATES[self.state_index]

    def advance(self, target: str, **values: Any) -> None:
        expected_index = self.state_index + 1
        if expected_index >= len(PROTOCOL_STATES) or PROTOCOL_STATES[expected_index] != target:
            raise ValueError(f"invalid transition {self.state} -> {target}")
        if target == "TRAJECTORY_COMMITTED":
            self.trainer_commitment = str(values.get("trainer_commitment", ""))
            if not self.trainer_commitment:
                raise ValueError("trainer commitment is required")
        elif target == "REPORT_COMMITTED":
            self.report_commitment = str(values.get("report_commitment", ""))
            if not self.report_commitment:
                raise ValueError("report commitment is required")
        elif target == "OWNER_RANDOMNESS_REVEALED":
            if not self.trainer_commitment or not self.report_commitment:
                raise ValueError("owner randomness requires both commitments")
            self.owner_nonce = str(values.get("owner_nonce", ""))
            if not self.owner_nonce:
                raise ValueError("owner nonce is required")
        elif target == "PRODUCTION_AUDIT_SELECTED":
            self.audit_indices = select_audit_indices(
                transition_count=int(values["transition_count"]),
                audit_count=int(values["audit_count"]),
                trainer_commitment=self.trainer_commitment,
                report_commitment=self.report_commitment,
                owner_nonce=self.owner_nonce,
            )
        self.state_index = expected_index


def hypergeometric_hit_probability(q: int, forged_count: int, audit_count: int) -> float:
    """Exact probability that a sample without replacement hits a fixed forged set."""

    if q <= 0 or not 0 <= forged_count <= q or not 0 <= audit_count <= q:
        raise ValueError("invalid hypergeometric dimensions")
    if forged_count == 0 or audit_count == 0:
        return 0.0
    if audit_count > q - forged_count:
        return 1.0
    return 1.0 - math.comb(q - forged_count, audit_count) / math.comb(q, audit_count)


def leakage_adjusted_lower_bound(
    q: int, forged_count: int, audit_count: int, leakage_advantage: float
) -> float:
    if not 0.0 <= leakage_advantage <= 1.0:
        raise ValueError("leakage advantage must lie in [0, 1]")
    return max(
        0.0,
        hypergeometric_hit_probability(q, forged_count, audit_count)
        - leakage_advantage,
    )


FIXED_FORGED_SET_ATTACKS = {
    "trainer_zero_effort",
    "trainer_half_effort",
    "trainer_three_quarter_effort",
    "false_endorsement",
    "false_accusation",
}


def map_attack_to_bound(attack: str, forged_count: int | None) -> Mapping[str, Any]:
    """Map only pre-commit fixed-set deviations to the hypergeometric bound."""

    mapped = attack in FIXED_FORGED_SET_ATTACKS and forged_count is not None
    return {
        "attack": attack,
        "fixed_precommit_forged_set": mapped,
        "forged_count": int(forged_count) if mapped else None,
        "bound": "hypergeometric_leakage_lower_bound" if mapped else "NONE",
        "reason": (
            "fixed before both commitments"
            if mapped
            else "attack has no fixed pre-commit forged-transition set"
        ),
    }


def audit_public_fields(
    public_records: Sequence[Mapping[str, Any]],
    private_membership_digests: Sequence[str],
) -> Mapping[str, Any]:
    """Detect exact-link fields and report a deterministic membership advantage.

    Advantage is the improvement over chance of the best exact field-link rule.
    This intentionally narrow audit is complemented by explicit timing/cache/network
    obligations in the evidence packet; those side channels are not claimed solved.
    """

    private = set(private_membership_digests)
    link_fields: dict[str, int] = {}
    for record in public_records:
        for field, value in record.items():
            if isinstance(value, str) and value in private:
                link_fields[field] = link_fields.get(field, 0) + 1
    exact_link = bool(link_fields)
    return {
        "record_count": len(public_records),
        "private_membership_count": len(private),
        "exact_link_fields": link_fields,
        "exact_membership_oracle_available": exact_link,
        "membership_accuracy": 1.0 if exact_link else 0.5,
        "membership_advantage": 0.5 if exact_link else 0.0,
    }


# Paid-replay admission adapters reuse the canonical sampling primitive above.


@dataclass(frozen=True)
class AdmissionPolicy:
    probes: bool
    audit_count: int


ADMISSION_POLICIES = Registry("paid replay admission policy")
for _key, _probes, _count in (
    ("rcmp-source-coupled", True, 0),
    ("rcmp-production-audit-8", True, 8),
    ("rcmp-production-audit-all", True, 32),
    ("production-audit-only-8", False, 8),
    ("production-audit-only-all", False, 32),
):
    ADMISSION_POLICIES.add(_key, AdmissionPolicy(_probes, _count))


def audit_sample(secret_hex: str, report_commitment: str, production_ids, count: int):
    ids = tuple(sorted(production_ids))
    if len(ids) != len(set(ids)) or not 0 <= count <= len(ids):
        raise ValueError("invalid production sampling population/budget")
    if len(bytes.fromhex(secret_hex)) != 32:
        raise ValueError("audit salt must contain 256 private random bits")
    indices = select_audit_indices(
        transition_count=len(ids), audit_count=count,
        trainer_commitment=identity(ids), report_commitment=report_commitment,
        owner_nonce=secret_hex,
    )
    return tuple(ids[index] for index in indices)


class CommittedProductionAudit:
    """Single-use private draw; construction precedes the verifier report."""
    def __init__(self, assignment_id: str, emit: Callable[[dict], None]):
        self.assignment_id, self.emit = assignment_id, emit
        self._secret = secrets.token_hex(32)
        self._used = False
        self.salt_commitment = identity(["production-audit-salt-v1", assignment_id, self._secret])
        emit({"phase": "AUDIT_SALT_COMMITTED", "assignment_id": assignment_id,
              "salt_commitment": self.salt_commitment, "monotonic": time.monotonic()})

    def open(self, report: CommittedVerifierReport, production_ids, count: int):
        if self._used:
            raise ValueError("an audit challenge cannot be resampled")
        if not report.committed or not report.revealed:
            raise ValueError("audit sampling requires a committed, revealed report")
        self._used = True
        selected = audit_sample(self._secret, report.commitment, production_ids, count)
        receipt = {"phase": "AUDIT_OPENED", "assignment_id": self.assignment_id,
            "monotonic": time.monotonic(),
            "salt_commitment": self.salt_commitment, "salt": self._secret,
            "report_commitment": report.commitment, "production_ids": sorted(production_ids),
            "selected_ids": list(selected), "audit_count": count}
        self.emit(receipt)
        return selected, receipt


def delivered_policy_tasks(prepared, policy_key):
    policy = ADMISSION_POLICIES.get(policy_key)
    tasks = prepared.mechanisms["rcmp-source-coupled"]
    production = {r.task_id for r in prepared.production_references.items}
    return tasks if policy.probes else tuple(t for t in tasks if t.task_id in production)


def settle_audited_service(report: CommittedVerifierReport, *, policy_key: str,
        production: OwnerProductionReferences, probes: OwnerProbeReferences,
        challenge: CommittedProductionAudit, cost_seconds: float, effort_fraction: float):
    """Use only typed owner receipts; evaluator rows never enter this interface."""
    if type(production) is not OwnerProductionReferences or type(probes) is not OwnerProbeReferences:
        raise TypeError("restricted owner production/probe references required")
    if len(production.items) != 32:
        raise ValueError("the frozen production audit policies require exactly 32 production tasks")
    # Reconstruct to recheck a report even if a caller illicitly modified a frozen instance.
    report = CommittedVerifierReport(**asdict(report))
    if any(v is not None and type(v) is not bool for v in report.verdicts):
        raise TypeError("report answers must be booleans or explicit omissions")
    policy = ADMISSION_POLICIES.get(policy_key)
    refs = {r.task_id: r for r in production.items}
    expected = set(refs) | (set(dict(probes.answers)) if policy.probes else set())
    if set(report.ordered_segment_ids) != expected:
        raise ValueError("report does not cover the exact policy delivery population")
    if not report.committed or not report.revealed:
        raise ValueError("unrevealed report supplies no eligible audited production")
    selected, receipt = challenge.open(report, refs, policy.audit_count)
    if not policy.audit_count:
        return settle_service(report, probes, cost_seconds=cost_seconds,
                              effort_fraction=effort_fraction), receipt
    truth = SealedEvaluationTruth(report.scenario_id, tuple(
        SealedSegmentTruth(k, refs[k].answer if k in selected else False, k in selected)
        for k in report.ordered_segment_ids))
    audited = settle_threshold_assignment(report, truth, failure_threshold=1, fee=1.25, bond=.5,
        cost=cost_seconds, effort=effort_fraction, sentinel_only=True)
    if policy.probes:
        probed = settle_service(report, probes, cost_seconds=cost_seconds,
                                effort_fraction=effort_fraction)
        result = probed if probed.status != "PASS" else audited
        probe_status = probed.status
    else:
        result, probe_status = audited, "NOT_USED"
    return replace(result, job_id=report.job_id, diagnostics=result.diagnostics + (
        ("production_audit_status", audited.status), ("probe_status", probe_status),
        ("audit_count", policy.audit_count))), receipt
