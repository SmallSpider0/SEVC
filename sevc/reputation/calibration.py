"""Owner-referenced, budgeted calibration before production admission."""
from __future__ import annotations

from dataclasses import dataclass, field
import math

from sevc.core.statistical_bounds import clopper_pearson_bound


@dataclass
class PaidCalibrationLedger:
    verifier_id: str
    budget: float
    fee: float
    owner_reserve: float = 0.
    reservations: dict = field(default_factory=dict, init=False)
    observations: list = field(default_factory=list, init=False)

    def __post_init__(self):
        if not self.verifier_id or any(not math.isfinite(x) or x < 0
                                      for x in (self.budget, self.fee, self.owner_reserve)):
            raise ValueError("invalid calibration identity or budget")

    def reserve(self, assignment_id: str, independent_block_id: str):
        """Calibration needs funds, not an already-qualified production score."""
        if not assignment_id or not independent_block_id or assignment_id in self.reservations:
            raise ValueError("calibration assignment identity already used or empty")
        if independent_block_id in self.reservations.values():
            raise ValueError("repeated source block cannot be an independent calibration trial")
        if (len(self.reservations) + 1) * (self.fee + self.owner_reserve) > self.budget + 1e-12:
            raise ValueError("calibration budget exhausted")
        self.reservations[assignment_id] = independent_block_id

    def record(self, assignment_id, settlement, *, timely: bool,
               reference_correct: bool | None, reference_receipt_sha256: str | None,
               owner_cost: float = 0.):
        if assignment_id not in self.reservations or any(
                r["assignment_id"] == assignment_id for r in self.observations):
            raise ValueError("calibration observation lacks an unused reservation")
        if settlement.verifier_id != self.verifier_id or type(timely) is not bool:
            raise ValueError("calibration verifier or timeliness mismatch")
        if (not math.isfinite(owner_cost) or not 0 <= owner_cost <= self.owner_reserve
                or settlement.service_fee > self.fee + 1e-12):
            raise ValueError("calibration payment exceeded reservation")
        admitted = settlement.status == "PASS" and settlement.accepted_report
        if reference_correct is not None:
            if not admitted or type(reference_correct) is not bool:
                raise ValueError("correctness calibration requires an admitted audited report")
            if (not isinstance(reference_receipt_sha256, str) or len(reference_receipt_sha256) != 64
                    or any(c not in "0123456789abcdef" for c in reference_receipt_sha256)):
                raise ValueError("correctness requires a bound owner replay receipt")
        self.observations.append({"assignment_id": assignment_id,
            "independent_block_id": self.reservations[assignment_id],
            "timely_admitted": timely and admitted, "admitted": admitted,
            "correct": reference_correct, "reference_receipt_sha256": reference_receipt_sha256,
            "status": settlement.status, "fee_paid": settlement.service_fee, "owner_cost": owner_cost})

    def summary(self, *, alpha: float = .05, independent_stationary_blocks: bool = False):
        if not 0 < alpha < 1:
            raise ValueError("alpha must lie in (0, 1)")
        rows = self.observations
        audited = [r for r in rows if r["admitted"] and r["correct"] is not None]
        all_admitted_audited = all(not r["admitted"] or r["correct"] is not None for r in rows)
        attributable = [r for r in rows if r["status"] in {"PASS", "FAIL_CONFIRMED", "DROPOUT"}]
        available, correct = sum(r["timely_admitted"] for r in rows), sum(r["correct"] for r in audited)
        def lower(k, n):
            return clopper_pearson_bound(k, n, side="lower", alpha=alpha) if n and independent_stationary_blocks else 0.
        return {"verifier_id": self.verifier_id, "completed_opportunities": len(rows),
            "pending_opportunities": len(self.reservations) - len(rows),
            "timely_admitted": available, "audited_admitted": len(audited), "correct_audited": correct,
            "attributable_opportunities": len(attributable),
            "availability": available / len(rows) if rows else None,
            "conditional_correctness": correct / len(audited) if audited and all_admitted_audited else None,
            "service_score": sum(r["status"] == "PASS" for r in attributable) / len(attributable) if attributable else None,
            "availability_lower": lower(available, len(rows)),
            "correctness_lower": lower(correct, len(audited)) if all_admitted_audited else 0.,
            "all_admitted_audited": all_admitted_audited,
            "confidence_assumption_declared": independent_stationary_blocks, "alpha": alpha,
            "reserved_expenditure": len(self.reservations) * (self.fee + self.owner_reserve),
            "actual_expenditure": sum(r["fee_paid"] + r["owner_cost"] for r in rows)}

    def production_admissible(self, *, minimum_correctness: float, minimum_availability: float,
                             alpha: float = .05, independent_stationary_blocks: bool = False):
        if not .5 < minimum_correctness <= 1 or not 0 < minimum_availability <= 1:
            raise ValueError("invalid production reliability thresholds")
        s = self.summary(alpha=alpha, independent_stationary_blocks=independent_stationary_blocks)
        return (s["pending_opportunities"] == 0 and s["audited_admitted"] > 0
                and s["correctness_lower"] >= minimum_correctness
                and s["availability_lower"] >= minimum_availability)
