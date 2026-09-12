"""Real callback-driven execution of the existing certified recovery policies."""
from __future__ import annotations

from dataclasses import asdict, replace
import time
from typing import Callable, Mapping

from sevc.committee import majority_success_probability
from sevc.core.registry import Registry
from sevc.core.task_lanes import TaskLanes
from sevc.incentives.verifier_protocol import (
    PublicVerifierOffer, VerificationJob, CommitteeAssignment, ReserveAvailabilityTerms,
)
from sevc.incentives.verifier_reserve_recovery import (
    CC_SARE_IMPLEMENTATION_KEY, reconstitute_pass_committee, select_global_recovery_matching,
)
from sevc.incentives.verifier_roster_certification import (
    MULTI_JOB_CAPACITY_CERTIFICATE_POLICY_KEY, certify_registered_multi_job_capacity,
)

RECOVERY_CONTROLLERS = Registry("executed recovery controller")
RECOVERY_CONTROLLERS.add("current-certified-ecs", True)
RECOVERY_CONTROLLERS.add("fixed-order-same-reserve", False)
FAULTS = ("no-missing", "one-missing", "correlated-missing", "insufficient-reserve")


def build_execution_roster(seed: int, owner_cost_reserve_per_assignment: float = 0.):
    if owner_cost_reserve_per_assignment < 0:
        raise ValueError("negative owner work reserve")
    offers = tuple(PublicVerifierOffer(f"v{i}", .9, 1, True, "assignment", 1.25, .5)
                   for i in range(9))
    jobs = tuple(VerificationJob(f"j{i}", 40, .95) for i in range(2))
    committees = tuple(CommitteeAssignment(j.job_id, tuple(f"v{k}" for k in range(i*3, i*3+3)),
                                           majority_success_probability([.9]*3))
                       for i, j in enumerate(jobs))
    terms = ReserveAvailabilityTerms(1., 0., 0., 0., 1.25, .5)
    roster, certificate = certify_registered_multi_job_capacity(
        policy_key=MULTI_JOB_CAPACITY_CERTIFICATE_POLICY_KEY, implementation_key=CC_SARE_IMPLEMENTATION_KEY,
        scenario_id=str(seed), parameter_set_id="five-rq-controlled-nine-v1", offers=offers,
        primary_committees=committees, jobs=jobs,
        parameters={"reserve_cap": 9, "activation_batch": 1, "removal_budget": 0,
                    "committed_expenditure_multiplier_cap": 3.}, order_seed=str(seed),
        availability_terms=terms, m_candidates=(9,), minimum_certificate_probability=.925,
        frozen_lazy_fraction=0., frozen_dropout_fraction=2/9,
        sentinel_generation_cost_per_assignment=0.,
        adjudication_reserve_per_assignment=owner_cost_reserve_per_assignment)
    return offers, jobs, roster, certificate


def required_segment_decision(required_ids, reports, eligible_ids, required_votes=3):
    """No audit labels: require a quorum for every segment and accept their conjunction."""
    eligible = set(eligible_ids)
    maps = [dict(zip(r.ordered_segment_ids, r.verdicts)) for r in reports if r.verifier_id in eligible
            and r.committed and r.revealed]
    if not required_ids or len(maps) < required_votes:
        return None
    outcomes = []
    for segment_id in required_ids:
        votes = [row[segment_id] for row in maps if row.get(segment_id) is not None]
        if len(votes) < required_votes:
            return None
        outcomes.append(sum(v is True for v in votes) > len(votes)/2)
    return all(outcomes)


def execute_recovery(*, policy: str, fault: str, seed: int, required_ids: tuple[str, ...],
                     execute: Callable, clock, timeout_seconds: float, deadline_seconds: float,
                     sleep: Callable = time.sleep, activation_lanes: int = 1,
                     owner_cost_reserve_per_assignment: float = 0., owner_prepaid_cost_per_job: float = 0.,
                     minimum_pass_count: int = 0, public_conflicts=None, missing_ids=None,
                     virtual_fault_wait: bool = False):
    """All policy-selected candidates invoke real service; fault waits occupy real wall time."""
    matching = RECOVERY_CONTROLLERS.get(policy)
    if ((fault not in FAULTS and missing_ids is None) or timeout_seconds <= 0 or deadline_seconds <= 0
            or owner_cost_reserve_per_assignment < 0 or owner_prepaid_cost_per_job < 0):
        raise ValueError("invalid frozen fault/deadline")
    if virtual_fault_wait and activation_lanes != 1:
        raise ValueError("logical fault clock requires serial activation")
    physical_started = time.monotonic()
    virtual_wait = 0.
    def now():
        return time.monotonic() + virtual_wait
    def wait_timeout(seconds):
        nonlocal virtual_wait
        if virtual_fault_wait:
            virtual_wait += seconds
        else:
            sleep(seconds)
    started = now()
    (offers, jobs, roster, certificate), _ = clock.call("recovery_certificate", "owner",
        build_execution_roster, seed, owner_cost_reserve_per_assignment)
    if not certificate["certificate_passed"]:
        raise ValueError("controlled execution roster failed the canonical certificate")
    if public_conflicts is not None:
        from sevc.core.artifacts import canonical_json_text, sha256_text
        offers = tuple(replace(o, conflict_job_ids=tuple(public_conflicts.get(o.verifier_id, ()))) for o in offers)
        entries = tuple(replace(e, conflict_job_ids=tuple(public_conflicts.get(e.verifier_id, ()))) for e in roster.entries)
        roster = replace(roster, entries=entries, roster_sha256=sha256_text(canonical_json_text(
            {"nominal_roster": roster.roster_sha256, "public_conflicts": public_conflicts})))
        certificate = {**certificate, "scope": "nominal-capacity-only; graph feasibility separately audited",
                       "graph_guarantee_inherited": False}
    # Public domains place the first two primary members of j0 and first of j1 together.
    missing = {"no-missing": (), "one-missing": ("v0",),
               "correlated-missing": ("v0", "v1", "v3"),
               "insufficient-reserve": ("v0", "v1", "v3", "v4")}.get(fault, ()) if missing_ids is None else tuple(missing_ids)
    if not set(missing) <= {o.verifier_id for o in offers}:
        raise ValueError("unknown missing identity")
    activation = {j.job_id: [] for j in jobs}
    reports = {j.job_id: [] for j in jobs}
    statuses = {j.job_id: {} for j in jobs}
    usage, published, receipts, decisions = {}, [], [], {}
    spent = {j.job_id: owner_prepaid_cost_per_job for j in jobs}
    # The existing zero-retainer certificate accounts per-assignment fee exposure.
    assignment_reserve = 1.25 + owner_cost_reserve_per_assignment
    cap = 3 * (3 * assignment_reserve + owner_prepaid_cost_per_job)
    wave = 0

    def execute_member(pair):
        job_id, verifier_id = pair
        clock.context.update(job_id=job_id,verifier_id=verifier_id)
        report = None
        if verifier_id in missing:
            clock.call("fault_timeout_wait", "wait", wait_timeout, timeout_seconds)
            status = "DROPOUT"
            receipt = {"job_id": job_id, "verifier_id": verifier_id, "status": status,
                       "fee": 0., "bond_return": 0., "slashed_bond": .5}
        else:
            (report, settlement, detail), _ = clock.call("recovery_assignment", "verifier",
                                                          execute, job_id, verifier_id)
            status = settlement.status
            owner_work = dict(settlement.diagnostics).get("owner_work_cost", 0.)
            if owner_work < 0 or owner_work > owner_cost_reserve_per_assignment:
                raise ValueError("actual owner admission work exceeded the frozen reservation")
            receipt = {"job_id": job_id, "verifier_id": verifier_id, "status": status,
                       "settlement": asdict(settlement), "detail": detail}
        return report,status,receipt,now()

    def activate_many(pairs):
        ready = []
        # Capacity and expenditure are reserved serially before any member runs.
        for job_id,verifier_id in pairs:
            if usage.get(verifier_id,0) >= 1:
                raise ValueError("duplicate capacity consumption")
            if spent[job_id]+assignment_reserve > cap or now()-started >= deadline_seconds:
                continue
            usage[verifier_id] = usage.get(verifier_id,0)+1
            activation[job_id].append(verifier_id)
            spent[job_id] += assignment_reserve
            ready.append((job_id,verifier_id))
            if activation_lanes == 1:
                # Preserve the baseline's serial reservation/deadline semantics.
                collect([(job_id,verifier_id)],[execute_member((job_id,verifier_id))])
        if activation_lanes != 1:
            lanes = TaskLanes(activation_lanes,"cpu")
            try:
                collect(ready,lanes.map(execute_member,ready,clock))
            finally:
                lanes.close()
        clock.context.pop("job_id",None);clock.context.pop("verifier_id",None)

    def collect(pairs,results):
        for (job_id,verifier_id),(report,status,receipt,published_at) in zip(pairs,results):
            if report is not None:
                reports[job_id].append(report)
            statuses[job_id][verifier_id] = status
            published.append({"job_id":job_id,"verifier_id":verifier_id,"status":status,
                              "published_at":published_at})
            receipts.append(receipt)

    activate_many([(committee.job_id,verifier_id) for committee in roster.primary_committees
                   for verifier_id in committee.verifier_ids])
    while True:
        current = {}
        for job in jobs:
            if job.job_id in decisions:
                continue
            result = reconstitute_pass_committee(roster=roster, job=job, wave_index=wave,
                activated_verifier_ids=activation[job.job_id],
                committed_verifier_ids=[r.verifier_id for r in reports[job.job_id]],
                settlement_statuses=statuses[job.job_id])
            if result.status == "RECONSTITUTED":
                verdict = required_segment_decision(required_ids, reports[job.job_id], result.verifier_ids)
                if verdict is not None and now()-started <= deadline_seconds:
                    decisions[job.job_id] = {"route": "accept" if verdict else "reject",
                        "trainer_reward": 1. if verdict else 0., "misconduct": int(not verdict),
                        "committee": list(result.verifier_ids), "wave": wave}
                    continue
            current[job.job_id] = [r.verifier_id for r in reports[job.job_id]
                                  if statuses[job.job_id].get(r.verifier_id) == "PASS"]
        if len(decisions) == len(jobs) or now()-started >= deadline_seconds:
            break
        unresolved = tuple(j for j in jobs if j.job_id not in decisions)
        if matching:
            selected = select_global_recovery_matching(jobs=unresolved, offers=offers,
                current_committees=current, used_capacity=usage, published_statuses=published,
                scheduler_read_time=now(), minimum_pass_count=minimum_pass_count)["assignments"]
        else:
            selected, reserved = [], set()
            for job in unresolved:
                choices = [o.verifier_id for o in offers if usage.get(o.verifier_id, 0) < 1
                           and o.verifier_id not in reserved and job.job_id not in o.conflict_job_ids]
                if choices:
                    selected.append((job.job_id, choices[0])); reserved.add(choices[0])
        selected = [(j, v) for j, v in selected if spent[j]+assignment_reserve <= cap]
        if not selected:
            break
        wave += 1
        activate_many(selected)
    for job in jobs:
        decisions.setdefault(job.job_id, {"route": "safe-defer", "trainer_reward": 0.,
                                         "misconduct": 0, "committee": [], "wave": wave})
    return {"policy": policy, "fault": fault, "seed": seed, "certificate": certificate,
            "roster_sha256": roster.roster_sha256, "decisions": decisions, "receipts": receipts,
            "usage": usage, "reserved_expenditure": spent, "budget_cap_per_job": cap,
            "owner_work_reserve_per_assignment": owner_cost_reserve_per_assignment,
            "owner_prepaid_cost_per_job": owner_prepaid_cost_per_job,
            "minimum_pass_count": minimum_pass_count,
            "timeout_seconds": timeout_seconds, "deadline_seconds": deadline_seconds,
            "wall_seconds": time.monotonic()-physical_started,
            "logical_elapsed_seconds": now()-started, "virtual_wait_seconds": virtual_wait,
            "clock_mode": "actual-callback-plus-virtual-timeout" if virtual_fault_wait else "physical",
            "single_host_control": True,
            "within_certified_unavailability_bound": len(missing) <= 2,
            "public_conflicts": public_conflicts, "missing_identities": list(missing),
            "required_ids": list(required_ids),
            "published_events": published}
