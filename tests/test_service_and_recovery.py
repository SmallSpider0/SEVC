"""Selected canonical scientific routines; deployment wrappers are omitted."""
from dataclasses import replace

import pytest

from sevc.committee.executed_recovery import execute_recovery, required_segment_decision, FAULTS

from sevc.evaluation.workload_performance import integrate_utilization

from sevc.incentives.verifier_protocol import CommittedVerifierReport

from sevc.verification.paid_replay_service import OwnerProbeReferences, settle_service, WorkClock

def report(verifier="v0", job="j0", production=True, probes=None):
    answers = tuple([True]*4+[False]*4) if probes is None else tuple(probes)
    return CommittedVerifierReport.create(scenario_id="known-fixture", verifier_id=verifier,
        job_id=job, ordered_segment_ids=("production",)+tuple(f"p{i}" for i in range(8)),
        verdicts=(production,)+answers, nonce="fixed-known-fixture")


REF = OwnerProbeReferences(tuple((f"p{i}", i < 4) for i in range(8)))


def test_production_truth_cannot_enter_online_service_score():
    first = settle_service(report(production=True), REF, cost_seconds=1., effort_fraction=1.)
    second = settle_service(report(production=False), REF, cost_seconds=1., effort_fraction=1.)
    assert first.status == second.status == "PASS"
    with pytest.raises(TypeError):
        settle_service(report(), {"production": False}, cost_seconds=1., effort_fraction=1.)
    assert settle_service(report(probes=[True]*8), REF, cost_seconds=0., effort_fraction=0.).status == "FAIL_CONFIRMED"
    pending = replace(report(), revealed=False)
    assert not settle_service(pending, REF, cost_seconds=0., effort_fraction=0.).accepted_report


def test_missing_required_segment_cannot_pay_trainer():
    reports = [report(f"v{i}") for i in range(3)]
    assert required_segment_decision(("production",), reports, ("v0","v1","v2")) is True
    assert required_segment_decision(("production","missing"), reports, ("v0","v1","v2")) is None
    assert required_segment_decision(("production",), reports, ("v0","v1")) is None
    bad = [report(f"v{i}", production=False) for i in range(3)]
    assert required_segment_decision(("production",), bad, ("v0","v1","v2")) is False


@pytest.mark.parametrize("policy", ["current-certified-ecs", "fixed-order-same-reserve"])
@pytest.mark.parametrize("fault", FAULTS)
@pytest.mark.parametrize("activation_lanes",[1,8])
def test_control_uses_actual_callbacks_and_bounded_capacity(policy, fault, activation_lanes):
    calls, waits = [], []
    def execute(job, verifier):
        calls.append((job,verifier))
        r = report(verifier,job)
        return r, settle_service(r,REF,cost_seconds=1.,effort_fraction=1.), []
    clock = WorkClock(lambda: None, lambda row: None)
    result = execute_recovery(policy=policy, fault=fault, seed=7, required_ids=("production",),
        execute=execute, clock=clock, timeout_seconds=.01, deadline_seconds=10., sleep=waits.append,
        activation_lanes=activation_lanes)
    assert len(calls)+len(waits) == sum(result["usage"].values())
    assert max(result["usage"].values()) <= 1
    if fault == "no-missing":
        assert len(calls) == 6
        assert all(v["route"] == "accept" for v in result["decisions"].values())
    if fault == "insufficient-reserve":
        assert any(v["route"] == "safe-defer" for v in result["decisions"].values())
    assert all(v <= result["budget_cap_per_job"] for v in result["reserved_expenditure"].values())


def test_full_window_utilization_includes_idle_and_clips_boundaries():
    samples = [{"monotonic":0.,"utilization_percent":100.},
               {"monotonic":1.,"utilization_percent":0.},
               {"monotonic":2.,"utilization_percent":100.}]
    assert integrate_utilization(samples,.5,1.5)["average_percent"] == 50.
    with pytest.raises(ValueError):
        integrate_utilization(samples,-1.,1.)


