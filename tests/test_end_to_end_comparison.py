from __future__ import annotations

import pytest

from sevc.evaluation import end_to_end_comparison as e2e
from sevc.evaluation.f_rq_audit import GOLD, RCMP

SEGMENTS = ("s0", "s1", "s2", "s3")
PARAMETERS = {"service_fee": 2.5, "service_bond": .5, "owner_cost_reserve_per_assignment": 20.,
              "deadline_seconds": 1200., "timeout_seconds": 1., "virtual_fault_wait": True, "minimum_pass_count": 3}


def _settlement(passed):
    return {"status": "PASS" if passed else "FAIL_CONFIRMED", "service_fee": 2.5 if passed else 0.,
            "refundable_bond": .5 if passed else 0., "slashed_bond": 0. if passed else .5,
            "owner_expenditure": 2.5 if passed else 0., "verifier_cost": 1., "effort_fraction": 1.,
            "committed": True, "revealed": True, "accepted_report": passed, "technical_failure": False,
            "abstained": False, "diagnostics": [["mismatch_count", 0 if passed else 3]],
            "job_id": "x", "verifier_id": "x"}


def _job(invalid, context="init-invalid"):
    honest = {s: not (invalid and s == "s0") for s in SEGMENTS}
    verdicts = {
        "honest": (honest, True),
        "constant-accept": ({s: True for s in SEGMENTS}, False),
        "constant-reject": ({s: False for s in SEGMENTS}, False),
        "sgd-consistency-shortcut": ({s: True for s in SEGMENTS}, False),
        "prefix-one-step-shortcut": ({s: True for s in SEGMENTS}, False),
        "uniform-k32": (dict(honest), True),
        "uniform-k39": ({s: True for s in SEGMENTS}, True),  # admitted, misses the invalid segment
    }
    services = {b: e2e.Service(b, v, _settlement(p)) for b, (v, p) in verdicts.items()}
    return e2e.Job("mnist", context, 0, invalid, 7, SEGMENTS, services)


VALID, INVALID = _job(0, "init-valid"), _job(1)


def _one(scheme, job, slots, companion=None):
    kwargs = {"parameters": PARAMETERS} if scheme in e2e.CONTROLLER_SCHEMES else {}
    (weight, outcome), = e2e.SCHEMES.get(scheme)(job, slots, companion or (INVALID if job is VALID else VALID), **kwargs)
    assert weight == 1.
    return outcome, e2e.classify(outcome, job.invalid)


def test_compositions_are_frozen_and_never_duplicate_partial_replay():
    comps = e2e.compositions()
    assert len(comps) == 18
    assert [f for f, _, _ in comps].count("shared-shortcut") == 4
    for family, _, slots in comps:
        assert all(slots.count(b) <= 1 for b in e2e.PARTIAL)
        if family == "shared-shortcut":
            assert slots[0] == slots[1] and slots[0] in e2e.DETERMINISTIC


def test_arrival_majority_breaks_ties_toward_rejection():
    outcome, row = _one("arrival-majority", VALID, ("missing", "constant-reject", "honest"))
    assert outcome["terminal"] == "reject" and row["false_accusation"] and row["honest_unpaid"]


def test_report_comparison_stalls_on_a_missing_report():
    outcome, row = _one("report-comparison", VALID, ("missing", "honest", "honest"))
    assert outcome["terminal"] == "no-terminal" and row["no_terminal"] and row["honest_unpaid"]
    assert not any(m["paid"] for m in outcome["members"])


def test_report_comparison_pays_a_shared_shortcut_and_unpays_the_honest_minority():
    _, row = _one("report-comparison", VALID, ("constant-reject", "constant-reject", "honest"))
    assert row["false_accusation"] and row["det_shortcut_paid"] and row["honest_unpaid"]


def test_single_verifier_weights_each_slot_equally():
    outcomes = e2e.SCHEMES.get("single-verifier")(INVALID, ("constant-accept", "honest", "honest"))
    assert [w for w, _ in outcomes] == [1 / 3] * 3
    assert [e2e.classify(o, 1)["false_release"] for _, o in outcomes] == [True, False, False]


def test_sevc_replaces_failed_service_and_pays_only_supported_reports():
    outcome, row = _one("sevc", VALID, ("constant-reject", "honest", "honest"))
    assert outcome["terminal"] == "accept" and row["accountable"]
    assert not row["det_shortcut_paid"] and not row["honest_unpaid"]
    outcome, row = _one("sevc", INVALID, ("missing", "constant-accept", "honest"))
    assert outcome["terminal"] == "reject" and row["accountable"]


@pytest.mark.parametrize("policy", ["value-preserving-rc", "all-response-certified-ecs", "no-recovery"])
def test_every_controller_serves_the_composition_to_the_evaluated_job(policy):
    outcome, _ = e2e.controller_outcome(VALID, INVALID, ("missing", "constant-reject", "honest"),
                                        policy=policy, admission=True, parameters=PARAMETERS)
    first_three = [m["behavior"] for m in outcome["members"]][:3]
    assert sorted(first_three) == ["constant-reject", "honest", "missing"]
    assert outcome["terminal"] == ("safe-defer" if policy == "no-recovery" else "accept")


def test_recovery_without_admission_decides_on_a_shared_shortcut():
    outcome, row = _one("recovery-without-rcmp", VALID, ("constant-reject", "constant-reject", "honest"))
    assert outcome["terminal"] == "reject" and row["false_accusation"] and row["det_shortcut_paid"]


def test_rcmp_without_recovery_defers_instead_of_deciding():
    outcome, row = _one("rcmp-no-recovery", VALID, ("missing", "honest", "honest"))
    assert outcome["terminal"] == "safe-defer" and row["trainer_accountable"] and not row["completed"]


def test_partial_replay_payment_is_reported_separately():
    _, row = _one("sevc", INVALID, ("uniform-k39", "honest", "honest"))
    assert row["partial_paid"] and row["accountable"] and not row["false_release"]


def test_decide_routes_any_sevc_failure_negative():
    cell = {"n": 1., "accountable": 1.}
    summary = {"mnist": {v: {f: {"sevc": dict(cell)} for f in e2e.FAMILIES} for v in ("valid", "invalid")}}
    assert e2e.decide(summary, ["mnist"], True)["terminal"] == "PASS"
    summary["mnist"]["valid"]["one-missing"]["sevc"]["accountable"] = 0.
    assert e2e.decide(summary, ["mnist"], True)["terminal"] == "ACCEPTED_NEGATIVE"
    assert e2e.decide(summary, ["mnist"], False)["terminal"] == "VOID_RERUN"


def _unit(method, behavior, verdicts, *, passed=True, invalid=0):
    uid = f"{method}-{behavior}"
    tasks = [{"task_id": f"{uid}-t{i}", "source_sha256": s, "role": "production"} for i, s in enumerate(SEGMENTS)]
    unit = {"unit_id": uid, "phase": "production", "context": "init-valid", "dataset": "mnist", "block": 0,
            "invalid": invalid, "seed": 7, "method": method, "behavior": behavior,
            "production_source_ids": list(SEGMENTS), "tasks": tasks}
    if method == e2e.DIRECT:
        unit["owner_direct_answers"] = dict(verdicts)
        return unit, None
    assignment = {"assignment_id": uid + "-v0", "settlement": _settlement(passed),
                  "report": {"ordered_segment_ids": [t["task_id"] for t in tasks],
                             "verdicts": [verdicts[s] for s in SEGMENTS]}}
    unit["assignments"] = [assignment]
    row = {"assignment_id": assignment["assignment_id"], "admitted": passed,
           "production_verdicts": {t["task_id"]: verdicts[t["source_sha256"]] for t in tasks}}
    return unit, row


def test_load_jobs_requires_identical_honest_replays():
    honest = {s: True for s in SEGMENTS}
    built = [_unit(RCMP, b, honest if b == "honest" else {s: True for s in SEGMENTS}, passed=b == "honest")
             for b in e2e.BEHAVIORS]
    built += [_unit(GOLD, "honest", honest), _unit(e2e.DIRECT, "honest", honest)]
    units = {u["unit_id"]: u for u, _ in built}
    rows = [r for _, r in built if r]
    jobs, integrity = e2e.load_jobs(units, rows)
    assert not integrity["failures"] and len(jobs) == 1
    units[f"{GOLD}-honest"]["assignments"][0]["report"]["verdicts"][0] = False
    rows = [r for _, r in built if r]
    rows[-1]["production_verdicts"][f"{GOLD}-honest-t0"] = False
    _, integrity = e2e.load_jobs(units, rows)
    assert any(f.startswith("I5") for f in integrity["failures"])


def test_classify_rejects_unknown_terminals():
    with pytest.raises(ValueError):
        e2e.classify({"terminal": "restart", "members": []}, 0)


# experiment-tdsc-single-verifier-e2e-v1: calibrated correctness makes ECS decide on one report.
CALIBRATED = {**PARAMETERS, "calibrated_correctness": .99}


def _calibrated(job, slots):
    (weight, outcome), = e2e.SCHEMES.get("sevc-calibrated")(
        job, slots, INVALID if job is VALID else VALID, parameters=CALIBRATED)
    return outcome, e2e.classify(outcome, job.invalid)


def test_calibrated_sevc_decides_on_the_first_supported_report():
    outcome, row = _calibrated(VALID, ("honest", "honest", "honest"))
    assert outcome["terminal"] == "accept" and row["accountable"]
    assert [m["behavior"] for m in outcome["members"]] == ["honest"]


def test_calibrated_sevc_replaces_a_failed_single_verifier():
    outcome, row = _calibrated(VALID, ("constant-reject", "honest", "honest"))
    assert outcome["terminal"] == "accept" and row["accountable"] and not row["det_shortcut_paid"]
    outcome, row = _calibrated(INVALID, ("missing", "constant-accept", "honest"))
    assert outcome["terminal"] == "reject" and row["accountable"]


def test_calibrated_sevc_passes_an_admitted_partial_replay_error_to_the_trainer():
    outcome, row = _calibrated(INVALID, ("uniform-k39", "honest", "honest"))
    assert outcome["terminal"] == "accept" and row["false_release"] and row["partial_paid"]
    # The three-report SEVC outvotes the same partial replay.
    _, row3 = _one("sevc", INVALID, ("uniform-k39", "honest", "honest"))
    assert not row3["false_release"]


def test_default_controller_parameters_keep_the_three_report_quorum():
    outcome, _ = _one("sevc", VALID, ("honest", "honest", "honest"))
    assert len([m for m in outcome["members"] if m["arrived"]]) == 3


def test_decide_calibrated_confines_failures_to_partial_replay():
    ok = {"n": 1., "accountable": 1., "false_accusation": 0., "det_shortcut_paid": 0., "completed": 1.}
    bad = {**ok, "accountable": 0.}
    scopes = {"all-honest": {"sevc-calibrated": dict(ok)}, "one-deviator/uniform-k39": {"sevc-calibrated": dict(bad)},
              "one-deviator": {"sevc-calibrated": dict(bad)}}
    summary = {"mnist": {"valid": scopes, "invalid": scopes}}
    assert e2e.decide_calibrated(summary, ["mnist"], True)["terminal"] == "PASS"
    scopes["one-deviator/constant-accept"] = {"sevc-calibrated": dict(bad)}
    assert e2e.decide_calibrated(summary, ["mnist"], True)["terminal"] == "ACCEPTED_NEGATIVE"
    assert e2e.decide_calibrated(summary, ["mnist"], False)["terminal"] == "VOID_RERUN"


def test_regression_check_reports_any_changed_cell():
    reference = {"mnist": {"valid": {"all-honest": {"sevc": {"n": 1., "accountable": 1.}}}}}
    assert e2e.regression_check(reference, reference)["differences"] == []
    changed = {"mnist": {"valid": {"all-honest": {"sevc": {"n": 1., "accountable": 0.}}}}}
    assert e2e.regression_check(changed, reference)["differences"] == ["mnist/valid/all-honest/sevc"]


def test_replay_residuals_deduplicate_receipts_and_split_by_verdict(tmp_path):
    audits = tmp_path / "reference-audits"
    audits.mkdir()
    receipt = lambda passed, diff: {"passed": passed, "max_abs_difference": diff, "max_model_abs_difference": diff,
                                    "max_optimizer_abs_difference": diff, "tolerance": 1e-5, "state_complete": True}
    for uid, rows in (("u1", {"p1": receipt(True, 0.), "p2": receipt(False, 4e-5)}),
                      ("u2", {"p1": receipt(True, 0.), "p3": receipt(True, 0.)})):
        (audits / (uid + ".json")).write_text(__import__("json").dumps({"production_receipts": rows}))
    units = {"u1": {"dataset": "mnist", "assignments": [1]}, "u2": {"dataset": "mnist", "assignments": [1]}}
    out = e2e.replay_residuals(tmp_path, units, ["mnist"])["datasets"]["mnist"]
    assert (out["unique_receipts"], out["passed"], out["failed"]) == (3, 2, 1)
    assert out["passed_max_abs_difference"] == 0. and out["failed_min_abs_difference"] == 4e-5
