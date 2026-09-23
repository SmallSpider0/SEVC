"""Trace-driven end-to-end comparison of complete settlement schemes.

Every F v2 production job was served by all seven registered RCMP verifier behaviors, one
hidden-gold honest verifier and the owner's own direct replay, each on the same 32 production
source segments.  A three-member committee is therefore assembled from services recorded on the
same job; a behavior occupies several slots only if it is deterministic.  ECS and its ablations
run through the canonical recovery controller with a callback that returns recorded services.
Read-only: no training, replay or tensor load.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, fields, replace
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from sevc.committee.executed_recovery import execute_recovery
from sevc.core.artifacts import (capture_environment, ensure_experiment_output_root, sha256_file, sha256_text,
                                 write_json)
from sevc.core.registry import Registry
from sevc.core.role_accounting import RoleClock
from sevc.core.statistical_bounds import clopper_pearson_bound
from sevc.evaluation.f_rq234_audit import majority_terminal
from sevc.evaluation.f_rq_audit import GOLD, RCMP, verify_packet
from sevc.incentives.verifier_protocol import CommittedVerifierReport, VerifierSettlement

CHANGE_ID = "experiment-tdsc-end-to-end-comparison-v1"
DIRECT = "owner-direct-v2"
CONTEXTS = ("init-valid", "init-invalid")
HONEST, MISSING = "honest", "missing"
DETERMINISTIC = ("constant-accept", "constant-reject", "sgd-consistency-shortcut", "prefix-one-step-shortcut")
PARTIAL = ("uniform-k32", "uniform-k39")
BEHAVIORS = (HONEST,) + DETERMINISTIC + PARTIAL
FAMILIES = ("all-honest", "one-deviator", "shared-shortcut", "one-missing", "missing-plus-deviator")
TERMINALS = ("accept", "reject", "safe-defer", "no-terminal")
METRICS = ("completed", "false_accusation", "false_release", "safe_defer", "no_terminal",
           "trainer_accountable", "det_shortcut_paid", "partial_paid", "honest_unpaid", "accountable")


@dataclass(frozen=True)
class Service:
    behavior: str
    verdicts: Mapping[str, bool]
    settlement: Mapping[str, Any]


@dataclass(frozen=True)
class Job:
    dataset: str
    context: str
    block: int
    invalid: int
    seed: int
    segments: tuple[str, ...]
    services: Mapping[str, Service]

    @property
    def key(self) -> str:
        return f"{self.dataset}/{self.context}/{self.block}"


def compositions() -> list[tuple[str, str, tuple[str, str, str]]]:
    """The frozen 18 committee compositions; a randomized behavior never fills two slots."""
    out = [("all-honest", "all-honest", (HONEST, HONEST, HONEST))]
    out += [("one-deviator", f"one-deviator/{b}", (b, HONEST, HONEST)) for b in DETERMINISTIC + PARTIAL]
    out += [("shared-shortcut", f"shared-shortcut/{d}", (d, d, HONEST)) for d in DETERMINISTIC]
    out += [("one-missing", "one-missing", (MISSING, HONEST, HONEST))]
    out += [("missing-plus-deviator", f"missing-plus-deviator/{b}", (MISSING, b, HONEST))
            for b in DETERMINISTIC + PARTIAL]
    for _, _, slots in out:
        if any(slots.count(b) > 1 for b in PARTIAL):
            raise ValueError("a randomized behavior cannot be duplicated")
    return out


# ------------------------------------------------------------------------ inputs

def load_jobs(units: Mapping[str, Mapping[str, Any]], service_rows: Sequence[Mapping[str, Any]]):
    """Group recorded production services by job; return jobs and integrity checks I2-I5."""
    rows_by_assignment = {r["assignment_id"]: r for r in service_rows}
    grouped: dict[tuple, dict] = defaultdict(lambda: {"rcmp": {}, "gold": [], "direct": [], "sources": set()})
    checks, failures = Counter(), []
    for unit in units.values():
        if unit.get("phase") != "production" or unit.get("context") not in CONTEXTS:
            continue
        key = (unit["dataset"], unit["context"], unit["block"])
        group = grouped[key]
        group["sources"].add(tuple(unit["production_source_ids"]))
        group.setdefault("invalid", set()).add(unit["invalid"])
        if unit["method"] == DIRECT:
            group["direct"].append(dict(unit["owner_direct_answers"]))
            continue
        assignment = unit["assignments"][0]
        source = {t["task_id"]: t["source_sha256"] for t in unit["tasks"] if t["role"] == "production"}
        report = dict(zip(assignment["report"]["ordered_segment_ids"], assignment["report"]["verdicts"]))
        verdicts = {source[t]: v for t, v in report.items() if t in source}
        checks["services"] += 1
        if set(verdicts) != set(unit["production_source_ids"]) or len(verdicts) != len(source):
            failures.append(f"I3 segment alignment: {unit['unit_id']}")
        row = rows_by_assignment.get(assignment["assignment_id"])
        if row is None or {source[t]: v for t, v in row["production_verdicts"].items()} != verdicts \
                or row["admitted"] != (assignment["settlement"]["status"] == "PASS"):
            failures.append(f"I4 service row mismatch: {unit['unit_id']}")
        if unit["method"] == GOLD:
            group["gold"].append(verdicts)
        elif unit["method"] == RCMP:
            if unit["behavior"] in group["rcmp"]:
                failures.append(f"I2 duplicate behavior: {key} {unit['behavior']}")
            group["rcmp"][unit["behavior"]] = (Service(unit["behavior"], verdicts, assignment["settlement"]),
                                               unit["seed"])
    jobs = {}
    for key, group in sorted(grouped.items()):
        if (set(group["rcmp"]) != set(BEHAVIORS) or len(group["gold"]) != 1 or len(group["direct"]) != 1
                or len(group["sources"]) != 1 or len(group["invalid"]) != 1):
            failures.append(f"I2 incomplete or unpaired job: {key}")
            continue
        segments = next(iter(group["sources"]))
        invalid = next(iter(group["invalid"]))
        honest = group["rcmp"][HONEST][0].verdicts
        truth_ok = (all(honest.values()) if invalid == 0 else sum(not v for v in honest.values()) == 1)
        if not (honest == group["gold"][0] == group["direct"][0]) or not truth_ok:
            failures.append(f"I5 honest replays disagree or differ from truth: {key}")
        checks["jobs"] += 1
        jobs[key] = Job(key[0], key[1], key[2], invalid, group["rcmp"][HONEST][1], segments,
                        {b: s for b, (s, _) in group["rcmp"].items()})
    return jobs, {"checks": dict(checks), "failures": failures}


# ----------------------------------------------------------------------- schemes

SCHEMES: Registry[Callable[..., list[tuple[float, dict]]]] = Registry("end-to-end settlement scheme")


def _report(job: Job, behavior: str) -> dict[str, bool]:
    return dict(job.services[behavior].verdicts)


def _agrees(verdicts: Mapping[str, bool], reports: Sequence[Mapping[str, bool]], segments) -> bool:
    return all((majority_terminal(reports, [s]) == "accept") == verdicts[s] for s in segments)


def _outcome(terminal: str, members: list[tuple[str, bool, bool]]) -> dict:
    return {"terminal": terminal, "members": [{"behavior": b, "arrived": a, "paid": p} for b, a, p in members]}


@SCHEMES.register("single-verifier")
def single_verifier(job: Job, slots, companion=None):
    out = []
    for behavior in slots:
        if behavior == MISSING:
            out.append((1 / 3, _outcome("no-terminal", [(MISSING, False, False)])))
        else:
            verdicts = _report(job, behavior)
            out.append((1 / 3, _outcome("accept" if all(verdicts[s] for s in job.segments) else "reject",
                                        [(behavior, True, True)])))
    return out


def _majority_scheme(job: Job, slots, stall_on_missing: bool):
    arrived = [(b, _report(job, b)) for b in slots if b != MISSING]
    if not arrived or (stall_on_missing and len(arrived) < len(slots)):
        return [(1., _outcome("no-terminal", [(b, b != MISSING, False) for b in slots]))]
    reports = [r for _, r in arrived]
    terminal = majority_terminal(reports, job.segments)
    members = [(b, True, _agrees(r, reports, job.segments)) for b, r in arrived]
    members += [(MISSING, False, False)] * (len(slots) - len(arrived))
    return [(1., _outcome(terminal, members))]


@SCHEMES.register("report-comparison")
def report_comparison(job: Job, slots, companion=None):
    return _majority_scheme(job, slots, stall_on_missing=True)


@SCHEMES.register("arrival-majority")
def arrival_majority(job: Job, slots, companion=None):
    return _majority_scheme(job, slots, stall_on_missing=False)


def _settlement(recorded: Mapping[str, Any], verifier_id: str, job_id: str, admission: bool,
                fee: float, bond: float) -> VerifierSettlement:
    names = {f.name for f in fields(VerifierSettlement)}
    values = {k: v for k, v in recorded.items() if k in names}
    values["diagnostics"] = tuple(tuple(x) for x in recorded.get("diagnostics", ()))
    settlement = VerifierSettlement(**{**values, "verifier_id": verifier_id, "job_id": job_id})
    if not admission:
        settlement = replace(settlement, status="PASS", service_fee=fee, refundable_bond=bond,
                             slashed_bond=0., owner_expenditure=fee, accepted_report=True)
    return settlement


def _dropout(verifier_id: str, job_id: str, bond: float) -> VerifierSettlement:
    return VerifierSettlement(verifier_id=verifier_id, status="DROPOUT", service_fee=0., refundable_bond=0.,
                              slashed_bond=bond, owner_expenditure=0., verifier_cost=0., effort_fraction=0.,
                              committed=False, revealed=False, accepted_report=False, technical_failure=False,
                              abstained=False, job_id=job_id)


def controller_outcome(job: Job, companion: Job, slots, *, policy: str, admission: bool,
                       parameters: Mapping[str, Any]) -> tuple[dict, dict]:
    """Run the canonical recovery controller on recorded services; j0 is the evaluated job.

    The controller chooses which identity serves which job.  The first three services that j0
    receives follow the composition, later ones (replacements) and all of j1's are honest, so the
    scenario reaches j0 whatever identities the controller activates.
    """
    assigned: dict[str, str] = {}

    def callback(job_id: str, verifier_id: str):
        if job_id == "j0":
            behavior = slots[len(assigned)] if len(assigned) < len(slots) else HONEST
            assigned[verifier_id] = behavior
        else:
            behavior = HONEST
        if behavior == MISSING:
            return None, _dropout(verifier_id, job_id, parameters["service_bond"]), []
        current = job if job_id == "j0" else companion
        service = current.services[behavior]
        report = CommittedVerifierReport.create(
            scenario_id=f"e2e/{job.key}", verifier_id=verifier_id, ordered_segment_ids=current.segments,
            verdicts=tuple(service.verdicts[s] for s in current.segments), nonce=f"{job_id}/{verifier_id}",
            job_id=job_id)
        return report, _settlement(service.settlement, verifier_id, job_id, admission,
                                   parameters["service_fee"], parameters["service_bond"]), []

    trace = execute_recovery(
        policy=policy, fault="structural", seed=job.seed, required_ids=job.segments, execute=callback,
        clock=RoleClock(lambda: None, lambda row: None), missing_ids=[],
        required_ids_by_job={"j0": job.segments, "j1": companion.segments}, **parameters)
    members = []
    for receipt in trace["receipts"]:
        if receipt["job_id"] != "j0":
            continue
        members.append((assigned[receipt["verifier_id"]], receipt["status"] != "DROPOUT",
                        receipt["settlement"]["service_fee"] > 0))
    route = trace["decisions"]["j0"]["route"]
    return _outcome(route, members), {"route": route, "members": sorted(members),
                                      "wave": trace["decisions"]["j0"]["wave"]}


def _controller_scheme(policy: str, admission: bool, calibrated: bool = False):
    def scheme(job: Job, slots, companion=None, parameters=None):
        controller = {k: v for k, v in (parameters or {}).items() if k != "calibrated_correctness"}
        if calibrated:
            # Only the roster's correctness input changes; ECS derives the decision-set size from it.
            controller["verifier_reputations"] = {f"v{i}": float(parameters["calibrated_correctness"])
                                                  for i in range(9)}
        return [(1., controller_outcome(job, companion, slots, policy=policy, admission=admission,
                                        parameters=controller)[0])]
    return scheme


SCHEMES.add("recovery-without-rcmp", _controller_scheme("value-preserving-rc", admission=False))
SCHEMES.add("rcmp-no-recovery", _controller_scheme("no-recovery", admission=True))
SCHEMES.add("sevc", _controller_scheme("value-preserving-rc", admission=True))
SCHEMES.add("sevc-calibrated", _controller_scheme("value-preserving-rc", admission=True, calibrated=True))
SCHEME_ORDER = ("single-verifier", "report-comparison", "arrival-majority", "recovery-without-rcmp",
                "rcmp-no-recovery", "sevc")
CONTROLLER_SCHEMES = {"recovery-without-rcmp", "rcmp-no-recovery", "sevc", "sevc-calibrated"}


# ----------------------------------------------------------------------- metrics

def classify(outcome: Mapping[str, Any], invalid: int) -> dict[str, bool]:
    terminal = outcome["terminal"]
    if terminal not in TERMINALS:
        raise ValueError("unknown terminal: " + terminal)
    target = "reject" if invalid else "accept"
    members = outcome["members"]
    row = {
        "completed": terminal in ("accept", "reject"),
        "false_accusation": not invalid and terminal == "reject",
        "false_release": bool(invalid) and terminal == "accept",
        "safe_defer": terminal == "safe-defer",
        "no_terminal": terminal == "no-terminal",
        "trainer_accountable": terminal in (target, "safe-defer"),
        "det_shortcut_paid": any(m["behavior"] in DETERMINISTIC and m["paid"] for m in members),
        "partial_paid": any(m["behavior"] in PARTIAL and m["paid"] for m in members),
        "honest_unpaid": any(m["behavior"] == HONEST and m["arrived"] and not m["paid"] for m in members),
    }
    row["accountable"] = row["trainer_accountable"] and not row["det_shortcut_paid"] and not row["honest_unpaid"]
    return row


def aggregate(rows: Sequence[Mapping[str, Any]], datasets: Sequence[str]) -> dict[str, Any]:
    cells: dict[tuple, dict] = {}
    for row in rows:
        validity = "invalid" if row["invalid"] else "valid"
        for key in {(row["dataset"], validity, row["family"], row["scheme"]),
                    (row["dataset"], validity, row["composition"], row["scheme"])}:
            cell = cells.setdefault(key, {"n": 0., "blocks": set(), "failed_blocks": set(),
                                          **{m: 0. for m in METRICS}, **{t: 0. for t in TERMINALS}})
            cell["n"] += row["weight"]
            cell["blocks"].add(row["block"])
            cell[row["terminal"]] += row["weight"]
            for m in METRICS:
                cell[m] += row["weight"] * row[m]
            if not row["accountable"]:
                cell["failed_blocks"].add(row["block"])
    out = {}
    for (dataset, validity, scope, scheme), cell in sorted(cells.items()):
        blocks, failed = len(cell.pop("blocks")), len(cell.pop("failed_blocks"))
        cell.update(blocks=blocks, blocks_with_nonaccountable=failed,
                    cp_upper_95_blocks=clopper_pearson_bound(0, blocks, side="upper", alpha=.05) if failed == 0 else None)
        out.setdefault(dataset, {}).setdefault(validity, {}).setdefault(scope, {})[scheme] = cell
    return {d: out.get(d, {}) for d in datasets}


def decide(summary: Mapping[str, Any], datasets: Sequence[str], integrity_ok: bool) -> dict[str, Any]:
    if not integrity_ok:
        return {"terminal": "VOID_RERUN", "routing": "NO_PAPER_CHANGE"}
    negative = []
    for dataset in datasets:
        for validity in ("valid", "invalid"):
            for family in FAMILIES:
                cell = summary[dataset][validity][family]["sevc"]
                if cell["accountable"] < cell["n"] - 1e-9:
                    negative.append(f"{dataset}/{validity}/{family}")
    return {"terminal": "PASS" if not negative else "ACCEPTED_NEGATIVE", "routing": "PAPER_CHANGE_REQUIRED",
            "sevc_nonaccountable_cells": negative}


def decide_calibrated(summary: Mapping[str, Any], datasets: Sequence[str], integrity_ok: bool) -> dict[str, Any]:
    """Single-verifier SEVC: no accusation, no shortcut paid, every job completed, and any
    non-accountable outcome confined to compositions containing a partial replay."""
    if not integrity_ok:
        return {"terminal": "VOID_RERUN", "routing": "NO_PAPER_CHANGE"}
    violations = []
    for dataset in datasets:
        for validity in ("valid", "invalid"):
            for scope, cells in summary[dataset][validity].items():
                if "/" not in scope and scope not in ("all-honest", "one-missing"):
                    continue
                cell = cells["sevc-calibrated"]
                partial = any(b in scope for b in PARTIAL)
                where = f"{dataset}/{validity}/{scope}"
                if cell["false_accusation"] > 1e-9 or cell["det_shortcut_paid"] > 1e-9:
                    violations.append(where + ": accusation or shortcut paid")
                if cell["completed"] < cell["n"] - 1e-9:
                    violations.append(where + ": incomplete")
                if cell["accountable"] < cell["n"] - 1e-9 and not partial:
                    violations.append(where + ": non-accountable without partial replay")
    return {"terminal": "PASS" if not violations else "ACCEPTED_NEGATIVE",
            "routing": "PAPER_CHANGE_REQUIRED", "violations": violations}


DECISION_RULES = {"sevc-all-accountable": decide, "single-verifier-residual": decide_calibrated}


def regression_check(summary: Mapping[str, Any], reference: Mapping[str, Any]) -> dict[str, Any]:
    """Every scheme cell of the reference run must reappear with identical metrics."""
    compared, differences = 0, []
    for dataset, by_validity in reference.items():
        for validity, by_scope in by_validity.items():
            for scope, by_scheme in by_scope.items():
                for scheme, cell in by_scheme.items():
                    compared += 1
                    now = summary.get(dataset, {}).get(validity, {}).get(scope, {}).get(scheme)
                    if now != cell:
                        differences.append(f"{dataset}/{validity}/{scope}/{scheme}")
    return {"compared_cells": compared, "differences": differences}


def replay_residuals(packet: Path, units: Mapping[str, Mapping[str, Any]], datasets: Sequence[str]) -> dict[str, Any]:
    """Owner reference replays of committed production segments, deduplicated by proof."""
    receipts: dict[str, dict[str, Mapping[str, Any]]] = {d: {} for d in datasets}
    files = {}
    for uid, unit in sorted(units.items()):
        path = packet / "reference-audits" / (uid + ".json")
        if not unit.get("assignments") or not path.is_file():
            continue
        files[uid] = sha256_file(path)
        for proof, receipt in json.loads(path.read_text()).get("production_receipts", {}).items():
            receipts[unit["dataset"]].setdefault(proof, receipt)
    out = {}
    for dataset in datasets:
        rows = list(receipts[dataset].values())
        passed = [r for r in rows if r["passed"]]
        failed = [r for r in rows if not r["passed"]]
        out[dataset] = {
            "unique_receipts": len(rows), "passed": len(passed), "failed": len(failed),
            "passed_max_abs_difference": max((r["max_abs_difference"] for r in passed), default=None),
            "passed_max_model_abs_difference": max((r["max_model_abs_difference"] for r in passed), default=None),
            "passed_max_optimizer_abs_difference": max((r["max_optimizer_abs_difference"] for r in passed), default=None),
            "failed_min_abs_difference": min((r["max_abs_difference"] for r in failed), default=None),
            "tolerances": sorted({r["tolerance"] for r in rows}),
            "state_complete": all(r.get("state_complete", False) for r in rows)}
    return {"datasets": out, "reference_audit_files": len(files),
            "reference_audit_manifest_sha256": sha256_text(json.dumps(files, sort_keys=True))}


def direct_replay_reference(cost_summary: Mapping[str, Any], datasets: Sequence[str]) -> dict[str, Any]:
    """Owner direct replay at SEVC's owner time: valid jobs accepted; an invalid one missed w.p. 1-q."""
    out = {}
    for dataset in datasets:
        q = cost_summary["E2_matched_budget"][f"{dataset}/init-invalid"]["direct_detection"]["1"]
        segments = cost_summary["E2_matched_budget"][f"{dataset}/init-invalid"]["direct_segments_at_equal_owner_time"]
        out[dataset] = {"segments_at_equal_owner_time": segments, "valid_accept": 1.,
                        "invalid_false_release": 1 - q, "invalid_reject": q}
    return out


# --------------------------------------------------------------------------- run

def _check_input(path: Path, expected: str, name: str, failures: list[str]) -> None:
    if sha256_file(path) != expected:
        failures.append(f"input hash mismatch: {name}")


def run(config_path: Path, output_root: Path, *, repo_root: Path, command: Sequence[str]) -> dict[str, Any]:
    config = json.loads(config_path.read_text())
    change_id = config.get("change_id", CHANGE_ID)
    schemes = tuple(config.get("schemes", SCHEME_ORDER))
    out = ensure_experiment_output_root(output_root, repo_root, change_id)
    if any(out.iterdir()):
        raise FileExistsError("output root must be empty")
    inputs = config["inputs"]
    failures: list[str] = []
    packet = Path(inputs["packet_root"])
    producer_config_path = repo_root / inputs["producer_config"]["path"]
    for path, spec, name in ((producer_config_path, inputs["producer_config"]["sha256"], "producer config"),
                             (packet / "result_index.json", inputs["result_index_sha256"], "result index"),
                             (Path(inputs["service_rows"]["path"]), inputs["service_rows"]["sha256"], "service rows"),
                             (Path(inputs["cost_summary"]["path"]), inputs["cost_summary"]["sha256"], "cost summary")):
        _check_input(path, spec, name, failures)
    units, _, packet_integrity = verify_packet(packet, json.loads(producer_config_path.read_text()))
    service_rows = json.loads(Path(inputs["service_rows"]["path"]).read_text())
    jobs, job_integrity = load_jobs(units, service_rows)
    failures += job_integrity["failures"]
    datasets = list(config["datasets"])
    if job_integrity["checks"].get("jobs") != config["expected_jobs"] \
            or job_integrity["checks"].get("services") != config["expected_services"]:
        failures.append("I2 job or service count differs from the frozen protocol")
    parameters = dict(config["controller"])
    # The equivalence check runs the canonical controller directly on the registered roster.
    canonical = {k: v for k, v in parameters.items() if k != "calibrated_correctness"}
    rows, equivalence = [], {"compared": 0, "disagreements": []}
    if not failures:
        for key, job in jobs.items():
            other = CONTEXTS[1 - CONTEXTS.index(job.context)]
            companion = jobs[(job.dataset, other, job.block)]
            for family, composition, slots in compositions():
                for scheme in schemes:
                    implementation = SCHEMES.get(scheme)
                    kwargs = {"parameters": parameters} if scheme in CONTROLLER_SCHEMES else {}
                    try:
                        outcomes = implementation(job, slots, companion, **kwargs)
                    except Exception as exc:  # I6: a scheme that cannot decide voids the run
                        failures.append(f"I6 {scheme} {job.key} {composition}: {exc!r}")
                        continue
                    for index, (weight, outcome) in enumerate(outcomes):
                        rows.append({"dataset": job.dataset, "context": job.context, "block": job.block,
                                     "invalid": job.invalid, "family": family, "composition": composition,
                                     "scheme": scheme, "slot": index if scheme == "single-verifier" else None,
                                     "weight": weight, "terminal": outcome["terminal"],
                                     "members": outcome["members"], **classify(outcome, job.invalid)})
                primary = controller_outcome(job, companion, slots, policy="value-preserving-rc", admission=True,
                                             parameters=canonical)[1]
                check = controller_outcome(job, companion, slots, policy=config["equivalence_policy"],
                                           admission=True, parameters=canonical)[1]
                equivalence["compared"] += 1
                if (primary["route"], primary["members"]) != (check["route"], check["members"]):
                    equivalence["disagreements"].append({"job": job.key, "composition": composition,
                                                         "primary": primary, "check": check})
    summary = aggregate(rows, datasets) if not failures else {}
    regression = None
    if "regression" in config and not failures:
        reference_path = Path(config["regression"]["summary"]["path"])
        if sha256_file(reference_path) != config["regression"]["summary"]["sha256"]:
            failures.append("regression reference hash mismatch")
        else:
            reference = json.loads(reference_path.read_text())["summary"]
            regression = regression_check(summary, reference)
            if regression["differences"]:
                failures.append(f"regression: {len(regression['differences'])} cells differ from the reference run")
    residuals = replay_residuals(packet, units, datasets) if config.get("residual_audit") else None
    integrity = {"packet": packet_integrity, "jobs": job_integrity["checks"], "failures": failures,
                 "status": "PASS" if not failures else "FAIL"}
    decision = DECISION_RULES[config.get("decision_rule", "sevc-all-accountable")](summary, datasets, not failures)
    cost_summary = json.loads(Path(inputs["cost_summary"]["path"]).read_text())
    write_json(out / "integrity.json", integrity)
    write_json(out / "per-job.json", rows)
    write_json(out / "summary.json", {
        "change_id": change_id, "decision": decision, "schemes": list(schemes),
        "regression": regression, "residuals": residuals,
        "compositions": [{"family": f, "composition": c, "slots": list(s)} for f, c, s in compositions()],
        "summary": summary, "equivalence": {"policy": config["equivalence_policy"], **equivalence},
        "direct_replay_reference": direct_replay_reference(cost_summary, datasets)})
    code = [Path(__file__), repo_root / "scripts/run_end_to_end_comparison.py", config_path,
            repo_root / "sevc/committee/executed_recovery.py", repo_root / "sevc/committee/recovery_game.py",
            repo_root / "sevc/evaluation/f_rq_audit.py", repo_root / "sevc/evaluation/f_rq234_audit.py"]
    write_json(out / "manifest.json", {
        "change_id": change_id, "command": list(command),
        "code_sha256": {str(p.resolve().relative_to(repo_root.resolve())): sha256_file(p) for p in code},
        "outputs": {n: sha256_file(out / n) for n in ("integrity.json", "per-job.json", "summary.json")},
        "environment": capture_environment()})
    return decision
