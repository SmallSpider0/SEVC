"""Offline audit and block-level evidence calculations for paid replay."""
from __future__ import annotations

from dataclasses import asdict
import math

from sevc.incentives.verifier_protocol import CommittedVerifierReport


def evidence_registry(*, formal, status):
    """Stable seven-slot artifact routing; eligibility requires later independent audit."""
    carriers = {
        "P2SOURCE":("source-task-identities.jsonl",),
        "P2MECH":("assignment-audit.jsonl","source-task-identities.jsonl"),
        "P2REF":("report-lifecycle.jsonl","workload-units.jsonl"),
        "P2JOINT":("assignment-audit.jsonl","phase-timing.jsonl"),
        "P2ECS":("assignment-audit.jsonl","workload-units.jsonl"),
        "P2COST":("phase-timing.jsonl","workload-units.jsonl"),
        "P2UTILITY":("assignment-audit.jsonl","phase-timing.jsonl"),
    }
    return {"full_matrix":formal,"run_status":status,"paper_eligible":False,
            "pending_independent_scientific_audit":True,
            "slots":{key:{"artifacts":list(paths),"status":"RAW_AWAITING_AUDIT" if formal else "NONFORMAL_NOT_SCIENTIFIC_EVIDENCE"}
                     for key,paths in carriers.items()}}


def audit_closed_screen(root):
    """Recompute the historical screen from raw reports, not its summary/gate.

    This is a recorded-receipt audit; original tensor payloads are unavailable.
    It neither resumes the stopped campaign nor certifies unrun datasets.
    """
    import json
    import hashlib
    from pathlib import Path
    from sevc.verification.replay_coupled_probes import verify_mutation_invalidity_witness
    root = Path(root)
    index = json.loads((root / "result_index.json").read_text())
    for name, digest in index.items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"screen artifact drift: {name}")
    banks = [json.loads(s) for s in (root / "source-task-identities.jsonl").read_text().splitlines()]
    tasks = {}
    for bank in banks:
        for source in bank["sources"]:
            receipt = source["owner_reference"]
            if receipt["state_complete"] is not True or receipt["proof_sha256"] != source["commitment"]:
                raise ValueError("screen source receipt identity failure")
        for task in bank["tasks"]:
            if task["witness"] and not verify_mutation_invalidity_witness(task["witness"]):
                raise ValueError("screen mutation witness failure")
            tasks[task["task_id"]] = task
    results, reports = [], []
    for line in (root / "assignment-audit.jsonl").read_text().splitlines():
        row = json.loads(line)
        if "report" not in row:
            continue
        report = row["report"]
        checked = audit_assignment(report, row["settlement"],
            [tasks[k] for k in report["ordered_segment_ids"]], row["execution"])
        reports.append(report)
        if "block_index" in row:
            results.append({"dataset": row["dataset"], "block_index": row["block_index"],
                            "mechanism": row["mechanism"], "behavior": row["behavior"], **checked})
    committed = {}
    reveals = 0
    for line in (root / "report-lifecycle.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row["phase"] == "COMMITTED":
            if row["assignment_id"] in committed:
                raise ValueError("duplicate commitment identity")
            committed[row["assignment_id"]] = row["commitment"]
        elif row["phase"] == "REVEALED":
            report = CommittedVerifierReport(**row["report"])
            if committed.get(row["assignment_id"]) != report.commitment:
                raise ValueError("screen commitment/reveal order failure")
            reveals += 1
    core = [r for r in results if r["behavior"] == "joint-view-targeted-cover"]
    if len(core) != 24 or not all(r["invalid_production_report_admitted"] for r in core):
        raise ValueError("historical counterexample identity no longer reproduces")
    return {"status": "AUDIT_PASS", "terminal": "ACCEPTED_NEGATIVE",
            "scope": "CIFAR10_SCREEN_ONLY", "verified_files": len(index),
            "report_reveals": reveals, "audited_assignments": len(reports),
            "core_counterexamples": core, "confirmation_blocks_run": 0,
            "unrun_confirmation_blocks": {d: 120 for d in ("mnist", "cifar10", "cifar100")},
            "independent_gpu_replay": False, "paper_route": "PAPER_CHANGE_REQUIRED"}


def workload_schedule(config, profile):
    """One immutable three-dataset campaign, or a disjoint technical schedule."""
    if config.get('change_id') in {'experiment-tdsc-scoped-five-rq-confirmation-v1','experiment-tdsc-scoped-runtime-optimization-v1','experiment-tdsc-balanced-five-rq-v1','experiment-tdsc-bounded-memory-v1'}:
        return [{'dataset':d,'stage':profile['namespace']} for d in config['scoped_candidate']['science']['dataset_order']]
    if config.get("change_id") == "experiment-tdsc-production-audit-repair-v1":
        study = config["repair_study"]
        return [{"dataset": dataset, "stage": profile["namespace"],
                 "seed_start": study["seed_starts"][dataset],
                 "blocks": study["blocks_per_dataset"], "invalid_source_count": invalid}
                for dataset in study["dataset_order"] for invalid in study["invalid_source_counts"]]
    if profile["full_matrix"]:
        return [
            {"dataset":"cifar10", "stage":"screen", "seed_start":2026094000, "blocks":12},
            {"dataset":"mnist", "stage":"confirmation", "seed_start":2026091000, "blocks":120},
            {"dataset":"cifar10", "stage":"confirmation", "seed_start":2026092000, "blocks":120},
            {"dataset":"cifar100", "stage":"confirmation", "seed_start":2026093000, "blocks":120},
        ]
    return [{"dataset":dataset,"stage":profile["namespace"],
             "seed_start":profile["seed_start"]+spec["seed_offset"],"blocks":profile["core_blocks"]}
            for dataset,spec in config["datasets"].items()]


def recovery_conditions(profile, workload, index):
    if not profile["full_matrix"]:
        return profile["recovery_conditions"] if index == 0 else []
    faults = ["no-missing"]*8+["one-missing"]*8+["correlated-missing"]*7+["insufficient-reserve"]*7
    if workload["stage"] == "screen":
        return ["no-missing","one-missing","correlated-missing","insufficient-reserve"][index:index+1]
    return faults[index:index+1]


def audit_assignment(report_dict, settlement_dict, task_rows, execution_rows):
    """Independently check commitments and transfers after all online decisions."""
    report = CommittedVerifierReport(**report_dict)
    answers = {r["task_id"]: r for r in task_rows}
    if set(report.ordered_segment_ids) != set(answers):
        raise ValueError("report/task identity coverage mismatch")
    mismatches = sum(v is None or v != answers[k]["expected"]
                     for k,v in zip(report.ordered_segment_ids, report.verdicts)
                     if answers[k]["role"] != "production")
    expected_status = "DROPOUT" if not report.committed or not report.revealed else "FAIL_CONFIRMED" if mismatches >= 2 else "PASS"
    if settlement_dict["status"] != expected_status:
        raise ValueError("independent service-status disagreement")
    passed = expected_status == "PASS"
    if settlement_dict["service_fee"] != (1.25 if passed else 0.) or settlement_dict["slashed_bond"] != (0. if passed else .5):
        raise ValueError("independent transfer disagreement")
    invalid_admitted = passed and any(answers[k]["role"] == "production" and not answers[k]["expected"] and v is True
                                     for k,v in zip(report.ordered_segment_ids, report.verdicts))
    return {"invalid_production_report_admitted": bool(invalid_admitted),
            "verifier_paid_without_service": passed and any(not r["replayed"] for r in execution_rows),
            "commitment_valid": True}


def exact_upper(events: int, blocks: int, alpha: float = .05/9):
    if not 0 <= events <= blocks or blocks < 1 or not 0 < alpha < 1:
        raise ValueError("invalid binomial block count")
    if events == blocks:
        return 1.
    if events == 0:
        return -math.expm1(math.log(alpha)/blocks)
    from scipy.stats import beta
    return float(beta.ppf(1-alpha, events+1, blocks-events))


def individual_utility(*, service_fee, slashed_bond, cost_seconds, cost_unit_seconds,
                       bond=.5, liquidity_rate=.01):
    if cost_unit_seconds <= 0 or cost_seconds < 0:
        raise ValueError("invalid registered cost unit")
    return service_fee - slashed_bond - cost_seconds/cost_unit_seconds - bond*liquidity_rate


def paired_interval(values, *, seed=2026095000, draws=20000, alpha=.05):
    import numpy as np
    a = np.asarray(values, dtype=float)
    if a.ndim != 1 or len(a) < 2 or not np.isfinite(a).all():
        raise ValueError("paired interval requires finite independent block differences")
    rng = np.random.default_rng(seed)
    means = np.empty(draws)
    for offset in range(0, draws, 1000):
        size = min(1000, draws-offset)
        means[offset:offset+size] = a[rng.integers(0, len(a), (size,len(a)))].mean(axis=1)
    return {"mean": float(a.mean()), "lower": float(np.quantile(means,alpha/2)),
            "upper": float(np.quantile(means,1-alpha/2)), "blocks": len(a),
            "method": "paired-block-percentile-bootstrap", "degenerate": bool(np.ptp(a)==0)}


def project_fixed_strategy_utility(assignment, task_rows, cost_unit_seconds,
                                   prevalences=(0., .01, .05, .5)):
    """Offline conditional-cost projection, never an adaptive attack experiment.

    Transfers and probe behavior are held fixed. Production conditional task costs
    are reweighted. This does not claim that information/role inference stays
    invariant under an actual population change.
    """
    truth = {r["task_id"]:r for r in task_rows}
    execution = assignment["execution"]
    settlement = assignment["settlement"]
    costs = {True:[], False:[]}
    for row in execution:
        t = truth[row["task_id"]]
        if t["role"] == "production":
            costs[t["expected"]].append(row["seconds"])
    if not all(costs.values()):
        raise ValueError("projection requires both measured production strata")
    constant = settlement["verifier_cost"]-sum(sum(v) for v in costs.values())
    if constant < -1e-6:
        raise ValueError("assignment cost omits measured execution")
    result = []
    for prevalence in prevalences:
        if not 0 <= prevalence <= 1:
            raise ValueError("invalid projection prevalence")
        seconds = max(0.,constant)+sum(map(len,costs.values()))*(
            prevalence*sum(costs[False])/len(costs[False])+
            (1-prevalence)*sum(costs[True])/len(costs[True]))
        result.append({"prevalence":prevalence, "projected_cost_seconds":seconds,
            "individual_utility":individual_utility(service_fee=settlement["service_fee"],
                slashed_bond=settlement["slashed_bond"],cost_seconds=seconds,
                cost_unit_seconds=cost_unit_seconds)})
    return {"kind":"ANALYTICAL_PROJECTION", "fixed_strategy":assignment["behavior"],
        "assumption":"fixed transfers, probe responses, conditional task costs and information; no adaptive reoptimization",
        "rows":result}


def primary_block_endpoints(assignments, task_rows):
    """Union the frozen attack family within each independent RCMP source block."""
    result = {"invalid_production_report_admitted":False,
              "verifier_paid_without_service":False,"honest_false_penalty":False}
    selected = [r for r in assignments if r.get("mechanism") == "rcmp-source-coupled"]
    if len(selected) != 4 or len({r["behavior"] for r in selected}) != 4:
        raise ValueError("full registered behavior family required for a block")
    rows = [r for r in task_rows if r["mechanism"] in {"shared","rcmp-source-coupled"}]
    for row in selected:
        audit = audit_assignment(row["report"],row["settlement"],rows,row["execution"])
        if row["behavior"] == "honest":
            result["honest_false_penalty"] = row["settlement"]["status"] != "PASS"
        else:
            for key in ("invalid_production_report_admitted","verifier_paid_without_service"):
                result[key] |= audit[key]
    return result


def exclusive_role_seconds(phases):
    """Attribute deepest measured call once; retain uninstrumented gaps separately."""
    if not phases:
        return {"roles":{},"unattributed_seconds":0.,"wall_seconds":0.}
    import heapq
    events = {}
    for index,phase in enumerate(phases):
        if phase["end"] < phase["start"]:
            raise ValueError("negative phase interval")
        events.setdefault(phase["start"],[]).append((True,index))
        events.setdefault(phase["end"],[]).append((False,index))
    edges, active, queue = sorted(events), set(), []
    totals, gap = {}, 0.
    for start,end in zip(edges,edges[1:]):
        for entering,index in events[start]:
            if entering:
                active.add(index)
                heapq.heappush(queue,(phases[index]["end"]-phases[index]["start"],index))
            else:
                active.discard(index)
        while queue and queue[0][1] not in active:
            heapq.heappop(queue)
        if queue:
            role = phases[queue[0][1]]["role"]
            totals[role] = totals.get(role,0.)+end-start
        else:
            gap += end-start
    return {"roles":totals,"unattributed_seconds":gap,"wall_seconds":edges[-1]-edges[0]}
