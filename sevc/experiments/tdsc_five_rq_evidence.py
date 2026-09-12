"""Registry-dispatched assembly for the reviewed five-RQ paid replay workload."""
from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import time

from sevc.core.artifacts import write_json, sha256_file, canonical_json_text, sha256_text
from sevc.verification.paid_replay_service import (
    WorkClock, prepare_service, execute_assignment, settle_service, BEHAVIOR_KEYS,
    verify_fixture_equivalence,
    SourceReferenceNotEstablished,
)
from sevc.committee.executed_recovery import execute_recovery, FAULTS, RECOVERY_CONTROLLERS
from sevc.evaluation.workload_performance import AppendLog, GPUSampler, integrate_utilization
from sevc.evaluation.five_rq_evidence import audit_assignment, workload_schedule, recovery_conditions, evidence_registry

CHANGE = "experiment-tdsc-five-rq-evidence-v1"
REPAIR_CHANGE = "experiment-tdsc-production-audit-repair-v1"
SCOPED_CHANGE = "experiment-tdsc-scoped-five-rq-confirmation-v1"
BALANCED_CHANGE = "experiment-tdsc-balanced-five-rq-v1"
BOUNDED_CHANGE = "experiment-tdsc-bounded-memory-v1"
RUNTIME_CHANGE = "experiment-tdsc-scoped-runtime-optimization-v1"
GPU_STATE_CHANGE = "experiment-tdsc-gpu-state-transform-v1"
STATE_COST_CHANGE = "experiment-tdsc-state-cost-breakdown-v1"


class ReplayDevice:
    def __init__(self, name):
        self.name = name

    def synchronize(self):
        if self.name.startswith("cuda"):
            import torch
            torch.cuda.current_stream().synchronize()


def scientific_configuration_sha256(config):
    """Exclude only activation wiring; retain every scientific/performance field."""
    projected = json.loads(json.dumps(config))
    for key in ("formal_execution_authorized", "protocol_lock_path", "protocol_lock_sha256",
                "confirmation_receipt_path", "ready_packet_path"):
        projected.pop(key, None)
    for key in tuple(projected):
        if key.startswith("_runtime_"):
            projected.pop(key)
    for item in projected["profiles"].values():
        item.pop("live_preflight_receipt", None)
    return sha256_text(canonical_json_text(projected))


def confirmation_barrier(config, profile, output_root):
    """Validate a new standalone run without using private deployment receipts."""
    from sevc.experiments.source_distribution import validate_public_run
    validate_public_run(config, profile, output_root)


def _repair_assignment(*, prepared, method, behavior, context, seed, assignment,
        clock, perf, commits, audit_log, job_id=None, verifier_id=None):
    from sevc.verification.production_audit import (
        CommittedProductionAudit, delivered_policy_tasks, settle_audited_service,
    )
    clock.context["assignment_id"] = assignment
    tasks = delivered_policy_tasks(prepared, method)
    challenge, commit_seconds = clock.call("owner_audit_commit", "owner",
        CommittedProductionAudit, assignment, audit_log)
    began = time.monotonic()
    report, executions = execute_assignment(tasks, behavior=behavior,
        trainer_hashes=prepared.trainer_committed_hashes,
        trainer_answer_cache=prepared.trainer_answer_cache, context=context, seed=seed,
        assignment_id=assignment, clock=clock, performance=perf,
        emit_commit=lambda row: commits({**row, "monotonic": time.monotonic()}),
        job_id=job_id, verifier_id=verifier_id)
    cost = time.monotonic() - began
    (settlement, audit_receipt), audit_seconds = clock.call("production_service_admission", "owner",
        settle_audited_service, report, policy_key=method,
        production=prepared.production_references, probes=prepared.references["rcmp-source-coupled"],
        challenge=challenge, cost_seconds=cost,
        effort_fraction=sum(r["replayed"] for r in executions)/len(executions))
    return report, settlement, executions, audit_receipt, commit_seconds + audit_seconds


def _run_repair_block(*, context, prepared, workload, index, seed, config, clock, perf,
        commits, audit_log, records, events, output_root, preparation_seconds):
    """Repair cells share the outer dataset/source runner and canonical service calls."""
    study = config["repair_study"]
    dataset, invalid = workload["dataset"], workload["invalid_source_count"]
    base = {"dataset": dataset, "block_index": index, "seed": seed, "invalid_source_count": invalid}
    block_id = f"{dataset}-{seed}-invalid{invalid}"
    methods = study["methods"]
    # Counterbalance deterministic timing order without adapting to measured outcomes.
    offset = index % len(methods)
    methods = methods[offset:] + methods[:offset]
    rows = []
    for method in methods:
        behaviors = study["behaviors"]
        shift = index % len(behaviors)
        for behavior in behaviors[shift:] + behaviors[:shift]:
            clock.context.update(mechanism=method, behavior=behavior)
            assignment = f"{block_id}-{method}-{behavior}"
            start = time.monotonic()
            report, settlement, detail, receipt, owner_seconds = _repair_assignment(
                prepared=prepared, method=method, behavior=behavior, context=context,
                seed=seed, assignment=assignment, clock=clock, perf=perf,
                commits=commits, audit_log=audit_log)
            row = {**base, "kind": "core", "assignment_id": assignment, "mechanism": method,
                "behavior": behavior, "report": asdict(report), "settlement": asdict(settlement),
                "execution": detail, "audit_receipt": receipt, "owner_admission_seconds": owner_seconds}
            records(row)
            rows.append(row)
            end = time.monotonic()
            events({**base, "unit": "assignment", "mechanism": method, "behavior": behavior,
                    "start": start, "end": end, "seconds": end-start})
    if index == study["retain_witnesses_block_index"] and invalid == study["retain_witnesses_invalid_count"]:
        import torch
        refs = {r.task_id: r for r in prepared.production_references.items}
        selected = [next(t for t in prepared.mechanisms["rcmp-source-coupled"]
                         if t.task_id in refs and refs[t.task_id].answer is answer) for answer in (True, False)]
        folder = output_root/"retained-witnesses"
        folder.mkdir(exist_ok=True)
        for task in selected:
            torch.save({"proof": task.proof, "descriptor": task.descriptor,
                        "envelope": task.envelope, "reference": asdict(refs[task.task_id])},
                       folder/f"{dataset}-{task.task_id}.pt")
    recovery = study["recovery"]
    if index not in recovery["block_indices"] or invalid not in recovery["invalid_source_counts"]:
        return
    cost_unit = next(r["settlement"]["verifier_cost"] for r in rows
                    if r["mechanism"] == "rcmp-source-coupled" and r["behavior"] == "honest")
    if cost_unit <= 0:
        raise ValueError("honest matched cost unit must be positive")
    required_ids = tuple(r.task_id for r in prepared.production_references.items)
    for method in recovery["methods"]:
        owner_reserve = max(recovery["owner_reserve_floor_units"], recovery["owner_reserve_factor"] *
            max(r["owner_admission_seconds"] for r in rows if r["mechanism"] == method)/cost_unit)
        for colluding in recovery["colluding_counts"]:
            recovery_id = f"{block_id}-{method}-coalition{colluding}"
            def activate(job_id, verifier_id):
                behavior = ("joint-view-targeted-cover" if invalid else "zero-effort-fixed-prior-constant") \
                    if job_id == "j0" and int(verifier_id[1:]) < colluding else "honest"
                clock.context.update(mechanism=method, behavior=behavior, recovery_id=recovery_id)
                assignment = f"{recovery_id}-{job_id}-{verifier_id}"
                report, settled, detail, receipt, seconds = _repair_assignment(
                    prepared=prepared, method=method, behavior=behavior, context=context,
                    seed=seed, assignment=assignment, clock=clock, perf=perf,
                    commits=commits, audit_log=audit_log, job_id=job_id, verifier_id=verifier_id)
                settled = replace(settled, diagnostics=settled.diagnostics + (("owner_work_cost", seconds/cost_unit),))
                records({**base, "kind": "recovery-assignment", "recovery_id": recovery_id,
                    "assignment_id": assignment, "mechanism": method, "behavior": behavior,
                    "report": asdict(report), "settlement": asdict(settled), "execution": detail,
                    "audit_receipt": receipt, "owner_admission_seconds": seconds})
                return report, settled, detail
            clock.context.update(mechanism=method, recovery_id=recovery_id)
            start = time.monotonic()
            result = execute_recovery(policy="current-certified-ecs", fault="no-missing", seed=seed,
                required_ids=required_ids, execute=activate, clock=clock,
                timeout_seconds=recovery["timeout_seconds"], deadline_seconds=recovery["deadline_seconds"],
                activation_lanes=1, owner_cost_reserve_per_assignment=owner_reserve,
                owner_prepaid_cost_per_job=preparation_seconds/cost_unit,
                minimum_pass_count=recovery["minimum_pass_count"])
            result.update(cost_unit_seconds=cost_unit, owner_prepaid_basis=
                "conservative whole preparation wall upper bound per job; includes trainer work; not measured owner-only time")
            records({**base, "kind": "recovery", "recovery_id": recovery_id,
                     "mechanism": method, "colluding_primary_count": colluding, "recovery": result})
            end = time.monotonic()
            events({**base, "unit": "recovery", "mechanism": method, "recovery_id": recovery_id,
                    "start": start, "end": end, "seconds": end-start})
    clock.context.pop("recovery_id", None)


def run_tdsc_five_rq_evidence(repo_root, config_path, output_root, config, **unused):
    entry_start = time.monotonic()
    from sevc.training.replay_sources import ReplayDatasetContext
    key = config.get("_runtime_profile", config["default_profile"])
    profile = config["profiles"][key]
    repair = config.get("change_id") == REPAIR_CHANGE
    scoped = config.get("change_id") in {SCOPED_CHANGE, RUNTIME_CHANGE, BALANCED_CHANGE, BOUNDED_CHANGE}
    output_root = Path(output_root)
    confirmation_barrier(config, profile, output_root)
    if config.get("_runtime_max_units") is not None:
        raise ValueError("five-RQ workload cannot be silently truncated with max-units")
    if not output_root.is_absolute() or repo_root == output_root or repo_root in output_root.parents:
        raise ValueError("evidence must be outside the checkout")
    if output_root.exists():
        raise FileExistsError("append-only run root already exists")
    import torch
    perf = config["performance"][profile["performance_key"]]
    torch.set_num_threads(perf["cpu_threads"])
    if torch.get_num_interop_threads()!=1:
        torch.set_num_interop_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = ReplayDevice(profile["device"])
    cuda = device.name.startswith("cuda")
    if cuda:
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
            raise ValueError("frozen deterministic cuBLAS workspace is required")
        from sevc.experiments.source_distribution import validate_cuda
        validate_cuda(profile)
    output_root.mkdir(parents=True, exist_ok=False)
    events, phases, records, sources, commits = [AppendLog(output_root/name) for name in
        ("workload-units.jsonl", "phase-timing.jsonl", "assignment-audit.jsonl", "source-task-identities.jsonl", "report-lifecycle.jsonl")]
    audit_log = AppendLog(output_root/"audit-lifecycle.jsonl") if repair else None
    logs = (events, phases, records, sources, commits) + ((audit_log,) if repair else ())
    clock = WorkClock(device.synchronize, phases)
    if scoped:
        from sevc.core.role_accounting import RoleClock
        from sevc.experiments.scoped_five_rq_units import ScopedStudy
        clock = RoleClock(device.synchronize,phases,cuda=cuda,selective_cuda=perf.get("selective_cuda_timing",False))
        study = ScopedStudy(config,profile,output_root,clock,records,sources,commits,events)
    sampler = GPUSampler(output_root, profile.get("gpu_uuid", "CPU-FIXTURE"), cuda)
    provenance = {"change_id": config.get("change_id", CHANGE), "profile": key, "full_matrix": profile["full_matrix"],
        "scientifically_eligible": False, "namespace": profile["namespace"],
        "config_sha256": sha256_file(config_path), "performance_sha256": sha256_text(canonical_json_text(perf)),
        "command": __import__("sys").argv, "python": platform.python_version(), "torch": torch.__version__,
        "device": device.name, "gpu_uuid": profile.get("gpu_uuid"), "worker_pid": os.getpid(),
        "created_at": datetime.now(timezone.utc).isoformat(), "configuration": config}
    write_json(output_root/"provenance.json", provenance)
    write_json(output_root/"status.json", {"status": "RUNNING_STANDALONE_FULL_MATRIX" if profile["full_matrix"] else "RUNNING_NONFORMAL", "pid": os.getpid()})
    windows = []
    screen_counterexample = False
    try:
        sampler.start()
        entire_start = time.monotonic()
        for workload in workload_schedule(config, profile):
            if profile["full_matrix"] and workload["stage"] == "confirmation" and screen_counterexample:
                break
            dataset = workload["dataset"]
            spec = config["datasets"][dataset]
            clock.context = {"dataset": dataset}
            context, _ = clock.call("dataset_load", "owner", ReplayDatasetContext,
                dataset, spec, Path(profile["data_root"]), device)
            partition = tuple(profile["sample_ranges"][dataset])
            if scoped:
                if not profile['full_matrix'] and profile.get('verify_equivalence',False):
                    result=verify_fixture_equivalence(context,partition,clock,perf)
                    write_json(output_root/f'equivalence-{dataset}.json',result)
                study.run_dataset(context,partition)
                context.close_task_lanes()
                continue
            if profile.get("equivalence_only"):
                result = verify_fixture_equivalence(context, partition, clock, perf)
                write_json(output_root/f"equivalence-{dataset}.json", result)
                context.close_task_lanes()
                continue
            for index in range(workload["blocks"]):
                seed = workload["seed_start"] + index
                clock.context = {"dataset": dataset, "block_index": index, "block_seed": seed, "stage": workload["stage"]}
                if repair:
                    clock.context["invalid_source_count"] = workload["invalid_source_count"]
                start = time.monotonic()
                prepared = prepare_service(context, seed=seed, partition=partition,
                    namespace=workload["stage"], clock=clock, performance=perf,
                    **({"mechanisms": ("rcmp-source-coupled",),
                        "invalid_source_count": workload["invalid_source_count"],
                        "collect_production_references": True} if repair else {}))
                sources({"dataset": dataset, "seed": seed, "sources": prepared.source_rows,
                         "tasks": prepared.audit_rows,
                         **({"invalid_source_count": workload["invalid_source_count"], "block_index": index,
                             "production_references": [asdict(r) for r in prepared.production_references.items],
                             "trainer_answer_cache": prepared.trainer_answer_cache} if repair else {})})
                source_end = time.monotonic()
                events({"dataset": dataset, "unit": "source_and_reference", "block_index": index,
                        "start": start, "end": source_end, "seconds": source_end-start,
                        **({"invalid_source_count": workload["invalid_source_count"]} if repair else {})})
                if repair:
                    _run_repair_block(context=context, prepared=prepared, workload=workload, index=index,
                        seed=seed, config=config, clock=clock, perf=perf, commits=commits, audit_log=audit_log,
                        records=records, events=events, output_root=output_root, preparation_seconds=source_end-start)
                    end = time.monotonic()
                    windows.append({"dataset": dataset, "unit": "core_block", "block_index": index,
                        "invalid_source_count": workload["invalid_source_count"], "start": start,
                        "end": end, "seconds": end-start, "includes_nested_recovery": True})
                    events(windows[-1])
                    del prepared
                    continue
                honest_max = 0.
                for mechanism, tasks in prepared.mechanisms.items():
                    for behavior in BEHAVIOR_KEYS:
                        clock.context.update(mechanism=mechanism, behavior=behavior)
                        assignment = f"{dataset}-{seed}-{mechanism}-{behavior}"
                        a_start = time.monotonic()
                        report, executions = execute_assignment(tasks, behavior=behavior,
                            trainer_hashes=prepared.trainer_committed_hashes, context=context,
                            seed=seed, assignment_id=assignment, clock=clock, performance=perf, emit_commit=commits)
                        cost = time.monotonic()-a_start
                        settlement, _ = clock.call("service_settlement", "owner", settle_service, report,
                            prepared.references[mechanism], cost_seconds=cost,
                            effort_fraction=sum(r["replayed"] for r in executions)/len(executions))
                        task_rows = [r for r in prepared.audit_rows if r["mechanism"] in {mechanism, "shared"}]
                        # Technical output is restricted to integrity; no detection/utility summaries.
                        audit, _ = clock.call("independent_assignment_audit", "evaluator", audit_assignment,
                            asdict(report), asdict(settlement), task_rows, executions)
                        if profile["full_matrix"] and workload["stage"] == "screen" and mechanism == "rcmp-source-coupled":
                            screen_counterexample |= bool(audit["invalid_production_report_admitted"] or
                                audit["verifier_paid_without_service"] or (behavior == "honest" and settlement.status != "PASS"))
                        records({"dataset": dataset, "block_index": index, "mechanism": mechanism,
                            "behavior": behavior, "report": asdict(report), "settlement": asdict(settlement),
                            "execution": executions, "audit_integrity": audit["commitment_valid"]})
                        a_end = time.monotonic()
                        events({"dataset": dataset, "unit": "assignment", "mechanism": mechanism,
                            "behavior": behavior, "block_index": index, "start": a_start, "end": a_end,
                            "seconds": a_end-a_start})
                        if behavior == "honest":
                            honest_max = max(honest_max, a_end-a_start)
                core_end = time.monotonic()
                windows.append({"dataset": dataset, "unit": "core_block", "block_index": index,
                                "start": start, "end": core_end, "seconds": core_end-start})
                events(windows[-1])
                if not profile["full_matrix"] or index < 30:
                    # Reference comparison is the actual same final eight RCMP tasks.
                    clock.context.update(mechanism="rcmp-source-coupled", behavior="honest")
                    reference_start = time.monotonic()
                    probe_ids = dict(prepared.references["rcmp-source-coupled"].answers)
                    probe_tasks = tuple(t for t in prepared.mechanisms["rcmp-source-coupled"] if t.task_id in probe_ids)
                    reference_report, _ = execute_assignment(probe_tasks, behavior="honest", trainer_hashes=frozenset(),
                        context=context, seed=seed, assignment_id=f"reference-{seed}", clock=clock,
                        performance=perf, emit_commit=commits)
                    if dict(zip(reference_report.ordered_segment_ids, reference_report.verdicts)) != probe_ids:
                        raise ValueError("same-task reference replay disagrees with derived reference")
                    r_end = time.monotonic()
                    events({"dataset": dataset, "unit": "same_task_reference", "block_index": index,
                            "start": reference_start, "end": r_end, "seconds": r_end-reference_start})
                conditions = recovery_conditions(profile, workload, index)
                if conditions:
                    timeout = profile.get("timeout_seconds", {}).get(dataset, max(1.,1.5*honest_max))
                    for fault in conditions:
                        recovery_prepared = prepared
                        material_start = time.monotonic()
                        if fault == "no-missing":
                            recovery_prepared = prepare_service(context, seed=seed+50, partition=partition,
                                namespace=profile["namespace"]+"-all-honest", clock=clock,
                                performance=perf, all_honest=True, mechanisms=("rcmp-source-coupled",))
                            sources({"dataset": dataset, "seed": seed+50, "all_honest": True,
                                     "sources": recovery_prepared.source_rows, "tasks": recovery_prepared.audit_rows})
                        material_end = time.monotonic()
                        events({"dataset": dataset, "unit": "recovery_source", "fault": fault,
                            "start": material_start, "end": material_end, "seconds": material_end-material_start})
                        mechanism = "rcmp-source-coupled"
                        tasks = recovery_prepared.mechanisms[mechanism]
                        required_ids = tuple(r["task_id"] for r in recovery_prepared.audit_rows if r["role"] == "production")
                        for policy in RECOVERY_CONTROLLERS.keys():
                            clock.context.update(fault=fault, recovery_policy=policy)
                            def activate(job_id, verifier_id):
                                assignment_started = time.monotonic()
                                report, detail = execute_assignment(tasks, behavior="honest", trainer_hashes=frozenset(),
                                    context=context, seed=seed,
                                    assignment_id=f"recovery-{seed}-{fault}-{policy}-{job_id}-{verifier_id}",
                                    clock=clock, performance=perf, emit_commit=commits,
                                    job_id=job_id, verifier_id=verifier_id)
                                assignment_cost = time.monotonic()-assignment_started
                                settlement, _ = clock.call("service_settlement", "owner", settle_service,
                                    report, recovery_prepared.references[mechanism],
                                    cost_seconds=assignment_cost, effort_fraction=1.)
                                return report, settlement, detail
                            rec_start = time.monotonic()
                            recovery = execute_recovery(policy=policy, fault=fault, seed=seed,
                                required_ids=required_ids, execute=activate, clock=clock,
                                timeout_seconds=timeout, deadline_seconds=12*timeout,
                                activation_lanes=perf.get("activation_lanes",1))
                            records({"dataset": dataset, "recovery": recovery})
                            rec_end = time.monotonic()
                            events({"dataset": dataset, "unit": "recovery", "fault": fault, "policy": policy,
                                "block_index": index, "start": rec_start, "end": rec_end, "seconds": rec_end-rec_start})
                        del recovery_prepared
                        clock.context.pop("fault", None); clock.context.pop("recovery_policy", None)
                cleanup_started = time.monotonic()
                del prepared
                # These loop/closure bindings otherwise keep a previous bank of
                # large replay states alive while the next bank is constructed.
                # All tasks and durable writes have completed at this boundary.
                tasks = probe_tasks = activate = None
                cleanup_end = time.monotonic()
                events({"dataset":dataset,"unit":"block_cleanup","block_index":index,
                        "start":cleanup_started,"end":cleanup_end,"seconds":cleanup_end-cleanup_started})
            context.close_task_lanes()
        if scoped:
            study.run_cpu()
        for log in logs:
            log.close()
        if scoped:
            from sevc.evaluation.scoped_result_audit import audit_and_summarize
            audit_and_summarize(output_root,config,profile)
        entire_end = time.monotonic()
        sampler.finish(output_root)
        utilization = integrate_utilization(sampler.rows, entire_start, entire_end) if cuda else None
        performance_window = {"start": entire_start, "end": entire_end,
            "wall_seconds": entire_end-entire_start, "utilization": utilization,
            "runner_entry_monotonic": entry_start,
            "entry_setup_seconds": entire_start-entry_start,
            "sampling_errors": sampler.errors, "windows": windows}
        write_json(output_root/"performance-window.json", performance_window)
        if scoped and cuda:
            from sevc.evaluation.scoped_forecast import assess_sampling
            write_json(output_root/'sampling-assessment.json',
                       assess_sampling(sampler.rows,performance_window,profile['gpu_uuid']))
        status = ("ACCEPTED_NEGATIVE_SCREEN" if screen_counterexample else "FORMAL_EVIDENCE_AWAITING_FINAL_AUDIT") if profile["full_matrix"] else "NONFORMAL_COMPLETE"
        if repair:
            status = "REPAIR_EVIDENCE_AWAITING_INDEPENDENT_AUDIT"
        if scoped:
            status = 'STANDALONE_FULL_MATRIX_AUDITED' if profile['full_matrix'] else 'NONFORMAL_COMPLETE'
        write_json(output_root/"status.json", {"status": status, "full_matrix_run_started": profile["full_matrix"]})
        write_json(output_root/"evidence-registry.json", evidence_registry(formal=profile["full_matrix"],status=status))
        write_json(output_root/"fixed-overhead.json", {"entry_setup_seconds": entire_start-entry_start,
            "export_preindex_seconds": time.monotonic()-entire_end,
            "python_import_startup_included": False,
            "note":"process startup must additionally be measured from supervisor launch to runner entry"})
        manifest = {str(p.relative_to(output_root)): sha256_file(p) for p in output_root.rglob("*") if p.is_file()}
        write_json(output_root/"result_index.json", manifest)
        return {"status": status, "output_root": str(output_root), "utilization": utilization}
    except BaseException as exc:
        if "context" in locals():
            context.close_task_lanes()
        write_json(output_root/"status.json", {"status": "SOURCE_REFERENCE_NOT_ESTABLISHED" if isinstance(exc, SourceReferenceNotEstablished) else "TECHNICAL_ATTEMPT_FAILED", "error": repr(exc),
                                              "full_matrix_run_started": profile["full_matrix"]})
        for log in logs:
            if not log.handle.closed:
                log.close()
        try:
            sampler.finish(output_root)
        except Exception:
            pass
        raise
