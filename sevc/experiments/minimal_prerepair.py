"""Tiny assembly on the shared five-RQ training/compiler/actor interfaces."""
from __future__ import annotations

from dataclasses import asdict
import gc
import platform
from pathlib import Path
import time

from sevc.core.artifacts import write_json, sha256_file
from sevc.training import verify_replay_proof
from sevc.training.replay_sources import ReplayDatasetContext, materialize_short_source
from sevc.experiments.tdsc_five_rq_evidence import ReplayDevice
from sevc.incentives.verifier_protocol import CommittedVerifierReport
from sevc.verification.paid_replay_service import (
    VERSION, WRAPPER_DOMAIN, WorkClock, DeliveredReplay, OwnerProbeReferences,
    execute_assignment, settle_service, identity,
)
from sevc.verification.public_replay_shortcuts import (
    opaque_source_handle, make_sgd_coherent_forgery,
)
from sevc.verification.replay_coupled_probes import (
    ATOM_KEYS, compile_canonical_replay_task, proof_component_hashes, mutate_replay_proof,
)


def run_structural_screen(repo_root, config_path, output_root, config, **runtime):
    """One source per dataset; sequential task delivery bounds resident memory."""
    if config.get("screen_kind") == "job-pilot":
        return run_job_pilot(repo_root, config_path, output_root, config, **runtime)
    import torch
    root = Path(output_root).resolve()
    repo = Path(repo_root).resolve()
    if (config.get("device") != "cpu" or config.get("scientifically_eligible") is not False
            or config.get("large_run_authorized") is not False
            or config.get("namespace") != "structural-fixture"):
        raise ValueError("structural screening is CPU-only and nonformal")
    if any(config.get(k) is not None for k in ("_runtime_max_units", "_runtime_profile")):
        raise ValueError("the three-dataset fixture cannot be truncated or reprofiled")
    if runtime.get("authorization_only") or runtime.get("authorization_receipt_path"):
        raise ValueError("structural screen does not issue remote activation receipts")
    if root == repo or repo in root.parents or root.exists():
        raise ValueError("new external output identity required")
    root.mkdir(parents=True)
    torch.set_num_threads(config["torch_threads"])
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    started = time.monotonic()
    source_files = sorted(p for p in (repo / "sevc").rglob("*.py"))
    write_json(root / "provenance.json", {
        "config_path": str(config_path), "config_sha256": sha256_file(Path(config_path)),
        "config": config, "python": platform.python_version(), "torch": torch.__version__,
        "platform": platform.platform(), "device": "cpu", "scientifically_eligible": False,
        "source_sha256": {str(p.relative_to(repo)): sha256_file(p) for p in source_files},
        "command": f"conda run -n py311 python scripts/sevc.py run --config {config_path} --output-root {root}",
    })
    datasets = []
    try:
        for spec in config["datasets"]:
            context = ReplayDatasetContext(spec["name"], spec, Path(config["data_root"]), ReplayDevice("cpu"))
            folder = root / spec["name"]
            folder.mkdir()
            times, commits = [], []
            clock = WorkClock(lambda: None, times.append)
            indices = list(range(spec["sample_start"], spec["sample_start"] + 8))
            recipe = {"source_seed": spec["seed"], "sample_indices": indices,
                      "sample_labels": [int(context.train[i][1]) for i in indices]}
            source, _ = clock.call("fixture_source", "trainer", materialize_short_source, context, recipe)
            hashes, _ = clock.call("fixture_hash", "trainer", proof_component_hashes, source)
            good, good_seconds = clock.call("fixture_honest", "owner", verify_replay_proof,
                source, context.factory, device="cpu", tolerance=config["tolerance"])
            bad, mutation = mutate_replay_proof(source, source_id=identity([spec["name"], "trainer"]),
                post_commit_seed=spec["seed"], atom_key="cp4-positive", magnitude=4e-5,
                protocol_version=VERSION)
            bad = make_sgd_coherent_forgery(bad, mutation)
            bad_hashes = proof_component_hashes(bad)
            bad_result, _ = clock.call("fixture_forgery_replay", "owner", verify_replay_proof,
                bad, context.factory, device="cpu", tolerance=config["tolerance"])
            if not good["passed"] or bad_result["passed"]:
                raise ValueError("known valid/forged source failed the frozen replay check")
            model = context.factory().cpu()
            variants = []
            for variant in config["variants"]:
                opaque = variant != "raw-source-v2"
                verdicts = {b: [] for b in config["behaviors"]}
                executions = {b: [] for b in config["behaviors"]}
                tasks, answers, material = [], [], []
                challenge_results = []
                for rank in range(9):
                    role = "control" if rank < 4 else "challenge" if rank < 8 else "production"
                    proof = bad if role == "production" else source
                    components = bad_hashes if role == "production" else hashes
                    source_id = identity([spec["name"], rank, "source"])
                    raw = components["proof_sha256"]
                    public = opaque_source_handle(config["owner_secret_domain"], source_id, raw) if opaque else raw
                    atom = ATOM_KEYS[rank - 4] if role == "challenge" else None
                    bundle, _ = clock.call("fixture_compile", "owner", compile_canonical_replay_task,
                        proof, model, context.build_key, source_id=source_id, source_commitment=raw,
                        post_commit_seed=spec["seed"] + rank, role=role, atom_key=atom,
                        permutation_seed=spec["seed"] + 100 + rank, protocol_version=VERSION,
                        wrapper_seed_domain=WRAPPER_DOMAIN, tamper_delta=4e-5,
                        source_component_hashes=components, delivery_profile="compact", identity_profile="fused",
                        schema_validator_profile="shared", public_source_commitment=public,
                        mutation_profile=config.get("mutation_profile", "legacy-suffix"),
                        wrapper_profile=config.get("wrapper_profile", "permuted"),
                        gradient_scale=config.get("gradient_scale", .99),
                        validated_source_verdict=False if role == "production" else True)
                    # The same canonical candidate is replayed once across identical wrappers/terms.
                    if atom is not None and (not opaque or config.get("candidate_version", 1) >= 2):
                        outcome, _ = clock.call("fixture_challenge_replay", "owner", verify_replay_proof,
                            bundle.canonical_candidate, context.factory, device="cpu", tolerance=config["tolerance"])
                        challenge_results.append(outcome)
                        if outcome["passed"]:
                            raise ValueError("registered challenge did not fail full replay")
                    task = DeliveredReplay(bundle.sealed.task_id, public, bundle.wrapped_proof,
                                           bundle.descriptor, bundle.public_envelope)
                    tasks.append(task.task_id)
                    if role != "production":
                        answers.append((task.task_id, role == "control"))
                    material.append({"rank": rank, "task_id": task.task_id, "role": role,
                        "source_sha256": raw, "public_binding": public, "mutation": bundle.mutation,
                        "envelope": bundle.public_envelope, "descriptor": bundle.descriptor,
                        "component_hashes": bundle.component_hashes, "compile_seconds": bundle.compile_seconds})
                    for behavior in config["behaviors"]:
                        report, details = execute_assignment((task,), behavior=behavior,
                            trainer_hashes=frozenset(), context=context, seed=spec["seed"],
                            assignment_id=f"{variant}/{behavior}/{rank}", clock=clock,
                            performance={"identity_profile": "fused", "lazy_payload_read": True},
                            emit_commit=commits.append)
                        verdicts[behavior].append(report.verdicts[0])
                        executions[behavior].extend(details)
                    del task, bundle
                    gc.collect()
                assessments = []
                references = OwnerProbeReferences(tuple(answers))
                for behavior in config["behaviors"]:
                    report = CommittedVerifierReport.create(scenario_id=str(spec["seed"]),
                        verifier_id=behavior, job_id=variant, ordered_segment_ids=tasks,
                        verdicts=verdicts[behavior], nonce=identity([variant, behavior, spec["seed"]]))
                    # Freeze the complete report before consulting the private reference vector.
                    write_json(folder / f"{variant}-{behavior}-report.json", asdict(report))
                    details = executions[behavior]
                    cost = sum(d["seconds"] for d in details)
                    settlement = settle_service(report, references, cost_seconds=cost,
                        effort_fraction=sum(d["replayed"] for d in details) / len(details),
                        failure_threshold=1 if opaque else 2, fee=2.5 if opaque else 1.25, bond=.5)
                    mismatches = sum(report.verdicts[i] != answer for i, (_, answer) in enumerate(answers))
                    bypass = (mismatches == 0 and report.verdicts[-1] and not any(d["replayed"] for d in details))
                    assessments.append({"behavior": behavior, "probe_mismatches": mismatches,
                        "probe_count": 8, "invalid_production_accepted": report.verdicts[-1],
                        "replayed_tasks": sum(d["replayed"] for d in details), "task_count": 9,
                        "structural_bypass": bypass, "seconds": cost, "settlement": asdict(settlement),
                        "execution": details})
                variants.append({"variant": variant, "assessments": assessments,
                    "challenge_replays": challenge_results, "private_material": material})
            row = {"dataset": spec["name"], "recipe": recipe, "source_hashes": hashes,
                   "valid_replay": good, "single_honest_replay_seconds": good_seconds,
                   "forgery": mutation, "forgery_hashes": bad_hashes, "forgery_replay": bad_result,
                   "variants": variants, "independent_source_count": 1, "scientifically_eligible": False}
            write_json(folder / "results.json", row)
            write_json(folder / "timing.json", times)
            write_json(folder / "commits.json", commits)
            datasets.append(row)
            context.close_task_lanes()
            del context, source, bad, model
            gc.collect()
        failed = any(a["structural_bypass"] for d in datasets for v in d["variants"]
                     if v["variant"] != "raw-source-v2" for a in v["assessments"])
        result = {"verdict": "ACCEPTED_NEGATIVE" if failed else "PASS",
            "route": "PAPER_CHANGE_REQUIRED", "gate": "STOP_BEFORE_JOB_PILOT" if failed else "REVIEW_JOB_PILOT_PROTOCOL",
            "scope": "NONFORMAL_STRUCTURAL_FIXTURE_ONLY", "formal_repair_ready": False,
            "datasets": datasets, "elapsed_seconds": time.monotonic() - started}
        write_json(root / "results.json", result)
        return {k: v for k, v in result.items() if k != "datasets"}
    except Exception as exc:
        write_json(root / "technical-failure.json", {"verdict": "VOID_RERUN", "error": repr(exc),
            "completed_datasets": [d["dataset"] for d in datasets], "elapsed_seconds": time.monotonic() - started})
        raise


def run_job_pilot(repo_root, config_path, output_root, config, **runtime):
    """Bounded local assembly of the shared source bank and OnlineJob interfaces."""
    import json
    import resource
    import sys
    import torch
    from sevc.verification.on_demand_service import build_source_bank, OnlineJob
    if (config.get('device') != 'cpu' or config.get('scientifically_eligible') is not False
            or config.get('large_run_authorized') is not False
            or config.get('namespace') != 'local-job-fixture'
            or config.get('blocks_per_dataset') != 1
            or [d['name'] for d in config['datasets']] != ['cifar10', 'mnist', 'cifar100']
            or config['behaviors'] != ['honest', 'partial-50', 'partial-90']
            or config['methods'] != ['rcmp-opaque-gradient-continuation-v3', 'owner-direct-v2']):
        raise ValueError('unreviewed local job fixture')
    if any(config.get(k) is not None for k in ('_runtime_profile', '_runtime_max_units')):
        raise ValueError('local fixture cannot be truncated or reprofiled')
    if runtime.get('authorization_only') or runtime.get('authorization_receipt_path'):
        raise ValueError('local fixture cannot authorize formal activation')
    repo, root = Path(repo_root).resolve(), Path(output_root).resolve()
    if root.exists() or root == repo or repo in root.parents:
        raise ValueError('new external output identity required')
    root.mkdir(parents=True)
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    write_json(root / 'provenance.json', {
        'config': config, 'config_sha256': sha256_file(Path(config_path)),
        'python': platform.python_version(), 'torch': torch.__version__,
        'platform': platform.platform(), 'device': 'cpu', 'mps_available': torch.backends.mps.is_available(),
        'device_reason': 'author-authorized bounded CPU engineering pilot; no accelerator timing claim',
        'source_sha256': {str(p.relative_to(repo)): sha256_file(p) for p in sorted((repo/'sevc').rglob('*.py'))},
        'command': f'conda run -n py311 python scripts/sevc.py run --config {config_path} --output-root {root}'})
    started, rows = time.monotonic(), []
    try:
        for spec in config['datasets']:
            folder = root/spec['name']; folder.mkdir()
            scratch = folder/'sources'; scratch.mkdir()
            delivery = folder/'delivery'; delivery.mkdir()
            context = ReplayDatasetContext(spec['name'], spec, Path(config['data_root']), ReplayDevice('cpu'))
            phases, events, commits = [], [], []
            clock = WorkClock(lambda: None, phases.append)
            performance = {'identity_profile': 'fused', 'comparison_device': 'cpu', 'task_lanes': 1,
                'reuse_replay_model': True, 'tensor_access': 'authenticated-mmap',
                'identity_disk_delivery': True, 'lazy_payload_read': True, 'owner_compile_lanes': 1}
            bank_started = time.monotonic()
            bank, sources, private_truth = build_source_bank(context, seed=spec['seed'],
                namespace=config['namespace'], partition=spec['partition'], steps=4, batch_size=2,
                invalid_count=config['invalid_count'], clock=clock, scratch_dir=scratch,
                access_profile='authenticated-mmap')
            bank_seconds = time.monotonic()-bank_started
            write_json(folder/'source-identities.json', sources)
            began = time.monotonic()
            candidate = OnlineJob(bank=bank, context=context, clock=clock, performance=performance,
                job_id=f"{spec['name']}-v3", method_key=config['methods'][0],
                role_secret=identity([spec['seed'], 'local-role-secret']), emit=events.append,
                delivery_scratch_dir=delivery)
            preparation_seconds = time.monotonic()-began
            if not candidate.issued:
                raise ValueError('candidate did not issue a complete job')
            services = []
            for behavior in config['behaviors']:
                began = time.monotonic()
                report, settlement, details = candidate.serve(behavior=behavior, seed=spec['seed'],
                    assignment_id=behavior, audit_secret=identity([spec['seed'], 'audit']),
                    trainer_cache=private_truth, commits=commits.append)
                service_wall = time.monotonic()-began
                # Persist the committed report before evaluating any private labels.
                write_json(folder/f'{behavior}-report.json', asdict(report))
                truth = {r['task_id']: (r['probe_answer'] if r['role'] != 'production'
                    else private_truth[r['source_sha256']]) for r in candidate.task_rows}
                errors = sum(v != truth[k] for k, v in zip(report.ordered_segment_ids, report.verdicts))
                verifier_seconds = float(dict(settlement.diagnostics)['cost']) if 'cost' in dict(settlement.diagnostics) else None
                # The authoritative actor wall is recorded by the common WorkClock.
                actor = [p for p in phases if p.get('phase') == 'verifier-service-interface']
                if not actor:
                    raise ValueError('missing shared actor timing')
                verifier_seconds = actor[-1]['end'] - actor[-1]['start']
                economic = [{'fee': fee, 'cost_per_second': k,
                    'utility': (fee if settlement.status == 'PASS' else 0) - settlement.slashed_bond - k*verifier_seconds}
                    for fee in config['fee_grid'] for k in config['cost_per_second_grid']]
                services.append({'behavior': behavior, 'report_errors': errors,
                    'replayed_tasks': sum(d['replayed'] for d in details), 'settlement': asdict(settlement),
                    'verifier_seconds': verifier_seconds, 'service_wall_seconds': service_wall,
                    'owner_service_seconds': max(0, service_wall-verifier_seconds),
                    'economic_scenarios': economic, 'execution': details})
            began = time.monotonic()
            direct = OnlineJob(bank=bank, context=context, clock=clock, performance=performance,
                job_id=f"{spec['name']}-direct", method_key='owner-direct-v2',
                role_secret=identity([spec['seed'], 'local-role-secret']), emit=events.append,
                paired_production=candidate.production_source_ids)
            direct_answers = {sid: direct.references.acquire(sid)['passed'] for sid in direct.production_source_ids}
            direct_seconds = time.monotonic()-began
            honest = services[0]
            owner_seconds = preparation_seconds + honest['owner_service_seconds']
            valid = (honest['report_errors'] == 0 and honest['settlement']['status'] == 'PASS'
                and all(direct_answers[k] == private_truth[k] for k in direct_answers))
            row = {'dataset': spec['name'], 'seed': spec['seed'], 'independent_blocks': 1,
                'device': 'cpu', 'scientifically_eligible': False, 'valid': valid,
                'shared_trainer_source_bank_seconds': bank_seconds,
                'owner_preparation_seconds': preparation_seconds, 'owner_complete_seconds': owner_seconds,
                'owner_direct_seconds': direct_seconds, 'owner_ratio': owner_seconds/direct_seconds,
                'owner_cost_pass': owner_seconds < direct_seconds, 'services': services,
                'direct_answers': direct_answers,
                'process_peak_rss_bytes_upper_bound': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024),
                'material_bytes': sum(p.stat().st_size for p in folder.rglob('*.pt'))}
            write_json(folder/'results.json', row); write_json(folder/'timing.json', phases)
            write_json(folder/'events.json', events); write_json(folder/'commits.json', commits)
            rows.append(row)
            print(json.dumps({'dataset': spec['name'], 'valid': valid, 'owner_ratio': row['owner_ratio']}), flush=True)
            context.close_task_lanes()
            del bank, candidate, direct, context
            gc.collect()
        passed = all(r['valid'] and r['owner_cost_pass'] for r in rows)
        result = {'verdict': 'PASS' if passed else 'ACCEPTED_NEGATIVE',
            'scope': 'NONFORMAL_LOCAL_JOB_PILOT_ONLY', 'formal_repair_ready': False,
            'route': 'PAPER_CHANGE_REQUIRED', 'datasets': rows,
            'elapsed_seconds': time.monotonic()-started}
        write_json(root/'results.json', result)
        return {k:v for k,v in result.items() if k != 'datasets'}
    except Exception as exc:
        write_json(root/'technical-failure.json', {'verdict': 'VOID_RERUN', 'error': repr(exc),
            'completed_datasets': [r['dataset'] for r in rows], 'elapsed_seconds': time.monotonic()-started})
        raise
