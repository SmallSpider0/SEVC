"""CPU enumeration lifecycle behind the canonical registry runner."""
from __future__ import annotations

from collections import defaultdict
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time

from sevc.core.artifacts import sha256_file, write_json
from sevc.evaluation.recovery_enumeration import (
    POLICIES, add_metrics, audit_saved, differences, execute_instance, instances, timing_benchmark,
)
from sevc.evaluation.recovery_graph_audit import arbitrary_conflict_witness, audit_trace

CHANGE = 'experiment-tdsc-rq3-recovery-enumeration-v1'
FILTER_CHANGE = 'experiment-tdsc-rq3-action-certificate-v3'
REPAIR_CHANGE = 'experiment-tdsc-rq3-executable-recovery-v2'
VALUE_CHANGE = 'experiment-tdsc-rq3-value-preserving-recovery-v1'
CPU_BUDGET_SECONDS = 7200


def validate_activation(repo, config_path, root, config):
    """Standalone activation: a new output root and, for formal runs, a lock beside the config."""
    if (config['change_id'] not in (CHANGE, REPAIR_CHANGE, FILTER_CHANGE, VALUE_CHANGE)
            or any(k.startswith('_runtime') for k in config)):
        raise PermissionError('enumeration identity/runtime override rejected')
    if root.exists():
        raise FileExistsError('preserve existing evidence')
    change = Path(config_path).resolve().parent
    if config['phase'] == 'formal':
        lock = json.loads((change/'protocol-lock.json').read_text())
        if sha256_file(config_path) != lock['config_sha256'] or config != lock['configuration']:
            raise PermissionError('configuration drift')
        for row in lock['source_files']+lock['input_files']:
            if sha256_file(repo/row['path']) != row['sha256']:
                raise PermissionError('source/input drift: '+row['path'])
        if not lock['ready'] or config['n'] < 1:
            raise PermissionError('protocol not frozen')
    elif config['phase'] != 'benchmark':
        raise ValueError('unknown phase')
    return change


def run_recovery_enumeration(repo_root, config_path, output_root, config):
    repo_root, config_path, output_root = map(Path, (repo_root, config_path, output_root))
    change = validate_activation(repo_root, config_path, output_root, config)
    output_root.mkdir(parents=True, exist_ok=False)
    write_json(output_root/'provenance.json', {
        'change_id': config['change_id'], 'configuration': config, 'config_sha256': sha256_file(config_path),
        'command': sys.argv, 'python': sys.version, 'platform': platform.platform(),
        'machine': platform.machine(), 'processor': platform.processor(), 'pid': os.getpid(),
        'device': 'cpu', 'workers': 1, 'cuda_used': False, 'mps_used': False})
    if config['change_id'] == VALUE_CHANGE:
        return run_value_preserving(repo_root, change, output_root, config)
    repair = config['change_id'] in (REPAIR_CHANGE, FILTER_CHANGE)
    filtered = config['change_id'] == FILTER_CHANGE
    from sevc.evaluation import action_certificate as filt
    from sevc.evaluation.recovery_enumeration import (
        REPAIR_POLICIES, add_repair_metrics, repair_differences, global_maximum,
        audit_repair_saved, repair_timing_benchmark)
    policies = filt.POLICIES if filtered else REPAIR_POLICIES if repair else POLICIES
    instance_source = filt.instances if filtered else instances
    aggregate = add_repair_metrics if repair else add_metrics
    compare = filt.differences if filtered else repair_differences if repair else differences
    auditor = audit_repair_saved if repair else audit_saved
    if config['phase'] == 'benchmark':
        result = (filt.benchmark(config['sampling_seed']) if filtered else
                  repair_timing_benchmark(config['sampling_seed']) if repair else timing_benchmark(config['sampling_seed']))
        write_json(output_root/'benchmark.json', result)
        return result
    snapshot_locked_sources(repo_root, change, output_root)
    totals = defaultdict(lambda: defaultdict(int))
    begun, wall = time.process_time(), time.monotonic()
    count = 0
    try:
        with (output_root/'traces.jsonl').open('x') as stream, (output_root/'differences.jsonl').open('x') as ds:
            for item in instance_source(config['n'], config['sampling_seed']):
                oracle = ({'maximum_completed_jobs': global_maximum(item['conflicts'], item['missing']),
                           'scope': 'global pre-assignment responsive capacity; no time/budget'} if repair
                          else arbitrary_conflict_witness(item['conflicts'], item['missing']))
                group = {}
                for policy in policies:
                    if time.process_time()-begun + config['benchmark_cpu_seconds'] > 7200:
                        raise TimeoutError('registered CPU budget exhausted')
                    policy_started = time.process_time()
                    row = execute_instance(item, policy)
                    if filtered:
                        row['policy_cpu_seconds'] = time.process_time()-policy_started
                    row['oracle'] = oracle
                    audit_trace(row['trace'], row['assignments'])
                    stream.write(json.dumps(row, sort_keys=True)+'\n')
                    aggregate(totals, row, oracle['maximum_completed_jobs'])
                    group[policy] = row
                    count += 1
                for d in compare(group):
                    ds.write(json.dumps(d, sort_keys=True)+'\n')
                if count % 500 == 0:
                    stream.flush()
                    print(json.dumps({'traces_written': count, 'cpu_seconds': time.process_time()-begun}), flush=True)
        write_json(output_root/'summary.json', {k: dict(v) for k, v in totals.items()})
        audit = auditor(output_root, {**config, 'audit_remaining_cpu_seconds':
            7200-config['benchmark_cpu_seconds']-(time.process_time()-begun)})
        write_json(output_root/'independent-audit.json', audit)
        negative = any(v[key] for v in totals.values() for key in (
            'wrong_terminal_jobs', 'over_budget_jobs', 'certificate_feasible_but_uncompleted_traces'))
        if repair:
            candidate = [v for k,v in totals.items() if k.endswith('/execution-certified-ecs-v2')]
            negative = (not sum(v['in_scope_certified_traces'] for v in candidate) or
                        any(v[k] for v in candidate for k in ('wrong_terminal_jobs', 'over_budget_jobs',
                                                               'in_scope_false_certificates')))
        if filtered:
            candidate = [v for k,v in totals.items() if k.endswith('/action-certified-ecs-v3')]
            negative = (any(v[k] for v in candidate for k in ('wrong_terminal_jobs','over_budget_jobs','count_guarantee_violations'))
                or not sum(v['interventions'] for v in candidate)
                or totals['D/action-certified-ecs-v3']['completed_jobs'] <= totals['D/current-state-matching-v3']['completed_jobs'])
        result = {'change_id': config['change_id'], 'verdict': 'ACCEPTED_NEGATIVE' if negative else 'PASS',
                  'route': 'PAPER_CHANGE_REQUIRED', 'traces': count, 'audit': audit['status'],
                  'cpu_seconds': time.process_time()-begun, 'wall_seconds': time.monotonic()-wall,
                  'benchmark_cpu_seconds': config['benchmark_cpu_seconds'],
                  'dataset_independent_construction': True}
        if result['cpu_seconds']+config['benchmark_cpu_seconds'] > 7200:
            raise TimeoutError('producer plus audit CPU budget exhausted')
    except BaseException as exc:
        write_json(output_root/'failure.json', {'type': type(exc).__name__, 'message': str(exc),
                                               'traces_written': count})
        write_json(output_root/'terminal.json', {'verdict': 'VOID_RERUN', 'route': 'NO_PAPER_CHANGE'})
        raise
    write_json(output_root/'terminal.json', result)
    manifest = [{'path': str(p.relative_to(output_root)), 'sha256': sha256_file(p)}
                for p in sorted(output_root.rglob('*')) if p.is_file()]
    write_json(output_root/'manifest.json', {'files': manifest})
    return result


def snapshot_locked_sources(repo_root, change, output_root):
    lock = json.loads((change/'protocol-lock.json').read_text())
    write_json(output_root/'protocol-lock.json', lock)
    for row in lock['source_files']+lock['input_files']:
        dest = output_root/'snapshot'/row['path']
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repo_root/row['path'], dest)
        if sha256_file(dest) != row['sha256']:
            raise PermissionError('snapshot drift')
    return lock


def _finish(output_root, result):
    write_json(output_root/'terminal.json', result)
    manifest = [{'path': str(p.relative_to(output_root)), 'sha256': sha256_file(p)}
                for p in sorted(output_root.rglob('*')) if p.is_file()]
    write_json(output_root/'manifest.json', {'files': manifest})
    return result


def run_value_preserving(repo_root, change, output_root, config):
    """E1 executed hold-out, E2 scale simulator, E4 bound and E5 carry-over; then audit."""
    from sevc.evaluation import value_preserving_recovery as vp
    from sevc.evaluation.recovery_enumeration import execute_instance, global_maximum
    if config['phase'] == 'benchmark':
        return _finish(output_root, vp.benchmark(config))
    snapshot_locked_sources(repo_root, change, output_root)
    begun, wall = time.process_time(), time.monotonic()

    def guard():
        if time.process_time()-begun+config['benchmark_cpu_seconds'] > CPU_BUDGET_SECONDS:
            raise TimeoutError('registered CPU budget exhausted')
    count = 0
    try:
        with (output_root/'e1_traces.jsonl').open('x') as stream:
            for item in vp.e1_instances():
                oracle = global_maximum(item['conflicts'], item['missing'])
                for arm in vp.E1_ARMS:
                    guard()
                    row = execute_instance(item, arm)
                    row['oracle'] = {'maximum_completed_jobs': oracle}
                    audit_trace(row['trace'], row['assignments'])
                    stream.write(json.dumps(row, sort_keys=True)+'\n')
                    count += 1
        with (output_root/'e2_rows.jsonl').open('x') as stream:
            for cell in vp.cells():
                for index in range(config['graphs_per_cell']):
                    guard()
                    for row in vp.run_graph(cell, index, config['e2_seed'])[2]:
                        stream.write(json.dumps(row, sort_keys=True)+'\n')
                        count += 1
        write_json(output_root/'e5_carry_over.json', vp.e5_carry_over())
        summary = vp.summarize_saved(output_root)
        write_json(output_root/'summary.json', summary)
        audit = vp.audit_saved(output_root, config,
                               CPU_BUDGET_SECONDS-config['benchmark_cpu_seconds']-(time.process_time()-begun))
        write_json(output_root/'independent-audit.json', audit)
        result = {'change_id': config['change_id'], **vp.verdict(summary), 'audit': audit['status'],
                  'rows': count, 'cpu_seconds': time.process_time()-begun,
                  'wall_seconds': time.monotonic()-wall,
                  'benchmark_cpu_seconds': config['benchmark_cpu_seconds'],
                  'dataset_independent_construction': True}
        if result['cpu_seconds']+config['benchmark_cpu_seconds'] > CPU_BUDGET_SECONDS:
            raise TimeoutError('producer plus audit CPU budget exhausted')
    except BaseException as exc:
        write_json(output_root/'failure.json', {'type': type(exc).__name__, 'message': str(exc),
                                               'rows_written': count})
        write_json(output_root/'terminal.json', {'verdict': 'VOID_RERUN', 'route': 'NO_PAPER_CHANGE'})
        raise
    return _finish(output_root, result)
