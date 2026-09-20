"""Independent per-RQ audit of a returned fixed-design packet.

Read-only: no training, replay, tensor load or GPU initialization. The packet
integrity layer is shared by later RQ audits; RQ1 (detection) is implemented here.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path
import platform
import statistics

from sevc.core.artifacts import sha256_file, write_json
from sevc.core.statistical_bounds import clopper_pearson_bound
from sevc.incentives.verifier_protocol import CommittedVerifierReport
from sevc.verification.paid_replay_service import OwnerProbeReferences, settle_service

CHANGE = 'experiment-tdsc-f-rq1-detection-audit-v1'
PRODUCER = 'experiment-tdsc-f-fixed-design-v2'
RUN_ID = 'f-fixed-design-20260916-v2-run01'
SNAPSHOT = 'cf29ab50606f5966ac40f2e8d11bdb8baa147e62603edbfe2720d3f5bda9b44f'
RCMP = 'rcmp-opaque-gradient-continuation-gpu-v1'
GOLD = 'hidden-gold-gradient-continuation-gpu-v1'
DEPOL = 'depol-local-verification-arbitration-v1'
DEVIATIONS = ('constant-accept', 'constant-reject', 'sgd-consistency-shortcut',
              'prefix-one-step-shortcut', 'uniform-k32', 'uniform-k39')
UNIFORM_QUOTA = {'uniform-k32': 32, 'uniform-k39': 39}
SETTLEMENT_KEYS = ('status', 'accepted_report', 'service_fee', 'slashed_bond',
                   'refundable_bond', 'owner_expenditure')
DEPOL_GRANULARITY_BASIS = (
    'sevc/verification/depol_local.py compares only proof.checkpoints[-1] per 4-step interval, while the '
    'registered invalid source (index 24, atom cp3-positive) perturbs checkpoint index 2 of 4 '
    '(sevc/verification/on_demand_service.py build_source_bank; replay_coupled_probes.py _atom_parts and '
    'mutate_replay_proof). The final checkpoint is unchanged, so the native endpoint view cannot observe it.')


class AuditError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise AuditError(message)


# --------------------------------------------------------------------------- logs

def _open_log(root, name):
    compressed, plain = Path(root) / (name + '.gz'), Path(root) / name
    if compressed.is_file():
        return gzip.open(compressed, 'rb')
    require(plain.is_file(), 'missing log: ' + name)
    return plain.open('rb')


def iter_log(root, name):
    with _open_log(root, name) as stream:
        for index, line in enumerate(stream):
            require(line.endswith(b'\n'), 'partial JSONL record: ' + name)
            yield index, json.loads(line)


def verify_log_prefixes(root, requirements, result_index=None):
    """Check every sealed (bytes, sha256) prefix with one streaming pass per log."""
    checked = {}
    for name in sorted(requirements):
        by_length = defaultdict(set)
        for length, digest in requirements[name]:
            require(isinstance(length, int) and length >= 0, 'invalid prefix length: ' + name)
            by_length[length].add(digest)
        digest = hashlib.sha256()
        position = 0
        with _open_log(root, name) as stream:
            for length in sorted(by_length):
                while position < length:
                    chunk = stream.read(min(length - position, 1 << 20))
                    require(bool(chunk), f'truncated sealed prefix: {name}@{length}')
                    digest.update(chunk)
                    position += len(chunk)
                require(by_length[length] == {digest.copy().hexdigest()},
                        f'sealed prefix hash mismatch: {name}@{length}')
            while True:
                chunk = stream.read(1 << 20)
                if not chunk:
                    break
                digest.update(chunk)
                position += len(chunk)
        full = digest.hexdigest()
        if result_index is not None and name in result_index:
            require(result_index[name] == full, 'full log differs from result index: ' + name)
        checked[name] = {'prefixes_checked': len(by_length), 'bytes': position, 'sha256': full}
    return checked


# ---------------------------------------------------------------------- integrity

def verify_packet(root, config):
    """Verify unit identity, stage seals, sealed log prefixes and audit bindings."""
    root = Path(root)
    units = config['fixed_design_units']
    expected = {u['unit_id']: u for u in units}
    require(len(expected) == len(units), 'duplicate registered unit')
    provenance = json.loads((root / 'provenance.json').read_text())
    require(provenance.get('change_id') == PRODUCER, 'producer change drift')
    require(provenance.get('snapshot_id') == SNAPSHOT, 'snapshot drift')
    require(provenance['configuration']['fixed_design_units'] == units, 'registered matrix drift')
    result_index = json.loads((root / 'result_index.json').read_text())
    rows, unit_hashes = {}, {}
    for uid, planned in expected.items():
        path = root / 'units' / (uid + '.json')
        require(path.is_file(), 'missing unit: ' + uid)
        digest = sha256_file(path)
        require(result_index.get(f'units/{uid}.json') == digest, 'unit differs from result index: ' + uid)
        row = json.loads(path.read_text())
        require(all(row.get(k) == v for k, v in planned.items()), 'unit identity drift: ' + uid)
        require(row.get('status') in {'MEASURED', 'INFEASIBLE_VALID_NATIVE_RESULT'}, 'unhandled unit status: ' + uid)
        if row.get('assignments'):
            audit_path = root / 'reference-audits' / (uid + '.json')
            require(audit_path.is_file(), 'missing reference audit: ' + uid)
            require(json.loads(audit_path.read_text())['unit_sha256'] == digest, 'reference audit binding: ' + uid)
            require(result_index.get(f'reference-audits/{uid}.json') == sha256_file(audit_path),
                    'reference audit differs from result index: ' + uid)
        rows[uid] = row
        unit_hashes[uid] = digest
    requirements = defaultdict(set)
    sealed_files, sealed_units, seal_hashes = {}, set(), {}
    for phase in config['fixed_design']['phase_order']:
        per_dataset = {}
        for dataset in config['datasets']:
            path = root / 'stage-seals' / f'{phase}-{dataset}.json'
            require(path.is_file(), 'missing stage seal: ' + path.name)
            seal = json.loads(path.read_text())
            require(seal['phase'] == phase and seal['dataset'] == dataset, 'seal identity: ' + path.name)
            registered = [u['unit_id'] for u in units if u.get('phase') == phase and u['dataset'] == dataset]
            require(seal['expected_units'] == seal['completed_units'] == len(registered), 'seal unit count: ' + path.name)
            require(all(f'units/{uid}.json' in seal['files'] for uid in registered), 'seal omits a unit: ' + path.name)
            for rel, digest in seal['files'].items():
                if rel not in sealed_files:
                    target = root / rel
                    require(target.is_file(), 'sealed file missing: ' + rel)
                    sealed_files[rel] = sha256_file(target)
                require(sealed_files[rel] == digest, 'sealed file hash mismatch: ' + rel)
            for log, info in seal['shared_log_prefixes'].items():
                requirements[log].add((info['bytes'], info['sha256']))
            sealed_units.update(registered)
            per_dataset[path.name] = sha256_file(path)
            seal_hashes[path.name] = per_dataset[path.name]
        index_path = root / 'stage-seals' / f'{phase}-all-datasets.json'
        require(index_path.is_file(), 'missing phase index: ' + phase)
        require(json.loads(index_path.read_text())['seals'] == per_dataset, 'phase index hash mismatch: ' + phase)
    unsealed = sorted(set(expected) - sealed_units)
    require(all(expected[uid]['package'] == 'M8' for uid in unsealed), 'unregistered unsealed unit')
    prefixes = verify_log_prefixes(root, requirements, result_index)
    integrity = {
        'status': 'PACKET_INTEGRITY_PASS',
        'producer_change_run_snapshot': {'change_id': PRODUCER, 'run_id': RUN_ID, 'snapshot_id': SNAPSHOT},
        'registered_units': len(expected), 'present_units': len(rows),
        'status_counts': dict(Counter(r['status'] for r in rows.values())),
        'stage_seals_verified': len(seal_hashes), 'stage_seal_sha256': seal_hashes,
        'sealed_files_verified': len(sealed_files),
        'unsealed_units': {'count': len(unsealed), 'package': 'M8 finite native game (dataset-independent)'},
        'shared_log_prefix_bytes_and_hashes': {name: sorted([b, s] for b, s in values)
                                               for name, values in sorted(requirements.items())},
        'full_logs': prefixes,
    }
    return rows, unit_hashes, integrity


# --------------------------------------------------------------- service records

def lifecycle_reports(root):
    committed, revealed = {}, {}
    for position, event in iter_log(root, 'report-lifecycle.jsonl'):
        aid = event['assignment_id']
        if event['phase'] == 'COMMITTED':
            require(aid not in committed, 'duplicate commitment: ' + aid)
            committed[aid] = (position, event['commitment'])
        elif event['phase'] == 'REVEALED':
            require(aid in committed and aid not in revealed, 'reveal without prior commitment: ' + aid)
            require(committed[aid][1] == event['report']['commitment'], 'revealed report commitment drift: ' + aid)
            revealed[aid] = event['report']
        else:
            raise AuditError('unknown lifecycle phase: ' + str(event['phase']))
    return revealed


def assignment_audit_index(root):
    selected, publications, references = {}, [], {}
    for position, event in iter_log(root, 'assignment-audit.jsonl'):
        kind = event.get('event')
        if kind == 'audit-selected':
            require(event['assignment_id'] not in selected, 'duplicate audit selection')
            selected[event['assignment_id']] = position
        elif kind == 'reference-answers-published':
            publications.append((position, event))
        elif kind in {'reference-replay', 'reference-cache-hit'}:
            receipt = event['receipt']
            key = (event['job_id'], receipt['proof_sha256'])
            references.setdefault(key, []).append(bool(receipt['passed']))
    return selected, publications, references


def source_truth(root):
    known, mutated = set(), set()
    for _, row in iter_log(root, 'source-task-identities.jsonl'):
        for source in row.get('sources', ()):
            known.add(source['proof_sha256'])
            if source.get('trainer_mutation'):
                mutated.add(source['proof_sha256'])
    return known, mutated


def recompute_service(unit, assignment, audit_row, receipts, revealed, terms):
    """Recompute one settlement from the revealed report and the eight probe answers."""
    aid = assignment['assignment_id']
    report = CommittedVerifierReport(**assignment['report'])
    require(revealed.get(aid) == assignment['report'], 'lifecycle report mismatch: ' + aid)
    tasks = {t['task_id']: t for t in unit['tasks']}
    probes = tuple((k, t['probe_answer']) for k, t in tasks.items() if t['role'] != 'production')
    references = OwnerProbeReferences(probes)
    recorded = assignment['settlement']
    again = asdict(settle_service(report, references, cost_seconds=recorded['verifier_cost'],
                                  effort_fraction=recorded['effort_fraction'], fee=terms['fee'], bond=terms['bond'],
                                  failure_threshold=terms['failure_threshold'], require_complete_probes=True))
    for key in SETTLEMENT_KEYS:
        require(again[key] == recorded[key], f'settlement mismatch ({key}): {aid}')
    verdicts = dict(zip(report.ordered_segment_ids, report.verdicts))
    probe_rows = [{'task_id': k, 'role': tasks[k]['role'], 'answer': answer,
                   'mismatch': verdicts.get(k) != answer} for k, answer in probes]
    wrong = sorted(k for k, t in tasks.items() if t['role'] == 'production'
                   and verdicts.get(k) != receipts[t['source_sha256']]['passed'])
    require(wrong == sorted(audit_row['wrong_task_ids']), 'wrong-production audit mismatch: ' + aid)
    admitted = recorded['status'] == 'PASS' and recorded['accepted_report']
    require(admitted == audit_row['admitted'], 'admission audit mismatch: ' + aid)
    status = recorded['status']
    return {'unit_id': unit['unit_id'], 'assignment_id': aid, 'phase': unit['phase'], 'dataset': unit['dataset'],
            'context': unit['context'], 'method': unit['method'], 'behavior': assignment['behavior'],
            'block': unit['block'], 'invalid': unit['invalid'], 'steps': unit['steps'], 'status': status,
            'admitted': admitted, 'non_service': status == 'FAIL_CONFIRMED',
            'unresolved': status not in {'PASS', 'FAIL_CONFIRMED'},
            'mismatches': sum(p['mismatch'] for p in probe_rows), 'probe_mismatches': probe_rows,
            'production_verdicts': {k: verdicts.get(k) for k, t in tasks.items() if t['role'] == 'production'},
            'production_truth': {k: receipts[t['source_sha256']]['passed'] for k, t in tasks.items() if t['role'] == 'production'},
            'wrong_production': bool(wrong), 'wrong_admitted': admitted and bool(wrong),
            'verifier_seconds': recorded['verifier_cost'], 'service_fee': recorded['service_fee'],
            'slashed_bond': recorded['slashed_bond']}


def reference_findings(unit, receipts, references, mutated, known):
    """Deterministic reference consistency and probe-construction witnesses for one unit."""
    inconsistencies, problems = [], []
    for sid, receipt in receipts.items():
        if sid not in known:
            problems.append('unregistered production source ' + sid)
        elif receipt['passed'] != (sid not in mutated):
            inconsistencies.append(sid)
    job_id = unit['unit_id'] + '-job'
    for task in unit['tasks']:
        if task['role'] == 'production':
            continue
        seen = references.get((job_id, task['source_sha256']), [])
        if not seen or not all(seen):
            problems.append(f"{task['role']} source not validated by owner replay: {task['task_id']}")
        if task['role'] == 'challenge':
            witness = task.get('witness') or {}
            if witness.get('kind') != 'measured-canonical-replay-v3' or witness.get('result', {}).get('passed') is not False:
                problems.append('challenge lacks a failing full-replay witness: ' + task['task_id'])
            if task.get('probe_answer') is not False:
                problems.append('challenge answer is not reject: ' + task['task_id'])
        elif task.get('probe_answer') is not True:
            problems.append('control answer is not accept: ' + task['task_id'])
    return inconsistencies, problems


def disclosure_findings(service_rows, units, selected, publications):
    """Every assignment is published once, after its audit selection, with matching answers."""
    by_assignment = {}
    for position, event in publications:
        for aid, state in event['assignment_states'].items():
            require(aid not in by_assignment, 'assignment published in two epochs: ' + aid)
            by_assignment[aid] = (position, state, dict((k, v) for k, v in event['answers']))
    problems = []
    for row in service_rows:
        aid = row['assignment_id']
        if aid not in by_assignment:
            problems.append('never published: ' + aid)
            continue
        position, state, answers = by_assignment[aid]
        if state != row['status']:
            problems.append('published state differs from settlement: ' + aid)
        if aid not in selected or selected[aid] >= position:
            problems.append('publication not after audit selection: ' + aid)
        for task in units[row['unit_id']]['tasks']:
            if task['role'] != 'production' and answers.get(task['task_id']) != task['probe_answer']:
                problems.append('published answer differs from probe answer: ' + aid)
                break
    return {'assignments_checked': len(service_rows), 'epochs': len(publications), 'problems': problems}


# ----------------------------------------------------------------------- statistics

def interval(successes, trials):
    from scipy.stats import beta
    if trials == 0:
        return [None, None]
    low = 0.0 if successes == 0 else float(beta.ppf(.025, successes, trials - successes + 1))
    high = 1.0 if successes == trials else float(beta.ppf(.975, successes + 1, trials - successes))
    return [low, high]


def uniform_admission(quota, total=40, challenges=4):
    return math.comb(total - challenges, quota - challenges) / math.comb(total, quota)


def uniform_wrong_admission(quota, total=40, challenges=4, invalid=1):
    return math.comb(total - challenges - invalid, quota - challenges) / math.comb(total, quota)


def summarize_stratum(rows, expected_blocks=None):
    blocks = sorted({r['block'] for r in rows})
    if expected_blocks is not None:
        require(len(blocks) == expected_blocks == len(rows), 'stratum is not one service per independent block')
    n = len(rows)
    admitted = sum(r['admitted'] for r in rows)
    non_service = sum(r['non_service'] for r in rows)
    unresolved = sum(r['unresolved'] for r in rows)
    wrong = sum(r['wrong_production'] for r in rows)
    wrong_admitted = sum(r['wrong_admitted'] for r in rows)
    return {'services': n, 'independent_blocks': len(blocks), 'admitted': admitted,
            'admitted_rate_ci95': interval(admitted, n), 'non_service': non_service,
            'non_service_rate_ci95': interval(non_service, n), 'unresolved': unresolved,
            'wrong_production_reports': wrong, 'wrong_production_admitted': wrong_admitted,
            'wrong_production_admitted_ci95': interval(wrong_admitted, n),
            'mismatch_histogram': {str(k): v for k, v in sorted(Counter(r['mismatches'] for r in rows).items())},
            'mean_verifier_seconds': statistics.mean(r['verifier_seconds'] for r in rows)}


def analytic_check(observed, trials, probability):
    from scipy.stats import binomtest
    return {'observed': observed, 'trials': trials, 'observed_rate': observed / trials,
            'analytic_rate': probability,
            'two_sided_exact_p': float(binomtest(observed, trials, probability).pvalue),
            'scope': 'descriptive consistency with the uniform selection law; no multiplicity claim'}


def honest_risk_verdict(failed_blocks, blocks, reference_inconsistencies, family_size=3):
    nominal = clopper_pearson_bound(failed_blocks, blocks, side='upper', alpha=.05)
    family = clopper_pearson_bound(failed_blocks, blocks, side='upper', alpha=.05 / family_size)
    if failed_blocks or reference_inconsistencies:
        verdict = 'ACCEPTED_NEGATIVE'
    elif family > .05:
        verdict = 'INCONCLUSIVE_BY_DESIGN'
    else:
        verdict = 'PASS'
    return {'failed_blocks': failed_blocks, 'independent_blocks': blocks,
            'reference_inconsistencies': reference_inconsistencies,
            'one_sided_upper_alpha_05': nominal, 'one_sided_upper_family_alpha': family,
            'family_alpha': .05 / family_size, 'criterion': 'CP upper <= .05 and zero deterministic reference inconsistencies',
            'verdict': verdict}


# ----------------------------------------------------------------- exploratory

def threshold_rescoring(probe_rows, thresholds=(1, 2, 3), probe_counts=(2, 4, 6, 8)):
    """EA-1: admission under other failure thresholds and balanced probe subsets."""
    controls = [p['mismatch'] for p in probe_rows if p['role'] == 'control']
    challenges = [p['mismatch'] for p in probe_rows if p['role'] == 'challenge']
    require(len(controls) == len(challenges) == 4, 'service does not carry four controls and four challenges')
    total = sum(controls) + sum(challenges)
    result = {'r8': {f'h{h}': float(total < h) for h in thresholds}}
    for r in probe_counts:
        half = r // 2
        subsets = [(c, h) for c in itertools.combinations(range(4), half) for h in itertools.combinations(range(4), half)]
        admitted = sum(1 for c, h in subsets if sum(controls[i] for i in c) + sum(challenges[i] for i in h) < 1)
        result[f'r{r}_h1'] = {'subsets': len(subsets), 'admitted_fraction': admitted / len(subsets)}
    return result


def uniform_exact(task_rows, quota, threshold=1):
    """EA-2: exact admission and wrong-admission probability for uniform replay of `quota` tasks.

    Each task carries (replay_error, skip_error, is_probe): the outcome when replayed honestly
    and when skipped and answered accept. Skipped tasks are answered accept.
    """
    types = Counter((row['is_probe'], row['replay_error'], row['skip_error']) for row in task_rows)
    keys = sorted(types)
    total = sum(types.values())
    require(0 <= quota <= total, 'invalid uniform quota')
    denominator = math.comb(total, quota)
    admit = wrong_admit = 0.0
    for picks in itertools.product(*[range(types[k] + 1) for k in keys]):
        if sum(picks) != quota:
            continue
        weight = math.prod(math.comb(types[k], x) for k, x in zip(keys, picks))
        probe_errors = sum(x * k[1] + (types[k] - x) * k[2] for k, x in zip(keys, picks) if k[0])
        production_errors = sum(x * k[1] + (types[k] - x) * k[2] for k, x in zip(keys, picks) if not k[0])
        if probe_errors < threshold:
            admit += weight
            if production_errors:
                wrong_admit += weight
    return {'admission': admit / denominator, 'wrong_production_admission': wrong_admit / denominator}


def honest_task_rows(service):
    rows = [{'is_probe': True, 'replay_error': int(p['mismatch']), 'skip_error': int(p['answer'] is not True)}
            for p in service['probe_mismatches']]
    for task_id, verdict in service['production_verdicts'].items():
        truth = service['production_truth'][task_id]
        rows.append({'is_probe': False, 'replay_error': int(verdict != truth), 'skip_error': int(truth is not True)})
    return rows


# ----------------------------------------------------------------------- RQ1 audit

def audit_rq1(root, config, rows, unit_hashes):
    root = Path(root)
    units = {u['unit_id']: u for u in config['fixed_design_units']}
    terms = config['scoped_candidate']['service_terms']
    revealed = lifecycle_reports(root)
    selected, publications, references = assignment_audit_index(root)
    known, mutated = source_truth(root)

    def in_core(u):
        return u['phase'] == 'production' and u['package'] in {'M1', 'M2'} and u['method'] in {RCMP, GOLD}

    def in_supplement(u):
        return ((u['phase'] == 'target_bridge' and u['package'] == 'M1') or u['phase'] == 'calibration')

    service_rows, inconsistencies, problems = [], defaultdict(list), []
    for uid, u in units.items():
        if not (in_core(u) or in_supplement(u)):
            continue
        row = rows[uid]
        audit = json.loads((root / 'reference-audits' / (uid + '.json')).read_text())
        receipts = audit['production_receipts']
        audits = {a['assignment_id']: a for a in audit['assignments']}
        found, issues = reference_findings(row, receipts, references, mutated, known)
        inconsistencies[(u['dataset'], u['phase'])].extend(found)
        problems.extend(issues)
        for assignment in row['assignments']:
            service_rows.append(recompute_service(row, assignment, audits[assignment['assignment_id']],
                                                  receipts, revealed, terms))
    require(not problems, 'probe construction or reference problems: ' + '; '.join(problems[:5]))
    disclosure = disclosure_findings(service_rows, rows, selected, publications)
    require(not disclosure['problems'], 'disclosure problems: ' + '; '.join(disclosure['problems'][:5]))

    core = [s for s in service_rows if s['phase'] == 'production']
    require(len(core) == 1296, 'RQ1 core must contain 1296 production services')
    strata = defaultdict(list)
    for s in core:
        strata[(s['dataset'], s['context'], 'R' if s['method'] == RCMP else 'G', s['behavior'])].append(s)
    detection = []
    for (dataset, context, method, behavior), group in sorted(strata.items()):
        item = {'dataset': dataset, 'context': context, 'method': method, 'behavior': behavior,
                **summarize_stratum(group, expected_blocks=24)}
        if behavior in UNIFORM_QUOTA:
            k = UNIFORM_QUOTA[behavior]
            item['analytic_admission'] = analytic_check(item['admitted'], item['services'], uniform_admission(k))
            if context != 'init-valid':
                item['analytic_wrong_admission'] = analytic_check(item['wrong_production_admitted'], item['services'],
                                                                  uniform_wrong_admission(k))
        detection.append(item)

    events = []
    for index, dataset in enumerate(('cifar10', 'cifar100', 'mnist')):
        honest = [s for s in core if s['dataset'] == dataset and s['behavior'] == 'honest']
        blocks = defaultdict(list)
        for s in honest:
            blocks[s['block']].append(s)
        require(len(blocks) == 24 and all(len(v) == 6 for v in blocks.values()),
                'honest production must have six services in each of 24 blocks')
        failed = sum(1 for v in blocks.values() if any(not s['admitted'] for s in v))
        events.append({'dataset': dataset, **honest_risk_verdict(failed, len(blocks),
                       len(inconsistencies[(dataset, 'production')]))})
    pooled_blocks = sum(e['independent_blocks'] for e in events)
    pooled_failed = sum(e['failed_blocks'] for e in events)
    pooled = {'failed_blocks': pooled_failed, 'independent_blocks': pooled_blocks,
              'one_sided_upper_alpha_05': clopper_pearson_bound(pooled_failed, pooled_blocks, side='upper', alpha=.05),
              'scope': 'descriptive pooled bound across datasets; not a registered event'}

    supplement_groups = defaultdict(list)
    for s in service_rows:
        if s['phase'] != 'production':
            supplement_groups[(s['phase'], s['dataset'], 'R' if s['method'] == RCMP else 'G')].append(s)
    supplement = [{'phase': phase, 'dataset': dataset, 'method': method, **summarize_stratum(group)}
                  for (phase, dataset, method), group in sorted(supplement_groups.items())]

    exploratory_threshold = []
    grouped_ea1 = defaultdict(list)
    for s in core:
        grouped_ea1[(s['dataset'], s['context'], 'R' if s['method'] == RCMP else 'G', s['behavior'])].append(
            (threshold_rescoring(s['probe_mismatches']), s['wrong_production']))
    for key, values in sorted(grouped_ea1.items()):
        summary = {'dataset': key[0], 'context': key[1], 'method': key[2], 'behavior': key[3], 'services': len(values)}
        for h in (1, 2, 3):
            summary[f'r8_h{h}_admission'] = statistics.mean(v[0]['r8'][f'h{h}'] for v in values)
        for r in (2, 4, 6, 8):
            summary[f'r{r}_h1_admission'] = statistics.mean(v[0][f'r{r}_h1']['admitted_fraction'] for v in values)
            summary[f'r{r}_h1_wrong_admission'] = statistics.mean(
                v[0][f'r{r}_h1']['admitted_fraction'] * v[1] for v in values)
        exploratory_threshold.append(summary)

    parity = []
    for dataset in ('cifar10', 'cifar100', 'mnist'):
        for context in ('init-valid', 'init-invalid'):
            entry = {'dataset': dataset, 'context': context}
            for method, key in ((RCMP, 'R'), (GOLD, 'G')):
                honest = [s for s in core if (s['dataset'], s['context'], s['method'], s['behavior'])
                          == (dataset, context, method, 'honest')]
                for behavior, quota in UNIFORM_QUOTA.items():
                    exact = [uniform_exact(honest_task_rows(s), quota) for s in honest]
                    entry[f'{key}_derived_{behavior}_admission'] = statistics.mean(x['admission'] for x in exact)
                    entry[f'{key}_derived_{behavior}_wrong_admission'] = statistics.mean(
                        x['wrong_production_admission'] for x in exact)
                entry[f'{key}_honest_probe_mismatches'] = sum(s['mismatches'] for s in honest)
                entry[f'{key}_derived_constant_non_service'] = 1.0
            for behavior in UNIFORM_QUOTA:
                observed = [s for s in core if (s['dataset'], s['context'], s['method'], s['behavior'])
                            == (dataset, context, RCMP, behavior)]
                entry[f'R_observed_{behavior}_admission'] = sum(s['admitted'] for s in observed) / len(observed)
                entry[f'R_observed_{behavior}_wrong_admission'] = sum(s['wrong_admitted'] for s in observed) / len(observed)
            parity.append(entry)

    depol = []
    for uid, u in sorted(units.items()):
        if u['package'] != 'DP' or u['phase'] != 'native':
            continue
        native = rows[uid]['native']
        final = native['final']
        includes_invalid = any(sid in mutated for sid in rows[uid]['production_source_ids'])
        depol.append({'unit_id': uid, 'dataset': u['dataset'], 'context': u['context'], 'behavior': u['behavior'],
                      'deviating_verifier_eligible': final['verifier_reward_eligible'][0],
                      'honest_verifiers_eligible': final['verifier_reward_eligible'][1:],
                      'fast_trainer_verdict': native['fast']['trainer_verdict'],
                      'final_trainer_verdict': final['trainer_verdict'],
                      'slow_path_run': native.get('slow') is not None,
                      'targets_include_registered_invalid_source': includes_invalid,
                      'trainer_verdict_applicability': ('NOT_APPLICABLE_ENDPOINT_GRANULARITY'
                                                        if u['invalid'] else 'APPLICABLE_VALID_CONTEXT'),
                      'native_wall_seconds': rows[uid].get('native_wall_seconds')})
    require(len(depol) == 36, 'DePoL native scope must contain 36 units')
    return {'service_rows': service_rows, 'detection': detection, 'events': events, 'pooled_honest': pooled,
            'supplement': supplement, 'exploratory_threshold': exploratory_threshold, 'exploratory_parity': parity,
            'depol': depol, 'disclosure': disclosure,
            'reference_inconsistencies': {f'{k[0]}/{k[1]}': len(v) for k, v in sorted(inconsistencies.items())},
            'unit_hashes': {uid: unit_hashes[uid] for uid in sorted({s['unit_id'] for s in service_rows})}}


def claim_ledger(result):
    detection = result['detection']
    honest = [d for d in detection if d['behavior'] == 'honest']
    certain = [d for d in detection if d['behavior'] in {'constant-accept', 'constant-reject',
                                                         'sgd-consistency-shortcut', 'prefix-one-step-shortcut'}]
    uniform = [d for d in detection if d['behavior'] in UNIFORM_QUOTA]
    inconsistent = [f"{d['dataset']}/{d['context']}/{d['behavior']}/{name}" for d in uniform
                    for name in ('analytic_admission', 'analytic_wrong_admission')
                    if name in d and d[name]['two_sided_exact_p'] < .05]
    wrong_outside_uniform = sum(d['wrong_production_admitted'] for d in detection if d['behavior'] not in UNIFORM_QUOTA)
    depol = result['depol']
    deviating = [d for d in depol if d['behavior'] != 'honest']
    claims = [
        {'id': 'RQ1-C1', 'claim': 'Honest full replay was never judged non-service under RCMP or hidden-gold.',
         'status': 'SUPPORTED_DESCRIPTIVE' if all(d['non_service'] == d['unresolved'] == 0 for d in honest) else 'ACCEPTED_NEGATIVE',
         'evidence': {'services': sum(d['services'] for d in honest), 'non_service': sum(d['non_service'] for d in honest),
                      'events': result['events'], 'pooled': result['pooled_honest']},
         'wording_limit': 'State per-dataset and pooled upper bounds; zero observed failures is not zero risk; single A6000 platform.'},
        {'id': 'RQ1-C2', 'claim': 'Constant accept, constant reject, SGD-consistency and one-step-prefix shortcuts were judged non-service in every service.',
         'status': 'SUPPORTED_DESCRIPTIVE' if all(d['non_service'] == d['services'] for d in certain) else 'ACCEPTED_NEGATIVE',
         'evidence': {'services': sum(d['services'] for d in certain), 'non_service': sum(d['non_service'] for d in certain)},
         'wording_limit': 'Registered content-independent strategies only; initial anchor only.'},
        {'id': 'RQ1-C3', 'claim': 'Uniform partial replay was detected at rates consistent with the analytic selection law.',
         'status': 'SUPPORTED_DESCRIPTIVE' if not inconsistent else 'PARTLY_INCONSISTENT_WITH_ANALYTIC',
         'evidence': {'strata': len(uniform), 'inconsistent_strata_p_below_05': inconsistent},
         'wording_limit': 'Descriptive consistency check without multiplicity control; applies to uniform selection only.'},
        {'id': 'RQ1-C4', 'claim': 'Incorrect production reports were admitted only under uniform partial replay.',
         'status': 'SUPPORTED_DESCRIPTIVE' if wrong_outside_uniform == 0 else 'ACCEPTED_NEGATIVE',
         'evidence': {'wrong_admitted_uniform': sum(d['wrong_production_admitted'] for d in uniform),
                      'wrong_admitted_other': wrong_outside_uniform},
         'wording_limit': 'Single-service admissions; final settlement is RQ3. Report counts and analytic rates.'},
        {'id': 'RQ1-C5', 'claim': 'For constant and uniform strategies, hidden-gold has the same detection law as RCMP by construction.',
         'status': 'DERIVED_EXPLORATORY', 'evidence': {'exploratory': 'EA-2'},
         'wording_limit': 'Derived from recorded honest services; not a measured hidden-gold deviation; SE-DET supplies measurements.'},
        {'id': 'RQ1-C6', 'claim': 'DePoL withheld reward eligibility from the deviating verifier in every native deviation case.',
         'status': 'SUPPORTED_DESCRIPTIVE' if all(d['deviating_verifier_eligible'] is False for d in deviating) else 'ACCEPTED_NEGATIVE',
         'evidence': {'deviation_units': len(deviating),
                      'ineligible': sum(d['deviating_verifier_eligible'] is False for d in deviating)},
         'wording_limit': 'One block per dataset with two honest peers; trainer verdict under the registered invalidity is not applicable (endpoint granularity).'},
        {'id': 'RQ1-GAP', 'claim': 'Measured hidden-gold deviations, content-dependent selection and per-step DePoL.',
         'status': 'INCOMPLETE_PENDING_SE_DET', 'evidence': {'change': 'experiment-tdsc-rq1-detection-supplement-v1'},
         'wording_limit': 'Do not claim until SE-DET closes.'},
    ]
    allowlist = [c['id'] for c in claims if c['status'] in {'SUPPORTED_DESCRIPTIVE', 'DERIVED_EXPLORATORY', 'ACCEPTED_NEGATIVE'}]
    return {'rq': 'RQ1', 'claims': claims, 'paper_claim_allowlist': allowlist}


def figure_inputs(result):
    detection = result['detection']
    panel_a = [{'dataset': d['dataset'], 'context': d['context'], 'method': d['method'], 'behavior': d['behavior'],
                'non_service_rate': d['non_service'] / d['services'], 'ci95': d['non_service_rate_ci95'],
                'analytic_non_service': (1 - d['analytic_admission']['analytic_rate']) if 'analytic_admission' in d else None}
               for d in detection]
    panel_b = {'observed': [{'dataset': d['dataset'], 'k': UNIFORM_QUOTA[d['behavior']],
                             'wrong_admitted': d['wrong_production_admitted'], 'services': d['services'],
                             'ci95': d['wrong_production_admitted_ci95']}
                            for d in detection if d['behavior'] in UNIFORM_QUOTA and d['context'] == 'init-invalid'
                            and d['method'] == 'R'],
               'analytic': [{'k': k, 'wrong_admission': uniform_wrong_admission(k), 'admission': uniform_admission(k)}
                            for k in range(4, 41)]}
    return {'panel_a_detection_by_strategy': panel_a, 'panel_b_incorrect_admission_vs_k': panel_b,
            'table_depol_native': result['depol'], 'status': 'AUDITED_INPUTS_NOT_RENDERED'}


def write_rq1_packet(root, config_path, output, command=None):
    root, output, config_path = Path(root), Path(output), Path(config_path)
    config = json.loads(config_path.read_text())
    output.mkdir(parents=True, exist_ok=False)
    rows, unit_hashes, integrity = verify_packet(root, config)
    result = audit_rq1(root, config, rows, unit_hashes)
    code = [Path(__file__), Path(__file__).resolve().parents[2] / 'scripts' / 'audit_tdsc_f_rq1.py']
    source_index = {'audit_change': CHANGE, 'producer': integrity['producer_change_run_snapshot'],
                    'config_path': str(config_path), 'config_sha256': sha256_file(config_path),
                    'packet_root': str(root),
                    'packet_files': {name: sha256_file(root / name) for name in
                                     ('provenance.json', 'result_index.json', 'expanded-units.json')},
                    'audit_code': {str(p.name): sha256_file(p) for p in code if p.exists()},
                    'python': platform.python_version(), 'platform': platform.platform(),
                    'command': command, 'new_measurements': 0}
    science = {'rq': 'RQ1', 'scope_core_or_full': 'core (F v2 production) plus F v2 full-scope supplements',
               'three_dataset_completeness': sorted({d['dataset'] for d in result['detection']}),
               'independent_block_definition': 'production source block; each stratum has one service per block (24)',
               'registered_statistical_family_and_alpha': {
                   'honest-reference-risk/001-003': 'CP upper bound, family alpha .05/3; nominal .05 also reported'},
               'metric_units_and_denominators': {
                   'non_service_rate': 'FAIL_CONFIRMED services / services in stratum',
                   'wrong_production_admitted': 'admitted services with at least one wrong production verdict / services in stratum',
                   'honest_block_failure': 'blocks with any non-admitted honest RCMP or hidden-gold service / 24'},
               'paired_source_identity': 'RCMP and hidden-gold share the 32 production targets within a source group (verified by the runner and reference audits)',
               'actual_vs_assumed_qualification': 'not applicable to RQ1 detection; no qualification input used',
               'detection': result['detection'], 'registered_events': result['events'],
               'pooled_honest_descriptive': result['pooled_honest'], 'supplementary_services': result['supplement'],
               'depol_native': result['depol'], 'disclosure_audit': result['disclosure'],
               'reference_inconsistencies': result['reference_inconsistencies'],
               'depol_granularity_basis': DEPOL_GRANULARITY_BASIS,
               'negative_and_inconclusive_endpoints': [e for e in result['events'] if e['verdict'] != 'PASS'],
               'unresolved_items': ['measured hidden-gold deviations (SE-DET)', 'content-dependent selection (SE-DET)',
                                    'per-step DePoL native comparison (SE-DET)', 'cross-hardware replay tolerance (not measured)'],
               'independent_audit_findings': {'integrity': integrity['status'], 'settlements_recomputed': len(result['service_rows']),
                                              'probe_construction_and_reference_problems': 0,
                                              'disclosure_problems': len(result['disclosure']['problems'])}}
    ledger = claim_ledger(result)
    exploratory = {'label': 'post-hoc exploratory; does not alter registered verdicts',
                   'EA-1_threshold_and_probe_count': result['exploratory_threshold'],
                   'EA-2_hidden_gold_uniform_parity': result['exploratory_parity']}
    readiness = {'rq': 'RQ1', 'audit_complete': True, 'paper_ready_core_after_change_close': True,
                 'closed_experiment_change_id': None, 'requires_change_closure': CHANGE,
                 'full_rq1_pending': ['experiment-tdsc-rq1-detection-supplement-v1'], 'paper_ready_full': False}
    write_json(output / 'source-index.json', source_index)
    write_json(output / 'integrity-audit.json', integrity)
    write_json(output / 'science-audit.json', science)
    write_json(output / 'exploratory.json', exploratory)
    write_json(output / 'claim-ledger.json', ledger)
    write_json(output / 'figure-inputs.json', figure_inputs(result))
    write_json(output / 'paper-readiness.json', readiness)
    write_json(output / 'service-rows.json', result['service_rows'])
    return {'integrity': integrity['status'], 'services': len(result['service_rows']),
            'events': [(e['dataset'], e['verdict']) for e in result['events']],
            'claims': [(c['id'], c['status']) for c in ledger['claims']]}
