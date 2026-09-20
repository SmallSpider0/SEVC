"""Finite recovery construction and independent, saved-evidence audit."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
import itertools
import json
import random
import time

from sevc.committee.executed_recovery import execute_recovery
from sevc.core.role_accounting import RoleClock
from sevc.evaluation.recovery_graph_audit import (
    arbitrary_conflict_witness, audit_trace, graph_conflicts,
)
from sevc.incentives.verifier_protocol import (
    CommittedVerifierReport, SealedEvaluationTruth, SealedSegmentTruth,
)
from sevc.verification.verifier_task_policies import settle_threshold_assignment

GRAPHS = ('all-compatible', 'reserve-specialists-j0', 'reserve-specialists-j1',
          'reserve-isolated-from-j1', 'three-only-j1')
POLICIES = ('all-response-certified-ecs', 'current-certified-ecs',
            'online-only-joint-matching-v1', 'fixed-order-same-reserve', 'no-recovery')
IDENTITIES = tuple(f'v{i}' for i in range(9))


def missing_sets(sizes):
    return [list(s) for k in sizes for s in itertools.combinations(IDENTITIES, k)]


def decode_map(index):
    if not 0 <= index < 3**9:
        raise ValueError('map outside registered population')
    result = {}
    for v in IDENTITIES:
        index, digit = divmod(index, 3)
        if digit:
            result[v] = [f'j{digit-1}']
    return result


def sampled_maps(n, seed):
    return random.Random(seed).sample(range(3**9), n)


def instances(n, seed):
    for graph in GRAPHS:
        for m, missing in enumerate(missing_sets(range(4))):
            yield {'id': f'A/{graph}/{m}', 'family': 'A', 'graph': graph,
                   'conflicts': graph_conflicts(graph), 'missing': missing,
                   'deadline': 1e6, 'owner_reserve': 0.}
    for index in sampled_maps(n, seed):
        for m, missing in enumerate(missing_sets((2, 3))):
            yield {'id': f'B/{index}/{m}', 'family': 'B', 'map_index': index,
                   'conflicts': decode_map(index), 'missing': missing,
                   'deadline': 1e6, 'owner_reserve': 0.}
    for name, missing, deadline in (
        ('replacement_failure', ['v0', 'v3', 'v6'], 1e6),
        ('deadline_exhaustion', ['v0', 'v3'], 1e-9),
        ('insufficient_response', ['v0', 'v1', 'v3', 'v4'], 1e6),
    ):
        yield {'id': f'C/{name}', 'family': 'C', 'conflicts': {},
               'missing': missing, 'deadline': deadline, 'owner_reserve': 20.}


def execute_instance(instance, policy):
    assignments = []
    def callback(job, verifier):
        report = CommittedVerifierReport.create(
            scenario_id='se-enum', verifier_id=verifier, job_id=job,
            ordered_segment_ids=('target',), verdicts=(True,), nonce=job+verifier)
        truth = SealedEvaluationTruth('se-enum', (SealedSegmentTruth('target', True, True),))
        settlement = settle_threshold_assignment(report, truth, failure_threshold=1,
                                                 fee=2.5, bond=.5, cost=0, effort=0)
        assignments.append({'report': asdict(report), 'settlement': asdict(settlement)})
        return report, settlement, []
    trace = execute_recovery(
        policy=policy, fault='structural', seed=0, required_ids=('target',),
        execute=callback, clock=RoleClock(lambda: None, lambda row: None),
        timeout_seconds=.001, deadline_seconds=instance['deadline'],
        virtual_fault_wait=True, missing_ids=instance['missing'], service_fee=2.5,
        service_bond=.5, public_conflicts=instance['conflicts'],
        owner_cost_reserve_per_assignment=instance['owner_reserve'], minimum_pass_count=3)
    return {'instance': instance, 'trace': trace, 'assignments': assignments}


def independent_maximum(conflicts, missing):
    """Independent quorum-subset oracle; does not call the producer oracle."""
    absent = set(missing)
    fixed = {j: set() for j in ('j0', 'j1')}
    consumed = set()
    for i, v in enumerate(IDENTITIES[:6]):
        j = f'j{i//3}'
        if j not in conflicts.get(v, ()):
            consumed.add(v)
            if v not in absent:
                fixed[j].add(v)
    available = set(IDENTITIES) - absent - consumed
    choices = {}
    for j in fixed:
        eligible = sorted(v for v in available if j not in conflicts.get(v, ()))
        need = max(0, 3-len(fixed[j]))
        choices[j] = [set(c) for c in itertools.combinations(eligible, need)]
    if any(a.isdisjoint(b) for a in choices['j0'] for b in choices['j1']):
        return 2
    return int(bool(choices['j0'] or choices['j1']))


def metrics(row, oracle):
    trace = row['trace']
    decisions = trace['decisions']
    complete = sum(d['route'] != 'safe-defer' for d in decisions.values())
    wrong = sum(d['route'] not in ('accept', 'safe-defer') or
                d['trainer_reward'] != float(d['route'] == 'accept') or d['misconduct'] != 0
                for d in decisions.values())
    certified = trace['certificate']['certificate_passed'] is True
    uncompleted = certified and complete < 2
    return {'traces': 1, 'completed_jobs': complete, 'oracle_completed_jobs': oracle,
            'oracle_gap_jobs': oracle-complete, 'safe_defer_jobs': 2-complete,
            'wrong_terminal_jobs': wrong,
            'certificate_feasible_but_uncompleted_traces': int(uncompleted),
            'certificate_uncompleted_absences_le2_nonconstraint': int(uncompleted and
                len(row['instance']['missing']) <= 2 and row['instance']['family'] != 'C'),
            'certificate_uncompleted_graph_guarantee': int(uncompleted and
                trace['policy'] == 'all-response-certified-ecs' and
                len(row['instance']['missing']) <= 2 and row['instance']['family'] != 'C'),
            'conservative_pre_defer_traces': int(trace['pre_defer'] and oracle > 0),
            'conservative_pre_defer_jobs': oracle if trace['pre_defer'] else 0,
            'over_budget_jobs': sum(x > trace['budget_cap_per_job']+1e-7
                                    for x in trace['reserved_expenditure'].values())}


def decision_signature(trace):
    return {j: d['route'] for j, d in trace['decisions'].items()}


def add_metrics(totals, row, oracle):
    key = row['instance']['family']+'/'+row['trace']['policy']
    for name, value in metrics(row, oracle).items():
        totals[key][name] += value


def differences(rows):
    online = decision_signature(rows['online-only-joint-matching-v1']['trace'])
    for policy in POLICIES[:2]:
        other = decision_signature(rows[policy]['trace'])
        if other != online:
            yield {'instance_id': rows[policy]['instance']['id'], 'policy': policy,
                   'ecs': other, 'online': online}


def timing_benchmark(seed):
    costs = []
    maps = [('named/'+g, graph_conflicts(g)) for g in GRAPHS]
    maps += [(str(i), decode_map(i)) for i in sampled_maps(2, seed)]
    begun = time.process_time()
    for name, conflicts in maps:
        for missing in ([], ['v0', 'v3'], ['v0', 'v3', 'v6']):
            for policy in POLICIES:
                item = {'id': name, 'family': 'benchmark', 'conflicts': conflicts,
                        'missing': missing, 'deadline': 1e6, 'owner_reserve': 0.}
                start = time.process_time()
                row = execute_instance(item, policy)
                row['oracle'] = arbitrary_conflict_witness(conflicts, missing)
                json.dumps(row, sort_keys=True)
                costs.append(time.process_time()-start)
    elapsed = time.process_time()-begun
    n = 128
    while n and 4*max(costs)*(3265+600*n)+elapsed > 7200:
        n //= 2
    return {'seed': seed, 'timed_traces': len(costs), 'cpu_seconds': elapsed,
            'maximum_cpu_seconds_per_trace': max(costs), 'n': n,
            'projected_cpu_seconds': 4*max(costs)*(3265+600*n)+elapsed,
            'budget_seconds': 7200, 'outcome_metrics_inspected': False}


def audit_saved(root, config):
    """Reload each row, rebuild every counter and pair comparison from evidence."""
    from pathlib import Path
    root = Path(root)
    totals = defaultdict(lambda: defaultdict(int))
    audit_cpu_started = time.process_time()
    count, diff_count = 0, 0
    with (root/'traces.jsonl').open() as stream, (root/'differences.jsonl').open() as diff_stream:
        for instance in instances(config['n'], config['sampling_seed']):
            group = {}
            oracle = independent_maximum(instance['conflicts'], instance['missing'])
            for policy in POLICIES:
                if time.process_time()-audit_cpu_started > config.get('audit_remaining_cpu_seconds', 7200):
                    raise TimeoutError('independent audit CPU budget exhausted')
                row = json.loads(next(stream))
                assert row['instance'] == json.loads(json.dumps(instance)), 'instance order/identity'
                t = row['trace']
                assert t['policy'] == policy, 'policy identity'
                assert t['missing_identities'] == instance['missing'], 'missing identity drift'
                assert t['public_conflicts'] == json.loads(json.dumps(instance['conflicts'])), 'graph drift'
                assert t['seed'] == 0 and t['service_fee'] == 2.5 and t['service_bond'] == .5
                assert t['deadline_seconds'] == instance['deadline']
                assert t['owner_work_reserve_per_assignment'] == instance['owner_reserve']
                assert row['oracle']['maximum_completed_jobs'] == oracle, 'oracle disagreement'
                witness = row['oracle']['witness']
                seen = set()
                for job, members in witness.items():
                    for member in members:
                        assert member not in seen and member not in instance['missing'], 'oracle capacity/absence'
                        assert job not in instance['conflicts'].get(member, ()), 'oracle conflict'
                        seen.add(member)
                assert sum(len(v) >= 3 for v in witness.values()) == oracle, 'oracle witness value'
                assert len(row['assignments']) == sum(e['status'] != 'DROPOUT' for e in t['published_events']), 'assignment completeness'
                for a in row['assignments']:
                    report = CommittedVerifierReport(**a['report'])
                    assert tuple(report.verdicts) == (True,) and tuple(report.ordered_segment_ids) == ('target',)
                    assert report.committed and report.revealed
                    s = a['settlement']
                    assert s['status'] == 'PASS' and s['service_fee'] == 2.5
                    assert s['refundable_bond'] == .5 and s['slashed_bond'] == 0
                a = audit_trace(t, row['assignments'])
                assert a['completed_jobs'] <= oracle, 'completion exceeds structural bound'
                for d in t['decisions'].values():
                    if d['route'] != 'safe-defer':
                        assert t['logical_elapsed_seconds'] <= t['deadline_seconds'], 'deadline violation'
                add_metrics(totals, row, oracle)
                count += 1
                group[policy] = row
            for diff in differences(group):
                assert json.loads(next(diff_stream)) == diff, 'difference list drift'
                diff_count += 1
        assert not stream.read().strip(), 'extra trace'
        assert not diff_stream.read().strip(), 'extra difference'
    totals = {k: dict(v) for k, v in totals.items()}
    assert totals == json.loads((root/'summary.json').read_text()), 'summary drift'
    return {'status': 'AUDIT_PASS', 'traces': count, 'instances': count//5,
            'difference_rows': diff_count, 'summary': totals,
            'oracle': 'independent disjoint quorum subset enumeration',
            'scheduler_rerun': False, 'external_reviewer': False,
            'independent_replication': False}


REPAIR_POLICIES = POLICIES + ('execution-certified-ecs-v2', 'execution-online-ablation-v2')


def global_maximum(conflicts, missing):
    """Two-job unit-capacity Hall oracle BEFORE any primary assignment."""
    available = set(IDENTITIES)-set(missing)
    a, b = [{v for v in available if j not in conflicts.get(v, ())} for j in ('j0', 'j1')]
    return 2 if len(a) >= 3 and len(b) >= 3 and len(a | b) >= 6 else int(max(len(a), len(b)) >= 3)


def independent_global_maximum(conflicts, missing):
    available = set(IDENTITIES)-set(missing)
    options = [[set(c) for c in itertools.combinations(
        sorted(v for v in available if j not in conflicts.get(v, ())), 3)] for j in ('j0', 'j1')]
    return 2 if any(a.isdisjoint(b) for a in options[0] for b in options[1]) else int(bool(options[0] or options[1]))


def repair_differences(rows):
    comparisons = [('execution-certified-ecs-v2', 'online-only-joint-matching-v1'),
                   ('execution-online-ablation-v2', 'online-only-joint-matching-v1'),
                   ('execution-certified-ecs-v2', 'execution-online-ablation-v2')]
    for left, right in comparisons:
        a, b = decision_signature(rows[left]['trace']), decision_signature(rows[right]['trace'])
        if a != b:
            yield {'instance_id': rows[left]['instance']['id'], 'left_policy': left,
                   'right_policy': right, 'left': a, 'right': b}


def repair_metrics(row, oracle):
    result = metrics(row, oracle)
    trace, item = row['trace'], row['instance']
    certified = trace['certificate']['certificate_passed'] is True
    candidate = trace['policy'] == 'execution-certified-ecs-v2'
    scoped = candidate and item['family'] != 'C' and len(item['missing']) <= 2
    result.update({'certified_traces': int(certified),
                   'in_scope_certified_traces': int(scoped and certified),
                   'in_scope_false_certificates': int(scoped and certified and result['completed_jobs'] < 2),
                   'assignment_attempts': len(trace['published_events']),
                   'reserved_expenditure': sum(trace['reserved_expenditure'].values())})
    if trace['policy'] in ('action-certified-ecs-v3','current-state-matching-v3'):
        root = trace['certificate'].get('worst_case_completed_jobs', 0)
        in_scope = item['family'] != 'C' and len(item['missing']) <= 2
        result.update(interventions=sum(s['intervention'] for s in trace['execution_steps']),
                      count_guarantee_violations=int(in_scope and result['completed_jobs'] < root))
    if 'policy_cpu_seconds' in row:
        result['policy_cpu_seconds'] = row['policy_cpu_seconds']
        result['policy_wall_seconds'] = trace['wall_seconds']
    return result


def add_repair_metrics(totals, row, oracle):
    key = row['instance']['family']+'/'+row['trace']['policy']
    for name, value in repair_metrics(row, oracle).items():
        totals[key][name] += value


from functools import lru_cache


@lru_cache(maxsize=8)
def hypothesis_auditor(compatibility):
    """Independent minimax verifier with explicit possible failure sets (not count DP)."""
    @lru_cache(maxsize=None)
    def solve(remaining, counts, hypotheses):
        done = sum(c >= 3 for c in counts)
        if done == 2 or remaining == 0:
            return done, None
        best = None
        for i in range(9):
            bit = 1 << i
            if not remaining & bit:
                continue
            for j in range(2):
                if counts[j] == 3 or not compatibility[i] & (1 << j):
                    continue
                rest = remaining ^ bit
                good = tuple(h for h in hypotheses if not h & bit)
                bad = tuple(sorted({h ^ bit for h in hypotheses if h & bit}))
                updated = tuple(c+int(k==j) for k,c in enumerate(counts))
                outcomes = []
                if good:
                    outcomes.append(solve(rest, updated, good)[0])
                if bad:
                    outcomes.append(solve(rest, counts, bad)[0])
                optimistic = solve(rest, updated, (0,))[0]
                key = (-min(outcomes), -optimistic, compatibility[i].bit_count(), j, i)
                if best is None or key < best[0]:
                    best = key, (j,i)
        return (done,None) if best is None else (-best[0][0],best[1])
    return solve


def failure_hypotheses(remaining, allowance):
    ids = [i for i in range(9) if remaining & (1<<i)]
    return tuple(sorted(sum(1<<i for i in c) for k in range(min(allowance,len(ids))+1)
                        for c in itertools.combinations(ids,k)))


def audit_executable_trace(trace):
    compat = tuple(sum(1<<j for j in range(2)
                       if f'j{j}' not in (trace['public_conflicts'] or {}).get(v,())) for v in IDENTITIES)
    solve = hypothesis_auditor(compat)
    value, action = solve(511, (0,0), failure_hypotheses(511,2))
    cert = trace['certificate']
    assert cert['worst_case_completed_jobs'] == value, 'causal certificate value'
    assert cert['root_action'] == (list(action) if action is not None else None), 'certificate root action'
    budget = trace['budget_cap_per_job']-trace['owner_prepaid_cost_per_job'] >= 9*(trace['service_fee']+trace['owner_work_reserve_per_assignment'])
    expected_cert = value == 2 and budget
    certified_arm = trace['policy'] == 'execution-certified-ecs-v2'
    assert cert['certificate_passed'] == (expected_cert if certified_arm else None)
    assert trace['pre_defer'] == (certified_arm and not expected_cert)
    remaining, counts, failures = 511, [0,0], 0
    events = trace['published_events']
    steps = trace['execution_steps']
    assert len(steps) >= len(events) and len(steps) <= len(events)+1, 'missing/excess causal actions'
    for index, step in enumerate(steps):
        b = max(0,2-failures)
        expected_value, chosen = solve(remaining, tuple(counts), failure_hypotheses(remaining,b))
        assert step == {'remaining_mask':remaining, 'pass_counts':counts,
                        'observed_failures':failures, 'remaining_failure_allowance':b,
                        'worst_case_completed_jobs':expected_value,
                        'action':list(chosen) if chosen is not None else None}, 'causal action witness mismatch'
        if index < len(events):
            event=events[index]
            assert chosen is not None
            j,i=chosen
            assert event['job_id']==f'j{j}' and event['verifier_id']==f'v{i}', 'future-information/action drift'
            remaining ^= 1<<i
            if event['status']=='PASS':counts[j]=min(3,counts[j]+1)
            else:failures+=1
    return True


def audit_repair_saved(root, config):
    """Audit seven-arm saved evidence, independently derive oracle/strategy and counters."""
    from pathlib import Path
    root=Path(root); totals=defaultdict(lambda:defaultdict(int)); count=diff_count=0
    action_filter = config.get('evaluation_variant') == 'action-filter-v3'
    from sevc.evaluation import action_certificate as filt
    policies = filt.POLICIES if action_filter else REPAIR_POLICIES
    instance_source = filt.instances if action_filter else instances
    compare = filt.differences if action_filter else repair_differences
    begun=time.process_time()
    predecessor = open(config['predecessor_traces']) if config.get('predecessor_traces') else None
    with (root/'traces.jsonl').open() as stream, (root/'differences.jsonl').open() as diffs:
        for item in instance_source(config['n'],config['sampling_seed']):
            oracle=independent_global_maximum(item['conflicts'],item['missing']); group={}
            for policy in policies:
                if time.process_time()-begun > config.get('audit_remaining_cpu_seconds',7200):
                    raise TimeoutError('repair audit CPU budget exhausted')
                row=json.loads(next(stream));t=row['trace']
                assert row['instance']==json.loads(json.dumps(item)) and t['policy']==policy
                assert t['public_conflicts']==json.loads(json.dumps(item['conflicts']))
                assert t['missing_identities']==item['missing']
                assert row['oracle']['maximum_completed_jobs']==oracle
                assert t['seed']==0 and t['service_fee']==2.5 and t['service_bond']==.5
                assert t['deadline_seconds']==item['deadline'] and t['owner_work_reserve_per_assignment']==item['owner_reserve']
                assert len(row['assignments'])==sum(e['status']!='DROPOUT' for e in t['published_events'])
                for a in row['assignments']:
                    report=CommittedVerifierReport(**a['report'])
                    assert tuple(report.verdicts)==(True,) and tuple(report.ordered_segment_ids)==('target',)
                    assert report.committed and report.revealed
                    s=a['settlement'];assert s['status']=='PASS' and s['service_fee']==2.5 and s['refundable_bond']==.5 and s['slashed_bond']==0
                audited=audit_trace(t,row['assignments']);assert audited['completed_jobs']<=oracle
                if policy in ('action-certified-ecs-v3','current-state-matching-v3'):filt.audit_trace(t)
                elif policy in REPAIR_POLICIES[-2:]:audit_executable_trace(t)
                elif predecessor is not None:
                    old=json.loads(next(predecessor));assert old['instance']==row['instance']
                    assert old['assignments']==row['assignments'], 'legacy assignment regression'
                    for field in ('policy','certificate','decisions','receipts','usage','reserved_expenditure','pre_defer'):
                        assert old['trace'][field]==t[field], 'legacy semantic regression: '+field
                # Independently derive core science counters; compare to shared producer metrics.
                complete=sum(d['route']=='accept' for d in t['decisions'].values())
                m=repair_metrics(row,oracle)
                assert m['completed_jobs']==complete and m['oracle_gap_jobs']==oracle-complete
                for name,value in m.items():totals[item['family']+'/'+policy][name]+=value
                group[policy]=row;count+=1
            for d in compare(group):
                assert json.loads(next(diffs))==d;diff_count+=1
        assert not stream.read().strip() and not diffs.read().strip()
    if predecessor is not None:
        assert not predecessor.read().strip(), 'extra legacy rows'
        predecessor.close()
    assert dict(totals)==json.loads((root/'summary.json').read_text()), 'summary drift'
    return {'status':'AUDIT_PASS','traces':count,'instances':count//len(policies),'difference_rows':diff_count,
            'independent_oracle':'disjoint quorum subsets','independent_strategy_audit':'explicit failure hypotheses',
            'external_reviewer':False,'independent_replication':False,'scheduler_rerun':False}


def repair_timing_benchmark(seed):
    """Outcome-blind whole-graph timing includes independent strategy precomputation."""
    selected=[]
    for item in instances(2,seed):
        if item['family']=='B' or item.get('graph')=='all-compatible':selected.append(item)
    started=time.process_time()
    for item in selected:
        for policy in REPAIR_POLICIES:
            row=execute_instance(item,policy)
            audit_trace(row['trace'],row['assignments'])
            if policy in REPAIR_POLICIES[-2:]:audit_executable_trace(row['trace'])
            json.dumps(row)
    elapsed=time.process_time()-started
    return {'cpu_seconds':elapsed,'timed_instances':len(selected),
            'timed_traces':len(selected)*7,'n':128,'seed':seed,
            'projected_cpu_seconds':elapsed*16013/len(selected)*3,
            'budget_seconds':7200,'outcome_metrics_inspected':False}
