"""Public graph witnesses and trace-only counterfactuals (no GPU claims)."""
from __future__ import annotations

import itertools


def graph_conflicts(graph):
    if graph == "all-compatible":
        return {}
    if graph == "reserve-specialists-j0":
        return {"v7": ("j1",), "v8": ("j1",)}
    if graph == "reserve-specialists-j1":
        return {"v7": ("j0",), "v8": ("j0",)}
    if graph == "reserve-isolated-from-j1":
        return {f"v{i}": ("j1",) for i in (6, 7, 8)}
    raise ValueError("unregistered compatibility graph")


def exhaustive_witness(graph, missing):
    """Enumerate all residual reserve assignments after fixed primary service."""
    absent = {f"v{i}" if isinstance(i, int) else i for i in missing}
    conflicts = graph_conflicts(graph)
    primary = {f"j{j}": [f"v{i}" for i in range(j*3, j*3+3) if f"v{i}" not in absent] for j in (0,1)}
    reserve = [f"v{i}" for i in (6,7,8) if f"v{i}" not in absent]
    best = (-1, None)
    for destinations in itertools.product((None, 'j0', 'j1'), repeat=len(reserve)):
        if any(j in conflicts.get(v, ()) for v,j in zip(reserve,destinations) if j):
            continue
        assigned = {j:primary[j]+[v for v,d in zip(reserve,destinations) if d==j] for j in primary}
        complete = sum(len(v)>=3 for v in assigned.values())
        if complete > best[0]:
            best = (complete, assigned)
    return {"maximum_completed_jobs":best[0],"witness":best[1],
            "scope":"fixed-primary reports followed by residual reserve capacity"}


def replay_counterfactual(trace, policy):
    published = trace['published_events']
    if policy not in {'no-recovery','zero-replacement-budget','deadline-at-first-timeout'}:
        raise ValueError('unregistered trace policy')
    cutoff = min((r['published_at'] for r in published if r['status']=='DROPOUT'), default=float('inf'))
    rows = []
    for r in published:
        primary = int(r['verifier_id'][1:]) // 3 == int(r['job_id'][1:])
        if policy in {'no-recovery','zero-replacement-budget'} and not primary:
            continue
        if policy == 'deadline-at-first-timeout' and r['published_at'] > cutoff:
            continue
        rows.append(r)
    counts = {j:sum(r['job_id']==j and r['status']=='PASS' for r in rows) for j in ('j0','j1')}
    return {'policy':policy,'available_pass_reports':counts,
            'completion_from_observed_reports':{j:n>=3 for j,n in counts.items()},
            'new_callbacks':0,'counterfactual_wall_seconds':None,
            'counterfactual_resource_cost':None,'unidentified':['unexecuted callbacks and retimed scheduling'],
            'source_event_count':len(published),'retained_events':rows}


def audit_trace(trace, assignments=()):
    """Recalculate capacity, payments and decisions from published receipts."""
    from collections import Counter
    events=trace['published_events']; receipts=trace['receipts']
    assert len(events)==len(receipts), 'unreceipted recovery event'
    counts=Counter(e['verifier_id'] for e in events)
    assert all(n==1 for n in counts.values()) and dict(counts)==trace['usage'], 'capacity reuse'
    conflicts=trace.get('public_conflicts') or {}
    reports={(r['report']['job_id'],r['report']['verifier_id']):r for r in assignments}
    for event,receipt in zip(events,receipts):
        for key in ('job_id','verifier_id','status'):
            assert event[key]==receipt[key], 'event/receipt drift'
        assert event['job_id'] not in conflicts.get(event['verifier_id'],()), 'incompatible callback'
        if event['status']=='DROPOUT':
            assert event['verifier_id'] in trace['missing_identities'], 'invented missing identity'
            assert receipt['fee']==0 and receipt['slashed_bond']==.5, 'missing payment drift'
        elif assignments:
            row=reports[event['job_id'],event['verifier_id']]
            assert row['settlement']['status']==event['status'], 'recovery admission drift'
            for key in ('service_fee','slashed_bond','verifier_cost'):
                assert receipt['settlement'][key]==row['settlement'][key], 'recovery payment drift'
    for job,decision in trace['decisions'].items():
        reserved=trace['owner_prepaid_cost_per_job']+sum(e['job_id']==job for e in events)*(1.25+trace['owner_work_reserve_per_assignment'])
        assert abs(reserved-trace['reserved_expenditure'][job])<1e-7, 'reservation conservation'
        assert reserved<=trace['budget_cap_per_job']+1e-7, 'budget exceeded'
        if decision['route']=='safe-defer':
            assert decision['trainer_reward']==0 and decision['misconduct']==0, 'defer punishes trainer'
            continue
        committee=decision['committee']
        assert len(committee)>=3 and len(committee)==len(set(committee)), 'unbacked quorum'
        assert all(any(e['job_id']==job and e['verifier_id']==v and e['status']=='PASS' for e in events) for v in committee), 'unpaid committee report'
        if assignments:
            rs=[reports[job,v]['report'] for v in committee]
            # Task roles are evidence metadata used solely by this auditor.
            target_ids=trace['required_ids']
            maps=[dict(zip(r['ordered_segment_ids'],r['verdicts'])) for r in rs]
            accept=all(sum(m[t] is True for m in maps)>len(maps)/2 for t in target_ids)
            assert decision['route']==('accept' if accept else 'reject'), 'trainer vote decision drift'
        assert decision['trainer_reward']==float(decision['route']=='accept'), 'trainer payment drift'
        assert decision['misconduct']==int(decision['route']=='reject'), 'trainer punishment drift'
    return {'status':'AUDIT_PASS','callbacks':len(events),'completed_jobs':sum(d['route']!='safe-defer' for d in trace['decisions'].values())}
