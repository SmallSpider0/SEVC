"""Frozen action-filter evaluation and trace-only causal witness audit."""
from functools import lru_cache
import itertools
import json
import random
import time

from sevc.evaluation import recovery_enumeration as base
from sevc.committee.executable_recovery import MATCHING, SHIELD

POLICIES=(MATCHING,SHIELD,'execution-online-ablation-v2','execution-certified-ecs-v2')


def holdout_maps():
    excluded=set(base.sampled_maps(128,20260917))
    for graph in base.GRAPHS:
        conflicts=base.graph_conflicts(graph)
        excluded.add(sum((int(conflicts.get(v,('j-1',))[0][1:])+1)*3**i
                         for i,v in enumerate(base.IDENTITIES)))
    return random.Random(20260918).sample(sorted(set(range(3**9))-excluded),128)


def instances(n,seed):
    yield from base.instances(n,seed)
    for index in holdout_maps():
        for m,missing in enumerate(base.missing_sets((2,3))):
            yield {'id':f'D/{index}/{m}','family':'D','map_index':index,
                   'conflicts':base.decode_map(index),'missing':missing,'deadline':1e6,'owner_reserve':0.}


def differences(rows):
    for left,right in ((SHIELD,MATCHING),(SHIELD,'execution-online-ablation-v2'),
                       (SHIELD,'execution-certified-ecs-v2')):
        a,b=base.decision_signature(rows[left]['trace']),base.decision_signature(rows[right]['trace'])
        if a!=b:
            yield {'instance_id':rows[left]['instance']['id'],'left_policy':left,'right_policy':right,
                   'left':a,'right':b}


@lru_cache(maxsize=8)
def audit_roster(encoded):
    from sevc.committee.executed_recovery import build_execution_roster
    conflicts={v:tuple(f'j{j}' for j in range(2) if not encoded[i]&(1<<j))
               for i,v in enumerate(base.IDENTITIES)}
    offers,jobs,_,_=build_execution_roster(0,service_fee=2.5,public_conflicts=conflicts,robust_certification=False)
    return offers,jobs


def audit_trace(trace):
    """Independent future-value recurrence; reconstruct public proposal at each prefix."""
    from sevc.committee.executable_recovery import matching_proposal
    compat=tuple(sum(1<<j for j in range(2) if f'j{j}' not in
                     (trace['public_conflicts'] or {}).get(v,())) for v in base.IDENTITIES)
    offers,jobs=audit_roster(compat)
    shield=trace['policy']==SHIELD
    solve=base.hypothesis_auditor(compat)
    if shield:
        value,root=solve(511,(0,0),base.failure_hypotheses(511,2))
        cert=trace['certificate'];assert cert['worst_case_completed_jobs']==value
        budget=trace['budget_cap_per_job']-trace['owner_prepaid_cost_per_job']>=9*(trace['service_fee']+trace['owner_work_reserve_per_assignment'])
        assert cert['certificate_passed']==(value==2 and budget)
        assert cert['root_action']==(list(root) if root is not None else None)
    else:
        assert trace['certificate']=={'certificate_passed':None,'status':'NO_ROBUST_PLANNER'}
        assert trace['execution_certificate_calls']==0
    assert not trace['pre_defer']
    remaining=511;counts=[0,0];failures=0;usage={};prefix=[]
    events=trace['published_events'];steps=trace['execution_steps']
    assert len(events)<=len(steps)<=len(events)+1
    for i,step in enumerate(steps):
        read_time=max((e['published_at'] for e in prefix),default=0.)
        proposal=matching_proposal(offers,jobs,usage,prefix,read_time)
        chosen=proposal
        expected={'remaining_mask':remaining,'pass_counts':counts.copy(),'observed_failures':failures,
                  'proposal':list(proposal) if proposal is not None else None}
        if shield:
            b=max(0,2-failures);hypotheses=base.failure_hypotheses(remaining,b)
            value,fallback=solve(remaining,tuple(counts),hypotheses)
            worst=value
            if proposal is not None:
                j,v=proposal;rest=remaining^(1<<v);updated=counts.copy();updated[j]+=1
                good=tuple(h for h in hypotheses if not h&(1<<v))
                bad=tuple(sorted({h^(1<<v) for h in hypotheses if h&(1<<v)}))
                outcomes=[]
                if good:outcomes.append(solve(rest,tuple(updated),good)[0])
                if bad:outcomes.append(solve(rest,tuple(counts),bad)[0])
                worst=min(outcomes)
            if worst<value:chosen=fallback
            expected.update(guaranteed_count=value,proposal_worst_successor=worst,
                            intervention=chosen!=proposal,remaining_failure_allowance=b)
        else:expected.update(guaranteed_count=None,proposal_worst_successor=None,
                             intervention=False,remaining_failure_allowance=None)
        expected['action']=list(chosen) if chosen is not None else None
        assert step==expected,'action-filter witness drift'
        if i<len(events):
            e=events[i];assert chosen is not None;j,v=chosen
            assert e['job_id']==f'j{j}' and e['verifier_id']==f'v{v}','action drift'
            remaining^=1<<v;usage[f'v{v}']=1;prefix.append(e)
            if e['status']=='PASS':counts[j]=min(3,counts[j]+1)
            else:failures+=1
    return True


def benchmark(seed):
    selected=[i for i in base.instances(2,seed) if i['family']=='B' or i.get('graph')=='all-compatible']
    begun=time.process_time()
    for item in selected:
        for policy in POLICIES:
            row=base.execute_instance(item,policy)
            base.audit_trace(row['trace'],row['assignments'])
            if policy in (MATCHING,SHIELD):audit_trace(row['trace'])
            else:base.audit_executable_trace(row['trace'])
            json.dumps(row)
    elapsed=time.process_time()-begun
    return {'cpu_seconds':elapsed,'timed_instances':len(selected),'timed_traces':len(selected)*4,
            'n':128,'seed':seed,'projected_cpu_seconds':3*elapsed*31373/len(selected),
            'budget_seconds':7200,'outcome_metrics_inspected':False}
