"""Exact causal recovery strategy for the registered nine-member, two-job domain."""
from functools import lru_cache

CERTIFIED = 'execution-certified-ecs-v2'
ABLATION = 'execution-online-ablation-v2'


@lru_cache(maxsize=8)
def planner(compatibility):
    """Cache public graph subproblems; neither actual failures nor verdicts are inputs."""
    if len(compatibility) != 9 or any(x not in (0, 1, 2, 3) for x in compatibility):
        raise ValueError('registered nine-identity compatibility required')
    from sevc.committee.recovery_game import exact_solver
    solve = exact_solver(tuple(compatibility), (3, 3))

    def legacy(remaining, c0, c1, failures):
        return solve(remaining, (c0, c1), failures)
    return legacy


def public_compatibility(offers):
    by_id = {o.verifier_id: o for o in offers}
    if set(by_id) != {f'v{i}' for i in range(9)}:
        raise ValueError('executable certificate is limited to nine explicit identities')
    if any(o.capacity != 1 or o.reputation != .9 for o in offers):
        raise ValueError('executable certificate requires capacity one and conditional p=.9')
    return tuple(sum(1 << j for j in (0, 1)
                     if o.accepts_offer and f'j{j}' not in o.conflict_job_ids)
                 for o in (by_id[f'v{i}'] for i in range(9)))


def execution_certificate(offers, budget_cap, assignment_reserve):
    compatibility = public_compatibility(offers)
    value, action = planner(compatibility)(511, 0, 0, 2)
    budget_sufficient = budget_cap >= 9*assignment_reserve
    return {'certificate_passed': value == 2 and budget_sufficient,
            'status': 'CERTIFIED' if value == 2 and budget_sufficient else 'INFEASIBLE',
            'scope': 'causal-policy completion under at most two failed assignments; adequate service/deadline required',
            'worst_case_completed_jobs': value, 'failure_allowance': 2,
            'compatibility': list(compatibility), 'budget_sufficient': budget_sufficient,
            'root_action': list(action) if action is not None else None,
            'static_hall_sufficiency_assumed': False}


def select_executable_action(offers, usage, published_events):
    """Only public settled statuses enter the state; one serial action at a time."""
    compatibility = public_compatibility(offers)
    counts = [0, 0]
    failures = 0
    for event in published_events:
        if event['status'] == 'PASS':
            counts[int(event['job_id'][1:])] += 1
        else:
            failures += 1
    remaining = sum(1 << i for i in range(9) if not usage.get(f'v{i}', 0))
    counts = [min(3, n) for n in counts]
    allowance = max(0, 2-failures)
    value, action = planner(compatibility)(remaining, *counts, allowance)
    evidence = {'remaining_mask': remaining, 'pass_counts': counts,
                'observed_failures': failures, 'remaining_failure_allowance': allowance,
                'worst_case_completed_jobs': value,
                'action': list(action) if action is not None else None}
    pairs = [] if action is None else [(f'j{action[0]}', f'v{action[1]}')]
    return pairs, evidence


MATCHING = 'current-state-matching-v3'
SHIELD = 'action-certified-ecs-v3'


def public_state(usage, events):
    counts = [min(3, sum(e['job_id']==f'j{j}' and e['status']=='PASS' for e in events))
              for j in range(2)]
    failures = sum(e['status']!='PASS' for e in events)
    remaining = sum(1<<i for i in range(9) if not usage.get(f'v{i}',0))
    return remaining, counts, failures


def matching_proposal(offers, jobs, usage, events, read_time):
    from sevc.incentives.verifier_reserve_recovery import select_global_recovery_matching
    current = {j.job_id:[e['verifier_id'] for e in events
                        if e['job_id']==j.job_id and e['status']=='PASS'] for j in jobs}
    active = tuple(j for j in jobs if len(current[j.job_id]) < 3)
    rows = select_global_recovery_matching(jobs=active, offers=offers,
        current_committees=current, used_capacity=usage, published_statuses=events,
        scheduler_read_time=read_time, minimum_pass_count=3)['assignments']
    return (int(rows[0][0][1:]), int(rows[0][1][1:])) if rows else None


def select_filtered_action(offers, jobs, usage, events, read_time, *, shield):
    proposal = matching_proposal(offers, jobs, usage, events, read_time)
    remaining, counts, failures = public_state(usage, events)
    step = {'remaining_mask':remaining, 'pass_counts':counts,
            'observed_failures':failures, 'proposal':list(proposal) if proposal else None}
    chosen = proposal
    if shield:
        solve = planner(public_compatibility(offers));b=max(0,2-failures)
        value, fallback = solve(remaining,*counts,b)
        successor = value
        if proposal is not None:
            j,i=proposal;updated=list(counts);updated[j]+=1;rest=remaining^(1<<i)
            successor=solve(rest,*updated,b)[0]
            if b:successor=min(successor,solve(rest,*counts,b-1)[0])
        if successor < value:chosen=fallback
        step.update(guaranteed_count=value, proposal_worst_successor=successor,
                    intervention=chosen!=proposal, remaining_failure_allowance=b)
    else:
        step.update(guaranteed_count=None, proposal_worst_successor=None,
                    intervention=False, remaining_failure_allowance=None)
    step['action']=list(chosen) if chosen is not None else None
    return ([] if chosen is None else [(f'j{chosen[0]}',f'v{chosen[1]}')]),step
