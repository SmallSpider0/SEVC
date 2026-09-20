"""Value-preserving recovery study: scale generator, adversary matrix and metrics."""
from __future__ import annotations

import itertools
import json
import math
import random

from sevc.core.artifacts import sha256_text

from sevc.committee import recovery_game as rg

CHANGE = 'experiment-tdsc-rq3-value-preserving-recovery-v1'
E1_SEED, E2_DEV_SEED, E2_CONFIRM_SEED = 20260920, 20260921, 20260922
RHO = .95
JOBS = (4, 8, 16)
SLACKS = (1.1, 1.25, 1.5, 2.0)
CONFLICTS = ('affiliation', 'random-0.3', 'specialist-2')
PROFILES = ('homogeneous-0.9', 'two-class-0.95-0.85')
ALLOWANCES = (2, 4)
E2_ARMS = (rg.RC, rg.MATCHING_WAVE)


def cells():
    return list(itertools.product(JOBS, SLACKS, CONFLICTS, PROFILES, ALLOWANCES))


def scale_graph(cell, index: int, seed: int):
    jobs, slack, conflict, profile, k = cell
    rng = random.Random(f'{seed}/{cell}/{index}')
    p_values = (.9,) if profile == 'homogeneous-0.9' else (.95, .85)
    need = rg.required_passes(min(p_values), RHO)
    n = math.ceil(slack*jobs*need)
    reliability = tuple(rng.choice(p_values) for _ in range(n))
    compat = []
    for _ in range(n):
        if conflict == 'affiliation':
            barred = {rng.randrange(jobs)}
        elif conflict == 'specialist-2':
            barred = set(range(jobs))-set(rng.sample(range(jobs), 2))
        else:
            density = float(conflict.split('-')[1])
            barred = {j for j in range(jobs) if rng.random() < density}
        compat.append(sum(1 << j for j in range(jobs) if j not in barred))
    # Per-job requirement uses the least reliable compatible identity (paper rule).
    need_by_job = []
    for j in range(jobs):
        members = [reliability[i] for i in range(n) if compat[i] >> j & 1]
        need_by_job.append(rg.required_passes(min(members), RHO) if members else need)
    return rg.Game(tuple(compat), tuple(need_by_job), k), {
        'cell': list(cell), 'index': index, 'identities': n, 'reliability': list(reliability)}


def adversary_runs(game: rg.Game, rng: random.Random):
    """Registered adversary matrix: (label, in_scope, factory)."""
    n, k = len(game.compat), game.allowance
    runs = []
    for draw in range(5):
        failed = rg.uniform_failures(n, k, rng)
        runs.append((f'static-uniform/{draw}', True, lambda f=failed: rg.Adversary('static', k, f)))
    for kind in ('adaptive-scarcity', 'adaptive-first'):
        runs.append((kind, True, lambda kind=kind: rg.Adversary(kind, k)))
    for draw in range(5):
        failed = rg.uniform_failures(n, k+1, rng)
        runs.append((f'static-uniform-k+1/{draw}', False,
                     lambda f=failed: rg.Adversary('static', k+1, f)))
    for kind in ('adaptive-scarcity', 'adaptive-first'):
        runs.append((kind+'-k+1', False, lambda kind=kind: rg.Adversary(kind, k+1)))
    for q in (.05, .10):
        for draw in range(5):
            failed = frozenset(i for i in range(n) if rng.random() < q)
            runs.append((f'bernoulli-{q:.2f}/{draw}', None,
                         lambda f=failed: rg.Adversary('static', n, f)))
    return runs


def run_graph(cell, index, seed, *, oracle_cap=20000):
    game, meta = scale_graph(cell, index, seed)
    rng = random.Random(f'{seed}/{cell}/{index}/adversary')
    rows = []
    for label, in_scope, factory in adversary_runs(game, rng):
        for arm in E2_ARMS:
            adversary = factory()
            result = rg.simulate(game, arm, adversary)
            failed = result['failed_identities']
            rows.append({'graph': [*cell, index], 'adversary': label, 'arm': arm,
                         'in_scope': in_scope if in_scope is not None else len(failed) <= game.allowance,
                         'bernoulli': in_scope is None,
                         'completed_jobs': result['completed_jobs'], 'jobs': len(game.need),
                         'root_guaranteed_count': result['root_guaranteed_count'],
                         'rounds': result['rounds'], 'attempts': result['attempts'],
                         'failed_identities': failed, 'identities': len(game.compat),
                         'allowance': game.allowance,
                         'trace_sha256': sha256_text(json.dumps(result['trace'], sort_keys=True)),
                         'oracle': rg.offline_maximum(game, failed, oracle_cap)})
    return game, meta, rows


# ---------------------------------------------------------------- E1: held-out nine-identity maps

E1_ARMS = ('current-certified-ecs', 'current-state-matching-v3', 'value-preserving-exact',
           'value-preserving-rc', 'execution-certified-ecs-v2', 'action-certified-ecs-v3')


def e1_maps():
    from sevc.evaluation import recovery_enumeration as base
    from sevc.evaluation.action_certificate import holdout_maps
    excluded = set(base.sampled_maps(128, 20260917)) | set(holdout_maps())
    for graph in base.GRAPHS:
        conflicts = base.graph_conflicts(graph)
        excluded.add(sum((int(conflicts.get(v, ('j-1',))[0][1:])+1)*3**i
                         for i, v in enumerate(base.IDENTITIES)))
    return random.Random(E1_SEED).sample(sorted(set(range(3**9))-excluded), 128)


def e1_instances():
    from sevc.evaluation import recovery_enumeration as base
    for index in e1_maps():
        for m, missing in enumerate(base.missing_sets((2, 3))):
            yield {'id': f'E/{index}/{m}', 'family': 'E', 'map_index': index,
                   'conflicts': base.decode_map(index), 'missing': missing,
                   'deadline': 1e6, 'owner_reserve': 0.}


SELF_SCHEDULED = {'current-state-matching-v3', 'action-certified-ecs-v3', 'value-preserving-exact',
                  'value-preserving-rc', 'execution-certified-ecs-v2', 'execution-online-ablation-v2'}


def rounds_of(trace):
    """Rounds including the primary round; fixed-primary arms activate primaries before wave 0."""
    last = max((d['wave'] for d in trace['decisions'].values()), default=0)
    return last if trace['policy'] in SELF_SCHEDULED else last+1


def committed_value(trace):
    cert = trace['certificate']
    if trace['policy'] == 'value-preserving-rc':
        return cert['guaranteed_count']
    if trace['policy'] in ('value-preserving-exact', 'execution-certified-ecs-v2',
                           'action-certified-ecs-v3'):
        return cert['worst_case_completed_jobs']
    return None


def e1_metrics(row, oracle):
    t, item = row['trace'], row['instance']
    decisions = t['decisions'].values()
    completed = sum(d['route'] != 'safe-defer' for d in decisions)
    value = committed_value(t)
    in_scope = len(item['missing']) <= 2
    return {'completed_jobs': completed, 'oracle_completed_jobs': oracle,
            'oracle_gap_jobs': oracle-completed, 'safe_defer_jobs': 2-completed,
            'wrong_terminal_jobs': sum(d['route'] not in ('accept', 'safe-defer') or d['misconduct'] != 0
                                       or d['trainer_reward'] != float(d['route'] == 'accept')
                                       for d in decisions),
            'over_budget_jobs': sum(x > t['budget_cap_per_job']+1e-7
                                    for x in t['reserved_expenditure'].values()),
            'committed_value': value,
            'count_guarantee_violations': int(value is not None and in_scope and completed < value
                                              and t['policy'] in ('value-preserving-exact',
                                                                  'value-preserving-rc')),
            'pre_defer_jobs': 2 if t['pre_defer'] else 0,
            'rounds': rounds_of(t),
            'assignment_attempts': len(t['published_events']),
            'reserved_expenditure': sum(t['reserved_expenditure'].values())}


def sign_test(wins: int, losses: int) -> float:
    """One-sided exact sign test P(X >= wins | n = wins+losses, 1/2); ties excluded."""
    n = wins+losses
    if n == 0:
        return 1.
    return sum(math.comb(n, x) for x in range(wins, n+1))/2**n


def paired(per_unit, left, right):
    wins = sum(v[left] > v[right] for v in per_unit.values())
    losses = sum(v[left] < v[right] for v in per_unit.values())
    return {'left': left, 'right': right, 'units': len(per_unit), 'wins': wins, 'losses': losses,
            'ties': len(per_unit)-wins-losses, 'p_one_sided': sign_test(wins, losses),
            'left_total': sum(v[left] for v in per_unit.values()),
            'right_total': sum(v[right] for v in per_unit.values())}


# ---------------------------------------------------------------- E2 summaries and E4 bound

def binomial_tail(n: int, q: float, k: int) -> float:
    """Pr(Binomial(n, q) > k)."""
    return max(0., 1.-sum(math.comb(n, x)*q**x*(1-q)**(n-x) for x in range(k+1)))


def _binomial_log_pmf(n: int, i: int, p: float) -> float:
    return (math.lgamma(n+1)-math.lgamma(i+1)-math.lgamma(n-i+1)
            + i*math.log(p) + (n-i)*math.log1p(-p))


def clopper_pearson(x: int, n: int, alpha: float = .05):
    """Exact two-sided interval by bisection on binomial tails, in log space (large n safe)."""
    def tail(p, lower):
        terms = range(x, n+1) if lower else range(0, x+1)
        return sum(math.exp(_binomial_log_pmf(n, i, p)) for i in terms)

    def bound(lower):
        lo, hi = 0., 1.
        for _ in range(60):
            mid = (lo+hi)/2
            value = tail(mid, lower)
            if lower:
                lo, hi = (mid, hi) if value < alpha/2 else (lo, mid)
            else:
                lo, hi = (mid, hi) if value > alpha/2 else (lo, mid)
        return (lo+hi)/2
    return (0. if x == 0 else bound(True)), (1. if x == n else bound(False))


def adversary_kind(label: str) -> str:
    return label.split('/')[0]


def summarize_e2(rows):
    from collections import defaultdict
    pairs = defaultdict(dict)
    for r in rows:
        pairs[(tuple(r['graph']), r['adversary'])][r['arm']] = r
    kinds = defaultdict(lambda: defaultdict(int))
    per_graph = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    violations = defaultdict(int)
    bound = defaultdict(lambda: defaultdict(float))
    for (graph, label), arms in sorted(pairs.items()):
        rc, mw = arms[rg.RC], arms[rg.MATCHING_WAVE]
        kind = adversary_kind(label)
        k = kinds[kind]
        k['runs'] += 1
        k['jobs'] += rc['jobs']
        k['guaranteed'] += rc['root_guaranteed_count']
        for name, r in (('rc', rc), ('mw', mw)):
            k[name+'_completed'] += r['completed_jobs']
            k[name+'_rounds'] += r['rounds']
            k[name+'_attempts'] += r['attempts']
            if r['oracle'] is None:
                k[name+'_oracle_unknown'] += 1
            else:
                k[name+'_oracle_gap'] += r['oracle']-r['completed_jobs']
        k['rc_wins'] += rc['completed_jobs'] > mw['completed_jobs']
        k['rc_losses'] += rc['completed_jobs'] < mw['completed_jobs']
        k['mw_below_rc_value_in_scope'] += int(mw['in_scope'] and mw['completed_jobs'] < rc['root_guaranteed_count'])
        if rc['in_scope'] and rc['completed_jobs'] < rc['root_guaranteed_count']:
            violations[kind] += 1
        per_graph[kind][graph][rg.RC] += rc['completed_jobs']
        per_graph[kind][graph][rg.MATCHING_WAVE] += mw['completed_jobs']
        if rc['bernoulli']:
            q = label.split('/')[0].split('-')[1]
            b = bound[q]
            b['runs'] += 1
            b['below_value'] += int(rc['completed_jobs'] < rc['root_guaranteed_count'])
            b['exceeded_allowance'] += int(len(rc['failed_identities']) > rc['allowance'])
            b['bound_sum'] += binomial_tail(rc['identities'], float(q), rc['allowance'])
    e4 = {}
    for q, b in sorted(bound.items()):
        runs, x = int(b['runs']), int(b['below_value'])
        low, high = clopper_pearson(x, runs)
        mean_bound = b['bound_sum']/runs
        e4[q] = {'runs': runs, 'observed_below_value': x, 'observed_rate': x/runs,
                 'clopper_pearson_95': [low, high],
                 'observed_exceeded_allowance': int(b['exceeded_allowance']),
                 'mean_binomial_bound': mean_bound, 'defect': low > mean_bound}
    tests = {kind: paired(graphs, rg.RC, rg.MATCHING_WAVE) for kind, graphs in per_graph.items()}
    return ({kind: dict(v) for kind, v in kinds.items()}, dict(violations), tests, e4)


def summarize_e2_by(rows, field_index, kind='adaptive-scarcity'):
    """Descriptive breakdown of one adversary kind by one generator dimension."""
    from collections import defaultdict
    out = defaultdict(lambda: defaultdict(int))
    for r in rows:
        if adversary_kind(r['adversary']) != kind:
            continue
        o = out[str(r['graph'][field_index])]
        o['jobs'] += r['jobs'] if r['arm'] == rg.RC else 0
        o[('rc' if r['arm'] == rg.RC else 'mw')+'_completed'] += r['completed_jobs']
        if r['arm'] == rg.RC:
            o['guaranteed'] += r['root_guaranteed_count']
    return {k: dict(v) for k, v in out.items()}


# ---------------------------------------------------------------- E5: F v2 fault scenarios

E5_FAULTS = {'no-missing': [], 'one-missing': ['v0'], 'correlated-missing': ['v0', 'v1', 'v3'],
             'insufficient-reserve': ['v0', 'v1', 'v3', 'v4']}
E5_ARMS = ('current-certified-ecs', 'value-preserving-exact', 'value-preserving-rc')


def e5_carry_over():
    from sevc.evaluation import recovery_enumeration as base
    rows = {}
    for fault, missing in E5_FAULTS.items():
        item = {'id': f'F/{fault}', 'family': 'F', 'conflicts': {}, 'missing': missing,
                'deadline': 1e6, 'owner_reserve': 0.}
        for arm in E5_ARMS:
            t = base.execute_instance(item, arm)['trace']
            rows[f'{fault}/{arm}'] = {
                'terminals': {j: d['route'] for j, d in sorted(t['decisions'].items())},
                'rounds': rounds_of(t),
                'attempts': len(t['published_events']),
                'reserved_expenditure': sum(t['reserved_expenditure'].values())}
    same = {fault: all(rows[f'{fault}/{a}']['terminals'] == rows[f'{fault}/current-certified-ecs']['terminals']
                       for a in E5_ARMS) for fault in E5_FAULTS}
    return {'rows': rows, 'identical_terminals': same,
            'carry_over': 'F v2 executed terminals carry over' if all(same.values())
            else 'executed arm needs rerun for differing faults'}


# ---------------------------------------------------------------- independent audit helpers

def independent_full_jobs(compat, need, failed, cap_checks=20000):
    """Separate Edmonds-Karp implementation of the E2 offline maximum."""
    from collections import deque
    alive = [i for i in range(len(compat)) if i not in set(failed)]

    def feasible(subset):
        total = sum(need[j] for j in subset)
        # nodes: 0 source, 1..J jobs, then identities, last sink
        jobs = list(subset)
        jid = {j: 1+n for n, j in enumerate(jobs)}
        iid = {i: 1+len(jobs)+n for n, i in enumerate(alive)}
        sink = 1+len(jobs)+len(alive)
        cap = {}
        adj = {x: [] for x in range(sink+1)}
        def arc(u, v, c):
            cap[u, v] = cap.get((u, v), 0)+c
            cap.setdefault((v, u), 0)
            adj[u].append(v)
            adj[v].append(u)
        for j in jobs:
            arc(0, jid[j], need[j])
            for i in alive:
                if compat[i] >> j & 1:
                    arc(jid[j], iid[i], 1)
        for i in alive:
            arc(iid[i], sink, 1)
        flow = 0
        while True:
            parent = {0: None}
            queue = deque([0])
            while queue and sink not in parent:
                u = queue.popleft()
                for v in adj[u]:
                    if v not in parent and cap[u, v] > 0:
                        parent[v] = u
                        queue.append(v)
            if sink not in parent:
                return flow == total
            v = sink
            while parent[v] is not None:
                u = parent[v]
                cap[u, v] -= 1
                cap[v, u] += 1
                v = u
            flow += 1

    viable = [j for j in range(len(need))
              if sum(compat[i] >> j & 1 for i in alive) >= need[j]]
    checks = 0
    for size in range(len(viable), 0, -1):
        for subset in itertools.combinations(viable, size):
            checks += 1
            if checks > cap_checks:
                return None
            if feasible(subset):
                return size
    return 0


def audit_rc_steps(compat, need, allowance, steps, actions_by_round, outcomes_by_round):
    """Independently re-derive the reserve invariant and count monotonicity along a trace."""
    unused = set(range(len(compat)))
    passes = [0]*len(need)
    failures, previous = 0, None
    for step, actions, outcomes in zip(steps, actions_by_round, outcomes_by_round):
        b = max(0, allowance-failures)
        assert step['remaining_failure_allowance'] == b, 'allowance drift'
        guaranteed = step['guaranteed_jobs']
        completed = sum(p >= q for p, q in zip(passes, need))
        assert step['guaranteed_count'] == completed+len(guaranteed), 'count drift'
        within = failures <= allowance
        if within and previous is not None:
            assert step['guaranteed_count'] >= previous, 'guaranteed count decreased'
        previous = step['guaranteed_count']
        used = {i for _, i in actions}
        assert used <= unused and len(used) == len(actions), 'reuse of identity'
        for j, i in actions:
            assert compat[i] >> j & 1, 'conflicting assignment'
        for j in guaranteed:
            mine = sum(1 for a, _ in actions if a == j)
            assert mine == need[j]-passes[j], 'guaranteed job not fully assigned'
            if within:
                cover = sum(1 for i in unused-used if compat[i] >> j & 1)
                assert cover >= b, 'reserve invariant violated'
        for (j, i), ok in zip(actions, outcomes):
            unused.discard(i)
            if ok:
                passes[j] = min(need[j], passes[j]+1)
            else:
                failures += 1
    return True


# ---------------------------------------------------------------- saved-evidence summary and verdict

ALPHA = .05/3
DIMENSIONS = ('jobs', 'slack', 'conflict', 'profile', 'allowance')


def _read_jsonl(path):
    with open(path) as stream:
        for line in stream:
            yield json.loads(line)


def summarize_e1(rows):
    from collections import defaultdict
    totals = defaultdict(lambda: defaultdict(int))
    per_map = defaultdict(lambda: defaultdict(int))
    values = defaultdict(dict)
    for row in rows:
        t, item = row['trace'], row['instance']
        m = e1_metrics(row, row['oracle']['maximum_completed_jobs'])
        arm = t['policy']
        for name, value in m.items():
            if name != 'committed_value':
                totals[arm][name] += value
        totals[arm]['traces'] += 1
        per_map[item['map_index']][arm] += m['completed_jobs']
        if m['committed_value'] is not None:
            values[item['map_index']].setdefault(arm, m['committed_value'])
    tests = {name: paired(per_map, left, right) for name, (left, right) in {
        'E1-P2': ('value-preserving-exact', 'current-certified-ecs'),
        'E1-P4': ('value-preserving-rc', 'current-certified-ecs'),
        'rc-vs-serial-matching': ('value-preserving-rc', 'current-state-matching-v3'),
        'rc-vs-exact': ('value-preserving-rc', 'value-preserving-exact'),
        'shield-vs-current': ('action-certified-ecs-v3', 'current-certified-ecs')}.items()}
    rc_above_exact = sorted(k for k, v in values.items()
                            if v['value-preserving-rc'] > v['value-preserving-exact'])
    value_totals = {arm: sum(v[arm] for v in values.values())
                    for arm in ('value-preserving-exact', 'value-preserving-rc')}
    return {'totals': {k: dict(v) for k, v in totals.items()}, 'map_tests': tests,
            'committed_value_totals': value_totals, 'maps_rc_value_above_exact': rc_above_exact,
            'maps_rc_value_equal_exact': sum(v['value-preserving-rc'] == v['value-preserving-exact']
                                             for v in values.values()),
            'maps': len(values)}


def summarize_saved(root):
    from pathlib import Path
    root = Path(root)
    e1 = summarize_e1(_read_jsonl(root/'e1_traces.jsonl'))
    rows = list(_read_jsonl(root/'e2_rows.jsonl'))
    kinds, violations, tests, e4 = summarize_e2(rows)
    by = {kind: {dim: summarize_e2_by(rows, n, kind) for n, dim in enumerate(DIMENSIONS)}
          for kind in ('adaptive-scarcity', 'static-uniform', 'bernoulli-0.05')}
    return {'E1': e1, 'E2': {'by_adversary': kinds, 'rc_in_scope_violations': violations,
                             'graph_tests': tests, 'by_dimension': by, 'rows': len(rows)},
            'E3_rounds': {'E1_mean': {arm: v['rounds']/v['traces'] for arm, v in e1['totals'].items()},
                          'E2_mean': {kind: {'rc': v['rc_rounds']/v['runs'], 'mw': v['mw_rounds']/v['runs']}
                                      for kind, v in kinds.items()}},
            'E4_bound': e4,
            'E5': json.loads((root/'e5_carry_over.json').read_text())}


def verdict(summary):
    e1, e2 = summary['E1'], summary['E2']
    tot = e1['totals']
    endpoints = {
        'E1-P1': tot['value-preserving-exact']['count_guarantee_violations'] == 0,
        'E1-P2': e1['map_tests']['E1-P2']['p_one_sided'] < ALPHA
                 and e1['map_tests']['E1-P2']['wins'] > e1['map_tests']['E1-P2']['losses'],
        'E1-P3': tot['value-preserving-rc']['count_guarantee_violations'] == 0
                 and not e1['maps_rc_value_above_exact'],
        'E1-P4': e1['map_tests']['E1-P4']['p_one_sided'] < ALPHA
                 and e1['map_tests']['E1-P4']['wins'] > e1['map_tests']['E1-P4']['losses'],
        'E1-safety': all(v['wrong_terminal_jobs'] == 0 and v['over_budget_jobs'] == 0
                         for v in tot.values()),
        'E2-P1': not any(e2['rc_in_scope_violations'].get(k, 0) for k in
                         ('static-uniform', 'adaptive-scarcity', 'adaptive-first')),
        'E2-P2': e2['graph_tests']['adaptive-scarcity']['p_one_sided'] < ALPHA
                 and e2['graph_tests']['adaptive-scarcity']['wins'] > e2['graph_tests']['adaptive-scarcity']['losses'],
    }
    return {'verdict': 'PASS' if all(endpoints.values()) else 'ACCEPTED_NEGATIVE',
            'route': 'PAPER_CHANGE_REQUIRED', 'endpoints': endpoints}


# ---------------------------------------------------------------- independent audit

def audit_saved(root, config, remaining_cpu):
    import time
    from pathlib import Path
    from sevc.evaluation import recovery_enumeration as base
    from sevc.evaluation import action_certificate as filt
    from sevc.evaluation.recovery_graph_audit import audit_trace
    root, begun = Path(root), time.process_time()

    def guard():
        if time.process_time()-begun > remaining_cpu:
            raise TimeoutError('audit CPU budget exhausted')
    stream = _read_jsonl(root/'e1_traces.jsonl')
    e1_count = 0
    for item in e1_instances():
        oracle = base.independent_global_maximum(item['conflicts'], item['missing'])
        for arm in E1_ARMS:
            guard()
            row = next(stream)
            t = row['trace']
            assert row['instance'] == json.loads(json.dumps(item)) and t['policy'] == arm
            assert t['missing_identities'] == item['missing'] and t['seed'] == 0
            assert row['oracle']['maximum_completed_jobs'] == oracle, 'oracle disagreement'
            completed = audit_trace(t, row['assignments'])['completed_jobs']
            assert completed <= oracle
            compat = tuple(sum(1 << j for j in range(2) if f'j{j}' not in item['conflicts'].get(v, ()))
                           for v in base.IDENTITIES)
            if arm in ('value-preserving-exact', 'execution-certified-ecs-v2'):
                base.audit_executable_trace(t)
            elif arm in ('current-state-matching-v3', 'action-certified-ecs-v3'):
                filt.audit_trace(t)
            elif arm == 'value-preserving-rc':
                exact = base.hypothesis_auditor(compat)(511, (0, 0), base.failure_hypotheses(511, 2))[0]
                assert t['certificate']['guaranteed_count'] <= exact, 'RC value above exact value'
                events = t['published_events']
                actions, outcomes, cursor = [], [], 0
                for step in t['execution_steps']:
                    planned = [tuple(a) for a in step['guaranteed_actions']+step['other_actions']]
                    done = events[cursor:cursor+len(planned)]
                    assert [(int(e['job_id'][1:]), int(e['verifier_id'][1:])) for e in done] == planned[:len(done)]
                    actions.append(planned[:len(done)])
                    outcomes.append([e['status'] == 'PASS' for e in done])
                    cursor += len(done)
                assert cursor == len(events), 'unplanned activation'
                audit_rc_steps(compat, (3, 3), 2, t['execution_steps'], actions, outcomes)
                if len(item['missing']) <= 2:
                    assert completed >= t['certificate']['guaranteed_count'], 'RC guarantee violated'
            e1_count += 1
    assert next(stream, None) is None, 'extra E1 rows'
    games, oracle_cache, resimulated = {}, {}, 0
    rows = list(_read_jsonl(root/'e2_rows.jsonl'))
    expected = [(tuple(c), i) for c in cells() for i in range(config['graphs_per_cell'])]
    seen = []
    for row in rows:
        guard()
        key = (tuple(row['graph'][:5]), row['graph'][5])
        if key not in games:
            games[key] = scale_graph(key[0], key[1], config['e2_seed'])[0]
            seen.append(key)
        game = games[key]
        cache = (key, tuple(row['failed_identities']))
        if cache not in oracle_cache:
            oracle_cache[cache] = independent_full_jobs(game.compat, game.need, row['failed_identities'])
        assert row['oracle'] == oracle_cache[cache], 'E2 oracle disagreement'
        assert row['completed_jobs'] <= (row['oracle'] if row['oracle'] is not None else row['jobs'])
    assert seen == expected, 'E2 graph matrix incomplete or reordered'
    by_graph = {}
    for row in rows:
        by_graph.setdefault((tuple(row['graph'][:5]), row['graph'][5]), []).append(row)
    for key in expected[::10]:
        guard()
        game, _, fresh = run_graph(key[0], key[1], config['e2_seed'])
        assert fresh == by_graph[key], 'E2 re-simulation drift'
        rng = random.Random(f"{config['e2_seed']}/{key[0]}/{key[1]}/adversary")
        for label, in_scope, factory in adversary_runs(game, rng):
            result = rg.simulate(game, rg.RC, factory())
            trace = result['trace']
            audit_rc_steps(game.compat, game.need, game.allowance, [r['step'] for r in trace],
                           [[tuple(a) for a in r['actions']] for r in trace],
                           [r['outcomes'] for r in trace])
        resimulated += 1
    recomputed = summarize_saved(root)
    assert recomputed == json.loads((root/'summary.json').read_text()), 'summary drift'
    return {'status': 'AUDIT_PASS', 'e1_traces': e1_count, 'e2_rows': len(rows),
            'e2_graphs_resimulated': resimulated, 'e2_oracle_evaluations': len(oracle_cache),
            'e1_oracle': 'independent disjoint quorum subsets',
            'e1_value_audit': 'explicit failure hypotheses (exact); independent reserve invariant (RC)',
            'e2_oracle': 'independent Edmonds-Karp subset search',
            'external_reviewer': False, 'independent_replication': False,
            'audit_cpu_seconds': time.process_time()-begun}


# ---------------------------------------------------------------- outcome-blind benchmark

def benchmark(config):
    """Time development instances only (B maps, dev seed); no confirmation instance is built."""
    import time
    from sevc.evaluation import recovery_enumeration as base
    from sevc.evaluation.recovery_graph_audit import audit_trace
    started = time.process_time()
    maps = base.sampled_maps(128, 20260917)[:2]
    for index in maps:
        for missing in base.missing_sets((2, 3)):
            item = {'id': 'bench', 'family': 'bench', 'conflicts': base.decode_map(index),
                    'missing': missing, 'deadline': 1e6, 'owner_reserve': 0.}
            for arm in E1_ARMS:
                row = base.execute_instance(item, arm)
                audit_trace(row['trace'], row['assignments'])
                json.dumps(row)
    e1 = time.process_time()-started
    started = time.process_time()
    sample = cells()[::12]
    for cell in sample:
        game, _, rows = run_graph(cell, 0, E2_DEV_SEED)
        for row in rows:
            independent_full_jobs(game.compat, game.need, row['failed_identities'])
    e2 = time.process_time()-started
    forecast = 2.5*e1*64 + 1.25*e2*len(cells())*config['graphs_per_cell']/len(sample)
    return {'phase': 'benchmark', 'e1_cpu_seconds': e1, 'e1_instances': 240, 'e2_cpu_seconds': e2,
            'e2_graphs': len(sample), 'projected_cpu_seconds': forecast, 'budget_seconds': 7200,
            'within_budget': forecast <= 7200, 'outcome_metrics_inspected': False,
            'instances': 'development only (B maps seed 20260917; E2 dev seed 20260921)'}
