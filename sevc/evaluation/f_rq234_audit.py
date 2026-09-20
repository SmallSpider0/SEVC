"""Independent RQ2, RQ3 and RQ4 audits of the returned F v2 packet, plus the event close-out.

Read-only: no training, replay, tensor load or GPU initialization. Packet integrity reuses the
shared layer of ``f_rq_audit``; single-service settlements come from the RQ1 audit's recomputed
service rows (hash-bound). Every verdict follows the frozen F v2 event criteria and alpha;
exploratory analyses (EA-3..EA-8) are labelled and never change a registered verdict.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import gzip
import json
import math
from pathlib import Path
import random
import statistics
import zlib

from sevc.core.artifacts import sha256_file
from sevc.core.statistical_bounds import clopper_pearson_bound
from sevc.evaluation.f_rq_audit import DEVIATIONS, GOLD, RCMP, require

DIRECT = 'owner-direct-v2'
CONTEXTS = ('init-valid', 'init-invalid', 'mid-invalid')
PRICES = (0.0, 0.01, 0.1, 1.0)
FEES = (1.25, 2.5)
MAIN = {'fee': 2.5, 'bond': 0.5, 'price': 0.01}
REPLAY = ('source_reference_replay', 'challenge_full_replay_validation')
PHASE_GROUPS = {
    'replay': REPLAY,
    'compile': ('task_compile',),
    'delivery': ('delivery_payload_write', 'delivery_payload_hash'),
    'read': ('source_payload_read',),
    'generation': ('source_recipe', 'source_materialization', 'source_payload_materialization',
                   'source_commitment_hash', 'source_payload_hash', 'source_compact_header_capture'),
}


def _group(phase):
    for name, phases in PHASE_GROUPS.items():
        if phase in phases:
            return name
    return 'other'


def iter_jsonl(path):
    path = Path(path)
    if not path.exists():
        path = Path(str(path) + '.gz')
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt') as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def sign_test(ratios, alpha):
    require(bool(ratios) and all(math.isfinite(x) and x > 0 for x in ratios), 'invalid paired ratios')
    n, wins = len(ratios), sum(x < 1 for x in ratios)
    p = sum(math.comb(n, i) for i in range(wins, n + 1)) / 2 ** n
    median = statistics.median(ratios)
    return {'n': n, 'median': median, 'min': min(ratios), 'max': max(ratios), 'wins_below_one': wins,
            'p_one_sided_ties_nonwins': p, 'alpha': alpha, 'criterion_met': median < 1 and p <= alpha}


def bootstrap_mean(values, blocks, seed, draws=4000):
    """Percentile interval of the mean, resampling independent blocks (exploratory)."""
    by_block = defaultdict(list)
    for v, b in zip(values, blocks):
        by_block[b].append(v)
    keys = sorted(by_block)
    rng = random.Random(seed)
    means = []
    for _ in range(draws):
        sample = [x for k in (rng.choice(keys) for _ in keys) for x in by_block[k]]
        means.append(sum(sample) / len(sample))
    means.sort()
    return [means[int(.025 * draws)], means[int(.975 * draws) - 1]]


# ------------------------------------------------------------------ shared inputs

def load_units(root, config, phases):
    root = Path(root)
    index = json.loads((root / 'result_index.json').read_text())
    out = {}
    for planned in config['fixed_design_units']:
        if planned['phase'] not in phases:
            continue
        uid = planned['unit_id']
        path = root / 'units' / (uid + '.json')
        require(index.get(f'units/{uid}.json') == sha256_file(path), 'unit differs from result index: ' + uid)
        row = json.loads(path.read_text())
        require(all(row.get(k) == v for k, v in planned.items()), 'unit identity drift: ' + uid)
        out[uid] = row
    return out


def unit_role_seconds(root, unit_ids):
    """Exclusive seconds per unit, role and phase for online events of the given units."""
    roles = defaultdict(lambda: defaultdict(float))
    phases = defaultdict(lambda: defaultdict(float))
    scoped = defaultdict(float)
    for event in iter_jsonl(Path(root) / 'phase-timing.jsonl'):
        scoped[(event.get('dataset', 'shared'), event.get('phase_scope', 'shared'), event['role'])] += event['exclusive_seconds']
        uid = event.get('unit_id')
        if uid in unit_ids and event.get('phase_scope') == 'online':
            roles[uid][event['role']] += event['exclusive_seconds']
            if event['role'] == 'owner':
                phases[uid][event['phase']] += event['exclusive_seconds']
    return roles, phases, scoped


# --------------------------------------------------------------------------- RQ4

def audit_rq4(root, config, units, roles, phases, scoped, services):
    datasets = list(config['datasets'])
    production = {u: r for u, r in units.items() if r['phase'] == 'production' and r['behavior'] == 'honest'}
    paired = defaultdict(dict)
    for uid, row in production.items():
        paired[(row['dataset'], row['context'], row['block'])][row['method']] = row
    owner_cells, events = {}, []
    reference_blocks = defaultdict(lambda: defaultdict(list))
    for d_index, dataset in enumerate(datasets):
        for c_index, context in enumerate(CONTEXTS):
            ratios = []
            for block in range(24):
                cell = paired[(dataset, context, block)]
                require(set(cell) == {RCMP, GOLD, DIRECT}, f'paired methods missing {dataset}/{context}/{block}')
                sources = [cell[m]['production_source_ids'] for m in (RCMP, GOLD, DIRECT)]
                require(sources[0] == sources[1] == sources[2], 'unpaired production sources')
                r, o = roles[cell[RCMP]['unit_id']]['owner'], roles[cell[DIRECT]['unit_id']]['owner']
                ratios.append(r / o)
                reference_blocks[dataset][block].append(cell[RCMP]['online_preparation_seconds']
                                                        / cell[GOLD]['online_preparation_seconds'])
            test = sign_test(ratios, .05 / 9)
            owner_cells[f'{dataset}/{context}'] = test
            events.append({'event_id': f'owner-cost/{3 * d_index + c_index + 1:03d}', 'dataset': dataset,
                           'context': context, **test, 'profile': 'as-run (no resident challenge delivery)',
                           'verdict': 'PASS_AS_RUN' if test['criterion_met'] else 'ACCEPTED_NEGATIVE_AS_RUN'})
    preparation = {}
    for d_index, dataset in enumerate(datasets):
        test = sign_test([statistics.mean(v) for _, v in sorted(reference_blocks[dataset].items())], .05 / 3)
        preparation[dataset] = test
        events.append({'event_id': f'reference-preparation/{d_index + 1:03d}', 'dataset': dataset, **test,
                       'profile': 'as-run', 'verdict': 'PASS_AS_RUN' if test['criterion_met'] else 'ACCEPTED_NEGATIVE_AS_RUN'})
    components = {}
    for dataset in datasets:
        for method in (RCMP, GOLD, DIRECT):
            ids = [u for u, r in production.items() if r['dataset'] == dataset and r['method'] == method]
            groups = defaultdict(list)
            for uid in ids:
                total = defaultdict(float)
                for phase, seconds in phases[uid].items():
                    total[_group(phase)] += seconds
                for name in list(PHASE_GROUPS) + ['other']:
                    groups[name].append(total[name])
            components[f'{dataset}/{method}'] = {k: statistics.mean(v) for k, v in groups.items()}
            components[f'{dataset}/{method}']['units'] = len(ids)
            components[f'{dataset}/{method}']['owner_total_mean'] = statistics.mean(roles[u]['owner'] for u in ids)
    # EA-7: hidden-gold preparation split and the "existing reference pool" sensitivity.
    ea7 = {}
    for dataset in datasets:
        r, g = components[f'{dataset}/{RCMP}'], components[f'{dataset}/{GOLD}']
        pool = g['owner_total_mean'] - g['generation']
        ea7[dataset] = {'gold_generation_mean_s': g['generation'], 'gold_validation_replay_mean_s': g['replay'],
                        'gold_compile_delivery_read_mean_s': g['compile'] + g['delivery'] + g['read'],
                        'R_over_G_owner_mean': r['owner_total_mean'] / g['owner_total_mean'],
                        'R_over_G_if_pool_exists_mean': r['owner_total_mean'] / pool,
                        'label': 'post-hoc; pool sensitivity removes only generation phases'}
    target = {}
    bridge = {u: r for u, r in units.items() if r['phase'] == 'target_bridge' and r['method'] in (RCMP, GOLD, DIRECT)}
    for dataset in datasets:
        for anchor in (128, 512):
            med = {}
            for method in (RCMP, GOLD, DIRECT):
                vals = [roles[u]['owner'] for u, r in bridge.items()
                        if r['dataset'] == dataset and r['anchor'] == anchor and r['method'] == method]
                med[method] = statistics.mean(vals)
            target[f'{dataset}/anchor-{anchor}'] = {'owner_mean_s': med, 'n_per_method': 2,
                                                     'R_over_O': med[RCMP] / med[DIRECT], 'R_over_G': med[RCMP] / med[GOLD]}
    depol = {}
    for dataset in datasets:
        ids = [u for u, r in units.items() if r['method'] == 'depol-local-verification-arbitration-v1'
               and r['dataset'] == dataset and r['phase'] in ('native', 'target_bridge')]
        per_role = defaultdict(list)
        for uid in ids:
            for role in ('owner', 'verifier', 'trainer'):
                per_role[f"{units[uid]['phase']}/{role}"].append(roles[uid][role])
        depol[dataset] = {k: statistics.mean(v) for k, v in sorted(per_role.items())}
    system = defaultdict(lambda: defaultdict(float))
    for (dataset, scope, role), seconds in scoped.items():
        system[dataset][f'{scope}/{role}'] += seconds
    transfers = defaultdict(lambda: defaultdict(float))
    for s in services:
        transfers[s['dataset']]['fees_paid'] += s['service_fee']
        transfers[s['dataset']]['bonds_forfeited'] += s['slashed_bond']
    return {'owner_R_over_O': owner_cells, 'preparation_R_over_G': preparation, 'events': events,
            'components_mean_seconds': components, 'EA-7': ea7, 'target_bridge_descriptive': target,
            'depol_native_role_seconds_unpaired': depol,
            'system_exclusive_seconds_by_dataset_scope_role': {k: dict(v) for k, v in system.items()},
            'single_service_transfers': {k: dict(v) for k, v in transfers.items()},
            'scope': 'F v2 as-run profile: resident challenge delivery absent; owner computation excludes payments'}


def ea8_break_even(se_cost_run):
    """Component model fitted on 4-step SE-COST units and checked on the 16-step target units."""
    root = Path(se_cost_run)
    units = {}
    for path in (root / 'units').glob('*.json'):
        row = json.loads(path.read_text())
        if row['status'] == 'MEASURED' and row['arm'] in ('R-repaired', 'O') and row['phase'] in ('primary', 'target'):
            units[row['unit_id']] = row
    parts = defaultdict(lambda: defaultdict(float))
    for event in iter_jsonl(root / 'phase-timing.jsonl'):
        uid = event.get('unit_id')
        if uid in units and event.get('phase_scope') == 'online' and event['role'] == 'owner':
            parts[uid][_group(event['phase'])] += event['exclusive_seconds']
    cells = defaultdict(lambda: defaultdict(list))
    for uid, row in units.items():
        key = f"{row['dataset']}/{row['steps']}"
        for name in list(PHASE_GROUPS) + ['other']:
            cells[key][row['arm'] + '/' + name].append(parts[uid][name])
        cells[key][row['arm'] + '/total'].append(sum(parts[uid].values()))
    mean = {k: {n: statistics.mean(v) for n, v in c.items()} for k, c in cells.items()}
    out = {}
    for dataset in sorted({k.split('/')[0] for k in mean}):
        base, far = mean[f'{dataset}/4'], mean[f'{dataset}/16']
        # Replay work scales with steps; payload handling with stored checkpoints (steps + 1).
        predict = {}
        for arm in ('R-repaired', 'O'):
            replay = base[f'{arm}/replay'] * 16 / 4
            payload = sum(base[f'{arm}/{n}'] for n in ('compile', 'delivery', 'read', 'other')) * 17 / 5
            predict[arm] = replay + payload
        payload_share = {arm: 1 - far[f'{arm}/replay'] / far[f'{arm}/total'] for arm in ('R-repaired', 'O')}
        out[dataset] = {
            'observed_R_over_O_4': base['R-repaired/total'] / base['O/total'],
            'observed_R_over_O_16': far['R-repaired/total'] / far['O/total'],
            'predicted_R_over_O_16': predict['R-repaired'] / predict['O'],
            'model_relative_error_R_16': predict['R-repaired'] / far['R-repaired/total'] - 1,
            'model_relative_error_O_16': predict['O'] / far['O/total'] - 1,
            'owner_payload_share_16': payload_share,
            'R_delivery_spill_seconds_16': far['R-repaired/delivery'],
            'R_components_16': {n: far[f'R-repaired/{n}'] for n in list(PHASE_GROUPS) + ['other']},
            'O_components_16': {n: far[f'O/{n}'] for n in list(PHASE_GROUPS) + ['other']},
            'R_components_4': {n: base[f'R-repaired/{n}'] for n in list(PHASE_GROUPS) + ['other']},
            'O_components_4': {n: base[f'O/{n}'] for n in list(PHASE_GROUPS) + ['other']},
            'break_even_condition': 'R < O iff owner payload handling (compile, read, delivery) is smaller than the '
                                    'replay it displaces: sum_payload_R < replay_O - replay_R + payload_O',
            'validated': abs(predict['R-repaired'] / far['R-repaired/total'] - 1) <= .2,
        }
    return {'label': 'EA-8 model extrapolation; not a registered endpoint', 'strata': out}


def se_cost_blocks(se_cost_run):
    """Per-block owner ratios and per-role resources of the SE-COST primary units (4 x 32)."""
    root = Path(se_cost_run)
    units = {}
    for path in (root / 'units').glob('*.json'):
        row = json.loads(path.read_text())
        if row['status'] == 'MEASURED' and row['phase'] == 'primary':
            units[row['unit_id']] = row
    usage = defaultdict(lambda: defaultdict(float))
    for event in iter_jsonl(root / 'phase-timing.jsonl'):
        row = units.get(event.get('unit_id'))
        if row is None or event.get('phase_scope') != 'online':
            continue
        role = event['role']
        usage[row['unit_id']][role + '/wall'] += event['exclusive_seconds']
        usage[row['unit_id']][role + '/cpu'] += event['exclusive_cpu_process_seconds']
        io = event['exclusive_io_bytes']
        usage[row['unit_id']][role + '/io_bytes'] += io['rchar'] + io['wchar']
        if role == 'owner':
            usage[row['unit_id']]['owner/' + _group(event['phase'])] += event['exclusive_seconds']
    cell = defaultdict(dict)
    for uid, row in units.items():
        cell[(row['dataset'], row['context'], row['block'])][row['arm']] = usage[uid]
    ratios = []
    for (dataset, context, block), arms in sorted(cell.items()):
        o = arms['O']['owner/wall']
        ratios.append({'dataset': dataset, 'context': context, 'block': block,
                       'R_repaired_over_O': arms['R-repaired']['owner/wall'] / o,
                       'R_as_run_over_O': arms['R-as-run']['owner/wall'] / o,
                       'R_over_G_owner': arms['R-repaired']['owner/wall'] / arms['G-repaired']['owner/wall']})
    resources = {}
    for dataset in sorted({k[0] for k in cell}):
        for arm in ('R-repaired', 'R-as-run', 'G-repaired', 'O'):
            rows = [a[arm] for k, a in cell.items() if k[0] == dataset]
            keys = sorted({key for r in rows for key in r})
            peak = [units[u]['absolute_resources'].get('cuda_unit_peak_allocated_bytes', 0)
                    for u, r in units.items() if r['dataset'] == dataset and r['arm'] == arm]
            resources[f'{dataset}/{arm}'] = {**{k: statistics.mean(r.get(k, 0.) for r in rows) for k in keys},
                                             'units': len(rows), 'peak_cuda_allocated_bytes_mean': statistics.mean(peak)}
    return {'label': 'SE-COST primary units re-aggregated from raw phase records', 'block_ratios': ratios,
            'role_resources_mean_per_job': resources}


# --------------------------------------------------------------------------- RQ2

def payoff(service, fee, bond, price):
    paid = 1 if service['admitted'] else 0
    slashed = service['slashed_bond'] / MAIN['bond'] * bond if MAIN['bond'] else 0.
    return fee * paid - slashed - price * service['verifier_seconds']


def audit_rq2(config, services, recovery_units, native_units):
    datasets = list(config['datasets'])
    honest = {(s['dataset'], s['context'], s['block']): s for s in services
              if s['phase'] == 'production' and s['method'] == RCMP and s['behavior'] == 'honest'}
    strata = []
    for dataset in datasets:
        for context in ('init-valid', 'init-invalid'):
            base = [honest[(dataset, context, b)] for b in range(24)]
            u_h = [payoff(s, **MAIN) for s in base]
            t_h = statistics.mean(s['verifier_seconds'] for s in base)
            pay_h = statistics.mean(MAIN['fee'] * s['admitted'] - s['slashed_bond'] for s in base)
            strata.append({'dataset': dataset, 'context': context, 'behavior': 'honest', 'n': 24,
                           'paid': sum(s['admitted'] for s in base), 'mean_seconds': t_h,
                           'mean_utility_main': statistics.mean(u_h),
                           'participation_break_even_price': pay_h / t_h,
                           'participation_break_even_price_by_fee': {
                               str(fee): statistics.mean(fee * s['admitted'] - s['slashed_bond'] for s in base) / t_h
                               for fee in FEES}})
            for behavior in DEVIATIONS:
                group = sorted((s for s in services if s['phase'] == 'production' and s['method'] == RCMP
                                and s['dataset'] == dataset and s['context'] == context and s['behavior'] == behavior),
                               key=lambda s: s['block'])
                require(len(group) == 24 and len({s['block'] for s in group}) == 24, 'not 24 blocks')
                diff = [payoff(s, **MAIN) - payoff(honest[(dataset, context, s['block'])], **MAIN) for s in group]
                d_pay = statistics.mean((MAIN['fee'] * s['admitted'] - s['slashed_bond'])
                                        - (MAIN['fee'] * honest[(dataset, context, s['block'])]['admitted']
                                           - honest[(dataset, context, s['block'])]['slashed_bond']) for s in group)
                d_time = statistics.mean(s['verifier_seconds'] - honest[(dataset, context, s['block'])]['verifier_seconds']
                                         for s in group)
                # Deviation gains iff d_pay - c*d_time > 0; with d_time < 0 this is c > d_pay/d_time.
                c_star = d_pay / d_time if d_time < 0 and d_pay < 0 else (0. if d_pay >= 0 else math.inf)
                paid_gap = statistics.mean(s['admitted'] - honest[(dataset, context, s['block'])]['admitted'] for s in group)
                forfeit_gap = statistics.mean(s['slashed_bond'] - honest[(dataset, context, s['block'])]['slashed_bond'] for s in group)
                by_fee = {}
                for fee in FEES:
                    pay = fee * paid_gap - forfeit_gap
                    by_fee[str(fee)] = pay / d_time if d_time < 0 and pay < 0 else (0. if pay >= 0 else math.inf)
                sweep = {f'{fee}/{price}': statistics.mean(payoff(s, fee, MAIN['bond'], price)
                                                          - payoff(honest[(dataset, context, s['block'])], fee, MAIN['bond'], price)
                                                          for s in group) for fee in FEES for price in PRICES}
                strata.append({'dataset': dataset, 'context': context, 'behavior': behavior, 'n': 24,
                               'paid': sum(s['admitted'] for s in group),
                               'forfeited': sum(s['slashed_bond'] > 0 for s in group),
                               'mean_seconds': statistics.mean(s['verifier_seconds'] for s in group),
                               'mean_deviation_minus_honest_main': statistics.mean(diff),
                               'block_bootstrap_95': bootstrap_mean(diff, [s['block'] for s in group],
                                                                    seed=zlib.crc32(f'{dataset}/{context}/{behavior}'.encode())),
                               'positive_gain_services': sum(x > 0 for x in diff),
                               'expected_payment_difference': d_pay, 'expected_time_difference': d_time,
                               'deviation_break_even_price': c_star, 'deviation_break_even_price_by_fee': by_fee,
                               'mean_difference_by_fee_price': sweep})
    events = []
    index = 0
    for dataset in datasets:
        for validity in ('valid', 'invalid'):
            for item in ('IR',) + DEVIATIONS:
                index += 1
                events.append({'event_id': f'honest-IR-and-effort/{index:03d}', 'dataset': dataset, 'validity': validity,
                               'item': item, 'verdict': 'INCONCLUSIVE',
                               'basis': 'no preregistered capped resource range in F v2; Hoeffding bound undefined'})
    # EA-3: Proposition 2 with observed parameters (h = 1, r = 8 probes, beta = u = 0).
    theory = []
    for s in strata:
        if s['behavior'] == 'honest':
            continue
        h = next(x for x in strata if x['dataset'] == s['dataset'] and x['context'] == s['context'] and x['behavior'] == 'honest')
        psi = s['forfeited'] / s['n']
        d_e = MAIN['price'] * (h['mean_seconds'] - s['mean_seconds'])
        theta = d_e / (MAIN['fee'] + MAIN['bond'])
        q = {'uniform-k32': (40 - 32) / 80, 'uniform-k39': (40 - 39) / 80}.get(s['behavior'])
        needed = (math.log(1 - theta) / math.log(1 - q)) if q and 0 <= theta < 1 else None
        theory.append({'dataset': s['dataset'], 'context': s['context'], 'behavior': s['behavior'],
                       'observed_psi': psi, 'D_e_main_price': d_e, 'theta': theta,
                       'effort_condition_holds_observed': psi * (MAIN['fee'] + MAIN['bond']) > d_e,
                       'q_e_by_construction': q, 'probes_needed_prop2': needed, 'probes_used': 8})
    # EA-6: liquidity cost of the bond (annual rate x lock time) relative to the fee.
    lock = max(s['mean_seconds'] for s in strata)
    ea6 = {'bond': MAIN['bond'], 'lock_seconds_upper': lock,
           'kappa_d_by_annual_rate': {str(rate): MAIN['bond'] * rate * lock / (365 * 86400) for rate in (.05, .2, 1.)},
           'label': 'analytic sensitivity; kappa_d = bond x rate x lock time'}
    # M2.4: realized transfers and utility in the executed recovery traces.
    roles = defaultdict(lambda: defaultdict(list))
    for row in recovery_units.values():
        rec = row['recovery']
        cost = {a['report']['verifier_id']: a['settlement']['verifier_cost'] for a in row['assignments']}
        behavior = {a['report']['verifier_id']: a['behavior'] for a in row['assignments']}
        order = defaultdict(list)
        for event in rec['published_events']:
            order[event['job_id']].append(event['verifier_id'])
        for receipt in rec['receipts']:
            vid = receipt['verifier_id']
            if receipt['status'] == 'DROPOUT':
                role, fee, slashed, seconds = 'missing', receipt['fee'], receipt['slashed_bond'], 0.
            else:
                settle = receipt['settlement']
                fee, slashed, seconds = settle['service_fee'], settle['slashed_bond'], cost.get(vid, settle['verifier_cost'])
                role = ('deviating' if behavior.get(vid, 'honest') != 'honest'
                        else 'replacement' if order[receipt['job_id']].index(vid) >= 3 else 'primary')
            roles[role]['fee'].append(fee)
            roles[role]['slashed'].append(slashed)
            roles[role]['utility_main'].append(fee - slashed - MAIN['price'] * seconds)
    recovery = {role: {'n': len(v['fee']), 'mean_fee': statistics.mean(v['fee']),
                       'mean_forfeit': statistics.mean(v['slashed']), 'mean_utility_main': statistics.mean(v['utility_main']),
                       'min_utility_main': min(v['utility_main'])} for role, v in sorted(roles.items())}
    ctf = []
    for row in native_units.values():
        if not row['method'].startswith(('ctf-pp', 'simple-agreement')):
            continue
        game = row.get('native_game') or {}
        score = (row.get('solution') or {}).get('score')
        ctf.append({'method': row['method'], 'epsilon': row.get('epsilon'), 'status': row['status'],
                    'honest_utility': (game.get('honest') or {}).get('utility'),
                    'honest_minus_best_deviation': game.get('honest_minus_best_deviation'),
                    'amplitude': max(abs(x) for r in score for x in r) if score else None,
                    'feasible': row['status'] == 'MEASURED'})
    ctf.sort(key=lambda x: (x['method'], x['epsilon']))
    return {'strata': strata, 'events': events, 'EA-3': theory, 'EA-6': ea6, 'recovery_utility': recovery,
            'ctf_pp': ctf, 'main_terms': MAIN}


# --------------------------------------------------------------------------- RQ3

def majority_terminal(reports, segments):
    """Accept iff every segment has a strict majority of accepting verdicts among the given reports."""
    for segment in segments:
        votes = [r[segment] for r in reports if r.get(segment) is not None]
        if not votes or not sum(v is True for v in votes) > len(votes) / 2:
            return 'reject'
    return 'accept'


def audit_rq3(config, recovery_units):
    datasets = list(config['datasets'])
    violations = Counter()
    checks = Counter()
    fees = defaultdict(float)
    wrong = defaultdict(int)
    blocks = defaultdict(set)
    for uid, row in recovery_units.items():
        rec = row['recovery']
        target = 'accept' if row['invalid'] == 0 else 'reject'
        cap = rec['budget_cap_per_job']
        for job, reserved in rec['reserved_expenditure'].items():
            checks['budget'] += 1
            violations['budget'] += reserved > cap + 1e-9
        paid = defaultdict(float)
        for receipt in rec['receipts']:
            settle = receipt.get('settlement') or receipt
            status = receipt['status']
            paid[receipt['job_id']] += settle.get('service_fee', settle.get('fee', 0.))
            checks['settlement'] += 1
            violations['settlement'] += (settle.get('service_fee', settle.get('fee', 0.)) > 0) != (status == 'PASS')
            violations['settlement'] += (settle['slashed_bond'] > 0) != (status in ('FAIL_CONFIRMED', 'DROPOUT'))
        for job, amount in paid.items():
            checks['expenditure'] += 1
            violations['expenditure'] += amount > rec['reserved_expenditure'][job] + 1e-9
            fees[(row['policy'], row['fault'])] += amount
        checks['deadline'] += 1
        violations['deadline'] += rec['logical_elapsed_seconds'] > rec['deadline_seconds']
        checks['capacity'] += len(rec['usage'])
        violations['capacity'] += sum(v > 1 for v in rec['usage'].values())
        conflicts = {(v, j) for v, jobs in rec['public_conflicts'].items() for j in jobs}
        checks['conflict'] += len(rec['published_events'])
        violations['conflict'] += sum((e['verifier_id'], e['job_id']) in conflicts for e in rec['published_events'])
        if rec['certificate'].get('status') == 'NOT_REQUESTED_ONLINE_ONLY':
            checks['certificate_not_requested_online_baseline'] += 1
        else:
            checks['certificate'] += 1
            violations['certificate'] += rec['certificate']['certificate_passed'] is not True
        checks['pre_defer'] += rec['pre_defer']
        blocks[(row['dataset'], row['invalid'])].add(row['block'])
        for job, decision in rec['decisions'].items():
            wrong[(row['dataset'], row['invalid'], row['block'])] += decision['route'] not in (target, 'safe-defer')
    risk = []
    index = 0
    for dataset in datasets:
        for invalid in (0, 1):
            index += 1
            n = len(blocks[(dataset, invalid)])
            failed = sum(wrong[(dataset, invalid, b)] > 0 for b in blocks[(dataset, invalid)])
            upper = clopper_pearson_bound(failed, n, side='upper', alpha=.05) if n else None
            risk.append({'event_id': f'final-settlement-risk/{index:03d}', 'dataset': dataset,
                         'validity': 'valid' if invalid == 0 else 'invalid', 'blocks': n, 'failed_blocks': failed,
                         'cp_upper_one_sided_95': upper,
                         'verdict': 'INCONCLUSIVE_BY_DESIGN' if failed == 0 else 'ACCEPTED_NEGATIVE'})
    invariant_event = {'event_id': 'ECS-causal-and-invariants/001', 'checks': dict(checks),
                       'violations': dict(violations), 'pre_defer_units': checks['pre_defer'],
                       'fees_paid_by_policy_fault': {f'{p}/{f}': v for (p, f), v in sorted(fees.items())},
                       'same_raw_roster': all(r.get('common_raw_roster') for r in recovery_units.values()),
                       'verdict': 'PASS' if not any(violations.values()) else 'ACCEPTED_NEGATIVE'}
    # EA-4: delete one recorded report and decide by majority of arrived reports without admission.
    ea4 = Counter()
    for row in recovery_units.values():
        rec = row['recovery']
        target = 'accept' if row['invalid'] == 0 else 'reject'
        order = defaultdict(list)
        for event in rec['published_events']:
            order[event['job_id']].append(event['verifier_id'])
        by_vid = {a['report']['verifier_id']: a for a in row['assignments']}
        for job, verifiers in order.items():
            first = [v for v in verifiers[:3] if v in by_vid]
            segments = rec['required_ids_by_job'][job]
            for drop in range(len(first)):
                kept = [by_vid[v] for i, v in enumerate(first) if i != drop]
                reports = [dict(zip(a['report']['ordered_segment_ids'], a['report']['verdicts'])) for a in kept]
                route = majority_terminal(reports, segments)
                deviant = any(a['behavior'] != 'honest' for a in kept)
                key = ('with_deviant' if deviant else 'honest_only', 'valid' if target == 'accept' else 'invalid')
                ea4[key + ('cases',)] += 1
                ea4[key + ('wrong',)] += route != target
                ea4[key + ('valid_accused',)] += target == 'accept' and route == 'reject'
    ea4_out = {'/'.join(k): v for k, v in sorted(ea4.items())}
    ea4_out['rule'] = 'drop one of the first three published reports; strict per-segment majority of the rest, no admission'
    # EA-5: decision-set size and residual error for conditionally independent p.
    def reli(p, n):
        return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(n // 2 + 1, n + 1))
    ea5 = []
    for p in (.7, .8, .9, .95):
        sizes = {rho: next(n for n in range(3, 101, 2) if reli(p, n) >= rho) for rho in (.95, .99)}
        ea5.append({'p': p, 'one_minus_reli_3': 1 - reli(p, 3), 'one_minus_reli_5': 1 - reli(p, 5),
                    'size_for_rho_95': sizes[.95], 'size_for_rho_99': sizes[.99]})
    return {'final_settlement_risk': risk, 'invariants': invariant_event, 'EA-4': ea4_out, 'EA-5': ea5}


# ------------------------------------------------------------------ event close-out

def close_events(event_map, verdicts):
    """Assign each of the 124 registered events exactly one terminal verdict."""
    out = []
    for event in event_map['events']:
        eid, family = event['event_id'], event['family']
        if eid in verdicts:
            verdict, source = verdicts[eid]
        elif family == 'qualification':
            verdict, source = 'AUTHOR_SCOPE_REVISED_NOT_CERTIFIED', 'F v2 SAP; conditional p=.9 roster'
        elif family == 'startup-economics':
            verdict, source = 'NOT_ESTABLISHED_SCOPE_REVISED', 'only the lightweight workflow startup was measured'
        else:
            raise ValueError('no verdict for registered event ' + eid)
        out.append({'event_id': eid, 'family': family, 'primary_rq': event['primary_rq'], 'verdict': verdict,
                    'source': source, 'original_criterion': event['original_criterion']})
    require(len(out) == 124 and len({e['event_id'] for e in out}) == 124, 'event count drift')
    return out

