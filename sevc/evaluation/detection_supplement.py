"""SE-DET scope gate and raw-record projection; independent return audit remains required."""
from collections import defaultdict
from dataclasses import asdict
import json
import math
from pathlib import Path

from sevc.core.artifacts import sha256_file, write_json


def validate_scope(config):
    from sevc.experiments.detection_supplement import CHANGE, COUNTS, expanded_units
    if config['change_id'] != CHANGE or config['fixed_design']['counts'] != COUNTS:
        raise PermissionError('detection count/change drift')
    if config['fixed_design']['phase_order'] != list(COUNTS):
        raise PermissionError('detection phase drift')
    if config['fixed_design_units'] != expanded_units(config['scoped_candidate']['science']['methods']):
        raise PermissionError('detection frozen matrix drift')
    if len(config['fixed_design_units']) != 846 or config['maximum_runner_wall_seconds'] != 43200:
        raise PermissionError('detection matrix/time drift')
    if config['detection_supplement']['billing_cap_seconds'] != {'mnist':30,'cifar10':300,'cifar100':300}:
        raise PermissionError('detection billing cap drift')
    if config['fixed_design']['depol'].get('interval_granularity') != 'per-step':
        raise PermissionError('detection native granularity drift')
    frozen = json.loads((Path(__file__).resolve().parents[2] / 'configs/tdsc_rq1_detection_supplement_v1.json').read_text())
    for key in ('datasets','performance','fixed_design','detection_supplement','scoped_candidate','reuse_paired_validity_sources'):
        if config[key] != frozen[key]:
            raise PermissionError('detection scientific configuration drift: '+key)


PARTICIPATION_CAPS = {'mnist': 10, 'cifar10': 60, 'cifar100': 60}
PARTICIPATION_PARTITIONS = {'cifar10': {'test': [8000, 14000]}, 'mnist': {'test': [18000, 24000]},
                            'cifar100': {'test': [8000, 14000]}}
PARTICIPATION_RUNNER_SECONDS = 21600


def validate_participation_scope(config):
    """Frozen honest-only follow-up: new blocks, tighter preregistered cap, same endpoint."""
    from sevc.experiments.detection_supplement import (PARTICIPATION_CHANGE, PARTICIPATION_COUNTS,
                                                       participation_units)
    fixed = config['fixed_design']
    if config['change_id'] != PARTICIPATION_CHANGE or fixed['counts'] != PARTICIPATION_COUNTS:
        raise PermissionError('participation count/change drift')
    if fixed['phase_order'] != ['test'] or fixed['partitions'] != PARTICIPATION_PARTITIONS:
        raise PermissionError('participation phase/partition drift')
    if config['fixed_design_units'] != participation_units(config['scoped_candidate']['science']['methods']):
        raise PermissionError('participation frozen matrix drift')
    if len(config['fixed_design_units']) != 144 or config['maximum_runner_wall_seconds'] != PARTICIPATION_RUNNER_SECONDS:
        raise PermissionError('participation matrix/time drift')
    if config['detection_supplement']['billing_cap_seconds'] != PARTICIPATION_CAPS:
        raise PermissionError('participation billing cap drift')
    frozen = json.loads((Path(__file__).resolve().parents[2] / 'configs/tdsc_rq2_honest_participation_v1.json').read_text())
    for key in ('datasets','performance','fixed_design','detection_supplement','scoped_candidate','reuse_paired_validity_sources'):
        if config[key] != frozen[key]:
            raise PermissionError('participation scientific configuration drift: '+key)


def capped_observation(settlement, cap, fee, bond, cost):
    """Author-approved billing endpoint; physical computation is NOT capped."""
    elapsed = settlement['verifier_cost']
    if not math.isfinite(elapsed) or elapsed < 0 or cap <= 0:
        raise ValueError('invalid service time')
    timeout = elapsed > cap
    paid = not timeout and settlement['status'] == 'PASS' and settlement['accepted_report']
    slashed = bond if timeout else (bond if settlement['slashed_bond'] > 0 else 0.)
    actual_fee = fee if paid else 0.
    transfer = actual_fee - slashed
    return {'paid':paid,'timeout':timeout,'service_fee':actual_fee,'slashed_bond':slashed,
            'actual_seconds':elapsed,'billed_seconds':min(elapsed, cap),
            'capped_billing_utility':transfer-cost*min(elapsed, cap),
            'uncapped_compute_utility':transfer-cost*elapsed}


def sign_test(differences):
    n = len(differences)
    if not n:
        raise ValueError('empty block observations')
    wins = sum(v > 0 for v in differences)
    return {'blocks':n,'wins':wins,'ties':sum(v == 0 for v in differences),
            'p_one_sided_ties_nonwins':sum(math.comb(n,i) for i in range(wins,n+1))/2**n}


def _records(path):
    with Path(path).open() as stream:
        for line in stream:
            yield json.loads(line)


def audit_and_summarize(root, config, profile):
    """Recompute settlements/references/selection from durable raw evidence, never pickle-load."""
    from sevc.evaluation.f_fixed_design import _inventory, full_cost_inventory
    from sevc.incentives.verifier_protocol import CommittedVerifierReport
    from sevc.verification.paid_replay_service import OwnerProbeReferences, settle_service
    from sevc.verification.cheap_discriminator import history_digest, select_development_models
    from sevc.verification.reference_acquisition import commitment
    from sklearn.metrics import roc_auc_score
    from sevc.experiments.detection_supplement import PARTICIPATION_CHANGE
    participation_only = config['change_id'] == PARTICIPATION_CHANGE
    root = Path(root)
    _, rows, errors = _inventory(root, config['fixed_design_units'])
    if errors:
        raise ValueError('detection inventory incomplete: '+str(errors[:3]))
    if json.loads((root/'expanded-units.json').read_text()) != config['fixed_design_units']:
        raise ValueError('expanded matrix drift')
    committed = {}; revealed = {}
    for event in _records(root/'report-lifecycle.jsonl'):
        aid = event['assignment_id']
        if event['phase'] == 'COMMITTED':
            if aid in committed: raise ValueError('duplicate report commitment')
            committed[aid] = event['commitment']
        elif event['phase'] == 'REVEALED':
            report = CommittedVerifierReport(**event['report'])
            if aid in revealed or committed.get(aid) != report.commitment:
                raise ValueError('report lifecycle mismatch')
            revealed[aid] = event['report']
        else:
            raise ValueError('unknown report lifecycle phase')
    by_id = {r['unit_id']:r for r in rows}
    expected_epochs = defaultdict(set)
    for row in rows:
        if row['package'] != 'DP':
            key = commitment([row['dataset'],row['phase'],(row['seed'],row['anchor'],row['steps'],row['batch_size'])])
            expected_epochs[key].update(a['assignment_id'] for a in row.get('assignments',[]))
    disclosed = {}; audit_seen = set()
    for event in _records(root/'assignment-audit.jsonl'):
        if event.get('event') == 'audit-selected':
            audit_seen.add(event['assignment_id'])
        if event.get('event') == 'reference-answers-published':
            epoch = event['epoch_id']; states = event['assignment_states']
            if epoch in disclosed or epoch not in expected_epochs or set(states) != expected_epochs[epoch]:
                raise ValueError('disclosure family mismatch')
            if not set(states) <= audit_seen or any(v in ('PLANNED','RUNNING') for v in states.values()):
                raise ValueError('premature disclosure')
            disclosed[epoch] = event
    if set(disclosed) != set(expected_epochs):
        raise ValueError('missing disclosure')
    entries = []; native = []
    terms = config['scoped_candidate']['science']['prices']
    for row in rows:
        if row['package'] == 'DP':
            n = row['native']
            if n['interval_granularity'] != 'per-step' or n['interval_count'] != 128 or len(n['interval_mapping']) != 128:
                raise ValueError('native interval coverage drift')
            from sevc.committee.depol_arbitration import arbitrate_distances
            if n['slow']:
                expected = arbitrate_distances(n['slow']['distance_matrices'], n['epsilon'])
                if any(n['final'][k] != v for k,v in expected.items()):
                    raise ValueError('native arbitration drift')
            native.append({'dataset':row['dataset'],'block':row['block'],'invalid':row['invalid'],
                'behavior':row['behavior'],'final':n['final'],'slow_path':n['slow'] is not None,
                'native_wall_seconds':row['native_wall_seconds'],'independent_blocks_per_dataset':1})
            continue
        audits = json.loads((root/'reference-audits'/f"{row['unit_id']}.json").read_text())
        receipts = audits['production_receipts']
        taskmap = {t['task_id']:t for t in row['tasks']}
        answers = tuple((t['task_id'],t['probe_answer']) for t in row['tasks'] if t['role'] != 'production')
        refs = OwnerProbeReferences(answers)
        for assignment in row['assignments']:
            aid = assignment['assignment_id']
            if revealed.get(aid) != assignment['report']:
                raise ValueError('unit report not revealed')
            report = CommittedVerifierReport(**assignment['report'])
            settlement = assignment['settlement']
            recomputed = asdict(settle_service(report,refs,cost_seconds=settlement['verifier_cost'],
                effort_fraction=sum(x['replayed'] for x in assignment['execution'])/40,
                fee=2.5,bond=.5,failure_threshold=1,require_complete_probes=True))
            for key in ('status','accepted_report','service_fee','slashed_bond','refundable_bond'):
                if settlement[key] != recomputed[key]: raise ValueError('settlement recomputation differs: '+key)
            verdicts = dict(zip(report.ordered_segment_ids, report.verdicts))
            wrong = [t['task_id'] for t in row['tasks'] if t['role']=='production'
                     and verdicts[t['task_id']] != receipts[t['source_sha256']]['passed']]
            reference_row = next(a for a in audits['assignments'] if a['assignment_id']==aid)
            if sorted(wrong) != sorted(reference_row['wrong_task_ids']): raise ValueError('production audit mismatch')
            cap = config['detection_supplement']['billing_cap_seconds'][row['dataset']]
            entry = {k:row[k] for k in ('unit_id','dataset','method','phase','block','invalid','behavior')}
            capped=capped_observation(settlement,cap,2.5,.5,.01)
            saved=next(x for x in row['capped_billing'] if x['assignment_id']==aid)
            if row['billing_deadline_seconds']!=cap or any(saved[k]!=v for k,v in capped.items()):
                raise ValueError('capped billing recomputation differs')
            entry.update(capped,wrong_production=bool(wrong))
            entry['prices'] = [{'fee':f,'cost':c,**capped_observation(settlement,cap,f,.5,c)}
                              for f in terms['fee_sensitivity'] for c in terms['cost_per_second']]
            if row['behavior']=='cheap-recognizer-k32':
                rec=row['recognizer']; labels=[int(taskmap[t]['role']!='production' and taskmap[t]['probe_answer'] is False) for t in rec['task_ids']]
                actual={t['task_id'] for t in assignment['execution'] if t['replayed']}
                if len(rec['selected'])!=32 or actual!=set(rec['selected']) or rec['online_fit_count']!=0:
                    raise ValueError('recognizer execution mismatch')
                entry.update(test_auc=float(roc_auc_score(labels,rec['scores'])),
                    challenge_coverage=sum(y for t,y in zip(rec['task_ids'],labels) if t in actual))
            entries.append(entry)
    selections=[]
    for dataset in (() if participation_only else config['datasets']):
        for method in (config['scoped_candidate']['science']['methods'][m] for m in ('R','G')):
            histories={split:[] for split in ('development','validation')}
            # Match execution ordering, which is seed-sorted within each phase.
            for split in histories:
                units=sorted([r for r in rows if r['dataset']==dataset and r['method']==method and r['phase']==split],key=lambda r:(r['seed'],r['invalid']))
                for row in units:
                    h=json.loads((root/'recognizer'/f"history-{row['unit_id']}.json").read_text())
                    tasks={t['task_id']:t for t in row['tasks']}
                    labels=[int(tasks[i]['role']!='production' and tasks[i]['probe_answer'] is False) for i in h['task_ids']]
                    if set(h['task_ids'])!=set(tasks) or h['labels']!=labels or not h['disclosed']:
                        raise ValueError('history/raw-task binding mismatch')
                    histories[split].append(h)
            receipt=json.loads((root/'recognizer'/f'selection-{dataset}-{method}.json').read_text())
            if receipt['model_sha256']!=sha256_file(root/'recognizer'/f'model-{dataset}-{method}.pkl'):
                raise ValueError('frozen model hash mismatch')
            for split in histories:
                if receipt[split+'_sha256']!=history_digest(histories[split]): raise ValueError('history digest mismatch')
            # Deterministic reconstruction uses frozen pre-test data only.
            _, rebuilt=select_development_models(histories['development'],histories['validation'],
                config['detection_supplement']['component_models'],config['detection_supplement']['random_state'])
            for old,new in zip(receipt['candidates'],rebuilt['candidates']):
                if any(old[k]!=new[k] for k in ('feature_variant','model','reject_coverage','per_service_reject_coverage','auc')):
                    raise ValueError('selection reconstruction mismatch')
            if any(receipt['selected'][k]!=rebuilt['selected'][k] for k in ('model','feature_variant')):
                raise ValueError('selected model mismatch')
            for row in rows:
                if (row['dataset'],row['method'],row['behavior'])==(dataset,method,'cheap-recognizer-k32'):
                    rec=row['recognizer']
                    if rec['selection_sha256']!=history_digest(receipt) or set(rec['task_ids'])&set(receipt['training_task_ids']+receipt['selection_task_ids']):
                        raise ValueError('test selection/label separation failed')
            selections.append({'dataset':dataset,'method':method,'selected':receipt['selected']})
    alpha=config['detection_supplement']['alpha']; endpoints=[]; participation=[]
    test=[e for e in entries if e['phase']=='test']
    for dataset in config['datasets']:
        for method in (config['scoped_candidate']['science']['methods'][m] for m in ('R','G')):
            selected=[e for e in test if e['dataset']==dataset and e['method']==method]
            lookup={(e['block'],e['invalid'],e['behavior']):e for e in selected}
            blocks=sorted({e['block'] for e in selected})
            if participation_only:
                if blocks!=list(range(config['fixed_design']['counts']['test'])) or len(lookup)!=2*len(blocks):
                    raise ValueError('incomplete honest block pairs')
            else:
                differences=[sum(int(lookup[b,i,'cheap-recognizer-k32']['paid'])-int(lookup[b,i,'uniform-k32']['paid']) for i in (0,1)) for b in blocks]
                endpoint={'dataset':dataset,'method':method,'differences':differences,**sign_test(differences),'alpha':alpha}
                endpoint['advantage_detected']=endpoint['p_one_sided_ties_nonwins']<=alpha
                endpoints.append(endpoint)
            cap=config['detection_supplement']['billing_cap_seconds'][dataset]
            for f in terms['fee_sensitivity']:
                for c in terms['cost_per_second']:
                    values=[sum(next(p['capped_billing_utility'] for p in lookup[b,i,'honest']['prices'] if p['fee']==f and p['cost']==c) for i in (0,1))/2 for b in blocks]
                    lower=sum(values)/len(values)-(f+.5+c*cap)*math.sqrt(math.log(1/alpha)/(2*len(values)))
                    participation.append({'dataset':dataset,'method':method,'fee':f,'cost':c,'blocks':len(values),'lower':lower,'positive':lower>=0,'scope':'capped billing only'})
    learning=[]; phase_rows=list(_records(root/'phase-timing.jsonl'))
    for choice in selections:
        dataset,method=choice['dataset'],choice['method']
        history_entries=[e for e in entries if e['dataset']==dataset and e['method']==method and e['phase'] in ('development','validation')]
        S=sum(p['seconds'] for p in phase_rows if p['phase'].startswith('development_')
              and ((p.get('dataset'),p.get('method'))==(dataset,method) or
                   (by_id.get(p.get('unit_id'),{}).get('dataset'),by_id.get(p.get('unit_id'),{}).get('method'))==(dataset,method)))
        H=sum(e['billed_seconds'] for e in history_entries); H_actual=sum(e['actual_seconds'] for e in history_entries)
        historical_net=sum(e['service_fee']-e['slashed_bond'] for e in history_entries)
        learning.append({**choice,'learning_seconds':S,'historical_billed_seconds':H,'historical_actual_seconds':H_actual,'historical_net_payment':historical_net})
        for e in test:
            if (e['dataset'],e['method'],e['behavior'])==(dataset,method,'cheap-recognizer-k32'):
                e['amortization']=[{'J':j,'fee':p['fee'],'cost':p['cost'],
                    'capped_billing_with_learning_history':p['capped_billing_utility']-p['cost']*(S+H)/j,
                    'uncapped_compute_with_learning_history':p['uncapped_compute_utility']-p['cost']*(S+H_actual)/j}
                    for j in config['detection_supplement']['amortization_services'] for p in e['prices']]
    full_cost_inventory(root,config)
    result={'status':'RAW_RECOMPUTATION_PASS_AWAITING_INDEPENDENT_RETURN_AUDIT','units':len(rows),
        'entries':entries,'P1':endpoints,'honest_capped_billing_bounds':participation,'native':native,
        'learning':learning,'uniform_k32_wrong_paid_analytic':math.comb(35,28)/math.comb(40,32),
        'paper_ready':False,'actual_compute_utility_bounded':False}
    if participation_only:
        # Primary endpoint only: honest capped-billing lower bound at the main price, per cell.
        result.pop('P1'); result.pop('learning')
        result['status']='RAW_RECOMPUTATION_PASS_AWAITING_INDEPENDENT_RETURN_AUDIT'
        result['primary_main_price']=[b for b in participation if b['fee']==2.5 and b['cost']==.01]
        write_json(root/'participation-summary.json',result)
        return result
    write_json(root/'detection-summary.json',result)
    return result
