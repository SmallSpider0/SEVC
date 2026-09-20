"""Descriptive tiny-attack projection after the common independent raw-record audit."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score,balanced_accuracy_score
from sevc.core.artifacts import write_json


def projection(rows, observations, full_cost, trainer_outcomes, science):
    entries=[]
    for uid,r in rows.items():
        if r['split']!='test':continue
        for a,o in zip(r['assignments'],observations.get(uid,[])):
            out={k:r[k] for k in ('unit_id','dataset','method','invalid','block','seed','behavior')}
            s=a['settlement']; out.update(paid=o['paid'],wrong_production_admitted=o['wrong_production_admitted'],
                verifier_seconds=s['verifier_cost'],service_fee=s['service_fee'],slashed_bond=s['slashed_bond'],
                replayed_tasks=sum(t['replayed'] for t in a['execution']))
            out['utilities']={str(c):s['service_fee']-s['slashed_bond']-c*s['verifier_cost'] for c in (0,.01,.1,1)}
            if r['behavior']=='cheap-recognizer-k32':
                rec=r['recognizer']; taskmap={t['task_id']:t for t in r['tasks']}
                labels=[int(taskmap[t]['role']!='production' and taskmap[t]['probe_answer'] is False) for t in rec['task_ids']]
                select=set(rec['selected']); picked=[int(t in select) for t in rec['task_ids']]
                out.update(reject_challenge_auc=float(roc_auc_score(labels,rec['scores'])),
                    top32_balanced_accuracy=float(balanced_accuracy_score(labels,picked)),
                    rejected_challenges_replayed=sum(y and p for y,p in zip(labels,picked)),
                    reject_challenge_count=sum(labels),history_sha256=rec['history_sha256'])
            entries.append(out)
    for e in entries:
        key=(e['dataset'],e['method'],e['invalid'])
        for base in ('honest','uniform-k32'):
            other=next((x for x in entries if (x['dataset'],x['method'],x['invalid'])==key and x['behavior']==base),None)
            if other:
                e['utility_delta_vs_'+base]={c:e['utilities'][c]-other['utilities'][c] for c in e['utilities']}
                e['cost_ratio_vs_'+base]=e['verifier_seconds']/other['verifier_seconds'] if other['verifier_seconds'] else None
    return {'entries':entries,'population_inference':'NOT_ESTABLISHED_N1_TEST_BLOCK_PER_DATASET',
            'expected_IC':'NOT_ESTABLISHED','cuda_cost_inference':False,'paper_ready':False}


def audit_and_summarize(root,config,profile):
    from sevc.evaluation.scoped_result_audit import audit_and_summarize as raw_audit
    from collections import defaultdict
    from sevc.experiments.scoped_five_rq_units import unit_assignment_ids
    grouped=defaultdict(list)
    for u in config['technical_units']['fixture']:grouped[u['dataset'],u['seed']].append(u)
    schedules={g[0]['unit_id']:[a for u in g for a in unit_assignment_ids(u)] for g in grouped.values()}
    audit=raw_audit(root,config,profile,descriptive_projection=projection,disclosure_schedules=schedules)
    root=Path(root);packet=json.loads((root/'claim-linked-results.json').read_text())
    expected=config['technical_units']['fixture']
    rows=packet['units'];assert {r['unit_id'] for r in rows}=={r['unit_id'] for r in expected}
    histories={}
    for p in (root/'recognizer').glob('development-*.json'):
        h=json.loads(p.read_text());assert h['disclosed'] and h['split']=='development'
        assert h['seed']==config['tiny_discriminator']['development_seed']
        histories[p.name]=h
    for r in rows:
        if r['behavior']=='cheap-recognizer-k32' and r['issued']:
            rec=r['recognizer'];assert len(rec['selected'])==32 and len(set(rec['selected']))==32
            assert all(not (set(h['task_ids']) & set(rec['task_ids'])) for h in histories.values())
            actual={t['task_id'] for a in r['assignments'] for t in a['execution'] if t['replayed']}
            assert actual==set(rec['selected'])
    result={'status':'AUDITED_NONFORMAL_DIAGNOSTIC','new_units':len(rows),'datasets':list(config['datasets']),
            'projection':packet['descriptive_projections'],'raw_audit':audit}
    if config['change_id'].endswith('-v2'):
        result['improved_costs']=improved_cost_report(root,config,rows,result['projection']['entries'])
    write_json(root/'tiny-attack-summary.json',result)
    return result


def improved_cost_report(root,config,rows,entries):
    """Reconcile measured startup and per-service costs without free learning."""
    from collections import defaultdict
    from sevc.core.artifacts import sha256_file
    root=Path(root);by_id={r['unit_id']:r for r in rows};startup=defaultdict(float)
    phases={'development_payload_read','development_feature_extraction',
            'development_feature_extraction_stride32','development_model_selection'}
    for line in (root/'phase-timing.jsonl').read_text().splitlines():
        r=json.loads(line)
        if r['phase'] in phases:
            u=by_id[r['unit_id']];assert u['split'] in ('development','validation')
            startup[u['dataset'],u['method']]+=r['seconds']
    learning=[]
    for dataset in config['datasets']:
        for method in config['scoped_candidate']['science']['methods'].values():
            relevant=[r for r in rows if r['dataset']==dataset and r['method']==method and r['split'] in ('development','validation')]
            if not relevant:continue
            record=json.loads((root/'recognizer'/f'selection-{dataset}-{method}.json').read_text())
            assert sha256_file(root/'recognizer'/f'model-{dataset}-{method}.pkl')==record['model_sha256']
            history={split:json.loads((root/'recognizer'/f'{split}-{dataset}-{method}.json').read_text()) for split in ('development','validation')}
            from sevc.verification.cheap_discriminator import history_digest
            for split,h in history.items():
                assert h['seed']==config['tiny_discriminator'][split+'_seed'] and h['disclosed']
                assert record[split+'_sha256']==history_digest(h)
            chosen=min(record['candidates'],key=lambda r:(-r['reject_coverage'],-r['auc'],r['feature_variant']!='stride32',
                config['tiny_discriminator']['model_candidates'].index(r['model'])))
            assert chosen==record['selected'] and not record['test_used']
            H=sum(a['settlement']['verifier_cost'] for r in relevant for a in r['assignments'])
            fees=sum(a['settlement']['service_fee']-a['settlement']['slashed_bond'] for r in relevant for a in r['assignments'])
            S=startup[dataset,method]
            learning.append({'dataset':dataset,'method':method,'learning_seconds':S,
                'historical_service_seconds':H,'historical_net_payment':fees,'selected':chosen,
                'model_sha256':record['model_sha256'],'candidate_count':len(record['candidates'])})
            expected_selection=__import__('hashlib').sha256(json.dumps(record,sort_keys=True).encode()).hexdigest()
            for r in rows:
                if r['dataset']==dataset and r['method']==method and r['behavior']=='cheap-recognizer-k32':
                    rec=r['recognizer'];assert rec['selection_sha256']==expected_selection and rec['online_fit_count']==0
                    assert (rec['model'],rec['feature_variant'])==(chosen['model'],chosen['feature_variant'])
                    forbidden=set(record['training_task_ids']+record['selection_task_ids'])
                    assert not forbidden&set(rec['task_ids'])
                    assert rec['payload_cache']['peak_logical_tensor_bytes']<=config['tiny_discriminator']['cache_limit_bytes']
            for e in entries:
                if (e['dataset'],e['method'],e['behavior'])!=(dataset,method,'cheap-recognizer-k32'):continue
                e['amortization']={}
                for J in config['tiny_discriminator']['amortization_services']:
                    e['amortization'][str(J)]={
                        'learning_cost_seconds_per_service':S/J,
                        'learning_plus_history_seconds_per_service':(S+H)/J,
                        'utility_with_learning':{c:u-float(c)*S/J for c,u in e['utilities'].items()},
                        'utility_with_learning_and_history_cost':{c:u-float(c)*(S+H)/J for c,u in e['utilities'].items()},
                        'utility_with_learning_and_history_net':{c:u+(fees-float(c)*(S+H))/J for c,u in e['utilities'].items()}}
    events=[json.loads(x) for x in (root/'workload-units.jsonl').read_text().splitlines()]
    resources=[e for e in events if e.get('event')=='LOCAL_RESOURCE_SAMPLE']
    from sevc.core.scratch_budget import disk_usage
    return {'learning_accounts':learning,'amortization_services':config['tiny_discriminator']['amortization_services'],
        'scratch_limit_bytes':config['tiny_discriminator']['scratch_limit_bytes'],
        'sampled_peak_scratch_bytes':max((r['scratch']['logical_bytes'] for r in resources),default=0),
        'peak_rss_bytes':max((r['process_peak_rss_bytes'] for r in resources),default=0),
        'scratch_remaining':disk_usage(config['scratch_parent']),
        'unit_of_inference':'one independent heldout block per dataset; scenarios are paired',
        'cold_learning_plus_history_is_conservative':True,
        'historical_payments_reported_separately':True,'formal':False}
