"""Identity guard and conservative fixed-design evidence inventory."""
import json
from pathlib import Path
from sevc.core.artifacts import sha256_file, write_json
from sevc.verification.reference_acquisition import commitment


def validate_activation(config,profile,root):
    from sevc.experiments.tdsc_five_rq_evidence import scientific_configuration_sha256
    if not config.get('formal_execution_authorized') or profile.get('formal') is not True:
        raise PermissionError('formal fixed-design authorization absent')
    lock_path=Path(config['protocol_lock_path']);lock=json.loads(lock_path.read_text())
    if sha256_file(lock_path)!=config['protocol_lock_sha256'] or lock.get('execution_ready') is not True:
        raise PermissionError('fixed-design lock absent or drifted')
    if lock['scientific_configuration_sha256']!=scientific_configuration_sha256(config):
        raise PermissionError('fixed configuration drift')
    if lock['output_root']!=str(root) or profile['device']!='cuda:0' or config['maximum_runner_wall_seconds']>71*3600:
        raise PermissionError('output/device/time identity invalid')
    if set(config['datasets'])!={'mnist','cifar10','cifar100'}:
        raise PermissionError('three datasets required')
    if config['change_id']=='experiment-tdsc-f-fixed-design-v2':
        f=config['fixed_design']
        if (f.get('qualification_mode')!='conditional-component-v1' or f['counts']['calibration']!=1
                or f.get('conditional_recovery_correctness')!=.9 or config['maximum_runner_wall_seconds']>47*3600):
            raise PermissionError('v2 scope drift')
    if config['change_id']=='experiment-tdsc-rq1-detection-supplement-v1':
        from sevc.evaluation.detection_supplement import validate_scope
        validate_scope(config)
    if config['change_id']=='experiment-tdsc-rq2-honest-participation-v1':
        from sevc.evaluation.detection_supplement import validate_participation_scope
        validate_participation_scope(config)
    if config['change_id']=='experiment-tdsc-rq4-overhead-supplement-v1':
        from sevc.evaluation.overhead_supplement import validate_scope
        validate_scope(config)
    units=config['fixed_design_units']
    if len({u['unit_id'] for u in units})!=len(units) or commitment(units)!=lock['matrix_commitment']:
        raise PermissionError('matrix drift')
    if any(lock['gates'].get(k) is not True for k in ('review','tests','source_integrity','fixed_samples','complete_budget')):
        raise PermissionError('required fixed-design gate absent')
    receipt=json.loads(Path(config['confirmation_receipt_path']).read_text())
    if receipt.get('status')!='AUTHOR_CONFIRMED' or receipt.get('protocol_sha256')!=config['protocol_lock_sha256'] or receipt.get('output_root')!=str(root):
        raise PermissionError('activation receipt does not bind this protocol')


def audit_and_summarize(root,config,profile):
    root=Path(root); units=config['fixed_design_units'];rows=[];errors=[]
    for u in units:
        path=root/'units'/f"{u['unit_id']}.json"
        if not path.exists():errors.append('missing:'+u['unit_id']);continue
        row=json.loads(path.read_text());rows.append(row)
        if any(row.get(k)!=u.get(k) for k in u):errors.append('identity:'+u['unit_id'])
        a=root/'reference-audits'/path.name
        if row.get('assignments'):
            if not a.exists():errors.append('missing_audit:'+u['unit_id'])
            elif json.loads(a.read_text())['unit_sha256']!=sha256_file(path):errors.append('audit_binding:'+u['unit_id'])
    summary={'status':'INTEGRITY_PASS_AWAITING_STATISTICAL_AUDIT' if not errors else 'INCOMPLETE',
        'expected_units':len(units),'completed_units':len(rows),'errors':errors,
        'measured':sum(x.get('status')=='MEASURED' for x in rows),
        'qualification_deferred':sum(x.get('status')=='DEFER_NOT_QUALIFIED' for x in rows),
        'formal_positive_scientific_claims_certified':False,
        'requires_event_by_event_analysis':124,
        'predictive_probability_was_not_a_launch_gate':True}
    full_cost_inventory(root,config)
    write_json(root/'fixed-design-integrity.json',summary)
    if errors:raise ValueError('fixed design evidence incomplete')
    return summary


def _inventory(root, units):
    """Re-read completed artifacts without trusting the runner's in-memory rows."""
    files={}; rows=[]; errors=[]
    for unit in units:
        path=root/'units'/f"{unit['unit_id']}.json"
        if not path.is_file():
            errors.append('missing:'+unit['unit_id']); continue
        row=json.loads(path.read_text()); rows.append(row)
        files[str(path.relative_to(root))]=sha256_file(path)
        if any(row.get(k)!=v for k,v in unit.items()):
            errors.append('identity:'+unit['unit_id'])
        audit=root/'reference-audits'/path.name
        if row.get('assignments'):
            if not audit.exists():errors.append('missing_audit:'+unit['unit_id'])
            else:
                files[str(audit.relative_to(root))]=sha256_file(audit)
                if json.loads(audit.read_text())['unit_sha256']!=files[str(path.relative_to(root))]:
                    errors.append('audit_binding:'+unit['unit_id'])
    return files, rows, errors


def _log_prefixes(root):
    import hashlib
    prefixes={}
    for path in sorted(root.glob('*.jsonl')):
        size=path.stat().st_size; remaining=size; digest=hashlib.sha256()
        with path.open('rb') as stream:
            while remaining:
                data=stream.read(min(1024*1024,remaining))
                if not data:raise ValueError('log prefix truncated')
                digest.update(data);remaining-=len(data)
        prefixes[path.name]={'bytes':size,'sha256':digest.hexdigest()}
    return prefixes


def stage_observations(root, config, rows):
    """Paired descriptive endpoints; scientific adjudication remains independent."""
    from collections import defaultdict
    import math
    import statistics
    selected={r['unit_id']:r for r in rows}
    roles=defaultdict(lambda:defaultdict(float))
    timing=root/'phase-timing.jsonl'
    if timing.exists():
        with timing.open() as stream:
            for line in stream:
                p=json.loads(line); uid=p.get('unit_id')
                if uid in selected and p.get('phase_scope')=='online':
                    roles[uid][p['role']]+=p.get('exclusive_seconds',0.)
    prices=config['fixed_design']['prices']; paired=defaultdict(dict); service=[]
    for row in rows:
        uid=row['unit_id']
        if row.get('behavior')=='honest' and row.get('phase') in ('production','target_bridge'):
            paired[(row['context'],row['block'])][row['method']]=roles[uid].get('owner',0.)
        for a in row.get('assignments',[]):
            settled=a['settlement']
            service.append({'unit_id':uid,'block':row['block'],'context':row['context'],
                'behavior':row['behavior'],'assignment_id':a['assignment_id'],
                'fee':settled['service_fee'],'slashed_bond':settled['slashed_bond'],
                'verifier_seconds':settled['verifier_cost'],
                'observed_utility':settled['service_fee']-settled['slashed_bond']-
                    prices['cost_per_second']*settled['verifier_cost']})
    ratios=defaultdict(list); methods=config['scoped_candidate']['science']['methods']
    for (context,block),values in paired.items():
        if methods['R'] in values and values.get(methods['O'],0)>0:
            ratios[context].append({'block':block,'ratio':values[methods['R']]/values[methods['O']]})
    owner=[]
    for context, values in ratios.items():
        n=len(values); wins=sum(v['ratio']<1 for v in values)
        owner.append({'context':context,'n':n,'paired_ratios':values,
            'median_R_over_O':statistics.median(v['ratio'] for v in values),
            'one_sided_sign_p_ties_as_nonwins':sum(math.comb(n,i) for i in range(wins,n+1))/2**n,
            'registered_nominal_alpha':.05/9,
            'scope':'online owner exclusive computation; excludes fee transfers and separately accounted startup/oracle'})
    return {'owner_pairs':owner,'service_observations':service,
        'online_exclusive_seconds_by_unit_role':dict(roles),
        'population_expected_IC':'INCONCLUSIVE_WITHOUT_VALID_COST_RANGE',
        'independent_scientific_audit':'REQUIRED',
        'paper_ready':False}


def seal_stage(root, config, phase, dataset):
    """Seal files by reference and shared log prefixes, without copying payloads."""
    root=Path(root); folder=root/'stage-seals';folder.mkdir(exist_ok=True)
    path=folder/f'{phase}-{dataset}.json'
    if path.exists():raise FileExistsError('stage seal is immutable')
    units=[u for u in config['fixed_design_units'] if u.get('phase')==phase and u['dataset']==dataset]
    files, rows, errors=_inventory(root,units)
    if errors:raise ValueError('cannot seal incomplete stage: '+str(errors[:3]))
    for extra in [root/'qualification'/f'{dataset}.json',root/'qualification'/f'{dataset}-payments.json',
                  root/'qualification'/'conditional-roster.json',
                  root/'provenance.json',root/'expanded-units.json',root/'stream-commitments.json',
                  root/'replay-environment-lock.json']:
        if extra.exists():files[str(extra.relative_to(root))]=sha256_file(extra)
    write_json(path,{'phase':phase,'dataset':dataset,'status':'SEALED_AWAITING_INDEPENDENT_SCIENTIFIC_AUDIT',
        'expected_units':len(units),'completed_units':len(rows),
        'deferred_units':sum(r.get('status')=='DEFER_NOT_QUALIFIED' for r in rows),
        'files':files,'shared_log_prefixes':_log_prefixes(root),
        'observations':stage_observations(root,config,rows),
        'dependency_seal':f'calibration-{dataset}.json' if phase=='recovery' else None,
        'qualification_mode':config['fixed_design'].get('qualification_mode','empirical'),
        'empirical_qualification_certified':False,
        'recovery_scope':'conditional component' if config['fixed_design'].get('qualification_mode')=='conditional-component-v1' else 'measured qualification required',
        'scientific_negative_does_not_trigger_rerun':True})
    seals=[folder/f'{phase}-{ds}.json' for ds in config['datasets']]
    if all(p.exists() for p in seals):
        write_json(folder/f'{phase}-all-datasets.json',{'phase':phase,
            'status':'THREE_DATASET_COLLECTION_SEALED_AWAITING_INDEPENDENT_SCIENTIFIC_AUDIT',
            'seals':{p.name:sha256_file(p) for p in seals},'paper_ready':False})


def full_cost_inventory(root, config):
    """Physical execution totals, with transfers separate from computation."""
    from collections import defaultdict
    root=Path(root); by_role=defaultdict(float); by_phase_role=defaultdict(float)
    unit_phase={u['unit_id']:u.get('phase','finite-native') for u in config['fixed_design_units']}
    with (root/'phase-timing.jsonl').open() as stream:
        for line in stream:
            row=json.loads(line); seconds=row.get('exclusive_seconds',0.)
            role=row['role'];by_role[role]+=seconds
            phase=unit_phase.get(row.get('unit_id'),row.get('phase_scope','shared-startup'))
            by_phase_role[f'{phase}/{role}']+=seconds
    fees=bonds=0.; measured_roles=defaultdict(float)
    for path in (root/'units').glob('*.json'):
        row=json.loads(path.read_text())
        for a in row.get('assignments',[]):
            s=a['settlement'];fees+=s['service_fee'];bonds+=s['slashed_bond']
            measured_roles[a['report']['verifier_id']]+=s['verifier_cost']
    value={'physical_exclusive_seconds_by_role':dict(by_role),
        'physical_exclusive_seconds_by_phase_role':dict(by_phase_role),
        'system_instrumented_seconds':sum(by_role.values()),
        'verifier_reported_compute_seconds_by_identity':dict(measured_roles),
        'fee_transfers':fees,'slashed_bond_transfers':bonds,
        'owner_computation_excludes_verifier_payments':True,
        'full_elapsed_and_uninstrumented_cost':'performance-window.json and supervisor process-window.json',
        'startup_J':('FULL_POPULATION_CERTIFICATION_STARTUP_NOT_ESTABLISHED; ONE_BLOCK_WORKFLOW_COST_ONLY'
            if config['fixed_design'].get('qualification_mode')=='conditional-component-v1'
            else 'AWAITING_INDEPENDENT_COST_SCOPE_AND_BREAK_EVEN_AUDIT'),
        'qualification_files':'qualification/',
        'notes':['Physical totals count shared source preparation once.',
                 'Verifier service totals overlap physical clocks; do not add them again.',
                 'Owner role physical totals include experimental oracle work; protocol online costs are separately reported in stage seals.',
                 'No assumption that every cost or cash endpoint improves.']}
    write_json(root/'full-cost-inventory.json',value)
    return value
