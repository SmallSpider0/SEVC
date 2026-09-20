"""Same-evidence checks and claim-wise local diagnostics; no extra samples."""
from collections import defaultdict
from dataclasses import asdict
import itertools
import json
import math
from pathlib import Path

from scipy.stats import binom
from sevc.core.artifacts import write_json
from sevc.evaluation.scoped_statistics import uniform_admission_probability
from sevc.evaluation.recovery_graph_audit import audit_trace, graph_conflicts, exhaustive_witness
from sevc.incentives.verifier_protocol import CommittedVerifierReport, SealedSegmentTruth, SealedEvaluationTruth, report_commitment
from sevc.verification.verifier_task_policies import settle_threshold_assignment
from sevc.committee.executed_recovery import execute_recovery


def analytical_screen(lock):
    rows=[]
    for r in lock['strategy_registry']['analytical_probe_counts']:
        for k in range(33+r):
            p=uniform_admission_probability(k,32+r,r//2)
            for fee,c,beta,mult in itertools.product(lock['cost_screen']['fee_grid'],
                    lock['cost_screen']['cost_grid'],lock['sensitivity']['honest_false_failure_beta'],
                    lock['sensitivity']['cost_multipliers']):
                rows.append({'total_probes':r,'rejecting_probes':r//2,'k':k,'admission':p,
                    'wrong_one_production_admission':p*(32+r-k)/(32+r-r//2),
                    'fee':fee,'cost_per_second':c,'beta':beta,'cost_multiplier':mult,
                    'normalized_honest_seconds':1,'assumed_deviation_seconds':k/(32+r),
                    'IR_assumed':fee-(fee+.5)*beta-c*mult,
                    'advantage_assumed':(fee+.5)*(1-p-beta)-c*mult*(1-k/(32+r)),
                    'time_kind':'assumed normalized proportional cost, not measured'})
    alpha=.05/54
    minimum=math.ceil(math.log(alpha)/math.log(.9))
    max_fail=max(f for f in range(97) if binom.cdf(f,96,.1)<=alpha)
    return {'analytical_uniform_grid':rows,'minimum_all_success_calibration_n':minimum,
        'single_observation_best_lower':alpha,'single_block_zero_error_upper95':.95,
        'formal_60_zero_error_scenarios':[{'block_error_rate':p,'zero_errors_probability':(1-p)**60}
            for p in lock['sensitivity']['block_error_rates']],
        'calibration_96_scenarios':[{'success_rate':p,'single_bound_pass_probability':float(binom.cdf(max_fail,96,1-p))}
            for p in lock['sensitivity']['calibration_success_rates']],
        'joint_54_bound_probability':None,'maximum_failures_at_96':max_fail,
        'J':None,'economic_gate':'HOLD_UNKNOWN_AMORTIZATION_HORIZON',
        'probabilities_are_scenarios_not_estimates':True,'formal_success_probability':None}


def structural_screen(clock):
    probes=tuple(f'p{i}' for i in range(8)); answers=(True,)*4+(False,)*4
    def report(jid,vid,ids=('target',),values=(True,)):
        return CommittedVerifierReport.create(scenario_id='local-structure',verifier_id=vid,
            job_id=jid,ordered_segment_ids=ids,verdicts=values,nonce=jid+vid)
    masks=[]
    for mask in range(256):
        truth=SealedEvaluationTruth('local-structure',tuple(SealedSegmentTruth(k,v,True,
            technical_failure=not bool(mask & 1<<i)) for i,(k,v) in enumerate(zip(probes,answers))))
        s=settle_threshold_assignment(report('mask',str(mask),probes,answers),truth,
            failure_threshold=1,fee=2.5,bond=.5,cost=0,effort=0,sentinel_only=True,required_probe_ids=probes)
        masks.append({'mask':mask,'settlement':asdict(s)})
    def callback(jid,vid):
        r=report(jid,vid)
        s=settle_threshold_assignment(r,SealedEvaluationTruth('local-structure',(SealedSegmentTruth('target',True,True),)),
            failure_threshold=1,fee=2.5,bond=.5,cost=0,effort=0)
        return r,s,[]
    def trace(policy,missing,graph='all-compatible',deadline=1e6,reserve=0):
        return execute_recovery(policy=policy,fault='structural',seed=0,required_ids=('target',),
            execute=callback,clock=clock,timeout_seconds=.001,deadline_seconds=deadline,
            virtual_fault_wait=True,missing_ids=missing,service_fee=2.5,service_bond=.5,
            public_conflicts=graph_conflicts(graph),owner_cost_reserve_per_assignment=reserve,
            minimum_pass_count=3)
    responses=[]
    for missing in itertools.combinations([f'v{i}' for i in range(9)],2):
        t=trace('all-response-certified-ecs',missing)
        responses.append({'missing':missing,'trace':t,'audit':audit_trace(t)})
    comparisons=[]
    for graph,missing in itertools.product(('all-compatible','reserve-specialists-j0',
            'reserve-specialists-j1','reserve-isolated-from-j1'),((),('v0','v3'),('v0','v3','v6'))):
        methods={}
        for policy in ('current-certified-ecs','fixed-order-same-reserve'):
            t=trace(policy,missing,graph)
            methods[policy]={'trace':t,'audit':audit_trace(t)}
        comparisons.append({'graph':graph,'missing':missing,'methods':methods,'oracle':exhaustive_witness(graph,missing)})
    constraints=[]
    for name,missing,deadline,reserve in (
        ('replacement_failure',('v0','v3','v6'),1e6,20.),
        ('deadline_exhaustion',('v0','v3'),1e-9,20.),
        ('insufficient_response',('v0','v1','v3','v4'),1e6,20.)):
        t=trace('current-certified-ecs',missing,deadline=deadline,reserve=reserve)
        constraints.append({'name':name,'trace':t,'audit':audit_trace(t)})
    return {'probe_masks':masks,'response_sets':responses,'greedy_comparisons':comparisons,
        'constraint_cases':constraints,'scope':'finite constructed fixtures; no real-service or independent-block count',
        'independent_replication':False}


def readlines(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x]


def serial_role_costs(online_seconds, phases):
    """Disjoint role time on the registered serial Python execution lane."""
    totals=defaultdict(float)
    for phase in phases:
        if 'exclusive_seconds' not in phase:
            raise ValueError('nested role accounting requires exclusive phase records')
        totals[phase['role']]+=phase['exclusive_seconds']
    if sum(totals.values()) > online_seconds + 1e-5:
        raise ValueError('exclusive measured phases exceed the online window')
    verifier,wait=totals['verifier'],totals['wait']
    return {'verifier_seconds':verifier, 'wait_seconds':wait,
            'owner_seconds':online_seconds-verifier-wait,
            'unattributed_owner_glue_seconds':max(0.,online_seconds-sum(totals.values())),
            'exclusive_instrumented_by_role':dict(totals)}


def inherited_reference_transfers(root, predecessor_root, sealed_manifest_path):
    """Authenticate a retained prefix and its predecessor's backend-transfer suffix."""
    from sevc.core.artifacts import sha256_file
    from sevc.experiments.submission_continuation import retained_log_row
    from sevc.verification.reference_acquisition import commitment
    root, parent = Path(root), Path(predecessor_root)
    receipt=json.loads((root/'continuation-receipt.json').read_text())
    manifest=json.loads(Path(sealed_manifest_path).read_text())
    assert sha256_file(Path(sealed_manifest_path))==receipt['manifest_sha256'], 'parent manifest drift'
    assert manifest['predecessor_root']==receipt['predecessor_root'], 'parent identity drift'
    indexed={r['path']:r['sha256'] for r in manifest['files']}
    for name in ('continuation-receipt.json','assignment-audit.jsonl'):
        assert sha256_file(parent/name)==indexed[receipt['predecessor_root']+'/'+name], 'parent file drift'
    old=json.loads((parent/'continuation-receipt.json').read_text())
    events=readlines(parent/'assignment-audit.jsonl')
    current=readlines(root/'assignment-audit.jsonl')
    completed=set(receipt['completed_units_reused_byte_exact'])
    all_ids={u['unit_id'] for u in json.loads((root/'protocol-lock.json').read_text())['expanded_units']}
    selected=[(i,e) for i,e in enumerate(events) if retained_log_row(e,completed,all_ids)]
    retained=receipt['effective_log_selection']['assignment-audit.jsonl']['retained']
    assert len(selected)==retained and [e for _,e in selected]==current[:retained], 'inherited prefix drift'
    boundary=old['effective_log_selection']['assignment-audit.jsonl']['retained']
    prior=set(old['predecessor_source_hashes']) & set(receipt['predecessor_source_hashes'])
    return {commitment(e):{'parent_event_index':i,'current_event_index':j,
        'parent_manifest_sha256':receipt['manifest_sha256'],
        'parent_receipt_sha256':sha256_file(parent/'continuation-receipt.json')}
        for j,(i,e) in enumerate(selected) if i>=boundary and
        e.get('event')=='reference-replay' and e['source_sha256'] in prior}


def reference_answers(units, audit_events, continuation=None, inherited_transfers=None):
    """Preserve measured owner decisions; isolate later backend transfer evidence."""
    answers={};disagreements=[]
    for unit in units:
        for sid,answer in unit.get('owner_direct_answers',{}).items():
            if sid in answers and answers[sid]!=answer:
                raise AssertionError('owner reference disagreement')
            answers[sid]=answer
    inherited=(continuation or {}).get('effective_log_selection',{}).get('assignment-audit.jsonl',{}).get('retained',0)
    prior=set((continuation or {}).get('predecessor_source_hashes',[]))
    from sevc.verification.reference_acquisition import commitment
    for index,event in enumerate(audit_events):
        if event.get('event')!='reference-replay':continue
        sid=event['source_sha256'];answer=event['receipt']['passed']
        if sid in answers and answers[sid]!=answer:
            historical=(inherited_transfers or {}).get(commitment(event))
            historical_valid=historical is not None and historical['current_event_index']==index
            if not (continuation and (index>=inherited or historical_valid) and sid in prior):
                raise AssertionError('source reference disagreement')
            disagreements.append({'source_sha256':sid,'original_owner_answer':answers[sid],
                'continuation_reference_answer':answer,'event_index':index,'event':event,
                'route':'HOLD_BACKEND_REFERENCE_TRANSFER','primary_old_answer_relabelled':False,
                'inherited_transfer_provenance':historical if historical_valid else None})
        else:answers[sid]=answer
    return answers,disagreements


def audit_and_summarize(root, config, profile, *, artifact_root=None, reference_predecessor=None,
                        reference_predecessor_manifest=None, allow_partial=False):
    root=Path(root); lock=json.loads((root/'protocol-lock.json').read_text())
    if config.get('server_only'):
        from sevc.core.replay_environment import validate_environment, require_continuation, require_source, validate_implementation_bridge
        provenance = json.loads((root/'provenance.json').read_text())['configuration']
        policy = {**config['server_only'], 'environment_lock_path':str(root/'replay-environment-lock.json')}
        environment_id = validate_environment({'change_id':config.get('change_id'),'server_only':policy}, provenance['_runtime_replay_environment'])
        bridge=validate_implementation_bridge(config,provenance['_runtime_replay_environment'],artifact_root=root)
        from sevc.core.evidence_rows import execution_identity_rows
        require_continuation({'source_groups':{}},
            execution_identity_rows([json.loads(p.read_text()) for p in (root/'units').glob('*.json')],lock['expanded_units']), environment_id, bridge)
        for bank in readlines(root/'source-task-identities.jsonl'):
            for source in bank['sources']:
                require_source(source, environment_id, bridge)
    destination=root if artifact_root is None else Path(artifact_root)
    if destination != root:
        destination.mkdir(parents=True,exist_ok=False)
    units=[json.loads(p.read_text()) for p in (root/'units').glob('*.json')]
    expected={u['unit_id']:u for u in lock['expanded_units']}
    present = {u['unit_id'] for u in units}
    assert present <= set(expected), 'unregistered cells'
    assert allow_partial or present==set(expected), 'registered cells not accounted for'
    missing = [expected[uid] for uid in sorted(set(expected)-present)]
    for u in units:
        assert all(u[k]==v for k,v in expected[u['unit_id']].items()), 'unit identity drift'
    lifecycle=readlines(root/'report-lifecycle.jsonl'); committed={}
    for event in lifecycle:
        aid=event['assignment_id']
        if event['phase']=='COMMITTED':
            assert aid not in committed, 'duplicate commitment'
            committed[aid]=event['commitment']
        else:
            r=CommittedVerifierReport(**event['report'])
            assert r.commitment==committed[aid] and r.commitment == report_commitment(r.scenario_id,r.verifier_id,r.ordered_segment_ids,r.verdicts,r.nonce,r.job_id), 'commit/reveal drift'
    banks=readlines(root/'source-task-identities.jsonl')
    truth={s['proof_sha256']:s['trainer_mutation'] is None for b in banks for s in b['sources']}
    phases=readlines(root/'phase-timing.jsonl'); by_uid=defaultdict(list)
    for p in phases:
        if p.get('phase_scope')=='online': by_uid[p.get('unit_id')].append(p)
    audit_events=readlines(root/'assignment-audit.jsonl')
    views={e['assignment_id']:e for e in audit_events if e.get('event')=='assignment-information-view'}
    continuation_path=root/'continuation-receipt.json'
    continuation=json.loads(continuation_path.read_text()) if continuation_path.exists() else None
    inherited_transfers=(inherited_reference_transfers(root,reference_predecessor,reference_predecessor_manifest)
        if reference_predecessor is not None else None)
    actual_source_answers,backend_disagreements=reference_answers(units,audit_events,continuation,inherited_transfers)
    expected_source_answers=dict(truth)
    truth.update(actual_source_answers)
    raw=[]; costs=[]; counterexamples=[]; recoveries=[]; reference=[]
    for u in units:
        if not u['issued']: continue
        # Every job may have its own opaque task IDs; include both delivery records.
        maps={t['task_id']:t for t in u['tasks']}
        for e in audit_events:
            if e.get('event')=='job-delivery-ready' and e.get('job_id','').startswith(u['unit_id']):
                maps.update({t['task_id']:t for t in e['tasks']})
        for a in u['assignments']:
            r=CommittedVerifierReport(**a['report']); verdict=dict(zip(r.ordered_segment_ids,r.verdicts))
            assert r.commitment == report_commitment(r.scenario_id,r.verifier_id,r.ordered_segment_ids,r.verdicts,r.nonce,r.job_id) and committed[a['assignment_id']]==r.commitment
            assert views[a['assignment_id']]['view']=='verifier-delivered-only' and views[a['assignment_id']]['trainer_cache_size']==0, 'private information leak'
            tasks={k:maps[k] for k in r.ordered_segment_ids}
            probe_mismatches=sum(verdict[k]!=t['probe_answer'] for k,t in tasks.items() if t['role']!='production')
            s=a['settlement']; passed=probe_mismatches==0
            assert s['status']==('PASS' if passed else 'FAIL_CONFIRMED'), 'raw admission mismatch'
            assert s['service_fee']==(2.5 if passed else 0) and s['slashed_bond']==(0 if passed else .5), 'payment mismatch'
            wrong=[k for k,t in tasks.items() if t['role']=='production' and verdict[k]!=truth[t['source_sha256']]]
            full=sum(d['replayed'] for d in a['execution'])
            if a['behavior'].startswith('uniform-k'):
                assert full==int(a['behavior'].split('k')[-1]), 'uniform executed k drift'
            if a['behavior']=='honest': assert full==40
            row={'dataset':u['dataset'],'block':0,'seed':u['seed'],'anchor':u.get('anchor',0),
                'steps':u['steps'],'batch_size':u['batch_size'],'invalid':u['invalid'],
                'unit_id':u['unit_id'],**a,'wrong_production_ids':wrong,'wrong_report_admitted':passed and bool(wrong),
                'full_replayed_tasks':full,'probe_mismatches':probe_mismatches}
            raw.append(row)
            if passed and wrong or a['behavior']=='honest' and not passed:
                counterexamples.append({'unit_id':u['unit_id'],'assignment_id':a['assignment_id'],
                    'dataset':u['dataset'],'behavior':a['behavior'],'kind':'WRONG_REPORT_ADMITTED' if passed else 'HONEST_FALSE_FAILURE',
                    'known_uniform_conditional_risk':a['behavior'].startswith('uniform-k'),
                    'population_IC_counterexample_established':False,'wrong_production_ids':wrong})
        for probe in u.get('probe_replays',[]):
            reference.append({'unit_id':u['unit_id'],'task_id':probe['task_id'],
                'agrees':probe['reference']==probe['actual']['passed'],'actual':probe['actual']})
        if 'recovery' in u:
            audit=audit_trace(u['recovery'],u['assignments'])
            routes={j:d['route'] for j,d in u['recovery']['decisions'].items()}
            expected_route='accept' if all(truth[s] for s in u['production_source_ids']) else 'reject'
            wrong_routes={j:r for j,r in routes.items() if r not in ('safe-defer',expected_route)}
            recoveries.append({'unit_id':u['unit_id'],'dataset':u['dataset'],'invalid':u['invalid'],
                'behavior':u['behavior'],'routes':routes,'wrong_routes':wrong_routes,'audit':audit,
                'constructed_qualification':True,'actual_service_fees':sum(a['settlement']['service_fee'] for a in u['assignments'])})
            if wrong_routes: counterexamples.append({'kind':'WRONG_ECS_SETTLEMENT','unit_id':u['unit_id'],'routes':wrong_routes})
        ps=by_uid[u['unit_id']]
        disjoint=serial_role_costs(u['online_suffix_seconds'],ps)
        verifier,owner=disjoint['verifier_seconds'],disjoint['owner_seconds']
        owner += u.get('owner_input_metadata_total_charge_seconds', 0.0)
        first=u['assignments'][0] if u['assignments'] else None
        single_owner=(u['online_preparation_seconds']+dict(first['settlement']['diagnostics'])['owner_admission_seconds'] if first else owner)
        costs.append({'unit_id':u['unit_id'],'dataset':u['dataset'],'anchor':u.get('anchor',0),
            'steps':u['steps'],'batch_size':u['batch_size'],'invalid':u['invalid'],'method':u['method'],
            'behavior':u['behavior'],'package':u['package'],'owner_seconds':owner,'single_owner_seconds':single_owner,
            'preparation_seconds':u['online_preparation_seconds'],'verifier_seconds':verifier,
            'shared_trainer_prefix_seconds':u['shared_trainer_prefix_seconds'],'online_suffix_seconds':u['online_suffix_seconds'],
            'service_fees':sum(a['settlement']['service_fee'] for a in u['assignments']),
            'role_accounting':disjoint,
            'owner_cost_revision':u.get('owner_cost_revision', 'pre-owner-cost-repair'),
            'owner_input_metadata_charge_seconds':u.get('owner_input_metadata_total_charge_seconds', 0.0),
            'shared_prefix_is_independent_repeat':False})
    for name,rows in [('raw-reports.jsonl',raw),('role-costs.jsonl',costs)]:
        with (destination/name).open('x') as f:
            for r in rows: f.write(json.dumps(r,sort_keys=True)+'\n')
    cal=([] if allow_partial and not (root/'calibration-ledger.jsonl').exists()
         else readlines(root/'calibration-ledger.jsonl'))
    for r in cal:
        s=r['summary']; assert s['completed_opportunities']==1 and s['audited_admitted']<=1
        assert not r['actual_qualification'] and not r['current_block_retroactive_qualification']
        assert s['audited_admitted']==sum(o['admitted'] and o['correct'] is not None for o in r['observations'])
    from sevc.reputation.calibration import PaidCalibrationLedger
    from sevc.incentives.verifier_protocol import VerifierSettlement
    from sevc.committee.executed_recovery import required_segment_decision
    raw_by_assignment={r['assignment_id']:r for r in raw}
    qualification_checks=[]
    for row in cal:
        actual=raw_by_assignment[row['assignment_id']]
        ledger=PaidCalibrationLedger(row['verifier_id'],22.5,2.5,20.)
        observation=row['observations'][0]
        ledger.reserve(row['assignment_id'],observation['independent_block_id'])
        ledger.record(row['assignment_id'],VerifierSettlement(**actual['settlement']),
            timely=observation['timely_admitted'] if observation['admitted'] else True,
            reference_correct=observation['correct'],
            reference_receipt_sha256=observation['reference_receipt_sha256'],owner_cost=observation['owner_cost'])
        criteria=dict(minimum_correctness=.9,minimum_availability=.9,alpha=.05/54)
        qualified=ledger.production_admissible(**criteria,independent_stationary_blocks=False)
        best_case=ledger.production_admissible(**criteria,independent_stationary_blocks=True)
        assert qualified==row['actual_qualification']==False and not best_case
        report=CommittedVerifierReport(**actual['report'])
        production=tuple(row['owner_receipts'])
        decision=required_segment_decision(production,[report],[]) if production else None
        assert decision is None
        qualification_checks.append({'dataset':row['dataset'],'verifier_id':row['verifier_id'],
            'assignment_id':row['assignment_id'],'qualified':qualified,'iid_best_case_qualified':best_case,
            'no_qualified_committee_decision':decision,'new_service_calls':0,'new_blocks':0})
    write_json(destination/'calibration-gate-check.json',qualification_checks)
    observed_utilities=[]
    for a in raw:
        peers=[h for h in raw if h['unit_id']==a['unit_id'] and h['behavior']=='honest'
               and h['report']['job_id']==a['report']['job_id']]
        for fee,c in itertools.product(lock['cost_screen']['fee_grid'],lock['cost_screen']['cost_grid']):
            def utility(row):
                st=row['settlement']
                return fee*int(st['accepted_report'])-(.5 if st['slashed_bond'] else 0)-c*st['verifier_cost']
            observed_utilities.append({'assignment_id':a['assignment_id'],'unit_id':a['unit_id'],
                'dataset':a['dataset'],'behavior':a['behavior'],'invalid':a['invalid'],'anchor':a['anchor'],
                'fee_scenario':fee,'cost_per_second':c,'observed_utility':utility(a),
                'observed_minus_first_same_job_honest':utility(a)-utility(peers[0]) if peers else None,
                'expected_IC_established':False,'new_measurements':False})
    write_json(destination/'observed-utility-grid.json',observed_utilities)
    methods=config['scoped_candidate']['science']['methods']
    pairs=[]
    for d,anchor,steps,batch,invalid in sorted({(c['dataset'],c['anchor'],c['steps'],c['batch_size'],c['invalid']) for c in costs}):
        group={c['method']:c for c in costs if (c['dataset'],c['anchor'],c['steps'],c['batch_size'],c['invalid'])==(d,anchor,steps,batch,invalid)
               and c['behavior']=='honest' and c['package'] in ('M1','M5','M7')}
        if set(methods.values())<=group.keys() and len({group[methods[k]]['owner_cost_revision'] for k in ('R','G','O')}) == 1:
            r,g,o=(group[methods[k]] for k in ('R','G','O'))
            pairs.append({'dataset':d,'anchor':anchor,'steps':steps,'batch_size':batch,'invalid':invalid,
                'owner_cost_revision':r['owner_cost_revision'],
                'R_O':r['single_owner_seconds']/o['owner_seconds'],
                'R_G_preparation':r['preparation_seconds']/g['preparation_seconds'],
                'R_owner_seconds':r['single_owner_seconds'],'O_owner_seconds':o['owner_seconds'],
                'R_preparation_seconds':r['preparation_seconds'],'G_preparation_seconds':g['preparation_seconds'],
                'reference_cost_scope':'actual reference acquisition, challenge construction/validation and delivery preparation',
                'formal_statistical_gate':False})
    sensitivity=[]
    for a in raw:
        if not a['behavior'].startswith('uniform-k'): continue
        peers=[h for h in raw if h['unit_id']==a['unit_id'] and h['behavior']=='honest' and h['report']['job_id']==a['report']['job_id']]
        if not peers: continue
        honest=peers[0]['settlement']['verifier_cost']; cost=a['settlement']['verifier_cost']
        p=uniform_admission_probability(a['full_replayed_tasks'])
        for fee,c,beta,mult in itertools.product(lock['cost_screen']['fee_grid'],lock['cost_screen']['cost_grid'],
                lock['sensitivity']['honest_false_failure_beta'],lock['sensitivity']['cost_multipliers']):
            sensitivity.append({'dataset':a['dataset'],'unit_id':a['unit_id'],'behavior':a['behavior'],
                'invalid':a['invalid'],'fee':fee,'cost_per_second':c,'beta':beta,'multiplier':mult,
                'honest_seconds':honest,'deviation_seconds':cost,'exact_uniform_admission':p,
                'IR_cost_plugin':fee-(fee+.5)*beta-c*mult*honest,
                'advantage_cost_plugin':(fee+.5)*(1-p-beta)-c*mult*(honest-cost),
                'population_expectation_estimated':False})
    forecast=json.loads((root/'forecast-sensitivity.json').read_text()); forecast['measured_cost_plugin']=sensitivity
    write_json(destination/'forecast-sensitivity.json',forecast)
    structures=json.loads((root/'structural-evidence.json').read_text())
    for row in structures['probe_masks']:
        s=row['settlement']; assert s['status']==('PASS' if row['mask']==255 else 'TECHNICAL_FAILURE')
        assert s['service_fee']==(2.5 if row['mask']==255 else 0) and s['slashed_bond']==0
    increments=[]
    for row in structures['greedy_comparisons']:
        count={k:audit_trace(v['trace'])['completed_jobs'] for k,v in row['methods'].items()}
        increments.append({'graph':row['graph'],'missing':row['missing'],**count,
            'oracle':row['oracle']['maximum_completed_jobs'],
            'increment':count['current-certified-ecs']-count['fixed-order-same-reserve']})
    bridge=[u for u in units if u.get('anchor',0)]; core=[u for u in units if not u.get('anchor',0)]
    holds=[{'unit_id':u['unit_id'],'dataset':u['dataset'],'anchor':u.get('anchor',0),'steps':u['steps'],
        'batch_size':u['batch_size'],'status':u['status']} for u in units if not u['issued'] and not u.get('protocol_measured',False)]
    holds.extend({**u, 'status':'HOLD_NOT_RUN_IN_PARTIAL_ATTEMPT'} for u in missing)
    stop=any(not e.get('known_uniform_conditional_risk',False) for e in counterexamples)
    readiness={'change_id':config['change_id'],'route':'STOP_REPAIR' if stop else 'HOLD',
        'terminal':'INCOMPLETE' if missing else ('ACCEPTED_NEGATIVE' if stop else 'PASS'),
        'terminal_scope':'partial evidence only' if missing else 'completed local diagnostic workflow only; scientific support below',
        'paper_route':'NO_PAPER_CHANGE','formal_launch_ready':False,'formal_success_probability':None,
        'registered_blocks_per_dataset':{d:1 for d in lock['dataset_order']},
        'blocks_per_dataset':{d:int(any(b.get('dataset')==d and 'shared_trainer_prefix_seconds' in b for b in banks)) for d in lock['dataset_order']},'independent_replication':False,
        'core_units_measured':sum(u['issued'] for u in core),'core_units_registered':len(core),
        'bridge_units_measured':sum(u['issued'] for u in bridge),'bridge_units_registered':len(bridge),
        'matched_cost_pairs':pairs,'reference_agreement':reference,'counterexamples':counterexamples,
        'controlled_real_service_settlements':recoveries,'calibration_observations':len(cal),
        'calibrated_qualified_identities':0,'real_qualified_production':'DEFER',
        'calibration_economics':'HOLD: one observation per identity and no justified amortization horizon',
        'content_attack_scope':'two fixed public rules; authenticated personal legal history not imported; no current-label fit',
        'scheduling_increments':increments,'holds':holds,
        'cross_device_transfer':'HOLD_PARTIAL_BACKEND_TRANSFER' if continuation else 'NOT_TESTED',
        'backend_reference_disagreements':backend_disagreements,
        'continue_recommendation':'repair demonstrated core issue before further evidence' if stop else
             'target missing qualification, economics and scale evidence with a new bounded protocol; no automatic expansion'}
    write_json(destination/'claim-readiness.json',readiness)
    write_json(destination/'counterexamples.json',{'observations':counterexamples,'uniform_events_are_known_conditional_risk':True,
        'technical_failures':[],'missing_evidence':holds,'no_negative_rows_removed':True})
    write_json(destination/'scale-transfer.json',{'status':'LOCAL_TRANSFER_SUPPORTED' if bridge and not missing and all(u['issued'] for u in bridge) and not stop else 'HOLD_SCALE_TRANSFER',
        'cells':[{'unit_id':u['unit_id'],'dataset':u['dataset'],'anchor':u['anchor'],'steps':u['steps'],
                  'batch_size':u['batch_size'],'status':u['status']} for u in bridge],
        'matched_cost_pairs':[p for p in pairs if p['anchor']], 'independent_replication':False,'stage_population_probability':None})
    write_json(destination/'independent-audit.json',{'status':'SAME_EVIDENCE_INTEGRITY_PASS','independent_replication':False,
        'independent_external_reviewer':False,'audited_units':len(units),'audited_assignments':len(raw),
        'calibration_observations':len(cal),'structural_masks':256,'response_sets':36,
        'actual_source_replay_receipts':len(actual_source_answers),
        'backend_reference_disagreements':backend_disagreements,
        'source_expected_actual_disagreements':[sid for sid in actual_source_answers if sid in expected_source_answers and expected_source_answers[sid]!=actual_source_answers[sid]],
        'scientific_pass_implied':False,'source_tensors_reexecuted_by_independent_auditor':False})
    return readiness
