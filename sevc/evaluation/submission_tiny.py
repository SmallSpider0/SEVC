"""Additional same-package audits for the submission local adaptation."""
from pathlib import Path
from itertools import product, combinations
import json
import math
import numpy as np
from sevc.core.artifacts import write_json
from sevc.verification.reference_acquisition import commitment
from sevc.committee.executed_recovery import execute_recovery
from sevc.evaluation.recovery_graph_audit import graph_conflicts,exhaustive_witness,audit_trace
from sevc.incentives.verifier_protocol import CommittedVerifierReport
from sevc.verification.paid_replay_service import OwnerProbeReferences,settle_service


def owner_amortization(*, startup_owner_seconds, startup_fees, saved_owner_seconds,
                       service_fee, cost_per_second):
    """Conditional accounting on observed costs, never a forecast of population J."""
    values=(startup_owner_seconds,startup_fees,saved_owner_seconds,service_fee,cost_per_second)
    if any(not math.isfinite(v) for v in values) or any(v<0 for v in
            (startup_owner_seconds,startup_fees,service_fee,cost_per_second)):
        raise ValueError('finite nonnegative expense inputs required')
    startup=startup_fees+cost_per_second*startup_owner_seconds
    saving=cost_per_second*saved_owner_seconds-service_fee
    threshold=startup/saving if saving>0 else None
    work_threshold=startup_owner_seconds/saved_owner_seconds if saved_owner_seconds>0 else None
    return {'visible_startup_expense':startup,'net_saving_per_paid_service':saving,
        'owner_work_only_break_even_J':work_threshold,
        'owner_work_only_minimum_integer_J':max(1,math.ceil(work_threshold)) if work_threshold is not None else None,
        'owner_work_only_excludes_fees_and_trainer_work':True,
        'continuous_break_even_J':threshold,
        'minimum_integer_J':max(1,math.ceil(threshold)) if threshold is not None else None,
        'conditional_status':'FINITE_CONDITIONAL_THRESHOLD' if threshold is not None else
            'NO_FINITE_BREAKEVEN_ON_OBSERVED_COSTS',
        'formal_J_lower':None,'formal_economics_supported':False,
        'formal_gate':'HOLD_UNIDENTIFIED_J_AND_CALIBRATION_FORECAST'}


def calibration_cost_report(root, *, audited_root=None):
    """Consume the same raw/audited package; no actor, training, or solver calls."""
    from sevc.evaluation.local_tiny_feasibility import readlines
    root=Path(root);audited=Path(audited_root) if audited_root else root
    units=[json.loads(p.read_text()) for p in (root/'units').glob('*.json')]
    costs={r['unit_id']:r for r in readlines(audited/'role-costs.jsonl')}
    ledger=readlines(root/'calibration-ledger.jsonl')
    lock=json.loads((root/'protocol-lock.json').read_text())
    readiness=json.loads((audited/'claim-readiness.json').read_text())
    startup={}
    for unit in units:
        if not unit.get('calibration') or not unit.get('issued'):continue
        dataset=unit['dataset']
        if dataset in startup:raise ValueError('duplicate dataset calibration source')
        rows=[r for r in ledger if r['unit_id']==unit['unit_id']]
        if len({r['verifier_id'] for r in rows})!=len(rows):
            raise ValueError('duplicate calibration identity observation')
        cost=costs[unit['unit_id']]
        startup[dataset]={'unit_id':unit['unit_id'],'method':unit['method'],'identity_observations':len(rows),
            'post_service_calibration_complete':len(rows)==len(unit['assignments']),
            'owner_service_seconds':cost['owner_seconds'],
            'post_service_owner_seconds':sum(r['owner_wall_seconds'] for r in rows),
            'actual_fee_outflow':cost['service_fees'],
            'paid_assignments':sum(bool(a['settlement']['accepted_report']) for a in unit['assignments']),
            'slashed_bonds_not_netted_as_owner_revenue':sum(a['settlement']['slashed_bond'] for a in unit['assignments']),
            'trainer_source_seconds_unpriced':cost['shared_trainer_prefix_seconds'],
            'full_startup_owner_expenditure':None,
            'unpriced_training_reward':'NOT_DEFINED_IN_THIS_OWNER_CASH_LEDGER',
            'cache_accounting':'sum actual observation costs; shared cached references are not multiplied by identity count'}
    scenarios=[]
    for pair in readiness['matched_cost_pairs']:
        if pair['dataset'] not in startup:continue
        if not startup[pair['dataset']]['post_service_calibration_complete']:continue
        start=startup[pair['dataset']]
        matching=[u for u in units if u.get('method')==start['method'] and
            u.get('package') in ('M1','M7') and u.get('behavior')=='honest' and
            all(u.get(k,0)==pair[k] for k in ('dataset','anchor','steps','batch_size','invalid'))]
        eligible=(len(matching)==1 and matching[0].get('issued') and
            bool(matching[0].get('assignments')) and
            all(a['settlement']['accepted_report'] for a in matching[0]['assignments']) and
            not any(e.get('unit_id')==matching[0]['unit_id'] for e in readiness['counterexamples']))
        for fee,c in product(lock['cost_screen']['fee_grid'],lock['cost_screen']['cost_grid']):
            accounting=owner_amortization(
                startup_owner_seconds=start['owner_service_seconds']+start['post_service_owner_seconds'],
                startup_fees=fee*start['paid_assignments'],
                saved_owner_seconds=pair['O_owner_seconds']-pair['R_owner_seconds'],
                service_fee=fee,cost_per_second=c)
            if not eligible:
                accounting.update(continuous_break_even_J=None,minimum_integer_J=None,
                    owner_work_only_break_even_J=None,owner_work_only_minimum_integer_J=None,
                    conditional_status='HOLD_FAILED_OR_INCOMPLETE_MATCHED_SERVICE')
            scenarios.append({**{k:pair[k] for k in ('dataset','anchor','steps','batch_size','invalid')},
                'observed_service_eligible':bool(eligible),
                'runtime_device':matching[0].get('runtime_device','cpu') if len(matching)==1 else None,
                'fee_scenario':fee,'cost_per_second_scenario':c,
                'price_anchor':fee==2.5 and c==.01,**accounting})
    return {'scope':'conditional observed single-verifier RCMP R/O cost accounting; not formal quorum economics',
        'startup':startup,'scenarios':scenarios,'J':None,'formal_startup_cost':None,
        'formal_economics_supported':False,'extra_scientific_blocks':0,
        'qualification_or_stationarity_implied':False,
        'limitations':['tiny startup did not qualify a production pool',
            'future workload cost and task reuse horizon are not identified',
            'trainer source computation is separately reported; no missing monetary transfer is filled with zero',
            'observed conditional thresholds are not population expectations or a joint success forecast']}


def static_oracle(graph,missing,budget=100.,deadline=100.):
    """Audit-only exhaustive allocation of known responders, no actor call."""
    conflicts=graph_conflicts(graph); available=[f'v{i}' for i in range(9) if f'v{i}' not in missing]
    possible={(0,0)}
    for v in available:
        possible=possible|{(a+(j==0),b+(j==1)) for a,b in possible for j in (0,1)
                           if f'j{j}' not in conflicts.get(v,())}
    if budget<7.5:return 0
    return max((a>=3)+(b>=3) for a,b in possible if a+b<=deadline)


def recovery_boundaries(clock):
    cases=[]
    graphs=('all-compatible','reserve-specialists-j0','reserve-specialists-j1','reserve-isolated-from-j1','three-only-j1')
    for graph,missing in product(graphs,((),('v0','v3'),('v0','v6'),('v0','v1','v3','v4'))):
        cases.append({'graph':graph,'missing':missing,'budget':100.,'deadline':100.})
    for budget,deadline in product((7.499,7.5,7.501),(5.999,6.,6.001)):
        cases.append({'graph':'all-compatible','missing':(),'budget':budget,'deadline':deadline})
    prior={g:{'all_response_sets':36,'passed':all(static_oracle(g,missing)==2 for missing in combinations([f'v{i}' for i in range(9)],2))} for g in graphs}
    rows=[]
    for case in cases:
        for policy in ('all-response-certified-ecs','online-only-joint-matching-v1','fixed-order-same-reserve','no-recovery'):
            timer=[0.]
            def callback(j,v):
                timer[0]+=1
                r=CommittedVerifierReport.create(scenario_id='submission-structure',verifier_id=v,job_id=j,
                    ordered_segment_ids=('target',)+tuple(f'p{i}' for i in range(8)),
                    verdicts=(True,)+(True,)*4+(False,)*4,nonce=v)
                s=settle_service(r,OwnerProbeReferences(tuple((f'p{i}',i<4) for i in range(8))),cost_seconds=0,effort_fraction=1,fee=2.5)
                return r,s,[]
            trace=execute_recovery(policy=policy,fault='structural',seed=17,required_ids=('target',),
                execute=callback,clock=clock,timeout_seconds=1,deadline_seconds=case['deadline'],
                sleep=lambda t:timer.__setitem__(0,timer[0]+t),monotonic=lambda:timer[0],
                service_fee=2.5,minimum_pass_count=3,all_response_sets=True,common_raw_roster=True,
                public_conflicts=graph_conflicts(case['graph']),missing_ids=case['missing'],
                budget_cap_per_job=case['budget'])
            audited=audit_trace(trace)
            if policy=='online-only-joint-matching-v1':assert trace['robust_certificate_calls']==0
            rows.append({'case':case,'policy':policy,'trace':trace,'audit':audited,
                'pre_static_oracle':prior[case['graph']],
                'post_static_maximum':static_oracle(case['graph'],case['missing'],case['budget'],case['deadline']),
                'post_oracle_minimum_legal_fees':7.5*static_oracle(case['graph'],case['missing'],case['budget'],case['deadline']),
                'post_fixed_primary_oracle':exhaustive_witness(case['graph'],case['missing']),
                'oracle_budget_deadline_scope':'exact static allocation of known responders; 1 time and 2.5 fee per successful callback; fixed-primary oracle is separately only a graph upper bound',
                'constructed_qualification':True,'clock_scope':'logical unit-duration fixture; not observed system latency',
                'new_scientific_blocks':0})
    return rows


def joint_assurance(parameter_evidence,required_events,hard_gates,**verification):
    """Delegate to the fail-closed checker; legacy numeric fields cannot unlock F."""
    from sevc.evaluation.joint_evidence import checked_joint_assurance
    return checked_joint_assurance(parameter_evidence,required_events,hard_gates,**verification)


def audit_and_summarize(root,config,profile,*,artifact_root=None,reference_predecessor=None,
                        reference_predecessor_manifest=None):
    from sevc.evaluation.local_tiny_feasibility import audit_and_summarize as base_audit
    root=Path(root);dest=root if artifact_root is None else Path(artifact_root)
    base=base_audit(root,config,profile,artifact_root=artifact_root,
        reference_predecessor=reference_predecessor,reference_predecessor_manifest=reference_predecessor_manifest)
    units=[json.loads(p.read_text()) for p in (root/'units').glob('*.json')]
    dp=[];pp=[]
    for u in units:
        if u['package']=='DP' and u.get('protocol_measured'):
            n=u['native'];commits={};reveals={}
            for e in n['native_commitment_events']:
                if e['phase']=='COMMITTED':assert not reveals;commits[e['party']]=e['commitment']
                else:
                    assert len(commits)==4 and commitment([e['party'],e['report']])==commits[e['party']]
                    reveals[e['party']]=e['report']
            assert n['numeric_payment'] is None and n['numeric_payment_status']=='NOT_DEFINED'
            from sevc.committee.depol_arbitration import arbitrate_digests
            order=next(e['sampled_interval_indices'] for e in n['protocol_events'] if e['phase']=='GROUP_SAMPLING')
            assert arbitrate_digests([reveals['trainer'][i] for i in order],[reveals[f'v{i}'] for i in range(3)])==n['fast']
            if n['slow']:
                from sevc.committee.depol_arbitration import arbitrate_distances
                actual=arbitrate_distances(n['slow']['distance_matrices'],n['epsilon'])
                assert actual['trainer_verdict']==n['final']['trainer_verdict']
                assert actual['verifier_reward_eligible']==n['final']['verifier_reward_eligible']
            dp.append({'unit_id':u['unit_id'],'dataset':u['dataset'],'invalid':u['invalid'],'behavior':u['behavior'],
                'final':n['final'],'slow_path':n['slow'] is not None,'wall_seconds':u['online_suffix_seconds'],
                'counterexample_to_full_source_decision':n['final']['trainer_verdict'] != (not u['invalid']),
                'endpoint_scope':'sampled interval endpoint weights; intermediate checkpoint mutation may be outside sampled target'})
        if u['package']=='M8':
            solution=u['solution']
            if solution['status']=='OPTIMAL':
                A,b,x,dual=map(np.asarray,(solution['A'],solution['b'],solution['primal'],solution['dual']))
                assert np.max(A@x-b)<1e-8 and abs(b@dual-solution['K'])<1e-8
            elif solution['status']=='INFEASIBLE':
                c=solution['certificate'];A,b,y=map(np.asarray,(c['A'],c['b'],c['multipliers']))
                assert np.min(y)>=0 and np.max(np.abs(A.T@y))<1e-8 and b@y<0
            pp.append({'method':u['method'],'epsilon':u['epsilon'],'status':u['status'],
                'honest_minus_best_deviation':u['native_game']['honest_minus_best_deviation'] if u['native_game'] else None})
    boundaries=json.loads((root/'recovery-boundaries.json').read_text())
    for b in boundaries:audit_trace(b['trace'])
    result={'base':base,'DP':dp,'PP':pp,'DP_numeric_payment':'NOT_DEFINED','DP_heterogeneous_gpu':'NOT_VALIDATED',
        'source_blocks':base['blocks_per_dataset'],'structural_cases':len(boundaries),
        'ON_zero_certificate_calls':all(b['trace']['robust_certificate_calls']==0 for b in boundaries if b['policy']=='online-only-joint-matching-v1'),
        'all_results_accounted':len(units)==len(config['technical_units']['fixture']),
        'F_started':False,'scientific_route':base['route'],'independent_replication':False}
    write_json(dest/'submission-readiness.json',result)
    if config.get('change_id') == 'experiment-tdsc-server-unlock-evidence-v1':
        from sevc.evaluation.server_unlock_evidence import full_cost_report
        full_cost_report(root, artifact_root=dest)
    return result
