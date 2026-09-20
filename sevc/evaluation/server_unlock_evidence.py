"""Complete same-evidence accounting for the server unlock diagnostic."""
from collections import defaultdict
import json
from pathlib import Path
from sevc.core.artifacts import write_json
from sevc.evaluation.local_tiny_feasibility import readlines
from sevc.evaluation.submission_tiny import calibration_cost_report


def full_cost_report(root, *, artifact_root=None):
    root = Path(root)
    destination = root if artifact_root is None else Path(artifact_root)
    lock = json.loads((root/'protocol-lock.json').read_text())
    phases = readlines(root/'phase-timing.jsonl')
    units = [json.loads(p.read_text()) for p in (root/'units').glob('*.json')]
    role = defaultdict(float); component = defaultdict(float); cpu = defaultdict(float)
    uid_phases = defaultdict(list)
    ids = set()
    for p in phases:
        if p['event_id'] in ids:
            raise ValueError('duplicate phase event would double-charge work')
        ids.add(p['event_id'])
        role[p['role']] += p['exclusive_seconds']
        cpu[p['role']] += p['exclusive_cpu_thread_seconds']
        component[p['phase']] += p['exclusive_seconds']
        uid_phases[p.get('unit_id')].append(p)
    rows=[]
    for u in units:
        parts=defaultdict(float)
        for p in uid_phases[u['unit_id']]:
            parts[p['role']] += p['exclusive_seconds']
        assignments=[{'assignment_id':a['assignment_id'],
            'verifier_id':a['report']['verifier_id'],'job_id':a['report']['job_id'],
            'behavior':a['behavior'],'verifier_interface_seconds':a['settlement']['verifier_cost'],
            'service_fee':a['settlement']['service_fee'],
            'returned_bond':a['settlement']['refundable_bond'],
            'slashed_bond':a['settlement']['slashed_bond'],
            'accepted_report':a['settlement']['accepted_report']} for a in u.get('assignments',[])]
        rows.append({'unit_id':u['unit_id'],'dataset':u['dataset'],'method':u['method'],
            'status':u['status'],'issued':u.get('issued',False),'assignments':assignments,
            'all_instrumented_exclusive_wall_by_role':dict(parts),
            'phase_components':{phase:sum(p['exclusive_seconds'] for p in uid_phases[u['unit_id']] if p['phase']==phase)
                                for phase in sorted({p['phase'] for p in uid_phases[u['unit_id']]})},
            'source_group':u.get('source_group'),
            'source_group_seconds_shared_not_summed_per_unit':u.get('shared_trainer_prefix_seconds'),
            'unit_wall_seconds':u['wall_seconds'],
            'failed_phase_seconds':sum(p['exclusive_seconds'] for p in uid_phases[u['unit_id']] if p.get('technical_failure')),
            'DP_numeric_fee': 'NOT_DEFINED' if u['package']=='DP' else 'NOT_APPLICABLE',
            'trainer_numeric_reward':'NOT_DEFINED_IN_REGISTERED_PAYMENT_INTERFACE'})
    result={'scope':'all attempts in this immutable run; no total saving assumption',
        'physical_instrumented_exclusive_wall_by_role':dict(role),
        'physical_instrumented_exclusive_cpu_thread_seconds_by_role':dict(cpu),
        'physical_instrumented_system_wall_seconds':sum(v for k,v in role.items() if k!='wait'),
        'components':dict(component),'units':rows,
        'cash_service_fees':sum(a['service_fee'] for r in rows for a in r['assignments']),
        'bond_principal_is_not_fee':True,'payment_is_transfer_not_system_compute':True,
        'unpriced_fields':['trainer numeric rewards','DP numeric payments','WAN/blockchain/energy'],
        'noninstrumented_process_setup_and_glue':'retain performance-window/fixed-overhead and supervisor; not assumed zero',
        'cuda_stream_events_are_not_kernel_busy_seconds':True,
        'shared_source_accounting':'physical phase ledger once; method comparisons must disclose separately allocated prefix',
        'formal_incentives_supported':False,'P3_started':False,'F_started':False}
    continuation_path = root/'continuation-receipt.json'
    if continuation_path.exists():
        continuation = json.loads(continuation_path.read_text())
        result['preserved_prior_partial_work'] = continuation.get('prior_partial_phase_costs', {})
        result['prior_partial_work_not_in_effective_phase_totals'] = True
    write_json(destination/'full-cost-ledger.json',result)
    economics=(calibration_cost_report(root, audited_root=destination)
        if (root/'calibration-ledger.jsonl').exists() else
        {'status':'HOLD_CALIBRATION_NOT_EXECUTED','startup':None,'J':None,
         'formal_economics_supported':False,'missing_cost_not_zero':True})
    pair_path=root/'owner-cost-repair-pairs.json'
    if pair_path.exists():
        pair=json.loads(pair_path.read_text())
        migration=sum(p['exclusive_seconds'] for p in phases
            if p.get('phase_scope')=='owner-repair-migration' and p['role']=='owner')
        economics['owner_repair_affected_pair']={**pair,
            'migration_owner_seconds':migration,
            'owner_excludes_verifier_payments':True,
            'startup_J_status':'HOLD_FULL_MATCHED_STARTUP_AND_REPEAT_SERVICE_EVIDENCE',
            'historical_costs_replaced':False}
    write_json(destination/'startup-economics.json',economics)
    write_json(destination/'steps-1-3-assessment.json',{
        'five_RQ_coverage':'see research-objective-evidence-matrix; executed unit completeness and missing cells in claim-readiness',
        'full_cost_file':'full-cost-ledger.json','economics_file':'startup-economics.json',
        'joint_pass_lower':None,'I0':'HOLD','P3_started':False,'F_started':False,
        'diagnostic_scale':lock.get('diagnostic_scale', 'target-16x32-v1'),
        'target_scale_evidence_status':('HOLD_SHORT_DIAGNOSTIC_CANNOT_REPLACE_TARGET'
            if lock.get('diagnostic_scale') == 'short-core-4x32-v2' else 'REQUIRES_COMPLETE_AUDIT'),
        'independent_parameter_stage_recommendation':'only after no unresolved core counterexample and supported fixed-price economics/J; no automatic launch'})
    return result
