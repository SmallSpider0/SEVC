"""SE-COST frozen scope and raw cost projection (not final independent certification)."""
from collections import defaultdict
import json
from pathlib import Path
import statistics

from sevc.core.artifacts import write_json
from sevc.verification.reference_acquisition import commitment


def validate_scope(config):
    from sevc.experiments.overhead_supplement import CHANGE, COUNTS, expanded_units
    if config['change_id'] != CHANGE or config['fixed_design']['counts'] != COUNTS:
        raise PermissionError('overhead scope/count drift')
    if config['fixed_design']['phase_order'] != list(COUNTS):
        raise PermissionError('overhead phase drift')
    if config['fixed_design_units'] != expanded_units(config['scoped_candidate']['science']['methods']):
        raise PermissionError('overhead matrix drift')
    if len(config['fixed_design_units']) != 264 or config['maximum_runner_wall_seconds'] != 28800:
        raise PermissionError('overhead matrix/time drift')
    if config['performance']['repaired'].get('resident_delivery_bytes') != 3221225472:
        raise PermissionError('resident budget drift')
    frozen = json.loads((Path(__file__).resolve().parents[2]/'configs/tdsc_rq4_overhead_supplement_v1.json').read_text())
    for key in ('datasets','performance','fixed_design','overhead_supplement','scoped_candidate',
                'trajectory','reuse_paired_validity_sources'):
        if config[key] != frozen[key]:
            raise PermissionError('overhead frozen config drift: '+key)


def paired_endpoint(numerators, denominators, alpha):
    from sevc.evaluation.detection_supplement import sign_test
    if set(numerators) != set(denominators) or any(v <= 0 for v in denominators.values()):
        raise ValueError('incomplete or nonpositive denominator')
    ratios = {k: numerators[k]/denominators[k] for k in sorted(numerators)}
    test = sign_test([1-v for v in ratios.values()])
    return {'ratios':ratios,'median':statistics.median(ratios.values()), **test,'alpha':alpha}


def audit_and_summarize(root, config, profile):
    from sevc.evaluation.f_fixed_design import _inventory, full_cost_inventory
    root = Path(root)
    _, rows, errors = _inventory(root, config['fixed_design_units'])
    if errors:
        raise ValueError('overhead inventory incomplete: '+str(errors[:3]))
    byid = {r['unit_id']:r for r in rows}
    roles = defaultdict(lambda:defaultdict(lambda:defaultdict(float)))
    preparation = defaultdict(float)
    first_service = {}
    phases = []
    with (root/'phase-timing.jsonl').open() as stream:
        for line in stream:
            p = json.loads(line)
            if p.get('unit_id') in byid and p.get('phase_scope') == 'online':
                if p.get('resource_accounting') != 'absolute-v1':
                    raise ValueError('absolute resources missing')
                phases.append(p)
                if p['phase'] == 'verifier-service-interface':
                    first_service[p['unit_id']] = p['start']
    for p in phases:
        uid = p['unit_id']
        for metric in ('exclusive_seconds','exclusive_cpu_thread_seconds','exclusive_cpu_process_seconds',
                       'cuda_stream_elapsed_seconds'):
            value = p.get(metric)
            if value is not None:
                roles[uid][p['role']][metric] += value
        for metric, value in (p.get('exclusive_io_bytes') or {}).items():
            roles[uid][p['role']]['io_'+metric] += value
        if p['role'] == 'owner' and p['end'] <= first_service.get(uid, float('-inf')):
            preparation[uid] += p['exclusive_seconds']
    receipts = defaultdict(list)
    for row in rows:
        if row['arm'].startswith('R-'):
            r = json.loads((root/'equivalence'/f"{row['unit_id']}.json").read_text())
            if r['unit_id'] != row['unit_id'] or commitment(r['projection']) != r['projection_sha256'] or commitment(r['delivered_proofs']) != r['proofs_sha256']:
                raise ValueError('equivalence receipt digest drift')
            receipts[r['pair_key']].append(r)
        if not row.get('absolute_resources'):
            raise ValueError('unit absolute resource missing')
        if row['profiler_sample'] and not row['absolute_resources'].get('kernel_self_seconds_by_role_phase'):
            raise ValueError('CUDA kernel profile absent')
    for group in receipts.values():
        if len(group) != 2 or len({(r['projection_sha256'],r['proofs_sha256']) for r in group}) != 1:
            raise ValueError('paired equivalence mismatch')
    owner = {}; reference = {}
    for dataset in config['datasets']:
        primary = [r for r in rows if r['dataset'] == dataset and r['phase'] == 'primary']
        for context in ('init-valid','init-invalid'):
            def values(arm):
                return {r['block']:roles[r['unit_id']]['owner']['exclusive_seconds'] for r in primary if r['context']==context and r['arm']==arm}
            owner[dataset+'/'+context] = paired_endpoint(values('R-repaired'),values('O'),config['overhead_supplement']['owner_alpha'])
        def prep(arm):
            result = defaultdict(float)
            for r in primary:
                if r['arm']==arm:result[r['block']] += preparation[r['unit_id']]
            return result
        reference[dataset] = paired_endpoint(prep('R-repaired'),prep('G-repaired'),config['overhead_supplement']['preparation_alpha'])
    full_cost_inventory(root, config)
    result = {'status':'COLLECTED_AWAITING_INDEPENDENT_AUDIT','units':len(rows),
              'owner_R_over_O':owner,'preparation_R_over_G':reference,
              'online_resources_by_unit_role':roles,'reference_preparation_seconds':dict(preparation),
              'equivalence_pairs':len(receipts),'paper_ready':False,
              'cuda_note':'event spans may nest/overlap; do not sum as kernel busy demand',
              'instrumentation_overhead':'see preflight engineering receipt; no subtraction'}
    write_json(root/'overhead-observations.json',result)
    return result
