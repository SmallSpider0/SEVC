"""RQ4 cost-scaling frozen scope and raw per-cell cost projection (descriptive, one block)."""
from collections import defaultdict
import json
from pathlib import Path

from sevc.core.artifacts import write_json

RESIDENT_BYTES = 8 * 1024 ** 3


def validate_scope(config):
    from sevc.experiments.cost_scaling import CHANGE, PHASES, expanded_units
    if config['change_id'] != CHANGE or config['fixed_design']['phase_order'] != list(PHASES):
        raise PermissionError('cost-scaling scope drift')
    if config['fixed_design_units'] != expanded_units(config['scoped_candidate']['science']['methods']):
        raise PermissionError('cost-scaling matrix drift')
    if len(config['fixed_design_units']) != 24 or config['maximum_runner_wall_seconds'] > 10800:
        raise PermissionError('cost-scaling matrix/time drift')
    resident = config['performance']['resident']
    if resident.get('resident_delivery_bytes') != RESIDENT_BYTES:
        raise PermissionError('resident budget drift')
    if {k: v for k, v in resident.items() if k != 'resident_delivery_bytes'} != {
            k: v for k, v in config['performance']['serial'].items()}:
        raise PermissionError('resident arm differs from the serial arm beyond its budget')
    if config.get('reuse_paired_validity_sources'):
        raise PermissionError('paired-validity source reuse is outside this matrix')
    frozen = json.loads((Path(__file__).resolve().parents[2]/'configs/tdsc_rq4_cost_scaling_v1.json').read_text())
    for key in ('datasets', 'performance', 'fixed_design', 'scoped_candidate', 'trajectory',
                'fixed_design_units', 'scratch_parent'):
        if config[key] != frozen[key]:
            raise PermissionError('cost-scaling frozen config drift: '+key)


def _online_roles(root, unit_ids):
    from sevc.evaluation.f_rq234_audit import _group
    roles = defaultdict(lambda: defaultdict(float))
    components = defaultdict(lambda: defaultdict(float))
    with (Path(root)/'phase-timing.jsonl').open() as stream:
        for line in stream:
            p = json.loads(line)
            if p.get('unit_id') not in unit_ids or p.get('phase_scope') != 'online':
                continue
            if p.get('resource_accounting') != 'absolute-v1':
                raise ValueError('absolute resources missing')
            seconds = p.get('exclusive_seconds') or 0.
            roles[p['unit_id']][p['role']] += seconds
            if p['role'] == 'owner':
                components[p['unit_id']][_group(p['phase'])] += seconds
    return roles, components


def audit_and_summarize(root, config, profile):
    from sevc.evaluation.f_fixed_design import _inventory, full_cost_inventory
    root = Path(root)
    _, rows, errors = _inventory(root, config['fixed_design_units'])
    if errors:
        raise ValueError('cost-scaling inventory incomplete: '+str(errors[:3]))
    for row in rows:
        if row.get('status') != 'MEASURED' or not row.get('absolute_resources'):
            raise ValueError('unmeasured cost-scaling unit: '+row['unit_id'])
    roles, components = _online_roles(root, {r['unit_id'] for r in rows})
    cells = {}
    for row in rows:
        key = f"{row['dataset']}/{row['phase']}/n{row['production_count']}/steps{row['steps']}"
        cell = cells.setdefault(key, {'dataset': row['dataset'], 'phase': row['phase'],
                                      'production_count': row['production_count'], 'steps': row['steps']})
        uid = row['unit_id']
        cell[row['arm']] = {'unit_id': uid, 'owner_seconds': roles[uid]['owner'],
                            'verifier_seconds': roles[uid]['verifier'],
                            'owner_components': dict(components[uid])}
        if row['arm'] == 'R-resident':
            delivered = row['tasks']
            cell['R-resident']['delivered_tasks'] = len(delivered)
            cell['R-resident']['spilled_tasks'] = sum(t.get('delivery_storage') == 'scratch'
                                                     and t['role'] == 'challenge' for t in delivered)
    for cell in cells.values():
        r, o = cell['R-resident'], cell['O']
        if r['spilled_tasks']:
            raise ValueError('challenge payload spilled to storage under the resident budget')
        n = cell['production_count']
        cell['owner_R_over_O'] = r['owner_seconds'] / o['owner_seconds']
        cell['verifier_over_O'] = r['verifier_seconds'] / o['owner_seconds']
        cell['direct_seconds_per_segment'] = o['owner_seconds'] / n
        cell['verifier_seconds_per_task'] = r['verifier_seconds'] / r['delivered_tasks']
    full_cost_inventory(root, config)
    result = {'status': 'COLLECTED_AWAITING_INDEPENDENT_AUDIT', 'units': len(rows), 'cells': cells,
              'blocks_per_cell': 1, 'inference': 'descriptive; one block per cell, no sign test',
              'primary_metric': 'online owner exclusive_seconds, verifier payments excluded',
              'paper_ready': False}
    write_json(root/'cost-scaling-observations.json', result)
    return result
