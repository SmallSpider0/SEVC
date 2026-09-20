"""Declarative successor matrix; execution remains SubmissionTinyStudy."""
from sevc.experiments.submission_tiny import expanded_units as submission_units
from sevc.verification.reference_acquisition import commitment

CHANGE = 'experiment-tdsc-server-unlock-evidence-v1'


def expanded_units(lock, methods, native):
    rows = submission_units(lock, methods, native)
    for index, row in enumerate(rows):
        generated_id = row.pop('unit_id')
        row['predecessor_unit_id'] = lock.get('predecessor_unit_ids', [])[index] if lock.get('predecessor_unit_ids') else generated_id
        if row['package'] != 'M8' and not row.get('anchor'):
            row.update(steps=16, batch_size=32)
        if row.get('reference_check'):
            row['replays'] = 2
    policies = ('all-response-certified-ecs', 'online-only-joint-matching-v1',
                'fixed-order-same-reserve', 'no-recovery')
    cases = [dict(scenario='graph-'+g, graph=g, missing_ids=['v0','v3']) for g in
             ('reserve-specialists-j0','reserve-specialists-j1',
              'reserve-isolated-from-j1','three-only-j1')]
    cases += [dict(scenario='replacement-failure', missing_ids=['v0','v3','v6']),
              dict(scenario='insufficient-response', missing_ids=['v0','v1','v3','v4']),
              dict(scenario='conditioned-one-report-error', missing_ids=[], conditioned_production_flip=True)]
    for dataset in lock['dataset_order']:
        common = dict(dataset=dataset, block=0, seed=lock['seeds'][dataset], anchor=0,
                      steps=16,batch_size=32,invalid=0,method=methods['R'],behavior='honest',
                      colluders=0,package='M4L',fault='no-missing',graph='all-compatible',
                      common_raw_roster=True,execution_order=13)
        for case in cases:
            for policy in policies:
                rows.append({**common, **case, 'policy':policy})
        for delta in (-.001,0.,.001):
            for policy in policies[:2]:
                rows.append({**common,'policy':policy,'scenario':f'fee-boundary-{delta}',
                             'budget_boundary_delta':delta})
        for deadline in (1e-9,2400.):
            for policy in policies[:2]:
                rows.append({**common,'policy':policy,'scenario':f'deadline-{deadline}',
                             'deadline_seconds':deadline,'fault':'one-missing'})
    for row in rows:
        if lock.get('diagnostic_scale') == 'short-core-4x32-v2':
            row['target_scale_unit_id'] = commitment([CHANGE, row])
            row['measurement_scope'] = 'SHORT_DIAGNOSTIC_NOT_TARGET_EVIDENCE'
            if row.get('anchor'):
                row['execution_disposition'] = 'HOLD_DEFERRED_TARGET_SCALE'
            elif row['package'] != 'M8':
                row.update(steps=4, batch_size=32)
        if row['package'] == 'DP' and lock.get('dp_partition_revision') == 3:
            row['estimator_partition'] = lock['depol']['estimator_partitions'][row['dataset']]
            row['dp_partition_revision'] = 3
        row['unit_id'] = commitment([CHANGE,row])
    return rows


def validate_estimator_partitions(lock, profile):
    """Reject invalid training-only calibration pools before opening data."""
    spec = lock['depol']
    needed = spec['estimator_steps'] * spec['estimator_batch_size']
    for dataset, total in (('cifar10', 50000), ('cifar100', 50000), ('mnist', 60000)):
        start, end = spec['estimator_partitions'][dataset]
        pstart, pend = profile['sample_ranges'][dataset]
        if not 0 <= start < end <= total or end-start < needed:
            raise PermissionError('DP estimator partition capacity insufficient: '+dataset)
        if max(start, pstart) < min(end, pend):
            raise PermissionError('DP estimator overlaps production partition: '+dataset)
