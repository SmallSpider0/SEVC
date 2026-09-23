"""RQ4 cost scaling: owner cost against job size and a resident 16-step point.

Reuses the SE-COST study on the single fixed-design runner; only the matrix and the
resident delivery budget differ.
"""
from sevc.experiments.f_fixed_design import FixedDesignStudy
from sevc.experiments.overhead_supplement import OverheadSupplementStudy
from sevc.verification.reference_acquisition import commitment

CHANGE = 'experiment-tdsc-rq4-cost-scaling-v1'
PHASES = ('scale', 'target')
PRODUCTION_COUNTS = (32, 64, 128)
ARMS = ('R-resident', 'O')
BLOCKS = 1


def expanded_units(methods):
    rows = []
    for phase in PHASES:
        for dataset in ('cifar10', 'mnist', 'cifar100'):
            for block in range(BLOCKS):
                cells = ([(n, 0, 0, 4) for n in PRODUCTION_COUNTS] if phase == 'scale'
                         else [(32, 128, 1, 16)])
                for production_count, anchor, invalid, steps in cells:
                    seed = int(commitment([CHANGE, dataset, phase, production_count, block, 'seed'])[:8], 16)
                    binding = commitment([CHANGE, dataset, phase, production_count, block, anchor, invalid,
                                          'paired-public'])
                    for execution_order, arm in enumerate(ARMS):
                        row = dict(dataset=dataset, phase=phase, block=block, seed=seed,
                                   anchor=anchor, steps=steps, batch_size=32, invalid=invalid,
                                   production_count=production_count,
                                   context=f'anchor{anchor}-invalid' if anchor else 'init-valid',
                                   method=methods[arm[0]], arm=arm,
                                   performance_arm='resident' if arm.startswith('R-') else 'serial',
                                   public_job_binding=binding if arm.startswith('R-') else binding+'-'+arm,
                                   behavior='honest', package='M1', verifier_count=1,
                                   execution_order=execution_order, reference_check=False,
                                   profiler_sample=False)
                        row['unit_id'] = commitment([CHANGE, row])
                        rows.append(row)
    return rows


class CostScalingStudy(OverheadSupplementStudy):
    PHASES = PHASES

    def __init__(self, *args, **kwargs):
        # No predecessor units are imported and there is a single R arm to pair.
        FixedDesignStudy.__init__(self, *args, **kwargs)
        self.pair_receipts = {}
        self.primary_reused = False

    def after_unit(self, unit, result, jobs, bank, scratch):
        result['absolute_resources'] = self.unit_resource.finish()
        FixedDesignStudy.after_unit(self, unit, result, jobs, bank, scratch)
