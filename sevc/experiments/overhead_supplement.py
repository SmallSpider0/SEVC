"""SE-COST matrix and hooks on the single fixed-design runner."""
from contextlib import contextmanager
from dataclasses import asdict

from sevc.core.artifacts import write_json
from sevc.experiments.f_fixed_design import FixedDesignStudy
from sevc.experiments.scoped_five_rq_units import ScopedStudy
from sevc.verification.reference_acquisition import commitment

CHANGE = 'experiment-tdsc-rq4-overhead-supplement-v1'
COUNTS = {'primary': 8, 'target': 2, 'profiler': 1}
ARMS = ('R-repaired', 'R-as-run', 'G-repaired', 'O')


def expanded_units(methods):
    rows = []
    for phase, count in COUNTS.items():
        for dataset in ('cifar10', 'mnist', 'cifar100'):
            for block in range(count):
                seed = int(commitment([CHANGE, dataset, phase, block, 'seed'])[:8], 16)
                order = ARMS[block % 4:] + ARMS[:block % 4]
                contexts = ((128, 1), (512, 1)) if phase == 'target' else ((0, 0), (0, 1))
                for anchor, invalid in contexts:
                    binding = commitment([CHANGE, dataset, phase, block, anchor, invalid, 'paired-public'])
                    for execution_order, arm in enumerate(order):
                        row = dict(dataset=dataset, phase=phase, block=block, seed=seed,
                                   anchor=anchor, steps=16 if phase == 'target' else 4,
                                   batch_size=32, invalid=invalid,
                                   context=f'anchor{anchor}-invalid' if anchor else
                                       ('init-invalid' if invalid else 'init-valid'),
                                   method=methods[arm[0]], arm=arm,
                                   performance_arm='repaired' if arm.endswith('-repaired') else 'serial',
                                   public_job_binding=binding if arm.startswith('R-') else binding+'-'+arm,
                                   behavior='honest', package='M1', verifier_count=1,
                                   execution_order=execution_order, reference_check=False,
                                   profiler_sample=phase == 'profiler')
                        row['unit_id'] = commitment([CHANGE, row])
                        rows.append(row)
    return rows


def equivalence_projection(job, result):
    """Timing/unique assignment identifiers remain raw; compare exact economic decisions."""
    fields = ('status', 'accepted_report', 'service_fee', 'slashed_bond')
    return {'public_tasks': [{'task_id': t.task_id, 'envelope': t.envelope,
                              'descriptor': t.descriptor} for t in job.tasks],
            'production_source_ids': list(job.production_source_ids),
            'verdicts': [a['report']['verdicts'] for a in result['assignments']],
            'settlement': [{k: a['settlement'][k] for k in fields} for a in result['assignments']]}


class OverheadSupplementStudy(FixedDesignStudy):
    PHASES = tuple(COUNTS)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pair_receipts = {}
        (self.root/'equivalence').mkdir()
        from sevc.experiments.overhead_continuation import import_primary
        self.primary_reused = import_primary(self) if self.profile['formal'] else False

    def run_dataset(self, context, partition):
        from sevc.evaluation.f_fixed_design import seal_stage
        self.context = context
        for phase in ([self.scheduled_phase] if hasattr(self, 'scheduled_phase') else list(self.PHASES)):
            if phase == 'primary' and self.primary_reused:
                continue
            self.phase = phase
            self.units = [u for u in self.all_units if u['phase'] == phase and u['dataset'] == context.dataset]
            ScopedStudy.run_dataset(self, context, tuple(self.lock['partitions'][context.dataset][phase]))
            seal_stage(self.root, self.config, phase, context.dataset)
        self.units = self.all_units

    @contextmanager
    def unit_context(self, unit):
        from sevc.core.absolute_resources import UnitResources
        with UnitResources(self.clock, self.root/'profiler', unit['unit_id'],
                           cuda=self.clock.cuda, profile=unit['profiler_sample']) as resource:
            self.unit_resource = resource
            yield

    def after_unit(self, unit, result, jobs, bank, scratch):
        result['absolute_resources'] = self.unit_resource.finish()
        if unit['arm'].startswith('R-'):
            from sevc.verification.paid_replay_service import StoredReplayProof
            from sevc.verification.replay_coupled_probes import proof_component_hashes
            old = dict(self.clock.context)
            self.clock.context['phase_scope'] = 'equivalence-audit'
            job = jobs[0]
            hashes = []
            for task in job.tasks:
                proof = task.proof.load() if isinstance(task.proof, StoredReplayProof) else task.proof
                digest, _ = self.clock.call('delivered-proof-equivalence-hash', 'audit', proof_component_hashes, proof)
                if digest != task.envelope['wrapped_proof_identity']:
                    raise ValueError('delivered proof differs from public identity')
                hashes.append(digest)
            projection = equivalence_projection(job, result)
            receipt = {'unit_id':unit['unit_id'], 'pair_key':unit['public_job_binding'],
                       'projection':projection, 'delivered_proofs':hashes,
                       'projection_sha256':commitment(projection), 'proofs_sha256':commitment(hashes)}
            other = self.pair_receipts.pop(unit['public_job_binding'], None)
            if other is None:
                self.pair_receipts[unit['public_job_binding']] = receipt
            elif (other['projection_sha256'], other['proofs_sha256']) != (receipt['projection_sha256'], receipt['proofs_sha256']):
                raise ValueError('R repaired/as-run byte identity or economic settlement mismatch')
            receipt['paired_with'] = other['unit_id'] if other else None
            write_json(self.root/'equivalence'/f"{unit['unit_id']}.json", receipt)
            self.clock.context = old
        super().after_unit(unit, result, jobs, bank, scratch)

    def run_cpu(self):
        if self.pair_receipts:
            raise ValueError('unpaired R equivalence receipts')
        from sevc.core.scratch_cleanup import finish_scratch_cleanup
        finish_scratch_cleanup(self)
