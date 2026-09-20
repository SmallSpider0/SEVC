"""SE-DET assembly on the canonical fixed-design/ScopedStudy lifecycle."""
from collections import defaultdict
import hashlib
import pickle

from sevc.core.artifacts import write_json
from sevc.experiments.f_fixed_design import FixedDesignStudy
from sevc.experiments.scoped_five_rq_units import ScopedStudy
from sevc.verification.reference_acquisition import commitment

CHANGE = 'experiment-tdsc-rq1-detection-supplement-v1'
COUNTS = {'development': 2, 'validation': 2, 'test': 8, 'native': 1}
BEHAVIORS = ('honest', 'constant-accept', 'constant-reject', 'uniform-k32',
             'uniform-k39', 'sgd-consistency-shortcut', 'prefix-one-step-shortcut',
             'cheap-recognizer-k32')


# Honest-only participation follow-up (RQ2): new independent test blocks and a tighter,
# preregistered billing cap. It reuses this study and runner unchanged.
PARTICIPATION_CHANGE = 'experiment-tdsc-rq2-honest-participation-v1'
PARTICIPATION_COUNTS = {'test': 12}


def _matrix(change, counts, arms_for):
    rows = []
    for phase, count in counts.items():
        for dataset in ('cifar10', 'mnist', 'cifar100'):
            for block in range(count):
                seed = int(commitment([change, dataset, phase, block, 'seed'])[:8], 16)
                arms = arms_for(phase)
                for invalid in (0, 1):
                    for order, (method, behavior) in enumerate(arms):
                        row = dict(dataset=dataset, phase=phase, split=phase, block=block,
                                   seed=seed, anchor=0, steps=4, batch_size=32, invalid=invalid,
                                   context='init-invalid' if invalid else 'init-valid',
                                   method=method, behavior=behavior,
                                   package='DP' if phase == 'native' else 'M1',
                                   verifier_count=3 if phase == 'native' else 1,
                                   execution_order=order, reference_check=False)
                        row['unit_id'] = commitment([change, row])
                        rows.append(row)
    return rows


def expanded_units(methods):
    from sevc.verification.depol_local import KEY
    return _matrix(CHANGE, COUNTS, lambda phase: [(KEY, b) for b in BEHAVIORS[:5]] if phase == 'native' else [
        (methods[m], b) for m in ('R', 'G') for b in (BEHAVIORS if phase == 'test' else ('honest',))])


def participation_units(methods):
    return _matrix(PARTICIPATION_CHANGE, PARTICIPATION_COUNTS,
                   lambda phase: [(methods[m], 'honest') for m in ('R', 'G')])


class DetectionSupplementStudy(FixedDesignStudy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.recognizer = self.config['detection_supplement']
        if self.profile['formal']:
            import importlib
            for name, version in self.recognizer['dependency_versions'].items():
                if importlib.import_module(name).__version__ != version:
                    raise PermissionError('recognizer dependency version drift: '+name)
        self.histories = defaultdict(list)
        self.history_pending = []
        self.fitted = {}
        self.selections = {}
        (self.root / 'recognizer').mkdir()

    def run_dataset(self, context, partition):
        from sevc.evaluation.f_fixed_design import seal_stage
        self.context = context
        phases = [self.scheduled_phase] if hasattr(self, 'scheduled_phase') else self.lock['phase_order']
        for phase in phases:
            self.phase = phase
            self.units = [u for u in self.all_units if u['phase'] == phase and u['dataset'] == context.dataset]
            ScopedStudy.run_dataset(self, context, tuple(self.lock['partitions'][context.dataset][phase]))
            if phase == 'validation':
                self.select_models(context.dataset)
            seal_stage(self.root, self.config, phase, context.dataset)
        self.units = self.all_units

    def before_unit(self, unit):
        from sevc.verification.cheap_discriminator import PersistentCheapSelector
        self.performance.pop('cheap_selector', None)
        if unit['behavior'] == 'cheap-recognizer-k32':
            key = (unit['dataset'], unit['method'])
            self.performance['cheap_selector'] = PersistentCheapSelector(
                self.fitted[key], self.selections[key], self.recognizer['cache_limit_bytes'])
        return True

    def after_unit(self, unit, result, jobs, bank, scratch):
        from sevc.verification.cheap_discriminator import collect_public_history
        super().after_unit(unit, result, jobs, bank, scratch)
        if not result['issued']:
            return
        if unit['phase'] in ('development', 'validation'):
            self.history_pending.append((unit, collect_public_history(unit, jobs[0], self.clock, improved=True)))
        if unit['behavior'] == 'cheap-recognizer-k32':
            selector = self.performance['cheap_selector']
            selector.finish()
            result['recognizer'] = selector.receipt
            write_json(self.root / 'recognizer' / f"{unit['unit_id']}.json", selector.receipt)

    def after_group(self, context, group, bank, group_id):
        super().after_group(context, group, bank, group_id)
        if self.epoch_key is not None:
            return
        from sevc.verification.cheap_discriminator import publish_history
        for unit, history in self.history_pending:
            publish_history(history, self.root / 'recognizer' / f"history-{unit['unit_id']}.json")
            self.histories[unit['dataset'], unit['method'], unit['phase']].append(history)
        self.history_pending.clear()

    def select_models(self, dataset):
        from sevc.verification.cheap_discriminator import select_development_models
        for method in (self.science['methods']['R'], self.science['methods']['G']):
            key = dataset, method
            if key in self.selections:
                raise ValueError('selection may occur only once')
            development = self.histories[dataset, method, 'development']
            validation = self.histories[dataset, method, 'validation']
            if len(development) != 2 * self.lock['counts']['development'] or len(validation) != 2 * self.lock['counts']['validation']:
                raise ValueError('incomplete disclosed training/validation services')
            self.clock.context = {'dataset': dataset, 'method': method, 'phase_scope': 'offline-attack-development'}
            (models, receipt), seconds = self.clock.call('development_model_selection', 'offline',
                select_development_models, development, validation,
                self.recognizer['component_models'], self.recognizer['random_state'])
            model_bytes = pickle.dumps(models, protocol=5)
            receipt.update(selection_wall_seconds=seconds, model_sha256=hashlib.sha256(model_bytes).hexdigest())
            (self.root / 'recognizer' / f'model-{dataset}-{method}.pkl').write_bytes(model_bytes)
            write_json(self.root / 'recognizer' / f'selection-{dataset}-{method}.json', receipt)
            self.fitted[key], self.selections[key] = models, receipt

    def finish_unit(self, unit, result, began, cpu_started):
        from sevc.evaluation.detection_supplement import capped_observation
        cap = self.recognizer['billing_cap_seconds'][unit['dataset']]
        result = {**result, 'billing_deadline_seconds': cap,
                  'capped_billing': [{'assignment_id': a['assignment_id'],
                      **capped_observation(a['settlement'], cap, 2.5, .5, .01)}
                      for a in result.get('assignments', [])]}
        super().finish_unit(unit, result, began, cpu_started)

    def run_cpu(self):
        # No additional scientific cells. Dataset stage seals already exist.
        from sevc.core.scratch_cleanup import release_scratch
        for path in list(getattr(self, 'pending_scratch_cleanup', ())):
            release_scratch(self, path)
