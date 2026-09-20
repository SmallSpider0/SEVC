"""Local feasibility assembly hooks for the single paid-replay runner."""
from dataclasses import asdict
import itertools
import json
from pathlib import Path
import shutil
import time

from sevc.core.artifacts import write_json, sha256_file
from sevc.experiments.scoped_five_rq_units import ScopedStudy
from sevc.verification.reference_acquisition import commitment, JobReferences
from sevc.verification.paid_replay_service import StoredReplayProof
from sevc.reputation.calibration import PaidCalibrationLedger

CHANGE = 'experiment-tdsc-local-tiny-feasibility-v1'


def validate_local_lock(config, profile, output_root):
    lock_path = Path(config['local_protocol_lock_path'])
    lock = json.loads(lock_path.read_text())
    continuation_cuda=(profile['device']=='cuda:0' and lock.get('continuation_science_ready') is True
        and config.get('continuation_manifest_path') and lock.get('review_scope')=='tiny_science'
        and lock.get('allowed_device')=='cuda:0')
    server_cuda=(profile['device']=='cuda:0' and config.get('server_only')
        and lock.get('server_only_science_ready') is True and lock.get('allowed_device')=='cuda:0')
    if profile['formal'] or (profile['device'] != 'cpu' and not (continuation_cuda or server_cuda)) or config.get('formal_execution_authorized'):
        raise PermissionError('only the frozen nonformal local CPU path is authorized')
    if not lock['execution_ready'] or sha256_file(lock_path) != config['local_protocol_lock_sha256']:
        raise PermissionError('local protocol lock is not ready or drifted')
    projection={k:v for k,v in config.items() if k!='local_protocol_lock_sha256' and not k.startswith('_runtime_')}
    if commitment(projection) != lock['configuration_commitment']:
        raise PermissionError('frozen configuration drift')
    if str(output_root) != lock['output_root'] or config['technical_units']['fixture'] != lock['expanded_units']:
        raise PermissionError('local run/output/matrix identity drift')
    if config['scoped_candidate']['science']['blocks_per_dataset'] != 1:
        raise PermissionError('one block per dataset is required')
    for row in lock['input_files'] + lock['source_files']:
        if sha256_file(Path(row['path'])) != row['sha256']:
            raise PermissionError('frozen input/source identity drift: ' + row['path'])
    if config.get('change_id') in {'experiment-tdsc-submission-tiny-v1','experiment-tdsc-server-unlock-evidence-v1'}:
        if not lock.get('science_ready') or lock.get('review_scope') != 'tiny_science':
            raise PermissionError('submission tiny requires scientific scope')
        if 'Scope: tiny_science' not in Path(config['review_path']).read_text():
            raise PermissionError('review scope is not scientific')
        if 'Ready: YES' not in Path(config['science_review_path']).read_text():
            raise PermissionError('science review is not ready')
        if set(lock['seeds']) != {'mnist','cifar10','cifar100'} or lock.get('extra_blocks',0):
            raise PermissionError('three single blocks only')
    if config.get('change_id') == 'experiment-tdsc-server-unlock-evidence-v1':
        if not config.get('server_only') or not lock.get('server_only_science_ready'):
            raise PermissionError('successor requires fresh server-only diagnostic identity')
        from sevc.experiments.server_unlock_evidence import validate_estimator_partitions
        validate_estimator_partitions(lock, profile)
        if config.get('continuation_manifest_path'):
            from sevc.experiments.submission_continuation import checked_manifest
            manifest = checked_manifest(config['continuation_manifest_path'])
            if (not lock.get('server_only_continuation_ready') or
                    lock.get('dp_partition_revision') != 3 or
                    sha256_file(Path(config['continuation_manifest_path'])) != lock.get('continuation_manifest_sha256') or
                    manifest.get('change_id') != config['change_id'] or
                    manifest.get('environment_id') != lock.get('predecessor_environment_id', commitment(lock['environment']))):
                raise PermissionError('unreviewed or foreign server continuation')
        if lock.get('diagnostic_scale') == 'short-core-4x32-v2':
            from sevc.experiments.server_unlock_evidence import expanded_units
            candidate = config['scoped_candidate']
            expected = expanded_units(lock, candidate['science']['methods'],
                                      next(p for p in candidate['packages'] if p['id'] == 'M8'))
            if lock['expanded_units'] != expected:
                raise PermissionError('short diagnostic matrix drift')
            unlimited = lock.get('time_budget_policy') == 'author-unlimited-until-complete-v5'
            cap = lock['resource_budget']['candidate_total_wall_seconds']
            remaining = None if unlimited else cap-lock['prior_attempt_wall_seconds']
            if unlimited and (lock.get('protocol_revision') != 5 or cap is not None or
                    lock['resource_budget']['unit_wall_seconds'] is not None):
                raise PermissionError('unreviewed unlimited time budget')
            if ((remaining is not None and remaining <= 0) or config.get('maximum_runner_wall_seconds') != remaining
                    or lock['depol'].get('estimator_steps') != 4
                    or lock['depol'].get('estimator_batch_size') != 32
                    or not config.get('trajectory', {}).get('execution_disabled')):
                raise PermissionError('short diagnostic budget or workload drift')
    if 'Ready: YES' not in Path(config['review_path']).read_text():
        raise PermissionError('review is not ready')


class LocalTinyStudy(ScopedStudy):
    """Reuse every source, train, assignment and recovery path of ScopedStudy."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.all_units = list(self.units)
        self.lock = json.loads(Path(self.config['local_protocol_lock_path']).read_text())
        self.phase = 'core'
        prior=self.lock.get('prior_attempt_wall_seconds',0.)
        if self.lock.get('candidate_first_started_unix'):
            prior=max(prior,time.time()-self.lock['candidate_first_started_unix'])
        self.started = time.monotonic() - prior
        self.calibration = {}
        self.calibration_jobs = []
        self.retained = set()
        self.core_counterexamples = []
        self.unit_budget_exceeded = False
        self.base_costs = {}
        write_json(self.root/'protocol-lock.json', self.lock)
        write_json(self.root/'input-manifest.json', self.lock['input_files'])

    def hold(self, unit, reason, **extra):
        if unit['unit_id'] not in self.results:
            now = time.monotonic()
            super().finish_unit(unit, {'status':reason, 'issued':False, 'assignments':[],
                'measured':False, **extra}, now, time.process_time())

    def before_group(self, context, group_key, group):
        if all(u['unit_id'] in self.results for u in group):
            return False
        if self.lock.get('diagnostic_scale') == 'short-core-4x32-v2' and group_key[1]:
            for unit in group:
                self.hold(unit, 'HOLD_DEFERRED_TARGET_SCALE',
                          target_evidence_required=True)
            return False
        seed, anchor, steps, batch, invalid = group_key
        elapsed = time.monotonic() - self.started
        reason = None
        estimate = None
        if self.lock['resource_budget']['candidate_total_wall_seconds'] is not None and elapsed >= self.lock['resource_budget']['candidate_total_wall_seconds']:
            reason = 'HOLD_RESOURCE'
        elif anchor:
            if self.core_counterexamples:
                reason = 'STOP_REPAIR'
            elif self.unit_budget_exceeded:
                reason = 'HOLD_RESOURCE'
            elif context.dataset in self.base_costs:
                base = self.base_costs[context.dataset]
                multiplier = steps * batch / self.lock.get('projection_base_workload', 8)
                estimate = base * multiplier
                if self.lock['resource_budget']['unit_wall_seconds'] is not None and estimate > self.lock['resource_budget']['unit_wall_seconds']:
                    reason = 'HOLD_RESOURCE_PROJECTED_UNIT'
                # Include checkpoint size growth, keeping an explicit local disk budget.
                disk_estimate = self.base_costs.get(context.dataset+'-bytes', 0) * (steps+1)/self.lock.get('projection_base_checkpoints', 5)
                if disk_estimate > self.lock['resource_budget']['bridge_source_disk_budget_bytes'] or disk_estimate > shutil.disk_usage(self.root).free * self.lock.get('disk_free_fraction', .8):
                    reason = 'HOLD_RESOURCE_PROJECTED_DISK'
        if reason:
            for unit in group:
                self.hold(unit, reason, projected_source_and_unit_seconds=estimate)
            return False
        return True

    def before_unit(self, unit):
        if unit['unit_id'] in self.results:
            return False
        if self.lock['resource_budget']['candidate_total_wall_seconds'] is not None and time.monotonic() - self.started >= self.lock['resource_budget']['candidate_total_wall_seconds']:
            self.hold(unit, 'HOLD_RESOURCE')
            return False
        if self.lock.get('unit_scratch_proof_multiplier'):
            largest = max((p.stat().st_size for p in Path(self.config['scratch_parent']).glob('*/*.pt')), default=0)
            required = largest*self.lock['unit_scratch_proof_multiplier'] + 1024**3
            if required > shutil.disk_usage(self.root).free:
                self.hold(unit, 'HOLD_RESOURCE_PROJECTED_WORKING_SET', required_scratch_bytes=required)
                return False
        if unit.get('calibration'):
            for i in range(9):
                vid = f'v{i}'
                ledger = PaidCalibrationLedger(vid, 22.5, 2.5, 20.)
                ledger.reserve(unit['unit_id']+'-'+vid, f"{unit['dataset']}-block0-{unit['seed']}")
                self.calibration[unit['dataset'],vid] = ledger
            self.records({'event':'calibration-reserved-before-service','unit_id':unit['unit_id'],
                          'dataset':unit['dataset'],'identities':9,'total_reserved':202.5})
        return True

    def after_unit(self, unit, result, jobs, bank, scratch):
        import torch
        if not result['issued']:
            return
        by_hash = {s.hashes['proof_sha256']:s for s in bank}
        # Owner/evaluator-only witness retention never feeds an actor's information view.
        task_map = {t['task_id']:t for j in jobs for t in j.task_rows}
        all_tasks = {t.task_id:t for j in jobs for t in getattr(j,'tasks',())}
        self.records({'event':'public-metadata-screen', 'unit_id':unit['unit_id'],
            'fit_current_labels':False, 'personal_legal_history':'UNAVAILABLE',
            'public_fields':[{'task_id':k, 'fields':dict(t.envelope)} for k,t in all_tasks.items()]})
        for assignment in result['assignments']:
            report = assignment['report']; settlement = assignment['settlement']
            verdicts = dict(zip(report['ordered_segment_ids'],report['verdicts']))
            wrong = [k for k,t in task_map.items() if k in verdicts and t['role']=='production'
                and verdicts[k] != (not (unit['invalid'] and by_hash[t['source_sha256']].recipe['source_index']==24))]
            core = (assignment['behavior']=='honest' and settlement['status']!='PASS' or
                    assignment['behavior'] in self.lock['frozen_content_strategies'] and settlement['status']=='PASS' and wrong)
            if core:
                self.core_counterexamples.append({'unit_id':unit['unit_id'],'assignment_id':assignment['assignment_id'],
                    'dataset':unit['dataset'],'behavior':assignment['behavior'],'wrong_production_ids':wrong,
                    'kind':'HONEST_FALSE_FAILURE' if assignment['behavior']=='honest' else 'CONTENT_WRONG_REPORT_ADMITTED'})
            if core or (settlement['status']=='PASS' and wrong):
                folder = self.root/'witnesses'/assignment['assignment_id']
                folder.mkdir(parents=True)
                keep = set(wrong[:1]) | {k for k,t in task_map.items() if k in verdicts and t['role']!='production'}
                paths=[]
                for key in sorted(keep):
                    t=all_tasks[key]; proof=t.proof.load() if isinstance(t.proof,StoredReplayProof) else t.proof
                    path=folder/(key+'.pt'); torch.save({'proof':proof,'descriptor':t.descriptor,'envelope':t.envelope},path)
                    paths.append({'path':str(path),'sha256':sha256_file(path),'task_id':key})
                write_json(folder/'witness.json',{'assignment':assignment,'tasks':[task_map[k] for k in keep],
                    'files':paths,'independent_block':False})
        if unit.get('calibration'):
            job=jobs[0]
            self.calibration_jobs.append({'dataset':unit['dataset'],'unit':unit,
                'production':dict(job.production),'sources':by_hash,'assignments':result['assignments'],
                'source_device':self.profile['device'],
                'replay_environment_id':self.config.get('_runtime_replay_environment_id')})
            self.retained.add(str(scratch))
        if unit.get('anchor',0)==0:
            self.base_costs[unit['dataset']] = max(self.base_costs.get(unit['dataset'],0),
                result['shared_trainer_prefix_seconds'] + result['online_suffix_seconds'])
            self.base_costs[unit['dataset']+'-bytes'] = sum(Path(s.path).stat().st_size for s in bank)

    def retain_group(self, group, scratch):
        return str(scratch) in self.retained

    def finish_unit(self, unit, result, began, cpu_started):
        super().finish_unit(unit, result, began, cpu_started)
        if self.lock['resource_budget']['unit_wall_seconds'] is not None and time.monotonic()-began > self.lock['resource_budget']['unit_wall_seconds']:
            self.unit_budget_exceeded = True

    def run_dataset(self, context, partition):
        self.units=[r for r in self.all_units if bool(r.get('anchor',0)) == (self.phase=='bridge')]
        try:
            super().run_dataset(context,partition)
        finally:
            self.units=self.all_units

    def run_structures(self):
        from sevc.evaluation.local_tiny_feasibility import analytical_screen, structural_screen
        write_json(self.root/'forecast-sensitivity.json',analytical_screen(self.lock))
        write_json(self.root/'structural-evidence.json',structural_screen(self.clock))

    def run_calibration(self):
        from sevc.training.replay_sources import ReplayDatasetContext
        from sevc.training import verify_replay_proof
        from sevc.experiments.tdsc_five_rq_evidence import ReplayDevice
        from sevc.incentives.verifier_protocol import VerifierSettlement, CommittedVerifierReport
        from sevc.verification.on_demand_service import complete_calibration_observation
        from sevc.core.replay_environment import require_pending_calibration
        environment_id = self.config.get('_runtime_replay_environment_id')
        for pending in self.calibration_jobs:
            require_pending_calibration(pending, environment_id, self.config.get('_runtime_replay_environment_bridge'))
        # Close every related service before paid calibration can observe references.
        self.disclosure_epoch.close()
        self.records({'event':'reference-answers-published',
                      **self.disclosure_epoch.release(tuple(sorted(self.disclosure_answers.items())))})
        rows=[]
        for pending in self.calibration_jobs:
            dataset=pending['dataset']; unit=pending['unit']
            source_device=pending.get('source_device','cpu')
            context=ReplayDatasetContext(dataset,self.config['datasets'][dataset],
                Path(self.profile['data_root']),ReplayDevice(source_device))
            context.replay_environment_id = environment_id
            context.replay_environment_bridge = self.config.get('_runtime_replay_environment_bridge')
            self.clock.context={'dataset':dataset,'unit_id':unit['unit_id'],'phase_scope':'post-service-calibration'}
            def replay(sid):
                proof=pending['sources'][sid].proof
                result,elapsed=self.clock.call('calibration_owner_replay','owner',verify_replay_proof,
                    proof,context.cached_replay_factory,device=source_device,tolerance=1e-5,
                    comparison_device='replay' if source_device.startswith('cuda') else 'cpu')
                return {**result,'proof_sha256':sid,'seconds':elapsed,
                        'replay_environment_id':environment_id}
            references=JobReferences(unit['unit_id']+'-post-service-owner-audit',replay,self.records)
            for assignment in pending['assignments']:
                vid=assignment['report']['verifier_id']; ledger=self.calibration[dataset,vid]
                settlement=VerifierSettlement(**assignment['settlement'])
                observation=complete_calibration_observation(ledger=ledger,
                    assignment_id=assignment['assignment_id'], report=CommittedVerifierReport(**assignment['report']),
                    settlement=settlement, production=pending['production'], references=references,
                    clock=self.clock, timely=True, owner_cost_per_second=.01)
                receipts=observation['receipts']; elapsed=observation['owner_wall_seconds']
                summary=ledger.summary(alpha=.05/54,independent_stationary_blocks=False)
                rows.append({'dataset':dataset,'block':0,'seed':unit['seed'],'verifier_id':vid,
                    'unit_id':unit['unit_id'],'assignment_id':assignment['assignment_id'],
                    'summary':summary,'observations':ledger.observations,'owner_receipts':receipts,
                    'owner_wall_seconds':elapsed,'actual_qualification':False,'production_route':'safe-defer',
                    'best_case_single_observation_lower':.05/54,'independent_replication':False,
                    'current_block_retroactive_qualification':False})
                rows[-1]['source_device']=source_device
                rows[-1]['replay_environment_id']=environment_id
            context.close_task_lanes()
        with (self.root/'calibration-ledger.jsonl').open('a' if self.config.get('dataset_scoped_disclosure') else 'x') as stream:
            for row in rows: stream.write(json.dumps(row,sort_keys=True)+'\n')
        write_json(self.root/'runtime-screen.json',{'core_counterexamples':self.core_counterexamples,
            'unit_budget_exceeded':self.unit_budget_exceeded,'base_costs':self.base_costs,
            'elapsed_seconds':time.monotonic()-self.started,'independent_replication':False})
        # Audit receipts and immutable recipes are durable; tensor counterexamples are separate.
        self.calibration_jobs.clear()
        for folder in self.retained:
            from sevc.core.scratch_cleanup import release_scratch
            release_scratch(self,folder)
            self.events({'event':'CALIBRATION_SOURCE_CACHE_RELEASED_AFTER_DURABLE_AUDIT','path':folder})
        self.retained.clear()

    def run_cpu(self):
        if not self.config.get('dataset_scoped_disclosure'):
            LocalTinyStudy.run_calibration(self)


def expanded_units(lock, methods):
    """Freeze all related jobs without changing the dataset-block denominator."""
    rows=[]
    behaviors=['constant-accept','constant-reject','uniform-k32','uniform-k39',*lock['frozen_content_strategies']]
    for dataset in lock['dataset_order']:
        common={'dataset':dataset,'block':0,'seed':lock['seeds'][dataset], 'anchor':0,'steps':4,'batch_size':2}
        for invalid in (0,1):
            for method in ('R','G','O'):
                rows.append({**common,'invalid':invalid,'method':methods[method], 'behavior':'honest',
                    'package':'M5' if method=='O' else 'M1','verifier_count':0 if method=='O' else 9 if method=='R' and invalid==0 else 1,
                    'calibration':method=='R' and invalid==0,'reference_check':method!='O'})
            for behavior in behaviors:
                rows.append({**common,'invalid':invalid,'method':methods['R'],'behavior':behavior,
                    'package':'M4L','scenario':('invalid' if invalid else 'valid')+'-unilateral-'+behavior,
                    'colluders':1,'fault':'no-missing','policy':'all-response-certified-ecs','graph':'all-compatible'})
        for anchor,(steps,batch) in itertools.product((128,512),((4,2),(8,8),(16,32))):
            for method,behavior in [('R','honest'),('G','honest'),('O','honest'),('R','prefix-one-step-shortcut')]:
                rows.append({**common,'anchor':anchor,'steps':steps,'batch_size':batch,'invalid':1,
                    'method':methods[method],'behavior':behavior,'package':'M5' if method=='O' else 'M7',
                    'verifier_count':0 if method=='O' else 1,'reference_check':behavior=='honest' and method!='O'})
    for row in rows: row['unit_id']=commitment([CHANGE,row])
    return rows
