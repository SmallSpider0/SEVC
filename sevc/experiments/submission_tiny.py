"""Submission tiny assembly; inherits the one ScopedStudy lifecycle."""
from pathlib import Path
import itertools
import shutil
import time

from sevc.core.artifacts import write_json, sha256_file
from sevc.experiments.local_tiny_feasibility import LocalTinyStudy, expanded_units as base_units
from sevc.verification.reference_acquisition import commitment
from sevc.verification.on_demand_service import build_source_bank
from sevc.verification.depol_local import KEY, execute_native_group
from sevc.incentives.native_peer_prediction import native_cell

CHANGE='experiment-tdsc-submission-tiny-v1'


def expanded_units(lock, methods, native):
    rows=base_units(lock,methods)
    for row in rows:
        row.pop('unit_id')
        row['common_raw_roster']=True
        if row['package']=='M4L':row['execution_order']=10
        elif not row['anchor']:
            order=list(methods.values()); shift=lock['dataset_order'].index(row['dataset'])%3
            row['execution_order']=(order[shift:]+order[:shift]).index(row['method'])
    for d in lock['dataset_order']:
        common={'dataset':d,'block':0,'seed':lock['seeds'][d],'anchor':0,'steps':4,'batch_size':2}
        for invalid in (0,1):
            for behavior in ('honest','constant-accept','constant-reject','uniform-k32','uniform-k39','prefix-one-step-shortcut'):
                rows.append({**common,'invalid':invalid,'method':KEY,'behavior':behavior,'package':'DP',
                             'verifier_count':3,'execution_order':20})
            for policy in ('all-response-certified-ecs','online-only-joint-matching-v1','fixed-order-same-reserve','no-recovery'):
                rows.append({**common,'invalid':invalid,'method':methods['R'],'behavior':'honest',
                    'package':'M4L','scenario':'paired-one-missing-'+policy,'colluders':0,
                    'fault':'one-missing','policy':policy,'graph':'all-compatible',
                    'common_raw_roster':True,'execution_order':12})
    for method,eps in itertools.product(native['methods'],native['epsilon']):
        rows.append({'dataset':'native','package':'M8','method':method,'epsilon':eps,'steps':0,'batch_size':0,'anchor':0})
    for row in rows:row['unit_id']=commitment([CHANGE,row])
    return rows


class SubmissionTinyStudy(LocalTinyStudy):
    def finish_unit(self,unit,result,began,cpu_started):
        prior=self.restored_groups.get(result.get('source_group'),{})
        origin=prior.get('source_device',self.profile['device'])
        super().finish_unit(unit,{**result,'runtime_device':self.profile['device'],
            'replay_environment_id':self.config.get('_runtime_replay_environment_id'),
            'owner_cost_revision':self.config.get('owner_cost_revision','pre-owner-cost-repair'),
            'source_origin_device':origin,
            'cross_backend_source_replay':origin!=self.profile['device']},began,cpu_started)

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        from sevc.experiments.submission_continuation import restore
        restore(self)

    def run_dataset(self, context, partition):
        if self.config.get('owner_cost_repair_pair') and self.phase == 'core' and context.dataset == 'cifar10':
            from sevc.experiments.owner_cost_repair import measure_restored_owner_pair
            measure_restored_owner_pair(self, context)
        if self.config.get('dataset_scoped_disclosure') and self.phase == 'core':
            from sevc.verification.disclosure import ProbeDisclosureEpoch
            from sevc.experiments.scoped_five_rq_units import unit_assignment_ids
            from sevc.verification.on_demand_service import METHODS
            planned = [a for u in self.all_units if u['dataset'] == context.dataset
                and u['package'] != 'M8' and METHODS.get(u['method']).require_complete_probes
                for a in unit_assignment_ids(u)]
            completed = [u for u in self.results.values() if u['dataset'] == context.dataset]
            old_answers = self.disclosure_answers
            self.disclosure_epoch = ProbeDisclosureEpoch(self.science['namespace']+'-'+context.dataset, planned)
            self.disclosure_answers = dict(old_answers) if completed else {}
            for unit in completed:
                for assignment in unit.get('assignments', []):
                    aid = assignment['assignment_id']
                    if self.disclosure_epoch.covers([aid]):
                        self.disclosure_epoch.begin(aid)
                        self.disclosure_epoch.finish(aid, assignment['settlement']['status'])
        super().run_dataset(context, partition)

    def retain_group(self, group, scratch):
        retained = super().retain_group(group, scratch)
        if retained and self.config.get('dataset_scoped_disclosure'):
            # After all group services, only production references are needed
            # by post-service calibration. Immutable recipes/hashes are logged.
            for pending in self.calibration_jobs:
                keep = set(pending['production'].values())
                pending['sources'] = {k:v for k,v in pending['sources'].items() if k in keep}
            keep_paths = {str(v.path) for p in self.calibration_jobs for v in p['sources'].values()}
            for path in scratch.glob('*.pt'):
                if str(path) not in keep_paths:
                    self.events({'event':'NONPRODUCTION_CACHE_RELEASED_AFTER_GROUP',
                        'path':str(path),'sha256':sha256_file(path)})
                    path.unlink()
        return retained

    def restored_source_group(self,context,group_id):
        from sevc.experiments.submission_continuation import restore_group
        return restore_group(self,group_id)

    def restored_production_targets(self,group_id):
        from sevc.experiments.submission_continuation import production_targets
        return production_targets(self.results.values(),group_id)

    def special_unit(self,context,unit,bank,targets,partition,scratch):
        if unit['package']!='DP':return None
        if targets is None:
            return {'status':'HOLD_PAIRED_PRODUCTION_NOT_ISSUED','issued':False,'assignments':[]}
        folder=scratch/unit['unit_id']; folder.mkdir()
        spec=self.lock['depol']
        estimators,source_rows,_=build_source_bank(context, seed=unit['seed'],
            namespace=self.profile['namespace']+'-depol-estimator',
            partition=tuple(spec['estimator_partitions'][unit['dataset']]),
            steps=unit['steps'] if spec.get('match_estimator_to_unit_workload') else spec.get('estimator_steps',4),
            batch_size=unit['batch_size'] if spec.get('match_estimator_to_unit_workload') else spec.get('estimator_batch_size',2),
            invalid_count=0,count=1,source_offset=80,role='owner',clock=self.clock,
            scratch_dir=folder,access_profile='authenticated-mmap')
        self.sources({'unit_id':unit['unit_id'],'kind':'depol-estimator-paid','sources':source_rows})
        selected={s.hashes['proof_sha256']:s for s in bank}
        start=time.monotonic()
        native=execute_native_group(sources=[selected[t] for t in targets],estimator=estimators[0],
            factory=context.cached_replay_factory,device=context.device.name,spec=spec,
            seed=int(self.secret(unit['dataset'],'depol')[:8],16),behavior=unit['behavior'],
            folder=folder/'native',clock=self.clock)
        native['estimator_source_rows']=source_rows
        native['production_source_ids']=list(targets)
        native['source_checkpoint_files']=[{'path':str(p),'sha256':sha256_file(p)} for p in (folder/'native').glob('*.pt')]
        # Retain one complete native witness if a deviation is still eligible.
        questionable=unit['behavior']!='honest' and native['final']['verifier_reward_eligible'][0]
        if questionable:
            witness=self.root/'native-witnesses'/unit['unit_id'];witness.mkdir(parents=True)
            files={}
            for name in ('trainer','v0','v1','v2'):
                source=folder/'native'/f'{name}-0.pt'
                if self.config.get('change_id') in {'experiment-tdsc-f-fixed-design-v1', 'experiment-tdsc-f-fixed-design-v2', 'experiment-tdsc-rq1-detection-supplement-v1'}:
                    digest=sha256_file(source)
                    shared=self.root/'native-witnesses'/'objects';shared.mkdir(exist_ok=True)
                    target=shared/(digest+'.pt')
                    if not target.exists():shutil.copy2(source,target)
                    if sha256_file(target)!=digest:raise ValueError('native witness digest mismatch')
                    files[source.name]={'path':str(target.relative_to(self.root)),'sha256':digest}
                else:
                    shutil.copy2(source,witness/source.name)
            if files:native['retained_witness_files']=files
            write_json(witness/'native.json',native)
        del estimators,selected
        if self.config.get('change_id') in {'experiment-tdsc-f-fixed-design-v1', 'experiment-tdsc-f-fixed-design-v2', 'experiment-tdsc-rq1-detection-supplement-v1'}:
            from sevc.core.scratch_cleanup import release_scratch
            release_scratch(self,folder)
        else:
            shutil.rmtree(folder)
        return {'status':'MEASURED','issued':False,'protocol_measured':True,'assignments':[],
                'native':native,'native_wall_seconds':time.monotonic()-start,
                'production_source_ids':list(targets),'endpoint_scope':'native checkpoint interval verification',
                'shared_prefix_is_independent_repeat':False}

    def run_structures(self):
        super().run_structures()
        from sevc.evaluation.submission_tiny import recovery_boundaries
        write_json(self.root/'recovery-boundaries.json',recovery_boundaries(self.clock))
        from sevc.committee.depol_arbitration import NativeCommitments
        timeout=NativeCommitments(['trainer','v0','v1','v2'])
        timeout.commit('trainer',commitment(['trainer',[]]))
        write_json(self.root/'native-timeout-boundary.json',{'result':timeout.expire(),
            'scope':'finite protocol fixture, no training block; missing monetary and adjudication rule unknown',
            'new_scientific_blocks':0})

    def run_cpu(self):
        super().run_cpu()
        native=next(p for p in self.candidate['packages'] if p['id']=='M8');scores={}
        for unit in self.all_units:
            if unit['package']!='M8':continue
            started=time.monotonic();cpu=time.process_time();method=unit['method']
            solution,value=native_cell(native,method,unit['epsilon'],scores)
            self.finish_unit(unit,{'status':'MEASURED' if value else 'INFEASIBLE_VALID_NATIVE_RESULT',
                'issued':False,'assignments':[],'protocol_measured':True,'solution':solution,'native_game':value},started,cpu)
