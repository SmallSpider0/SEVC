"""Package assembly plugged into tdsc_five_rq_evidence's single runner.

Owns no train/replay loop, device setup, launch policy or process supervisor.
"""
from __future__ import annotations

from dataclasses import asdict, replace
import gc
import itertools
import json
from pathlib import Path
import random
import tempfile
import time
import resource
import sys

from sevc.core.artifacts import write_json, sha256_file
from sevc.evaluation.scoped_campaign_contract import expand_units
from sevc.evaluation.recovery_graph_audit import graph_conflicts, exhaustive_witness, replay_counterfactual
from sevc.verification.reference_acquisition import commitment
from sevc.verification.on_demand_service import build_source_bank, OnlineJob
from sevc.verification.paid_replay_service import StoredReplayProof, VERSION, WRAPPER_DOMAIN
from sevc.verification.replay_coupled_probes import canonicalize_replay_proof, derive_int
from sevc.training import verify_replay_proof, produce_worker_update, WorkerBehavior
from sevc.committee.executed_recovery import execute_recovery, required_segment_decision
from sevc.incentives.native_peer_prediction import solve_score, evaluate_score


FUNCTIONAL_PACKAGES = frozenset({'M1','M2','M2C','M3','M6','M7'})


def order_source_group(rows, release_after_last_use=False):
    """Keep dependencies legal while retaining at most one method template."""
    if not release_after_last_use:
        return rows
    order = {p:i for i,p in enumerate(('M1','M2','M2C','M3','M6','M7'))}
    methods = list(dict.fromkeys(r['method'] for r in rows if r['package'] in order))
    return sorted(rows, key=lambda r: (0, methods.index(r['method']), order[r['package']])
                  if r['package'] in order else (1, 0, 0))


def workload_performance(performance, steps, batch_size):
    """Bound larger source/checkpoint workloads without changing science."""
    result = dict(performance)
    cap = result.get('large_workload_lane_cap')
    if cap and (steps > 4 or batch_size > 2):
        for key in ('task_lanes', 'owner_compile_lanes'):
            result[key] = min(result.get(key, 1), cap)
    return result


class ScopedStudy:
    def __init__(self, config, profile, root, clock, records, sources, commits, events):
        self.config,self.profile,self.root,self.clock=config,profile,root,clock
        self.records,self.sources,self.commits,self.events=records,sources,commits,events
        self.candidate=config['scoped_candidate']
        self.units=expand_units(self.candidate) if profile['full_matrix'] else config['technical_units'][profile['namespace']]
        self.secrets=json.loads(Path(config['private_streams_path']).read_text())
        self.results={}; self.performance=config['performance'][profile['performance_key']]
        (root/'units').mkdir(); (root/'milestones').mkdir()
        write_json(root/'expanded-units.json',self.units)
        write_json(root/'stream-commitments.json',{k:commitment(v) for k,v in self.secrets.items()})
        self.science=self.candidate['science']

    def secret(self, key, purpose):
        return commitment([self.secrets[purpose],key,purpose])

    def finish_unit(self, unit, result, began, cpu_started):
        ended=time.monotonic()
        row={**unit,**result,'wall_seconds':ended-began,
             'started_monotonic':began,'ended_monotonic':ended,
             'cpu_process_seconds':time.process_time()-cpu_started,
             'full_matrix':self.profile['full_matrix'],
             'process_peak_rss_bytes_upper_bound':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)}
        path=self.root/'units'/f"{unit['unit_id']}.json"
        if path.exists():
            raise FileExistsError('unit output identity already exists')
        write_json(path,row); self.results[unit['unit_id']]=row
        self.events({'event':'UNIT_COMPLETE','unit_id':unit['unit_id'],'package':unit['package'],
                     'dataset':unit['dataset'],'status':row['status'],'path':str(path),
                     'sha256':sha256_file(path),'wall_seconds':row['wall_seconds']})
        print(json.dumps({'event':'UNIT_COMPLETE','package':unit['package'],'dataset':unit['dataset'],
                          'unit_id':unit['unit_id'],'status':row['status']}),flush=True)

    def trajectory(self, context, seed, partition):
        import torch
        from sevc.core.runtime import set_global_seed
        began=time.monotonic(); cpu_started=time.process_time()
        self.clock.context={'dataset':context.dataset,'trajectory_seed':seed,'phase_scope':'trajectory-prefix'}
        folder=self.root/'milestones'/f'{context.dataset}-{seed}'
        folder.mkdir()
        set_global_seed(seed)
        rng=random.Random(derive_int(seed,'trajectory-data'))
        batches=[]; indices=[]
        for _ in range(2048):
            ids=rng.sample(range(*partition),64); samples=[context.train[i] for i in ids]
            batches.append((torch.stack([x for x,_ in samples]),torch.tensor([int(y) for _,y in samples])))
            indices.append(ids)
        write_json(folder/'data-order.json',{'seed':seed,'indices':indices,'partition':partition})
        def save(step,model,momentum):
            payload={'step':step,'model':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                     'momentum':momentum,'cpu_rng':torch.get_rng_state(),
                     'cuda_rng':torch.cuda.get_rng_state_all() if context.device.name.startswith('cuda') else [],
                     'data_order_sha256':sha256_file(folder/'data-order.json')}
            path=folder/f'step-{step}.pt'
            torch.save(payload,path)
            self.events({'event':'TRAINING_MILESTONE','dataset':context.dataset,'seed':seed,'step':step,
                         'path':str(path),'sha256':sha256_file(path)})
        model=context.factory().to(context.device.name)
        _,wall=self.clock.call('trajectory-training','trainer',produce_worker_update,
            model,context.factory,batches,WorkerBehavior.NORMAL,device=context.device.name,
            learning_rate=.01,momentum=.9,milestone_callback=save,milestone_steps=(32,512,2048))
        ended=time.monotonic()
        write_json(folder/'prefix-cost.json',{'wall_seconds':ended-began,'training_only_seconds':wall,
            'cpu_process_seconds':time.process_time()-cpu_started,'started_monotonic':began,
            'ended_monotonic':ended,'includes_input_and_milestone_materialization':True,'steps':2048,'batch_size':64})
        return folder

    def run_dataset(self,context,partition):
        import torch
        rows=[r for r in self.units if r['dataset']==context.dataset and r['package']!='M4T']
        # Group immutable source inputs; every method still has its own online
        # cache. M1 precedes exact references from M2/M3/M6.
        def bank_key(r):
            return r['seed'],r.get('anchor',0),r['steps'],r['batch_size'],r['invalid']
        rows.sort(key=lambda r:(bank_key(r), {'M1':0,'M2':1,'M3':2}.get(r['package'],3),r['package']))
        trajectories={}
        base_performance = self.config['performance'][self.profile['performance_key']]
        for group_key,group_iter in itertools.groupby(rows,key=bank_key):
            group=list(group_iter); seed,anchor,steps,size,invalid=group_key
            self.performance = workload_performance(base_performance, steps, size)
            release_early = self.performance.get('release_preparation_after_last_use', False)
            group = order_source_group(group, release_early)
            last_use = {r['method']:r['unit_id'] for r in group
                        if r['package'] in FUNCTIONAL_PACKAGES and not r.get('reuse_of')}
            active=[r for r in group if not r.get('reuse_of')]
            if not active:
                for unit in group:
                    original=self.results[unit['reuse_of']]
                    self.finish_unit(unit,{'status':'EXACT_REUSE','reuse_sha256':sha256_file(self.root/'units'/f"{unit['reuse_of']}.json")},time.monotonic(),time.process_time())
                continue
            initial=momentum=initial_rng=None
            if anchor:
                if seed not in trajectories:
                    trajectories[seed]=self.trajectory(context,seed,partition)
                saved=torch.load(trajectories[seed]/f'step-{anchor}.pt',map_location='cpu',weights_only=False)
                initial,momentum=saved['model'],saved['momentum']
                initial_rng={'cpu_rng':saved['cpu_rng'],'cuda_rng':saved['cuda_rng']}
            namespace=self.profile['namespace']+f'-anchor{anchor}'
            group_id=commitment([context.dataset,group_key,namespace])
            # Disposable tensor caches are outside the result tree. On technical
            # exception retain the path and its files for affected-only recovery.
            scratch=Path(tempfile.mkdtemp(prefix='sevc-scoped-'+group_id[:10]+'-',dir=self.config['scratch_parent']))
            self.events({'event':'SCRATCH_CREATED','group_id':group_id,'path':str(scratch)})
            self.clock.context={'dataset':context.dataset,'source_group':group_id,'phase_scope':'shared-trainer-prefix'}
            began=time.monotonic(); prefix_cpu_started=time.process_time()
            bank,source_rows,cache=build_source_bank(context,seed=seed,namespace=namespace,partition=partition,
                steps=steps,batch_size=size,invalid_count=invalid,clock=self.clock,initial_state=initial,
                initial_momentum=momentum,scratch_dir=scratch,initial_rng=initial_rng,
                access_profile=self.performance.get("tensor_access", "strict"))
            prefix=time.monotonic()-began
            prefix_cpu=time.process_time()-prefix_cpu_started
            self.sources({'group_id':group_id,'dataset':context.dataset,'seed':seed,'anchor':anchor,
                          'steps':steps,'batch_size':size,'invalid':invalid,'sources':source_rows,
                          'shared_trainer_prefix_seconds':prefix,'shared_trainer_prefix_cpu_seconds':prefix_cpu,
                          'started_monotonic':began,'ended_monotonic':began+prefix})
            templates={}
            retained_folders=set()
            template_folders={}
            targets=None
            role_secret=self.secret(group_id,'roles')
            for unit in group:
                began=time.monotonic(); cpu_started=time.process_time()
                if unit.get('reuse_of'):
                    self.finish_unit(unit,{'status':'EXACT_REUSE','reuse_sha256':sha256_file(self.root/'units'/f"{unit['reuse_of']}.json")},began,cpu_started)
                    continue
                uid=unit['unit_id']; self.clock.context={'dataset':context.dataset,'unit_id':uid,
                    'package':unit['package'],'source_group':group_id,'phase_scope':'online'}
                # Target construction is experiment work, never a free owner
                # reference. Each RCMP job independently repeats the selection.
                if targets is None and unit['method'] not in ('rcmp-probe-source-v2','rcmp-audit-8-v2','rcmp-audit-all-v2'):
                    self.clock.context['phase_scope']='pairing-fixture'
                    fixture=OnlineJob(bank=bank,context=context,clock=self.clock,performance=self.performance,
                        job_id=group_id+'-fixture',method_key='rcmp-probe-source-v2',role_secret=role_secret,
                        emit=self.records,delivery_scratch_dir=None,prepare_only=True)
                    targets=fixture.production_source_ids if fixture.issued else None
                    del fixture; gc.collect()
                    self.clock.context['phase_scope']='online'
                online_started=time.monotonic()
                online_cpu_started=time.process_time()
                jobs=[]; job_temp=[]
                reusable = (self.performance.get('conditional_preparation_reuse',False)
                            and unit['package'] in FUNCTIONAL_PACKAGES)
                def prepare(method=unit['method'],suffix='job'):
                    if reusable and method in templates:
                        result=templates[method].reuse_preparation(uid+'-'+suffix)
                        jobs.append(result)
                        return result
                    folder=scratch/(uid+'-'+suffix); folder.mkdir(); job_temp.append(folder)
                    def gold():
                        g,rs,_=build_source_bank(context,seed=seed,namespace=namespace+'-gold',partition=partition,
                            steps=steps,batch_size=size,invalid_count=0,clock=self.clock,
                            initial_state=initial,initial_momentum=momentum,count=8,source_offset=40,role='owner',scratch_dir=folder,
                            initial_rng=initial_rng,access_profile=self.performance.get("tensor_access", "strict"))
                        self.sources({'group_id':group_id,'unit_id':uid,'kind':'paid-hidden-gold','sources':rs})
                        return g
                    comparator=not method.startswith('rcmp-')
                    disk_policy=self.performance.get("disk_delivery",False)
                    disk_delivery=steps>4 or (reusable if disk_policy=="large-or-shared" else bool(disk_policy))
                    result=OnlineJob(bank=bank,context=context,clock=self.clock,performance=self.performance,
                        job_id=uid+'-'+suffix,method_key=method,role_secret=role_secret,emit=self.records,
                        paired_production=targets if comparator else None,gold_factory=gold,
                        delivery_scratch_dir=folder if disk_delivery else None,
                        prepare_only=unit['package']=='M0')
                    if self.performance.get('trim_cpu_arenas',False):
                        from sevc.core.runtime import release_process_memory
                        release_process_memory(cuda=False)
                    if result.issued and targets is not None and tuple(result.production_source_ids)!=tuple(targets):
                        raise ValueError('independent RCMP preparation differs from paired targets')
                    if reusable and result.issued:
                        templates[method]=result.preparation_template()
                        retained_folders.add(folder)
                        template_folders[method] = folder
                    jobs.append(result)
                    return result
                prep_start=time.monotonic(); job=prepare(); preparation=time.monotonic()-prep_start
                if targets is None and job.issued:
                    targets=job.production_source_ids
                result={'status':'MEASURED' if job.issued else job.preparation['status'],'issued':job.issued,
                        'source_group':group_id,'preparation':job.preparation,
                        'shared_trainer_prefix_seconds':prefix,'online_preparation_seconds':preparation,
                        'shared_trainer_prefix_cpu_seconds':prefix_cpu,
                        'preparation_measurement': 'conditional-on-prepared-input' if reusable else 'independent-preparation',
                        'preparation_donor_job_id': job.preparation.get('conditional_preparation_from'),
                        'tasks':job.task_rows,'assignments':[],'production_source_ids':list(job.production_source_ids) if job.issued else None}
                def serve(current,behavior,aid,job_id=None,verifier_id=None):
                    start=time.monotonic()
                    value=current.serve(behavior=behavior,seed=unit['seed'],assignment_id=aid,
                        audit_secret=self.secret(aid,'audits'),trainer_cache=cache,commits=self.commits,
                        job_id=job_id,verifier_id=verifier_id)
                    report,settled,detail=value
                    row={'assignment_id':aid,'method':current.method_key,'behavior':behavior,'report':asdict(report),
                         'settlement':asdict(settled),'execution':detail,'wall_seconds':time.monotonic()-start}
                    result['assignments'].append(row)
                    return value
                if job.issued and unit['package']!='M0':
                    if unit['package'] in {'M4L','M4S','M4X'}:
                        second=prepare(suffix='job1')
                        if not second.issued:
                            raise ValueError('same identity per-job reference outcome differs')
                        calibration=self.profile['calibration'][context.dataset]
                        def callback(jid,vid):
                            current=job if jid=='j0' else second
                            colluding=jid=='j0' and int(vid[1:])<unit.get('colluders',0)
                            behavior=unit['behavior'] if colluding else 'honest'
                            report,settled,detail=serve(current,behavior,uid+'-'+jid+'-'+vid,jid,vid)
                            actual=dict(settled.diagnostics)['owner_admission_seconds']/calibration['cost_unit_seconds']
                            settled=replace(settled,diagnostics=settled.diagnostics+(("owner_work_cost",actual),))
                            return report,settled,detail
                        result['recovery']=execute_recovery(policy=unit['policy'],fault=unit['fault'],seed=seed,
                            required_ids=tuple(job.production),execute=callback,clock=self.clock,
                            timeout_seconds=calibration['timeout_seconds'],deadline_seconds=calibration['deadline_seconds'],
                            owner_cost_reserve_per_assignment=calibration['owner_reserve_units'],
                            owner_prepaid_cost_per_job=preparation/calibration['cost_unit_seconds'],minimum_pass_count=3,
                            public_conflicts=graph_conflicts(unit['graph']),
                            missing_ids=('v0','v3') if unit['package']=='M4X' else None,
                            virtual_fault_wait=self.performance.get('virtual_fault_wait',False))
                    elif job.method.owner_direct:
                        def direct(sid):
                            return sid,job.references.acquire(sid)['passed']
                        answers=dict(context.task_lanes(self.performance.get('task_lanes',1)).map(direct,job.production_source_ids,self.clock))
                        result['owner_direct_answers']=answers
                        result['trainer_route']='accept' if all(answers.values()) else 'reject'
                    elif unit['package']=='M3':
                        result['probe_replays']=self.probe_replays(context,job)
                    else:
                        n=unit.get('verifier_count',1)
                        for v in range(n):
                            serve(job,unit['behavior'],uid+f'-v{v}',uid,f'v{v}')
                        if n==3:
                            from sevc.incentives.verifier_protocol import CommittedVerifierReport
                            reports=[CommittedVerifierReport(**r['report']) for r in result['assignments']]
                            eligible=[r['report']['verifier_id'] for r in result['assignments'] if r['settlement']['status']=='PASS']
                            decision=required_segment_decision(tuple(job.production),reports,eligible)
                            result['trainer_route']='safe-defer' if decision is None else 'accept' if decision else 'reject'
                        if unit['package']=='M7' and unit.get('replays'):
                            result['probe_replays']=self.probe_replays(context,job)
                result['online_suffix_seconds']=time.monotonic()-online_started
                result['online_cpu_process_seconds']=time.process_time()-online_cpu_started
                result['pairing_fixture_seconds']=online_started-began
                result['composed_prefix_plus_suffix_seconds']=prefix+result['online_suffix_seconds']
                result['shared_prefix_is_independent_repeat']=False
                self.finish_unit(unit,result,began,cpu_started)
                jobs.clear(); job=second=None; callback=serve=prepare=None; gc.collect()
                if self.performance.get('trim_cpu_arenas',False):
                    from sevc.core.runtime import release_process_memory
                    release_process_memory(cuda=False)
                import shutil
                if release_early and last_use.get(unit['method']) == uid:
                    templates.pop(unit['method'], None)
                    folder = template_folders.pop(unit['method'], None)
                    if folder is not None:
                        retained_folders.discard(folder)
                        gc.collect()
                        shutil.rmtree(folder)
                        self.events({'event':'PREPARATION_RELEASED_AFTER_LAST_USE',
                                     'unit_id':uid,'method':unit['method'],'path':str(folder)})
                for folder in job_temp:
                    if folder not in retained_folders and folder.exists():
                        shutil.rmtree(folder)
            templates.clear()
            del bank,cache,initial,momentum; gc.collect()
            import shutil
            shutil.rmtree(scratch)
            self.events({'event':'SCRATCH_RELEASED_AFTER_DURABLE_GROUP','group_id':group_id})

    def probe_replays(self,context,job):
        result=[]
        self.clock.context['phase_scope']='offline-reference-diagnostic'
        for task in job.tasks:
            if task.task_id not in dict(job.probes.answers):
                continue
            proof=task.proof.load() if isinstance(task.proof,StoredReplayProof) else task.proof
            (canonical,_),_=self.clock.call('diagnostic-canonicalize','offline',canonicalize_replay_proof,
                proof,context.factory(),task.descriptor,protocol_version=VERSION,wrapper_seed_domain=WRAPPER_DOMAIN)
            observed,wall=self.clock.call('diagnostic-probe-replay','offline',verify_replay_proof,canonical,
                context.cached_replay_factory,device=context.device.name,tolerance=1e-5,
                comparison_device=self.performance['comparison_device'])
            result.append({'task_id':task.task_id,'reference':dict(job.probes.answers)[task.task_id],
                           'actual':observed,'wall_seconds':wall})
        self.clock.context['phase_scope']='online'
        return result

    def run_cpu(self):
        from sevc.verification.paid_replay_service import OwnerProbeReferences, settle_service
        from sevc.incentives.verifier_protocol import CommittedVerifierReport
        scores={}; native=next(p for p in self.candidate['packages'] if p['id']=='M8')
        for unit in self.units:
            if unit['package'] not in {'M4B','M4T','M8'}:
                continue
            start=time.monotonic(); cpu_started=time.process_time(); self.clock.context={'unit_id':unit['unit_id'],'package':unit['package'],'phase_scope':'offline-CPU'}
            result={'status':'MEASURED'}
            if unit['package']=='M4T':
                original=self.results[unit['trace_from']]
                if not original['issued']:
                    result.update(status='UNIDENTIFIED_SOURCE_NOT_ISSUED',counterfactual_result=None,
                                  reason=original['status'])
                else:
                    result['counterfactual_result']=replay_counterfactual(original['recovery'],unit['counterfactual'])
            elif unit['package']=='M4B':
                refs=OwnerProbeReferences(tuple((f'p{i}',i<4) for i in range(8)))
                def callback(jid,vid):
                    report=CommittedVerifierReport.create(scenario_id='structural',verifier_id=vid,job_id=jid,
                        ordered_segment_ids=('target',)+tuple(f'p{i}' for i in range(8)),
                        verdicts=(True,)+tuple(i<4 for i in range(8)),nonce='structural')
                    return report,settle_service(report,refs,cost_seconds=0,effort_fraction=1),[]
                trace=execute_recovery(policy=unit['policy'],fault='structural',seed=0,required_ids=('target',),
                    execute=callback,clock=self.clock,timeout_seconds=1,deadline_seconds=1e6,
                    sleep=lambda _:None,minimum_pass_count=3,public_conflicts=graph_conflicts(unit['graph']),
                    missing_ids=tuple(f'v{i}' for i in unit['missing']))
                result.update(recovery=trace,oracle=exhaustive_witness(unit['graph'],unit['missing']),structural_only=True)
            else:
                method=unit['method']
                if method not in scores:
                    scores[method]=({'status':'PUBLISHED_ROUNDED','score':native['published_score']}
                        if 'published' in method else solve_score(native,simple_agreement=method.startswith('simple-agreement')))
                result['solution']=scores[method]
                if scores[method]['score'] is not None:
                    result['native']=evaluate_score(native,scores[method]['score'],unit['epsilon'])
                else:
                    result['status']='INFEASIBLE_VALID_NATIVE_RESULT'
            self.finish_unit(unit,result,start,cpu_started)
