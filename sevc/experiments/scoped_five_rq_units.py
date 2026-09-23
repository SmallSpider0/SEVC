"""Package assembly plugged into tdsc_five_rq_evidence's single runner.

Owns no train/replay loop, device setup, launch policy or process supervisor.
"""
from __future__ import annotations
from sevc.core.scratch_cleanup import release_scratch

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
from sevc.verification.on_demand_service import build_source_bank, OnlineJob, METHODS
from sevc.verification.paid_replay_service import StoredReplayProof, VERSION, WRAPPER_DOMAIN
from sevc.verification.replay_coupled_probes import canonicalize_replay_proof, derive_int
from sevc.training import verify_replay_proof, produce_worker_update, WorkerBehavior
from sevc.committee.executed_recovery import execute_recovery, required_segment_decision
from sevc.incentives.native_peer_prediction import native_cell


FUNCTIONAL_PACKAGES = frozenset({'M1','M2','M2C','M3','M6','M7'})


def unit_assignment_ids(unit):
    uid = unit['unit_id']
    return ([uid+'-'+j+'-'+f'v{i}' for j in ('j0','j1') for i in range(9)]
            if unit['package'] in {'M4L','M4S','M4X'}
            else [uid+f'-v{i}' for i in range(unit.get('verifier_count',1))])


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
    def recovery_reputations(self, dataset):
        """None is reserved for constructed fixtures; formal subclasses bind actual qualification."""
        return None

    def restored_production_targets(self, group_id):
        return None

    def restored_source_group(self, context, group_id):
        return None

    def special_unit(self, context, unit, bank, targets, partition, scratch):
        return None

    def before_group(self, context, group_key, group):
        return True

    def unit_context(self, unit):
        from contextlib import nullcontext
        return nullcontext()

    def before_unit(self, unit):
        return True

    def after_unit(self, unit, result, jobs, bank, scratch):
        pass

    def after_group(self, context, group, bank, group_id):
        pass

    def retain_group(self, group, scratch):
        return False

    def __init__(self, config, profile, root, clock, records, sources, commits, events):
        self.config,self.profile,self.root,self.clock=config,profile,root,clock
        self.records,self.sources,self.commits,self.events=records,sources,commits,events
        self.candidate=config['scoped_candidate']
        self.claim_linked = self.candidate.get('protocol_variant') == 'claim-linked-tiny-v1'
        self.units=(config['fixed_design_units'] if config.get('change_id') in {'experiment-tdsc-f-fixed-design-v1', 'experiment-tdsc-f-fixed-design-v2', 'experiment-tdsc-rq1-detection-supplement-v1', 'experiment-tdsc-rq4-overhead-supplement-v1', 'experiment-tdsc-rq2-honest-participation-v1', 'experiment-tdsc-rq4-cost-scaling-v1'}
            else expand_units(self.candidate) if profile['full_matrix'] else config['technical_units'][profile['namespace']])
        self.secrets=json.loads(Path(config['private_streams_path']).read_text())
        self.results={}; self.performance=config['performance'][profile['performance_key']]
        (root/'units').mkdir(); (root/'milestones').mkdir()
        write_json(root/'expanded-units.json',self.units)
        write_json(root/'stream-commitments.json',{k:commitment(v) for k,v in self.secrets.items()})
        self.science=self.candidate['science']
        self.disclosure_epoch, self.disclosure_answers = None, {}
        if self.claim_linked:
            from sevc.verification.disclosure import ProbeDisclosureEpoch
            planned = [a for unit in self.units if unit.get('method') and unit['package'] != 'M8' and METHODS.get(unit['method']).require_complete_probes
                       for a in unit_assignment_ids(unit)]
            self.disclosure_epoch = ProbeDisclosureEpoch(self.science['namespace'], planned)

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
        trajectory_spec = self.config.get('trajectory', {'steps':2048, 'batch_size':64, 'anchors':[32,512,2048]})
        for _ in range(trajectory_spec['steps']):
            ids=rng.sample(range(*partition),trajectory_spec['batch_size']); samples=[context.train[i] for i in ids]
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
            learning_rate=.01,momentum=.9,milestone_callback=save,milestone_steps=tuple(trajectory_spec['anchors']))
        ended=time.monotonic()
        write_json(folder/'prefix-cost.json',{'wall_seconds':ended-began,'training_only_seconds':wall,
            'cpu_process_seconds':time.process_time()-cpu_started,'started_monotonic':began,
            'ended_monotonic':ended,'includes_input_and_milestone_materialization':True,
            'steps':trajectory_spec['steps'],'batch_size':trajectory_spec['batch_size']})
        return folder

    def run_dataset(self,context,partition):
        import torch
        rows=[r for r in self.units if r['dataset']==context.dataset and r['package']!='M4T']
        # Group immutable source inputs; every method still has its own online
        # cache. M1 precedes exact references from M2/M3/M6.
        def bank_key(r):
            return r['seed'],r.get('anchor',0),r['steps'],r['batch_size'],r['invalid']
        rows.sort(key=lambda r:(bank_key(r), r.get('execution_order', {'M1':0,'M2':1,'M3':2}.get(r['package'],3)),r['package']))
        trajectories={}
        source_donors={}
        base_performance = self.config['performance'][self.profile['performance_key']]
        for group_key,group_iter in itertools.groupby(rows,key=bank_key):
            group=list(group_iter); seed,anchor,steps,size,invalid=group_key
            production_count={r.get('production_count',32) for r in group}
            if len(production_count)!=1:
                raise ValueError('one source bank serves one registered production count')
            production_count=production_count.pop()
            if not self.before_group(context, group_key, group):
                continue
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
            restored=self.restored_source_group(context,group_id)
            donor_key=(seed,anchor,steps,size,namespace,tuple(partition))
            # Paired-validity reuse derives exactly one of forty sources.
            donor=source_donors.get(donor_key) if invalid==1 and production_count==32 else None
            source_reuse=None
            if restored is None and donor is not None:
                from sevc.experiments.source_variants import derive_one_invalid_bank
                bank,source_rows,cache=derive_one_invalid_bank(context,
                    valid_bank=donor['bank'],valid_rows=donor['rows'],valid_cache=donor['cache'],
                    seed=seed,namespace=namespace,partition=partition,steps=steps,batch_size=size,
                    clock=self.clock,initial_state=initial,initial_momentum=momentum,
                    scratch_dir=scratch,initial_rng=initial_rng,
                    access_profile=self.performance.get("tensor_access", "strict"),
                    capture_compact_headers=self.performance.get("compact_source_headers",False))
                physical_prefix=time.monotonic()-began
                prefix=donor['prefix']+physical_prefix
                prefix_cpu=donor['prefix_cpu']+time.process_time()-prefix_cpu_started
                source_reuse={'donor_group_id':donor['group_id'],'unchanged_sources_reused':39,
                    'new_source_materializations':1,'physical_new_prefix_seconds':physical_prefix,
                    'logical_input_prefix_seconds':prefix,'independent_new_block':False}
                self.events({'event':'PAIRED_VALIDITY_TRAINER_INPUT_REUSE','group_id':group_id,**source_reuse})
            elif restored is None:
                bank,source_rows,cache=build_source_bank(context,seed=seed,namespace=namespace,partition=partition,
                    steps=steps,batch_size=size,invalid_count=invalid,clock=self.clock,initial_state=initial,
                    initial_momentum=momentum,scratch_dir=scratch,initial_rng=initial_rng,
                    count=production_count+8,population=production_count+8,
                    access_profile=self.performance.get("tensor_access", "strict"),
                    capture_compact_headers=self.performance.get("compact_source_headers", False))
                prefix=time.monotonic()-began
                prefix_cpu=time.process_time()-prefix_cpu_started
            else:
                bank,source_rows,cache,prefix,prefix_cpu=restored
                self.events({'event':'SOURCE_GROUP_HASH_REUSED','group_id':group_id,
                             'new_source_materializations':0})
            if (self.config.get('reuse_paired_validity_sources',False) and invalid==0 and restored is None
                    and any(r['seed']==seed and r.get('anchor',0)==anchor and r['steps']==steps
                            and r['batch_size']==size and r['invalid']==1 for r in rows)):
                source_donors[donor_key]={'bank':bank,'rows':source_rows,'cache':cache,
                    'prefix':prefix,'prefix_cpu':prefix_cpu,'scratch':scratch,'group_id':group_id}
            from sevc.core.replay_environment import require_sources
            require_sources(bank, getattr(context, 'replay_environment_id', None), getattr(context, 'replay_environment_bridge', None))
            self.sources({'group_id':group_id,'dataset':context.dataset,'seed':seed,'anchor':anchor,
                          'replay_environment_id':getattr(context, 'replay_environment_id', None),
                          'steps':steps,'batch_size':size,'invalid':invalid,'sources':source_rows,
                          'shared_trainer_prefix_seconds':prefix,'shared_trainer_prefix_cpu_seconds':prefix_cpu,
                          'source_reuse':source_reuse,
                          'started_monotonic':began,'ended_monotonic':time.monotonic()})
            templates={}
            group_disclosure, group_answers = self.disclosure_epoch, self.disclosure_answers
            retained_folders=set()
            template_folders={}
            targets=self.restored_production_targets(group_id)
            role_secret=self.secret(group_id,'roles')
            for unit in group:
                if unit.get('performance_arm'):
                    self.performance = workload_performance(
                        self.config['performance'][unit['performance_arm']], steps, size)
                if not self.before_unit(unit):
                    continue
                with self.unit_context(unit):
                    began=time.monotonic(); cpu_started=time.process_time()
                    if unit.get('reuse_of'):
                        self.finish_unit(unit,{'status':'EXACT_REUSE','reuse_sha256':sha256_file(self.root/'units'/f"{unit['reuse_of']}.json")},began,cpu_started)
                        continue
                    uid=unit['unit_id']; self.clock.context={'dataset':context.dataset,'unit_id':uid,
                        'package':unit['package'],'source_group':group_id,'phase_scope':'online'}
                    # Target construction is experiment work, never a free owner
                    # reference. Each RCMP job independently repeats the selection.
                    if targets is None and METHODS.get(unit['method']).probes != 'source':
                        self.clock.context['phase_scope']='pairing-fixture'
                        fixture=OnlineJob(bank=bank,context=context,clock=self.clock,performance=self.performance,
                            job_id=group_id+'-fixture',method_key=(self.science['methods']['R'] if self.claim_linked else 'rcmp-probe-source-v2'),role_secret=role_secret,
                            emit=self.records,delivery_scratch_dir=None,prepare_only=True,
                            production_count=production_count)
                        targets=fixture.production_source_ids if fixture.issued else None
                        del fixture; gc.collect()
                        self.clock.context['phase_scope']='online'
                    online_started=time.monotonic()
                    online_cpu_started=time.process_time()
                    special = self.special_unit(context, unit, bank, targets, partition, scratch)
                    if special is not None:
                        special.update(source_group=group_id, shared_trainer_prefix_seconds=prefix,
                            online_suffix_seconds=time.monotonic()-online_started)
                        self.finish_unit(unit, special, began, cpu_started)
                        continue
                    jobs=[]; job_temp=[]
                    reusable = (self.performance.get('conditional_preparation_reuse',False)
                                and unit['package'] in FUNCTIONAL_PACKAGES
                                and not METHODS.get(unit['method']).require_complete_probes)
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
                                initial_rng=initial_rng,access_profile=self.performance.get("tensor_access", "strict"),
                                capture_compact_headers=self.performance.get("compact_source_headers", False))
                            self.sources({'group_id':group_id,'unit_id':uid,'kind':'paid-hidden-gold','sources':rs})
                            return g
                        comparator=not method.startswith('rcmp-')
                        disk_policy=self.performance.get("disk_delivery",False)
                        disk_delivery=steps>4 or (reusable if disk_policy=="large-or-shared" else bool(disk_policy))
                        result=OnlineJob(bank=bank,context=context,clock=self.clock,performance=self.performance,
                            job_id=unit.get('public_job_binding',uid)+'-'+suffix,method_key=method,role_secret=role_secret,emit=self.records,
                            paired_production=targets if comparator else None,gold_factory=gold,
                            delivery_scratch_dir=folder if disk_delivery else None,
                            prepare_only=unit['package']=='M0',production_count=production_count)
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
                        if result.issued and result.method.require_complete_probes:
                            result.begin_service_epoch(unit_assignment_ids(unit), shared_epoch=group_disclosure)
                            for key, answer in result.probes.answers:
                                if key in group_answers and group_answers[key] != answer:
                                    raise ValueError('conflicting shared probe identity')
                                group_answers[key] = answer
                        return result
                    prep_start=time.monotonic(); job=prepare(); preparation_wall=time.monotonic()-prep_start
                    preparation=preparation_wall+job.header_capture_charge
                    if targets is None and job.issued:
                        targets=job.production_source_ids
                    result={'status':'MEASURED' if job.issued else job.preparation['status'],'issued':job.issued,
                            'source_group':group_id,'preparation':job.preparation,
                            'shared_trainer_prefix_seconds':prefix,'online_preparation_seconds':preparation,
                            'shared_trainer_prefix_cpu_seconds':prefix_cpu,
                            'online_preparation_measured_wall_seconds':preparation_wall,
                            'owner_input_metadata_total_charge_seconds':job.header_capture_charge,
                            'owner_input_metadata_charge_seconds':job.header_capture_charge,
                            'preparation_measurement': 'conditional-on-prepared-input' if reusable else 'independent-preparation',
                            'preparation_donor_job_id': job.preparation.get('conditional_preparation_from'),
                            'tasks':job.task_rows,'assignments':[],'production_source_ids':list(job.production_source_ids) if job.issued else None}
                    def serve(current,behavior,aid,job_id=None,verifier_id=None,precommit_fault_task_id=None):
                        start=time.monotonic()
                        previous_identity = {k:self.clock.context.get(k) for k in ('assignment_id','verifier_id')}
                        self.clock.context.update(assignment_id=aid, verifier_id=verifier_id)
                        try:
                            value=current.serve(behavior=behavior,seed=unit['seed'],assignment_id=aid,
                                audit_secret=self.secret(aid,'audits'),trainer_cache=cache,commits=self.commits,
                                job_id=job_id,verifier_id=verifier_id,precommit_fault_task_id=precommit_fault_task_id)
                        finally:
                            for k,v in previous_identity.items():
                                if v is None:
                                    self.clock.context.pop(k, None)
                                else:
                                    self.clock.context[k] = v
                        report,settled,detail=value
                        row={'assignment_id':aid,'method':current.method_key,'behavior':behavior,'report':asdict(report),
                             'settlement':asdict(settled),'execution':detail,'wall_seconds':time.monotonic()-start}
                        result['assignments'].append(row)
                        return value
                    if job.issued and unit['package']!='M0':
                        if unit['package'] in {'M4L','M4S','M4X'}:
                            second_started=time.monotonic()
                            second=prepare(suffix='job1')
                            second_preparation=time.monotonic()-second_started+second.header_capture_charge
                            result["owner_input_metadata_total_charge_seconds"] += second.header_capture_charge
                            result['second_job_preparation_seconds']=second_preparation
                            if not second.issued:
                                raise ValueError('same identity per-job reference outcome differs')
                            calibration=self.profile['calibration'][context.dataset]
                            def callback(jid,vid):
                                current=job if jid=='j0' else second
                                colluding=jid=='j0' and int(vid[1:])<unit.get('colluders',0)
                                behavior=unit['behavior'] if colluding else 'honest'
                                flip = (next(iter(current.production)) if
                                        unit.get('conditioned_production_flip') and jid=='j0' and vid=='v0' else None)
                                report,settled,detail=serve(current,behavior,uid+'-'+jid+'-'+vid,jid,vid,flip)
                                actual=dict(settled.diagnostics)['owner_admission_seconds']/calibration['cost_unit_seconds']
                                settled=replace(settled,diagnostics=settled.diagnostics+(("owner_work_cost",actual),))
                                return report,settled,detail
                            result['recovery']=execute_recovery(policy=unit['policy'],fault=unit['fault'],seed=seed,
                                required_ids=tuple(job.production),execute=callback,clock=self.clock,
                                timeout_seconds=calibration['timeout_seconds'],deadline_seconds=unit.get('deadline_seconds',calibration['deadline_seconds']),
                                owner_cost_reserve_per_assignment=calibration['owner_reserve_units'],
                                owner_prepaid_cost_per_job=(max(preparation,second_preparation) if self.claim_linked else preparation)/calibration['cost_unit_seconds'],minimum_pass_count=3,
                                public_conflicts=graph_conflicts(unit['graph']),
                                missing_ids=unit.get('missing_ids', ('v0','v3') if unit['package']=='M4X' else None),
                                virtual_fault_wait=self.performance.get('virtual_fault_wait',False),
                                service_fee=job.method.fee,service_bond=job.method.bond,
                                all_response_sets=self.claim_linked,
                                required_ids_by_job={'j0':tuple(job.production),'j1':tuple(second.production)},
                                common_raw_roster=unit.get('common_raw_roster',False),
                                verifier_reputations=self.recovery_reputations(context.dataset),
                                budget_cap_per_job=(max(preparation,second_preparation)/calibration['cost_unit_seconds']
                                    + 3*(job.method.fee+calibration['owner_reserve_units']) + unit['budget_boundary_delta']
                                    if 'budget_boundary_delta' in unit else unit.get('budget_cap_per_job')))
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
                    self.after_unit(unit, result, jobs, bank, scratch)
                    for current in jobs:
                        if current.disclosure_epoch is not None and group_disclosure is None:
                            current.close_service_epoch()
                    if job.issued and unit.get('reference_check'):
                        result['probe_replays']=self.probe_replays(context,job)
                    self.finish_unit(unit,result,began,cpu_started)
                    jobs.clear(); job=second=current=None; callback=serve=prepare=None; gc.collect()
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
                            release_scratch(self,folder)
                            self.events({'event':'PREPARATION_RELEASED_AFTER_LAST_USE',
                                         'unit_id':uid,'method':unit['method'],'path':str(folder)})
                    for folder in job_temp:
                        if folder not in retained_folders and folder.exists():
                            release_scratch(self,folder)
            templates.clear()
            self.after_group(context,group,bank,group_id)
            del bank,cache,initial,momentum; gc.collect()
            import shutil
            retained=self.retain_group(group,scratch)
            own_donor=source_donors.get(donor_key)
            if retained and (source_reuse is not None or own_donor is not None):
                raise ValueError('paired source reuse requires within-group audit completion')
            if not retained and not (invalid==0 and own_donor is not None):
                release_scratch(self,scratch)
                self.events({'event':'SCRATCH_RELEASED_AFTER_DURABLE_GROUP','group_id':group_id})
            if invalid==1 and donor is not None:
                donor_folder=donor['scratch']
                source_donors.pop(donor_key)
                donor=own_donor=None
                gc.collect()
                release_scratch(self,donor_folder)
            if anchor and self.config.get('release_trajectory_after_last_use', False):
                remaining=[r for r in rows if r['seed']==seed and r.get('anchor',0)
                           and r['unit_id'] not in self.results]
                if not remaining:
                    # References to mmap tensors must be gone before unlink on NFS.
                    saved=initial_rng=None
                    gc.collect()
                    from sevc.core.trajectory_storage import release_trajectory_tensors
                    release_trajectory_tensors(trajectories[seed],self.events)

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
        self.clock.context={'phase_scope':'offline-protocol-fixtures'}
        if self.disclosure_epoch is not None:
            self.disclosure_epoch.close()
            self.records({'event':'reference-answers-published',
                          **self.disclosure_epoch.release(tuple(sorted(self.disclosure_answers.items())))})
        if self.claim_linked:
            write_json(self.root/'protocol-state-evidence.json', self.protocol_state_evidence())
        scores={}; native=next((p for p in self.candidate['packages'] if p['id']=='M8'),None)
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
                solution,value=native_cell(native,unit['method'],unit['epsilon'],scores)
                result['solution']=solution
                if value is not None:
                    result['native']=value
                else:
                    result['status']='INFEASIBLE_VALID_NATIVE_RESULT'
            self.finish_unit(unit,result,start,cpu_started)

    def protocol_state_evidence(self):
        """Finite protocol fixtures; their denominators are not dataset samples."""
        from sevc.incentives.verifier_protocol import CommittedVerifierReport, SealedEvaluationTruth, SealedSegmentTruth
        from sevc.verification.verifier_task_policies import settle_threshold_assignment
        from sevc.reputation.calibration import PaidCalibrationLedger
        from sevc.evaluation.recovery_graph_audit import audit_trace
        probes=tuple(f'p{i}' for i in range(8)); answers=tuple(i<4 for i in range(8))
        masks=[]
        for mask in range(256):
            report=CommittedVerifierReport.create(scenario_id='completeness',verifier_id='v',job_id='j',
                ordered_segment_ids=probes,verdicts=answers,nonce=str(mask))
            truth=SealedEvaluationTruth('completeness',tuple(SealedSegmentTruth(k,v,True,
                technical_failure=not bool(mask & (1<<i))) for i,(k,v) in enumerate(zip(probes,answers))))
            settled=settle_threshold_assignment(report,truth,failure_threshold=1,fee=2.5,bond=.5,cost=0,effort=0,
                                                sentinel_only=True,required_probe_ids=probes)
            masks.append({'mask':mask,'settlement':asdict(settled)})
        def callback(jid,vid):
            report=CommittedVerifierReport.create(scenario_id='capacity',verifier_id=vid,job_id=jid,
                ordered_segment_ids=('target',),verdicts=(True,),nonce=jid+vid)
            truth=SealedEvaluationTruth('capacity',(SealedSegmentTruth('target',True,True),))
            settled=settle_threshold_assignment(report,truth,failure_threshold=1,fee=2.5,bond=.5,cost=0,effort=0)
            return report,settled,[]
        response_sets=[]
        for missing in itertools.combinations([f'v{i}' for i in range(9)],2):
            trace=execute_recovery(policy='all-response-certified-ecs',fault='structural',seed=0,
                required_ids=('target',),execute=callback,clock=self.clock,timeout_seconds=.001,
                deadline_seconds=1e6,virtual_fault_wait=True,missing_ids=missing,service_fee=2.5)
            response_sets.append({'missing':missing,'trace':trace,'audit':audit_trace(trace)})
        ledger=PaidCalibrationLedger('v0',200.,2.5)
        criteria=dict(minimum_correctness=.9,minimum_availability=.9,alpha=.05/9,independent_stationary_blocks=True)
        states=[{'observations':0,'admitted':ledger.production_admissible(**criteria)}]
        for i in range(64):
            ledger.reserve(str(i),f'constructed-independent-block-{i}')
            _,settled,_=callback('calibration','v0')
            ledger.record(str(i),settled,timely=True,reference_correct=True,reference_receipt_sha256=commitment(['fixture',i]))
            if i in (0,63):
                states.append({'observations':i+1,'admitted':ledger.production_admissible(**criteria),
                               'summary':ledger.summary(alpha=.05/9,independent_stationary_blocks=True)})
        return {'input_kind':'constructed finite-state fixtures; not learned population reliability',
                'probe_masks':masks,'response_sets':response_sets,'calibration_states':states,
                'no_dataset_sample_count_from_fixtures':True}
