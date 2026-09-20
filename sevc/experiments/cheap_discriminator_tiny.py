"""Nonformal diagnostic hooks for the sole ScopedStudy runner."""
from __future__ import annotations
import json
import time
from pathlib import Path
from sevc.core.artifacts import write_json,sha256_file
from sevc.experiments.scoped_five_rq_units import ScopedStudy,unit_assignment_ids
from sevc.verification.cheap_discriminator import CheapSelector,public_features
from sevc.verification.disclosure import ProbeDisclosureEpoch
from sevc.verification.paid_replay_service import StoredReplayProof

CHANGE='experiment-tdsc-cheap-discriminator-tiny-v1'
CHANGES={CHANGE,'experiment-tdsc-cheap-discriminator-tiny-v2'}


def validate_activation(config,profile,output_root):
    if (config['change_id'] not in CHANGES or profile['formal'] or profile['device']!='cpu'
            or config.get('formal_execution_authorized') is not False):
        raise PermissionError('nonformal local CPU diagnostic only')
    review=Path(config['review_path']);lock=json.loads((review.parent/'protocol-lock.json').read_text())
    frozen=Path(__file__).resolve().parents[2]/'configs'/('tdsc_cheap_discriminator_tiny_'+config['change_id'].rsplit('-',1)[1]+'.json')
    if sha256_file(frozen)!=lock['config_sha256'] or 'Ready: YES' not in review.read_text():
        raise PermissionError('tiny protocol has not been frozen and reviewed')
    expected=json.loads(frozen.read_text());runtime={k:v for k,v in config.items() if not k.startswith('_runtime')}
    if runtime!=expected:raise PermissionError('runtime scientific configuration drift')
    if output_root.exists():raise FileExistsError('append-only tiny output identity exists')
    if set(config['datasets'])!={'cifar10','cifar100','mnist'}:raise ValueError('three datasets required')


class CheapDiscriminatorStudy(ScopedStudy):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.all_units=list(self.units); self.lock=self.config['tiny_discriminator'];self.started=time.monotonic()
        self.improved=self.config['change_id'].endswith('-v2');self.selections={};self.fitted={}
        self.histories={};self.pending=[];self.current_epoch=None;self.disclosure_epoch=None;self.disclosure_answers={}
        (self.root/'recognizer').mkdir()

    def run_dataset(self,context,partition):
        from contextlib import nullcontext
        from sevc.core.scratch_budget import scratch_budget
        guard=scratch_budget(self.config['scratch_parent'],self.lock['scratch_limit_bytes']) if self.improved else nullcontext()
        with guard:
            for split in self.lock.get('split_order',('development','test')):
                self.split=split
                self.units=[u for u in self.all_units if u['dataset']==context.dataset and u['split']==split]
                super().run_dataset(context,tuple(self.lock[split+'_partition'][context.dataset]))
        self.units=list(self.all_units)
        self.record_resources('dataset-complete',dataset=context.dataset)

    def record_resources(self,event,**identity):
        if not self.improved:return
        import resource,sys
        from sevc.core.scratch_budget import disk_usage
        rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        self.events({'event':'LOCAL_RESOURCE_SAMPLE','boundary':event,**identity,
            'scratch':disk_usage(self.config['scratch_parent']),
            'process_peak_rss_bytes':int(rss if sys.platform=='darwin' else rss*1024),
            'disk_peak_is_boundary_sample':True})

    def before_group(self,context,group_key,group):
        if time.monotonic()-self.started>self.lock['wall_budget_seconds']:
            raise TimeoutError('HOLD_RESOURCE: frozen 3600-second budget reached; no new source group')
        if self.current_epoch is None:
            seed,anchor,steps,size,invalid=group_key
            related=[u for u in self.units if u['seed']==seed]
            self.disclosure_epoch=ProbeDisclosureEpoch(group[0]['unit_id'],[a for u in related for a in unit_assignment_ids(u)])
            self.current_epoch=self.disclosure_epoch;self.disclosure_answers={}
        return True

    def before_unit(self,unit):
        self.performance.pop('cheap_selector',None)
        self.record_resources('before-unit',unit_id=unit['unit_id'])
        if unit['behavior']=='cheap-recognizer-k32':
            key=(unit['dataset'],unit['method'])
            if self.improved:
                from sevc.verification.cheap_discriminator import PersistentCheapSelector
                self.performance['cheap_selector']=PersistentCheapSelector(self.fitted[key],self.selections[key],self.lock['cache_limit_bytes'])
            else:
                history=self.histories[key]
                self.performance['cheap_selector']=CheapSelector(history,self.lock['component_models'],self.lock['quota'])
        return True

    def after_unit(self,unit,result,jobs,bank,scratch):
        if not result['issued']:return
        job=jobs[0]
        if unit['split'] in ('development','validation'):
            from sevc.verification.cheap_discriminator import collect_public_history
            history=collect_public_history(unit,job,self.clock,improved=self.improved)
            self.pending.append((unit,history))
        if unit['behavior']=='cheap-recognizer-k32':
            selector=self.performance['cheap_selector']
            if self.improved:selector.finish()
            receipt=selector.receipt
            result['recognizer']=receipt
            write_json(self.root/'recognizer'/f"{unit['unit_id']}.json",receipt)
        self.record_resources('after-unit',unit_id=unit['unit_id'])
        # Preserve one source proof per admitted false production report before group cleanup.
        truth={s.hashes['proof_sha256']:s for s in bank}
        for a in result['assignments']:
            if a['settlement']['status']!='PASS':continue
            verdict=dict(zip(a['report']['ordered_segment_ids'],a['report']['verdicts']))
            for row in job.task_rows:
                if row['role']=='production' and verdict[row['task_id']] != (truth[row['source_sha256']].recipe['source_index'] != 24 or unit['invalid'] == 0):
                    # The raw independent auditor is authoritative; retain potential witnesses conservatively.
                    import torch
                    s=truth[row['source_sha256']];folder=self.root/'counterexamples';folder.mkdir(exist_ok=True)
                    path=folder/(row['source_sha256']+'.pt')
                    if not path.exists():torch.save(s.proof,path)

    def after_group(self,context,group,bank,group_id):
        if any(u['seed']==group[0]['seed'] and u['invalid']>group[0]['invalid'] for u in self.units):return
        self.current_epoch.close()
        event=self.current_epoch.release(tuple(sorted(self.disclosure_answers.items())))
        self.records({'event':'reference-answers-published',**event})
        for u,h in self.pending:
            from sevc.verification.cheap_discriminator import publish_history
            publish_history(h,self.root/'recognizer'/f"{u['split']}-{u['dataset']}-{u['method']}.json")
            key=(u['dataset'],u['method'])
            self.histories[(*key,u['split']) if self.improved else key]=h
        if self.improved and self.split=='validation':
            import pickle,hashlib
            from sevc.verification.cheap_discriminator import select_development_model
            for u,h in self.pending:
                key=(u['dataset'],u['method'])
                self.clock.context={'dataset':u['dataset'],'unit_id':u['unit_id'],'method':u['method'],'phase_scope':'offline-attack-development'}
                (models,receipt),seconds=self.clock.call('development_model_selection','offline',select_development_model,
                    self.histories[(*key,'development')],h,self.lock['component_models'],self.lock['random_state'])
                model_bytes=pickle.dumps(models,protocol=5)
                receipt.update(selection_wall_seconds=seconds,model_sha256=hashlib.sha256(model_bytes).hexdigest())
                model_path=self.root/'recognizer'/f"model-{u['dataset']}-{u['method']}.pkl"
                model_path.write_bytes(model_bytes)
                self.fitted[key]=models;self.selections[key]=receipt
                write_json(self.root/'recognizer'/f"selection-{u['dataset']}-{u['method']}.json",receipt)
        self.pending=[];self.current_epoch=None;self.disclosure_epoch=None;self.disclosure_answers={}
