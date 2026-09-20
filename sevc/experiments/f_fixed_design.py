"""Fixed-design assembly using the existing ScopedStudy train/replay lifecycle."""
from collections import defaultdict
from dataclasses import asdict
import json
from pathlib import Path
import time

from sevc.experiments.scoped_five_rq_units import ScopedStudy, unit_assignment_ids
from sevc.core.artifacts import write_json, sha256_file
from sevc.core.statistical_bounds import clopper_pearson_bound
from sevc.verification.reference_acquisition import JobReferences, commitment
from sevc.verification.disclosure import ProbeDisclosureEpoch
from sevc.verification.on_demand_service import METHODS
from sevc.training import verify_replay_proof

CHANGE = 'experiment-tdsc-f-fixed-design-v1'


def qualification(rows, expected_blocks, contexts, alpha=.05/54):
    """Conservative block-AND outcomes; never count scenarios as independent."""
    grouped=defaultdict(dict)
    for row in rows:
        key=row['block']; context=row['context']
        if context in grouped[key]: raise ValueError('duplicate calibration block/context')
        grouped[key][context]=row
    if set(grouped)!=set(range(expected_blocks)) or any(set(v)!=set(contexts) for v in grouped.values()):
        raise ValueError('incomplete calibration observations')
    available=correct=audited=0
    for block in grouped.values():
        admitted=all(x['admitted'] for x in block.values())
        available+=all(x['timely'] and x['admitted'] for x in block.values())
        if admitted:
            if any(type(x['correct']) is not bool for x in block.values()):
                raise ValueError('missing admitted reference audit')
            audited+=1; correct+=all(x['correct'] for x in block.values())
    a=clopper_pearson_bound(available,expected_blocks,side='lower',alpha=alpha)
    p=clopper_pearson_bound(correct,audited,side='lower',alpha=alpha) if audited else 0.
    return {'blocks':expected_blocks,'available_blocks':available,'audited_blocks':audited,
            'correct_blocks':correct,'availability_lower':a,'correctness_lower':p,
            'qualified':a>=.9 and p>=.9,'alpha':alpha,'aggregation':'all registered contexts per block'}


class FixedDesignStudy(ScopedStudy):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.all_units=list(self.units)
        self.lock=self.config['fixed_design']
        self.disclosure_epoch=None; self.disclosure_answers={}
        self.calibration_rows=defaultdict(list); self.qualified={}
        self.group_pending=[]; self.group_sources={}; self.epoch_key=None
        self.paid_ledgers={}
        from sevc.reputation.calibration import PaidCalibrationLedger
        for u in self.all_units:
            if u.get("phase")!="calibration":continue
            for i in range(9):
                vid=f"v{i}";key=(u["dataset"],vid,u["context"])
                ledger=self.paid_ledgers.setdefault(key,PaidCalibrationLedger(vid,budget=self.lock["counts"]["calibration"]*2.5,fee=2.5))
                ledger.reserve(u["unit_id"]+"-"+vid,commitment([u["dataset"],u["seed"],u["phase"]]))
        (self.root/'reference-audits').mkdir()
        (self.root/'qualification').mkdir()
        if self.lock.get('qualification_mode') == 'conditional-component-v1':
            write_json(self.root/'qualification'/'conditional-roster.json', {
                'source':'assumed-fixed-software-roster',
                'correctness_input':self.lock['conditional_recovery_correctness'],
                'identities':[f'v{i}' for i in range(9)],
                'empirical_qualification_certified':False,
                'scope':'conditional component execution; not population reliability'})

    def recovery_reputations(self,dataset):
        if self.lock.get('qualification_mode') == 'conditional-component-v1':
            if self.config['change_id'] != 'experiment-tdsc-f-fixed-design-v2':
                raise ValueError('conditional qualification requires reviewed v2 scope')
            return {f'v{i}':self.lock['conditional_recovery_correctness'] for i in range(9)}
        return {vid:r['correctness_lower'] for vid,r in self.qualified.get(dataset,{}).items() if r['qualified']}

    def run_dataset(self,context,partition):
        self.context=context
        for phase in ([self.scheduled_phase] if hasattr(self, 'scheduled_phase') else self.lock['phase_order']):
            self.phase=phase
            self.units=[u for u in self.all_units if u.get('phase')==phase and u['dataset']==context.dataset]
            domain='calibration' if phase=='calibration' else 'production'
            if self.units: super().run_dataset(context,tuple(self.lock['partitions'][context.dataset][domain]))
            if phase=='calibration':
                self.qualified[context.dataset]={vid:qualification(self.calibration_rows[context.dataset,vid],
                    self.lock['counts']['calibration'],self.lock['calibration_contexts']) for vid in [f'v{i}' for i in range(9)]}
                write_json(self.root/'qualification'/f'{context.dataset}.json',self.qualified[context.dataset])
                write_json(self.root/'qualification'/f'{context.dataset}-payments.json',{
                    f'{vid}/{ctx}':{'reservations':ledger.reservations,'observations':ledger.observations}
                    for (ds,vid,ctx),ledger in self.paid_ledgers.items() if ds==context.dataset})
                self.events({'event':'CALIBRATION_COMPLETE','dataset':context.dataset,
                    'qualified_count':sum(x['qualified'] for x in self.qualified[context.dataset].values()),
                    'conditional_roster_used':self.lock.get('qualification_mode')=='conditional-component-v1'})
            from sevc.evaluation.f_fixed_design import seal_stage
            seal_stage(self.root, self.config, phase, context.dataset)
        self.units=self.all_units

    def finish_unit(self,unit,result,began,cpu_started):
        if unit.get('phase') == 'recovery' and self.lock.get('qualification_mode') == 'conditional-component-v1':
            result={**result,'conditional_component':True,
                'qualification_source':'assumed-fixed-software-roster',
                'empirical_qualification_certified':False,
                'assumed_correctness':self.lock['conditional_recovery_correctness']}
        super().finish_unit(unit,result,began,cpu_started)

    def before_group(self,context,key,group):
        if self.phase=='recovery' and len(self.recovery_reputations(context.dataset))<9:
            for u in group:
                self.finish_unit(u,{'status':'DEFER_NOT_QUALIFIED','issued':False,'assignments':[],
                    'not_a_technical_failure':True},time.monotonic(),time.process_time())
            return False
        seed,anchor,steps,batch,invalid=key
        epoch_key=(seed,anchor,steps,batch)
        if epoch_key!=self.epoch_key:
            if self.group_pending: raise ValueError('unclosed audit epoch')
            related=[u for u in self.units if (u['seed'],u.get('anchor',0),u['steps'],u['batch_size'])==epoch_key
                and u['package']!='DP' and METHODS.get(u['method']).require_complete_probes]
            ids=[a for u in related for a in unit_assignment_ids(u)]
            self.disclosure_epoch=ProbeDisclosureEpoch(commitment([context.dataset,self.phase,epoch_key]),ids) if ids else None
            self.disclosure_answers={};self.epoch_key=epoch_key
        return True

    def after_unit(self,unit,result,jobs,bank,scratch):
        if not result['issued']:
            if unit['phase'] == 'calibration':
                for i in range(9):
                    self.calibration_rows[unit['dataset'], f'v{i}'].append({
                        'block':unit['block'], 'context':unit['context'],
                        'admitted':False, 'timely':False, 'correct':None,
                        'reason':'registered preparation did not issue service'})
            return
        mappings={j.job_id:dict(j.production) for j in jobs}
        self.group_pending.append((unit,result,mappings))
        self.group_sources.update({s.hashes['proof_sha256']:s for s in bank})

    def after_group(self,context,group,bank,group_id):
        key=self.epoch_key
        later=[u for u in self.units if (u['seed'],u.get('anchor',0),u['steps'],u['batch_size'])==key
               and u['invalid']>group[0]['invalid']]
        if later:return
        if self.disclosure_epoch:
            self.disclosure_epoch.close()
            self.records({'event':'reference-answers-published',**self.disclosure_epoch.release(tuple(sorted(self.disclosure_answers.items())))})
        self.clock.context={'dataset':context.dataset,'phase_scope':'post-epoch-owner-audit','source_group':group_id}
        def replay(sid):
            source=self.group_sources[sid]
            result,elapsed=self.clock.call('formal_owner_reference_audit','owner',verify_replay_proof,
                source.proof,context.cached_replay_factory,device=context.device.name,tolerance=1e-5,comparison_device='replay')
            return {**result,'proof_sha256':sid,'seconds':elapsed}
        references=JobReferences(commitment([group_id,'post-epoch']),replay,self.records)
        for unit,result,mappings in self.group_pending:
            receipts={sid:references.acquire(sid) for mapping in mappings.values() for sid in mapping.values()}
            audits=[]
            for a in result['assignments']:
                report=a['report'];mapping=next((v for k,v in mappings.items() if k==report['job_id']),None)
                if mapping is None:
                    jid='job1' if report['job_id']=='j1' else 'job'
                    mapping=next(v for k,v in mappings.items() if k.endswith('-'+jid))
                values=dict(zip(report['ordered_segment_ids'],report['verdicts']))
                wrong=[tid for tid,sid in mapping.items() if values.get(tid)!=receipts[sid]['passed']]
                admitted=a['settlement']['accepted_report'] and a['settlement']['status']=='PASS'
                row={'assignment_id':a['assignment_id'],'verifier_id':report['verifier_id'],
                     'block':unit['block'],'context':unit['context'],'admitted':admitted,
                     'timely':a['wall_seconds']<=self.lock['calibration_deadline_seconds'],
                     'correct':not wrong if admitted else None,'wrong_task_ids':wrong,
                     'fee_paid':a['settlement']['service_fee'],'verifier_seconds':a['settlement']['verifier_cost']}
                audits.append(row)
                if unit['phase']=='calibration':
                    from sevc.incentives.verifier_protocol import VerifierSettlement
                    ledger=self.paid_ledgers[unit['dataset'],report['verifier_id'],unit['context']]
                    ledger.record(a['assignment_id'],VerifierSettlement(**a['settlement']),timely=row['timely'],
                        reference_correct=row['correct'],reference_receipt_sha256=commitment(receipts) if admitted else None)
                    self.calibration_rows[unit['dataset'],report['verifier_id']].append(row)
                if admitted and wrong:
                    import torch
                    folder=self.root/'counterexamples';folder.mkdir(parents=True,exist_ok=True)
                    for tid in wrong:
                        sid=mapping[tid];path=folder/(sid+'.pt')
                        if not path.exists():torch.save(self.group_sources[sid].proof,path)
                        self.events({'event':'WRONG_REPORT_ADMITTED','unit_id':unit['unit_id'],
                            'path':str(path),'sha256':sha256_file(path)})
            write_json(self.root/'reference-audits'/f"{unit['unit_id']}.json",{'unit_id':unit['unit_id'],
                'unit_sha256':sha256_file(self.root/'units'/f"{unit['unit_id']}.json"),'assignments':audits,
                'production_receipts':receipts,'audit_is_independent_sample':False})
        self.group_pending.clear();self.group_sources.clear();self.epoch_key=None
        self.disclosure_epoch=None;self.disclosure_answers={}

    def special_unit(self,*args):
        from sevc.experiments.submission_tiny import SubmissionTinyStudy
        return SubmissionTinyStudy.special_unit(self,*args)

    def run_cpu(self):
        # Preserve the existing finite structural domain and native PP interface.
        from sevc.evaluation.local_tiny_feasibility import structural_screen
        write_json(self.root/'structural-evidence.json',structural_screen(self.clock))
        from sevc.evaluation.submission_tiny import recovery_boundaries
        write_json(self.root/'recovery-boundaries.json',recovery_boundaries(self.clock))
        from sevc.incentives.native_peer_prediction import native_cell
        native=next(x for x in self.candidate['packages'] if x['id']=='M8');scores={}
        for u in self.all_units:
            if u['package']!='M8':continue
            began=time.monotonic();cpu=time.process_time()
            solution,value=native_cell(native,u['method'],u['epsilon'],scores)
            self.finish_unit(u,{'status':'MEASURED' if value else 'INFEASIBLE_VALID_NATIVE_RESULT',
                'issued':False,'assignments':[],'solution':solution,'native_game':value},began,cpu)
