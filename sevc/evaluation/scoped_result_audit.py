"""Independent event/coverage/accounting audit and frozen descriptive analysis."""
from __future__ import annotations

from collections import defaultdict, Counter
import itertools
import json
import math
import hashlib
from pathlib import Path

import numpy as np
from scipy.stats import beta

from sevc.core.artifacts import write_json, sha256_file
from sevc.incentives.verifier_protocol import CommittedVerifierReport


def lines(path):
    with Path(path).open() as stream:
        for line in stream:
            yield json.loads(line)


def interval_union(intervals):
    if not intervals:
        return 0.
    ordered=sorted(intervals); start,end=ordered[0]; total=0.
    for a,b in ordered[1:]:
        if a>end:
            total+=end-start; start,end=a,b
        else:
            end=max(end,b)
    return total+end-start


def require(condition,message):
    if not condition:
        raise ValueError(message)


def resource_breakdown(root,rows):
    """Integrate disjoint measured component windows with sampler boundary rows."""
    import bisect
    from sevc.evaluation.workload_performance import integrate_utilization
    samples=list(lines(root/'gpu-utilization.jsonl'))
    ticks=[s['monotonic'] for s in samples]
    components=[(r['dataset'],r['package'],r['started_monotonic'],r['ended_monotonic'],r['cpu_process_seconds']) for r in rows.values()]
    for bank in lines(root/'source-task-identities.jsonl'):
        if 'shared_trainer_prefix_seconds' in bank:
            components.append((bank['dataset'],'SHARED_SOURCE',bank['started_monotonic'],bank['ended_monotonic'],bank['shared_trainer_prefix_cpu_seconds']))
    for path in (root/'milestones').glob('*/prefix-cost.json'):
        r=json.loads(path.read_text());dataset=path.parent.name.rsplit('-',1)[0]
        components.append((dataset,'TRAJECTORY',r['started_monotonic'],r['ended_monotonic'],r['cpu_process_seconds']))
    groups=defaultdict(lambda:{'wall_seconds':0.,'cpu_process_seconds':0.,'gpu_percent_seconds':0.,'components':0})
    for dataset,package,start,end,cpu_seconds in components:
        g=groups[str(dataset)+'/'+package];g['wall_seconds']+=end-start;g['cpu_process_seconds']+=cpu_seconds;g['components']+=1
        if samples and end>start:
            left=max(0,bisect.bisect_right(ticks,start)-1);right=bisect.bisect_left(ticks,end)+1
            measured=integrate_utilization(samples[left:right],start,end)
            g['gpu_percent_seconds']+=measured['average_percent']*(end-start)
    for g in groups.values():
        g['gpu_average_percent']=g.pop('gpu_percent_seconds')/g['wall_seconds'] if samples and g['wall_seconds'] else None
        g['scope']='sum of measured component windows; shared input/trajectory prefixes separate; whole process setup/export reported separately'
    return dict(groups)


def audit_and_summarize(root,config,profile):
    root=Path(root); specification=json.loads((root/'expanded-units.json').read_text())
    registered={r['unit_id']:r for r in specification}
    files=list((root/'units').glob('*.json'))
    require({p.stem for p in files}==set(registered),'unit coverage mismatch')
    rows={p.stem:json.loads(p.read_text()) for p in files}
    from sevc.verification.reference_acquisition import commitment
    private=json.loads(Path(config['private_streams_path']).read_text())
    locked_streams=json.loads((root/'stream-commitments.json').read_text())
    require(all(locked_streams[k]==commitment(v) for k,v in private.items()),'private stream identity drift')
    job_groups={r['unit_id']+'-'+suffix:r['source_group'] for r in rows.values() if 'source_group' in r for suffix in ('job','job1')}
    source_truth={}; source_count=0; source_populations={}
    for bank in lines(root/'source-task-identities.jsonl'):
        if 'shared_trainer_prefix_seconds' in bank:
            source_populations[bank['group_id']]={s['proof_sha256'] for s in bank['sources']}
        for source in bank['sources']:
            digest=source['proof_sha256']
            constructed=source['trainer_mutation'] is None
            require(digest not in source_truth or source_truth[digest]==constructed,'source identity conflict')
            source_truth[digest]=constructed; source_count+=1
    commitments={}; reveals={}
    for event in lines(root/'report-lifecycle.jsonl'):
        aid=event['assignment_id']
        if event['phase']=='COMMITTED':
            require(aid not in commitments,'duplicate assignment commitment')
            commitments[aid]=event['commitment']
        else:
            require(event['phase']=='REVEALED' and aid in commitments,'invalid commitment order')
            report=CommittedVerifierReport(**event['report'])
            require(report.commitment==commitments[aid] and aid not in reveals,'report binding failure')
            reveals[aid]=event['report']
    references=defaultdict(dict); audits={}; salt_commitments={}; deliveries={}; role_populations={}; priorities={}
    reference_new=reference_hits=0; views={}; prepared_references={}; shared_preparations={}
    for event in lines(root/'assignment-audit.jsonl'):
        kind=event.get('event')
        if kind=='role-seed-committed':
            ids=event['source_ids']; require(len(set(ids))==40,'reference population is not all40')
            job=event['job_id']; group=job[:-8] if job.endswith('-fixture') else job_groups[job]
            secret=commitment([private['roles'],group,'roles'])
            require(set(ids)==source_populations[group],'priority population differs from committed source bank')
            require(event['commitment']==commitment(['scoped-role-v2',job,secret]),'private priority commitment drift')
            require(job not in role_populations and not references[job],'source priority committed too late')
            role_populations[job]=set(ids)
            priorities[job]=sorted(ids,key=lambda sid:commitment(['scoped-priority-v2',secret,sid]))
        elif kind=='reference-preparation-outcome':
            job=event['job_id']; attempts=event['attempts']; selected=event['selected_ids']
            require(len(attempts)<=40 and len({a['source_sha256'] for a in attempts})==len(attempts),'candidate repeated or cap exceeded')
            require(selected==[a['source_sha256'] for a in attempts if a['passed']],'selection not actual valid candidates')
            require([a['source_sha256'] for a in attempts]==priorities[job][:len(attempts)],'hidden priority population or attempt order drift')
            require(all(references[job][a['source_sha256']]['passed']==a['passed'] for a in attempts),'preparation answer not backed by replay')
            require((len(selected)==8 and attempts[-1]['passed']) if event['issued'] else (len(selected)<8 and len(attempts)==40),'invalid stopping rule')
            require(set(event['production_ids'])==role_populations[job]-set(selected),'production complement drift')
        elif kind in {'reference-replay','reference-cache-hit'}:
            job,sid=event['job_id'],event['source_sha256']; receipt=event['receipt']
            require(receipt['proof_sha256']==sid and receipt['state_complete'] is True,'unbound reference receipt')
            require(type(receipt['passed']) is bool,'unmeasured reference answer')
            if kind=='reference-cache-hit':
                require(references[job].get(sid)==receipt,'cross-job or unearned reference cache hit')
                reference_hits+=1
            else:
                require(sid not in references[job],'unrecorded duplicate reference replay')
                reference_new+=1
            references[job][sid]=receipt
        elif kind=='conditional-preparation-reused':
            job,donor=event['job_id'],event['donor_job_id']
            require(config['performance'][profile['performance_key']].get('conditional_preparation_reuse') is True,'unregistered preparation reuse')
            uid=job.rsplit('-job',1)[0]; donor_uid=donor.rsplit('-job',1)[0]
            require(rows[uid]['package'] in {'M1','M2','M2C','M3','M6','M7'},'complete cost preparation reused')
            require(job_groups[job]==job_groups[donor] and rows[uid]['method']==rows[donor_uid]['method'],'cross-group or cross-method preparation reuse')
            require(event['reference_receipts']==prepared_references[donor],'unearned or later audit receipt inherited')
            require(not references[job] and job not in shared_preparations,'duplicate preparation inheritance')
            references[job]=dict(event['reference_receipts']); shared_preparations[job]=donor
        elif kind=='job-delivery-ready':
            job=event['job_id']
            if job in shared_preparations:
                donor=deliveries[shared_preparations[job]]
                require(event['tasks']==donor['tasks'] and event['production_source_ids']==donor['production_source_ids'],'shared delivery content drift')
            prepared_references[job]=dict(references[job])
            deliveries[job]=event
        elif kind=='audit-seed-committed':
            salt_commitments[event['assignment_id']]=event['commitment']
        elif kind=='assignment-information-view':
            views[event['assignment_id']]=event
        elif kind=='audit-selected':
            from sevc.verification.paid_replay_service import identity
            aid=event['assignment_id']
            require(event['secret']==commitment([private['audits'],aid,'audits']),'audit salt not from frozen independent stream')
            require(salt_commitments.get(aid)==identity([aid,event['secret']]),'audit secret commitment mismatch')
            require(commitments.get(aid)==event['report_commitment'],'audit not bound to report')
            audits[aid]=event
    timed=defaultdict(lambda:defaultdict(list)); cpu=defaultdict(lambda:defaultdict(float))
    events_seen=set()
    for phase in lines(root/'phase-timing.jsonl'):
        require(phase['event_id'] not in events_seen,'duplicate resource event'); events_seen.add(phase['event_id'])
        require(phase['end']>=phase['start'] and not phase['technical_failure'],'invalid measured phase')
        uid=phase.get('unit_id'); scope=phase.get('phase_scope')
        if uid and scope=='online':
            if phase['phase']=='verifier-service-interface':
                timed[uid]['verifier'].append((phase['start'],phase['end']))
            elif phase['role']=='wait':
                timed[uid]['wait'].append((phase['start'],phase['end']))
            cpu[uid][phase['role']]+=phase['exclusive_cpu_thread_seconds']
    assignment_count=0; risks=defaultdict(dict); utilities=defaultdict(list); full_cost=[]; reference_checks=[]
    observations={}; recovery_checks=[]; trainer_outcomes=[]
    for uid,row in rows.items():
        require(all(row[k]==v for k,v in registered[uid].items()),'unit scientific fields drifted')
        require(abs(row['ended_monotonic']-row['started_monotonic']-row['wall_seconds'])<1e-6,'unit timer does not close')
        require(row['cpu_process_seconds']>=0,'negative process CPU cost')
        if row.get('reuse_of'):
            require(row['reuse_sha256']==sha256_file(root/'units'/f"{row['reuse_of']}.json"),'reuse identity mismatch')
            continue
        if row['package'] in {'M4B','M4T','M8'}:
            if row['package']=='M4B':
                from sevc.evaluation.recovery_graph_audit import audit_trace, exhaustive_witness
                require(row['oracle']==exhaustive_witness(row['graph'],row['missing']),'structural witness drift')
                recovery_checks.append({'unit_id':uid,**audit_trace(row['recovery'])})
            elif row['package']=='M4T':
                from sevc.evaluation.recovery_graph_audit import replay_counterfactual
                original=rows[row['trace_from']]
                expected=replay_counterfactual(original['recovery'],row['counterfactual']) if original['issued'] else None
                require(row['counterfactual_result']==expected,'trace counterfactual drift')
            if row['package']=='M8':
                solution=row['solution']
                if solution['status']=='INFEASIBLE':
                    cert=solution['certificate']; a,b,y=map(np.asarray,(cert['A'],cert['b'],cert['multipliers']))
                    require(y.min()>=0 and np.max(np.abs(a.T@y))<1e-8 and b@y < -1e-8,'invalid Farkas certificate')
                elif solution['status']=='OPTIMAL':
                    a,b,x,y=map(np.asarray,(solution['A'],solution['b'],solution['primal'],solution['dual']))
                    require(np.max(a@x-b)<1e-8 and y.max()<=1e-8 and abs(b@y-x[-1])<1e-8,'LP primal/dual gap')
                    target=np.zeros(len(x)); target[-1]=1
                    require(np.max(np.abs(a.T@y-target))<1e-8,'LP dual stationarity failure')
                if 'native' in row:
                    spec=next(p for p in config['scoped_candidate']['packages'] if p['id']=='M8')
                    obs=spec['observation_matrix']; eps=row['epsilon']; prior=[.5-eps,.25,.25,eps]
                    joint=[[sum(prior[t]*obs[t][x]*obs[t][y] for t in range(4)) for y in range(4)] for x in range(4)]
                    score=solution['score']
                    for strategy in row['native']['strategies']:
                        reward=sum(joint[x][y]*score[strategy['mapping'][x]][y] for x in range(4) for y in range(4))
                        cost=sum(sum(joint[x])*spec['costs'][x] for x in range(4)) if strategy['observed'] else 0.
                        require(abs(reward-cost-strategy['utility'])<1e-8,'native strategy utility disagreement')
            continue
        if not row['issued']:
            require(row['assignments']==[],'unissued service has invented observations')
            continue
        task_map={t['task_id']:t for t in row['tasks']}
        per_row=[]
        for assignment in row['assignments']:
            assignment_count+=1; aid=assignment['assignment_id']; report=assignment['report']
            joint=assignment['behavior'].startswith('joint-')
            require(views[aid]['view']==('trainer-plus-verifier' if joint else 'verifier-delivered-only'),'information view drift')
            require(joint or views[aid]['trainer_cache_size']==0,'ordinary verifier received trainer-private cache')
            require(not joint or views[aid]['trainer_cache_size']==40,'joint view was deprived of committed trainer sources')
            require(reveals.get(aid)==report,'raw lifecycle and exported report differ')
            verdict=dict(zip(report['ordered_segment_ids'],report['verdicts']))
            require(set(verdict)==set(task_map),'assignment delivery population mismatch')
            require(set(d['task_id'] for d in assignment['execution'])==set(verdict),'execution identity mismatch')
            job=uid+'-job1' if report['job_id']=='j1' else uid+'-job'
            probes={k:t['probe_answer'] for k,t in task_map.items() if t['role']!='production'}
            mismatches=sum(verdict[k]!=v for k,v in probes.items())
            sampled=audits[aid]['selected']
            production_ids=sorted(k for k,t in task_map.items() if t['role']=='production')
            count=8 if row['method'] in ('rcmp-audit-8-v2','production-audit-8-v2') else 32 if row['method'] in ('rcmp-audit-all-v2','production-audit-all-v2') else 0
            from sevc.verification.paid_replay_service import identity
            prefix=f"sevc-production-audit-v1|{identity(tuple(production_ids))}|{report['commitment']}|{audits[aid]['secret']}|"
            rank=sorted(range(len(production_ids)),key=lambda i:hashlib.sha256((prefix+str(i)).encode()).digest())
            require(sampled==[production_ids[i] for i in sorted(rank[:count])],'production audit sample/budget drift')
            wrong_audit=any(verdict[k]!=references[job][task_map[k]['source_sha256']]['passed'] for k in sampled)
            passed=mismatches<2 and not wrong_audit
            settlement=assignment['settlement']
            require(settlement['status']==('PASS' if passed else 'FAIL_CONFIRMED'),'independent service decision mismatch')
            require(settlement['service_fee']==(1.25 if passed else 0.) and settlement['slashed_bond']==(0. if passed else .5),'payment conservation failure')
            wrong_production=any(verdict[k]!=source_truth[t['source_sha256']] for k,t in task_map.items() if t['role']=='production')
            per_row.append({'paid':passed,'wrong_production_admitted':passed and wrong_production,
                            'honest_false_penalty':assignment['behavior']=='honest' and not passed,
                            'cost_seconds':settlement['verifier_cost']})
            if row['package'] in {'M1','M2','M2C','M6','M7'}:
                key=(row['dataset'],row['package'],row['method'],assignment['behavior'],row['invalid'],
                     row['steps'],row['batch_size'],row.get('anchor',0))
                utilities[key].append({'block':row['block'],'fee':settlement['service_fee'],
                    'slashed':settlement['slashed_bond'],'cost_seconds':settlement['verifier_cost']})
        observations[uid]=per_row
        if 'recovery' in row:
            from sevc.evaluation.recovery_graph_audit import audit_trace
            recovery_checks.append({'unit_id':uid,**audit_trace(row['recovery'],row['assignments'])})
        expected_accept=all(source_truth[sid] for sid in row.get('production_source_ids',[]))
        decisions=({j:d['route'] for j,d in row['recovery']['decisions'].items()} if 'recovery' in row
                   else {'job':row['trainer_route']} if 'trainer_route' in row else {})
        for job,route in decisions.items():
            trainer_outcomes.append({'unit_id':uid,'dataset':row['dataset'],'package':row['package'],
                'block':row['block'],'method':row['method'],'job_id':job,'invalid':row['invalid'],
                'behavior':row['behavior'],'colluders':row.get('colluders'),'fault':row.get('fault'),
                'route':route,'source_bundle_valid':expected_accept,'safe_defer':route=='safe-defer',
                'wrong_reward':route=='accept' and not expected_accept,
                'wrong_misconduct':route=='reject' and expected_accept,
                'truth_used_only_by_offline_auditor':True})
        if row['package']=='M1':
            require(len(per_row)==1,'M1 must contain one service')
            risks[row['dataset'],row['block']][row['method'],row['behavior'],row['invalid']]=per_row[0]
        for observed in row.get('probe_replays',[]):
            reference_checks.append({'dataset':row['dataset'],'package':row['package'],
                'block':row['block'],'task_id':observed['task_id'],
                'agrees':observed['reference']==observed['actual']['passed']})
        if row['package'] in {'M5','M6C'}:
            require(not row.get('preparation_donor_job_id'),'complete cost uses shared preparation')
            verifier=interval_union(timed[uid]['verifier']); wait=interval_union(timed[uid]['wait'])
            owner=row['online_suffix_seconds']-verifier-wait
            require(owner>=-1e-5,'cost timeline does not close')
            full_cost.append({'unit_id':uid,'dataset':row['dataset'],'block':row['block'],'method':row['method'],
                'invalid':row['invalid'],'steps':row['steps'],'batch_size':row['batch_size'],
                'shared_trainer_prefix_seconds':row['shared_trainer_prefix_seconds'],
                'online_suffix_seconds':row['online_suffix_seconds'],'owner_busy_wall_seconds':max(0.,owner),
                'composed_full_job_seconds':row['shared_trainer_prefix_seconds']+row['online_suffix_seconds'],
                'verifier_busy_wall_seconds':verifier,'wait_seconds':wait,
                'cpu_thread_seconds_by_role':dict(cpu[uid]),'trainer_route':row.get('trainer_route'),
                'online_cpu_process_seconds':row['online_cpu_process_seconds'],
                'unattributed_online_cpu_process_seconds':max(0.,row['online_cpu_process_seconds']-sum(cpu[uid].values())),
                'shared_trainer_prefix_cpu_seconds':row['shared_trainer_prefix_cpu_seconds'],
                'owner_basis':'controlled serial suffix minus union of actual verifier interfaces and waits; residual owner orchestration retained',
                'resource_and_payment_transfers_separate':True})
    primary=[]; methods=config['scoped_candidate']['science']['methods']; alpha=.05/9
    for dataset in config['scoped_candidate']['science']['dataset_order']:
        blocks=sorted(b for d,b in risks if d==dataset)
        for endpoint in ('honest_false_penalty','constant_paid','partial_wrong_production_admitted'):
            events=issued=complete_issuance=0
            for b in blocks:
                bank=risks[dataset,b]
                behavior=('honest',) if endpoint=='honest_false_penalty' else ('constant-accept','constant-reject') if endpoint=='constant_paid' else ('partial-50',)
                selected=[bank[methods['R'],a,m] for a,m in itertools.product(behavior,(0,1)) if (methods['R'],a,m) in bank]
                issued+=bool(selected)
                complete_issuance+=len(selected)==len(behavior)*2
                field='honest_false_penalty' if endpoint=='honest_false_penalty' else 'paid' if endpoint=='constant_paid' else 'wrong_production_admitted'
                events+=any(r[field] for r in selected)
            n=config['scoped_candidate']['science']['datasets'][dataset]['blocks'] if profile['full_matrix'] else len(blocks)
            upper=float(beta.ppf(1-alpha,events+1,n-events)) if events<n else 1.
            primary.append({'dataset':dataset,'endpoint':endpoint,'events':events,'prescribed_blocks':n,
                            'issued_blocks':issued,'rate':events/n if n else None,'upper':upper,
                            'complete_issuance_blocks':complete_issuance,'issued_condition':'at least one prescribed opportunity issued',
                            'issued_conditional_rate':events/issued if issued else None,
                            'common_rate_gate':None,'alpha':alpha})
    grid=[]; prices=config['scoped_candidate']['science']['prices']
    for dataset,package,method,behavior,invalid,steps,size,anchor in sorted(utilities):
        values=utilities[dataset,package,method,behavior,invalid,steps,size,anchor]
        calibration=profile.get('calibration',{}).get(dataset,{})
        unit=calibration.get('cost_unit_seconds')
        if not unit:
            continue
        for fee,bond,mult in itertools.product(prices['fee_sensitivity'],prices['bond_sensitivity'],prices['cost_multipliers']):
            payoffs=[(fee if r['fee'] else 0.)-(bond if r['slashed'] else 0.)-mult*r['cost_seconds']/unit-prices['liquidity_rate']*bond for r in values]
            grid.append({'dataset':dataset,'method':method,'behavior':behavior,'invalid':invalid,
                         'package':package,'steps':steps,'batch_size':size,'anchor':anchor,
                         'independent_unit':'trajectory' if package=='M7' else 'source-block',
                         'fee':fee,'bond':bond,'cost_multiplier':mult,'observations':len(values),
                         'mean_utility':sum(payoffs)/len(payoffs),'fixed_strategy_reweighting':True})
    write_json(root/'primary-risks.json',primary)
    write_json(root/'full-job-costs.json',full_cost)
    write_json(root/'fixed-strategy-cost-grid.json',grid)
    write_json(root/'probe-reference-agreement.json',reference_checks)
    from sevc.evaluation.scoped_statistics import descriptive_analysis
    summaries=descriptive_analysis(rows,observations,config['scoped_candidate']['science'],profile)
    for key,value in summaries.items():
        write_json(root/(key+'.json'),value)
    write_json(root/'recovery-independent-audit.json',recovery_checks)
    write_json(root/'trainer-settlement-outcomes.json',trainer_outcomes)
    from sevc.evaluation.scoped_statistics import cost_contrasts, claim_packet
    costs=cost_contrasts(full_cost,config['scoped_candidate']['science'])
    write_json(root/'paired-cost-contrasts.json',costs)
    write_json(root/'claim-packet.json',claim_packet(rows,observations,reference_checks,summaries,
               recovery_checks,profile,config['scoped_candidate']['science']))
    write_json(root/'package-resource-windows.json',resource_breakdown(root,rows))
    audit={'status':'AUDIT_PASS','units':len(rows),'assignments':assignment_count,'source_records':source_count,
           'reference_replays':reference_new,'reference_cache_hits':reference_hits,'resource_events':len(events_seen),
           'package_counts':dict(Counter(r['package'] for r in rows.values())),
           'full_matrix':profile['full_matrix'],'independent_calculation':True,'independent_researcher':False,
           'tensor_replay_audit':'technical equivalence plus actual recorded owner/diagnostic replay; scratch payloads not retained',
           'publication_ready':False,'close_archive_complete':False}
    write_json(root/'independent-audit.json',audit)
    return audit
