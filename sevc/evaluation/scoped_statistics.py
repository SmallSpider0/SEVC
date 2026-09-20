"""Frozen block/trajectory analyses; no controller consumes these outputs."""
from collections import defaultdict, Counter
import itertools
import numpy as np
from scipy.stats import beta


def uniform_admission_probability(k, total=40, rejecting=4):
    """Exact fixed-quota admission for uniform replay and accept-on-omission."""
    from math import comb
    if not 0 <= rejecting <= total or not 0 <= k <= total:
        raise ValueError('invalid fixed-quota dimensions')
    return comb(k, rejecting) / comb(total, rejecting) if k >= rejecting else 0.


def claim_linked_descriptive(rows, observations, full_cost, trainer_outcomes, science):
    """Paired small-sample projections for the improved protocol, with raw points."""
    from math import comb
    methods=science['methods']
    prices=science.get('prices', {'fee_sensitivity':[1.25,2.5], 'bond':.5,
                                 'cost_per_second':[0,.01,.1,1]})
    services=defaultdict(list); cost_by_id={r['unit_id']:r for r in full_cost}
    utilities=[]; service_points=[]; recovery_points=[]; owner_ratios=[]
    honest={}
    for uid,row in rows.items():
        if row['package']=='M1' and row['issued'] and row['behavior']=='honest':
            honest[row['dataset'],row['block'],row['method']]=observations[uid][0]['cost_seconds']
    for uid,row in rows.items():
        if row['package']=='M1':
            actual=observations.get(uid,[])
            key=row['dataset'],row['method'],row['behavior']
            services[key].append(row)
            point={'unit_id':uid,'dataset':row['dataset'],'method':row['method'],
                   'behavior':row['behavior'],'block':row['block'],'issued':row['issued'],
                   'paid':sum(a['paid'] for a in actual),
                   'wrong_production_admitted':sum(a['wrong_production_admitted'] for a in actual),
                   'honest_false_penalty':sum(a['honest_false_penalty'] for a in actual),
                   'replayed_tasks':sum(d['replayed'] for a in row['assignments'] for d in a['execution'])}
            service_points.append(point)
            if not actual:
                continue
            cost=actual[0]['cost_seconds']; hcost=honest.get((row['dataset'],row['block'],row['method']))
            k={'honest':40,'partial-50':20,'partial-90':36}[row['behavior']]
            chance=uniform_admission_probability(k)
            settlement=row['assignments'][0]['settlement']
            for fee,unit_cost in itertools.product(prices['fee_sensitivity'],prices['cost_per_second']):
                bond=prices['bond']
                utilities.append({'unit_id':uid,'dataset':row['dataset'],'block':row['block'],
                    'method':row['method'],'behavior':row['behavior'],'fee':fee,'bond':bond,
                    'cost_per_second':unit_cost,'measured_cost_seconds':cost,
                    'observed_utility':(fee if actual[0]['paid'] else 0.)
                        -(bond if settlement['slashed_bond'] else 0.)-unit_cost*cost,
                    'combinatorial_pass_probability':chance,
                    'combinatorial_wrong_report_admission_probability':chance*(40-k)/36,
                    'expected_utility_cost_plugin':fee*chance-bond*(1-chance)-unit_cost*cost,
                    'honest_IR_cost_plugin':None if hcost is None else fee-unit_cost*hcost,
                    'honest_minus_deviation_cost_plugin':None if hcost is None else
                        (fee+bond)*(1-chance)-unit_cost*(hcost-cost),
                    'statistical_scope':'exact uniform-selection probability; measured cost point, not a population IC estimate'})
        if row['package']=='M4L' and row['issued']:
            trace=row['recovery']; outcome=[r for r in trainer_outcomes if r['unit_id']==uid]
            recovery_points.append({'unit_id':uid,'dataset':row['dataset'],'block':row['block'],
                'scenario':row['scenario'],'policy':row['policy'],'jobs':2,
                'completed_jobs':sum(not r['safe_defer'] for r in outcome),
                'safe_defer_jobs':sum(r['safe_defer'] for r in outcome),
                'wrong_reward':sum(r['wrong_reward'] for r in outcome),
                'wrong_misconduct':sum(r['wrong_misconduct'] for r in outcome),
                'callbacks':len(trace['receipts']),
                'actual_service_fees':sum(a['settlement']['service_fee'] for a in row['assignments']),
                'reserved_expenditure':trace['reserved_expenditure'],
                'budget_cap_per_job':trace['budget_cap_per_job'],
                'terminal_routes':{j:d['route'] for j,d in trace['decisions'].items()}})
    direct={(r['dataset'],r['seed'],r['block'],r['invalid']):cost_by_id[uid] for uid,r in rows.items()
            if r['package']=='M5' and r['issued'] and uid in cost_by_id}
    for uid,row in rows.items():
        baseline=direct.get((row['dataset'],row['seed'],row['block'],row['invalid']))
        if uid not in cost_by_id or baseline is None or row['method']==methods['O']:
            continue
        if row['package']=='M1' and row['behavior']!='honest':
            continue
        count=2 if row['package']=='M4L' else 1
        owner=cost_by_id[uid]['owner_busy_wall_seconds']
        denominator=count*baseline['owner_busy_wall_seconds']
        if denominator<=0:
            raise ValueError('owner-direct comparison has no measured denominator')
        owner_ratios.append({'unit_id':uid,'dataset':row['dataset'],'block':row['block'],
            'method':row['method'],'scenario':row.get('scenario','single-verifier'),
            'jobs_in_numerator':count,'owner_seconds':owner,
            'matched_direct_seconds':denominator,'owner_to_direct_ratio':owner/denominator,
            'direct_unit_id':baseline['unit_id']})
    return {'service_points':service_points,'recovery_points':recovery_points,
            'owner_cost_points':owner_ratios,'all_price_grid_points':utilities,
            'service_opportunity_counts':[{'dataset':d,'method':m,'behavior':b,
                'prescribed_opportunities':len(rs),'issued':sum(r['issued'] for r in rs),
                'independent_blocks':len({r['block'] for r in rs})} for (d,m,b),rs in sorted(services.items())],
            'independent_unit':'paired source block within dataset',
            'replicate_unit_warning':'Two jobs and repeated verifier callbacks are paired observations, not independent blocks.'}


def exact_interval(k,n,alpha):
    if not n:
        return [None,None]
    return [0. if k==0 else float(beta.ppf(alpha/2,k,n-k+1)),
            1. if k==n else float(beta.ppf(1-alpha/2,k+1,n-k))]


def paired_bootstrap(values,seed,draws=20000):
    values=np.asarray(values,dtype=float)
    if not len(values):
        return {'n':0,'mean':None,'interval':[None,None]}
    rng=np.random.default_rng(seed)
    means=values[rng.integers(len(values),size=(draws,len(values)))].mean(axis=1)
    return {'n':len(values),'mean':float(values.mean()),
            'interval':np.quantile(means,[.025,.975]).tolist(),'draws':draws,
            'seed':seed,'coverage':'descriptive pointwise 95%; no global guarantee'}


def descriptive_analysis(rows,observations,science,profile):
    """Only independently reconstructed assignment observations enter endpoints."""
    methods=science['methods']; seed=science['bootstrap']['seed']; draws=science['bootstrap']['draws']
    opportunity=defaultdict(list); primary=defaultdict(dict); mixed=defaultdict(dict)
    for uid,row in rows.items():
        original=rows[row['reuse_of']] if row.get('reuse_of') else row
        if 'issued' in original:
            opportunity[row['dataset'],row['package'],row['method']].append({**original,**row})
        if row['package']=='M1':
            primary[row['dataset'],row['block']][row['method'],row['behavior'],row['invalid']]=row
        if row['package'] not in {'M1','M2','M2C','M6','M7'}:
            continue
        values=observations.get(row.get('reuse_of') or uid,[])
        if len(values)!=1:
            continue
        a={**values[0],'source_group':original['source_group'],'origin_unit_id':row.get('reuse_of') or uid}
        key=(row['dataset'],row['package'],row['method'],row['steps'],row['batch_size'],row.get('anchor'))
        mixed[key][row['block'],row['behavior'],row['invalid']]=a
    availability=[{'dataset':d,'package':p,'method':m,'opportunities':len(rs),
                   'issued':sum(r['issued'] for r in rs),'not_issued':sum(not r['issued'] for r in rs),
                   'not_issued_reasons':dict(Counter(r['status'] for r in rs if not r['issued'])),
                   'assignment_measurements':sum(len(r['assignments']) for r in rs),
                   'exact_reuse_opportunities':sum(bool(r.get('reuse_of')) for r in rs),
                   'new_assignment_measurements':sum(len(r['assignments']) for r in rs if not r.get('reuse_of')),
                   'unissued_measurements':None}
                  for (d,p,m),rs in sorted(opportunity.items())]
    paired=[]
    for dataset in science['dataset_order']:
        blocks=sorted(b for d,b in primary if d==dataset)
        n=science['datasets'][dataset]['blocks'] if profile['full_matrix'] else len(blocks)
        quadrants=Counter(); issuance=Counter(); both_events=[]
        for block in blocks:
            bank=primary[dataset,block]; events=[]; issued=[]
            for m in (methods['R'],methods['G']):
                rs=[r for (method,behavior,_),r in bank.items() if method==m and behavior!='honest']
                issued.append(bool(rs) and all(r['issued'] for r in rs))
                events.append(any(a['wrong_production_admitted'] for r in rs for a in observations.get(r['unit_id'],[])))
            quadrants[tuple(events)]+=1; issuance[tuple(issued)]+=1
            if all(issued): both_events.append(tuple(events))
        ro,go=quadrants[True,False],quadrants[False,True]
        ri,gi=exact_interval(ro,n,.05/6),exact_interval(go,n,.05/6)
        paired.append({'dataset':dataset,'prescribed_blocks':n,'observed_blocks':len(blocks),
                       'R_only_error':ro,'G_only_error':go,'both_error':quadrants[True,True],
                       'neither_error':quadrants[False,False],'risk_difference':(ro-go)/n if n else None,
                       'R_only_interval':ri,'G_only_interval':gi,
                       'difference_interval':[ri[0]-gi[1],ri[1]-gi[0]] if n else [None,None],
                       'paired_complete_issuance':{f'R{int(r)}_G{int(g)}':issuance[r,g] for r,g in itertools.product((False,True),repeat=2)},
                       'issued_only_n':len(both_events),
                       'issued_only_difference':sum(int(r)-int(g) for r,g in both_events)/len(both_events) if both_events else None,
                       'noninferiority_margin':None,'unissued_is_service_success':False})
    prices=science['prices']; utility=[]
    for key,bank in sorted(mixed.items(),key=lambda x:str(x[0])):
        dataset,package,method,steps,size,anchor=key
        unit=profile['calibration'][dataset]['cost_unit_seconds']
        behaviors=sorted({a for _,a,_ in bank})
        blocks=sorted({b for b,_,_ in bank})
        for fee,bond,cost,prior in itertools.product(prices['fee_sensitivity'],prices['bond_sensitivity'],prices['cost_multipliers'],prices['invalid_bundle_priors']):
            def payoff(block,behavior):
                needed=(0,) if prior==0 else (0,1)
                if any((block,behavior,m) not in bank for m in needed): return None
                total=0.
                for m in needed:
                    a=bank[block,behavior,m]; weight=(1-prior if m==0 else prior)
                    total+=weight*((fee if a['paid'] else 0.)-(bond if not a['paid'] else 0.)-cost*a['cost_seconds']/unit-prices['liquidity_rate']*bond)
                return total
            honest={b:payoff(b,'honest') for b in blocks}
            for behavior in behaviors:
                values=[]; ids=[]
                for b in blocks:
                    value=payoff(b,behavior)
                    if value is not None and honest[b] is not None:
                        values.append(value if behavior=='honest' else value-honest[b]);ids.append(b)
                estimate=paired_bootstrap(values,seed,draws)
                utility.append({'dataset':dataset,'package':package,'method':method,'steps':steps,'batch_size':size,'anchor':anchor,
                    'behavior':behavior,'quantity':'honest_IR' if behavior=='honest' else 'deviation_minus_honest',
                    'fee':fee,'bond':bond,'cost_multiplier':cost,'invalid_bundle_prior':prior,
                    'independent_unit':'trajectory' if package=='M7' else 'source-block','block_ids':ids,
                    'estimate':estimate,'missing_pair_reason':'no registered matched m0/m1 measurements' if not values else None,
                    'fixed_strategy_reweighting':True})
    conditional=[]
    for key,bank in sorted(mixed.items(),key=lambda x:str(x[0])):
        dataset,package,method,steps,size,anchor=key
        baseline_key=(dataset,'M2' if package=='M2C' else package,method,steps,size,anchor)
        baseline=mixed.get(baseline_key,{})
        unit=profile['calibration'][dataset]['cost_unit_seconds']
        for invalid,behavior in sorted({(m,a) for _,a,m in bank}):
            pairs=[]
            for block in sorted({b for b,_,_ in bank}):
                a=bank.get((block,behavior,invalid));h=baseline.get((block,'honest',invalid))
                if a is not None and h is not None and a['source_group']==h['source_group']:
                    pairs.append((block,a,h))
            for fee,bond,cost in itertools.product(prices['fee_sensitivity'],prices['bond_sensitivity'],prices['cost_multipliers']):
                def net(a):return (fee if a['paid'] else -bond)-cost*a['cost_seconds']/unit-prices['liquidity_rate']*bond
                conditional.append({'dataset':dataset,'package':package,'method':method,'steps':steps,'batch_size':size,
                    'anchor':anchor,'invalid':invalid,'behavior':behavior,'fee':fee,'bond':bond,'cost_multiplier':cost,
                    'independent_unit':'trajectory' if package=='M7' else 'source-block','block_ids':[b for b,_,_ in pairs],
                    'honest_baseline_package':baseline_key[1],
                    'honest_unit_ids':[h['origin_unit_id'] for _,_,h in pairs],
                    'honest_IR':paired_bootstrap([net(h) for _,_,h in pairs],seed,draws),
                    'deviation_minus_honest':paired_bootstrap([net(a)-net(h) for _,a,h in pairs],seed,draws),
                    'scope':'conditional on registered invalid stratum; unmatched prior mixtures remain unidentified'})
    cells=defaultdict(list)
    for row in utility:
        if row['package']=='M1':
            cells[row['dataset'],row['method'],row['fee'],row['bond'],row['cost_multiplier'],row['invalid_bundle_prior']].append(row)
    conditions=[]
    for key,rs in sorted(cells.items()):
        honest=next((r for r in rs if r['quantity']=='honest_IR'),None)
        deviations=[r for r in rs if r['quantity']=='deviation_minus_honest']
        available=honest is not None and honest['estimate']['n'] and len(deviations)==3 and all(r['estimate']['n'] for r in deviations)
        supported=bool(available and honest['estimate']['interval'][0]>0 and all(r['estimate']['interval'][1]<0 for r in deviations))
        conditions.append({'dataset':key[0],'method':key[1],'fee':key[2],'bond':key[3],
            'cost_multiplier':key[4],'invalid_bundle_prior':key[5],
            'all_registered_M1_pairs_available':bool(available),'honest_IR_and_specified_deviation_intervals_support_cell':supported,
            'strategies':[r['behavior'] for r in deviations],'scope':'M1 individual fixed strategies only; no coalition equilibrium or simultaneous grid guarantee'})
    return {'availability':availability,'paired_risks':paired,'utility_sensitivity':utility,
            'conditional-strategy-contrasts':conditional,'nonempty-incentive-conditions':conditions}


def cost_contrasts(costs,science):
    groups=defaultdict(dict)
    for row in costs:
        groups[row['dataset'],row['steps'],row['batch_size'],row['invalid']][row['block'],row['method']]=row
    contrasts=[]
    for key,bank in sorted(groups.items()):
        methods=sorted({m for _,m in bank}); blocks=sorted({b for b,_ in bank})
        for method in methods:
            if method==science['methods']['O']:continue
            for metric in ('owner_busy_wall_seconds','online_suffix_seconds','composed_full_job_seconds'):
                pairs=[(bank[b,method][metric],bank[b,science['methods']['O']][metric]) for b in blocks
                       if (b,method) in bank and (b,science['methods']['O']) in bank]
                contrasts.append({'dataset':key[0],'steps':key[1],'batch_size':key[2],'invalid':key[3],
                    'method':method,'baseline':science['methods']['O'],'metric':metric,
                    'difference':paired_bootstrap([a-b for a,b in pairs],**science['bootstrap']),
                    'paired_ratio':paired_bootstrap([a/b for a,b in pairs if b>0],**science['bootstrap']),
                    'same_run_same_worker_only':True})
    return contrasts


def claim_packet(rows,observations,references,summaries,recovery,profile,science):
    functional=[]; scope=[]
    for uid,values in observations.items():
        row=rows[uid]
        for value in values:
            if value['honest_false_penalty']:
                functional.append({'unit_id':uid,'target':'honest admission','event':'honest false penalty'})
            if value['wrong_production_admitted']:
                full=row['method'] in (science['methods']['AA'],science['methods']['PA'])
                (functional if full else scope).append({'unit_id':uid,'target':'full audit report correctness' if full else 'bounded probe/sample protection',
                                                       'event':'wrong production admitted'})
    disagreements=[r for r in references if not r['agrees']]
    functional.extend({'target':'reference agreement','event':r} for r in disagreements)
    base_util=[r for r in summaries['utility_sensitivity'] if r['package']=='M1' and r['method']==science['methods']['R']
               and r['fee']==science['prices']['fee'] and r['bond']==science['prices']['bond'] and r['cost_multiplier']==1]
    utility_negative=[r for r in base_util if r['estimate']['n'] and
        (r['estimate']['interval'][1]<0 if r['quantity']=='honest_IR' else r['estimate']['interval'][0]>0)]
    ids={'P2SOURCE':['availability.json','source-task-identities.jsonl','assignment-audit.jsonl'],
         'P2MECH':['primary-risks.json','paired_risks.json','paired-cost-contrasts.json'],
         'P2REF':['probe-reference-agreement.json','phase-timing.jsonl'],
         'P2JOINT':['trainer-settlement-outcomes.json','report-lifecycle.jsonl','recovery-independent-audit.json'],
         'P2ECS':['recovery-independent-audit.json','units/'],
         'P2COST':['full-job-costs.json','paired-cost-contrasts.json'],
         'P2UTILITY':['utility_sensitivity.json','conditional-strategy-contrasts.json','fixed-strategy-cost-grid.json','units/']}
    all_measured=all(r.get('issued',True) for r in rows.values() if r['package']!='M0')
    negative=bool(functional or utility_negative)
    claims=[]
    for key,files in ids.items():
        state='INDETERMINATE'
        if key=='P2SOURCE':state='SUPPORTED'
        if key=='P2REF':state='NEGATIVE' if disagreements else 'SUPPORTED' if references else 'INDETERMINATE'
        if key in ('P2MECH','P2JOINT') and functional:state='NEGATIVE'
        if key=='P2UTILITY' and utility_negative:state='NEGATIVE'
        claims.append({'id':key,'state':state,'evidence':files,
                       'scope':'observed registered finite experiments; paper wording requires later author review'})
    return {'full_matrix':profile['full_matrix'],'terminal':'ACCEPTED_NEGATIVE' if negative else None,
            'terminal_review_pending':not negative,
            'route':'USER_INTERPRETATION_REQUIRED' if profile['full_matrix'] else 'NO_PAPER_CHANGE_TECHNICAL_ONLY',
            'EVIDENCE_ATTEMPTS_COMPLETE':True,'ALL_REQUESTED_MEASUREMENTS_AVAILABLE':all_measured,
            'CLAIMS_SUPPORTED':False,'PUBLICATION_READY':False,'claims':claims,
            'functional_counterexamples':functional,'registered_boundary_events':scope,
            'base_price_utility_negative_cells':utility_negative,'complete_registered_grid_retained':True,
            'native_reproduction':'four-observation scoring/LP model only; not full CTF system',
            'training_reproduction':'three trajectories per dataset and three anchors; not convergence evidence',
            'predecessor':'historical results are not included in this source distribution',
            'close_archive_complete':False}
