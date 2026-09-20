"""Bounded paired engineering measurements assembled from the canonical job."""
from dataclasses import asdict
import gc
import time
from pathlib import Path
from sevc.core.artifacts import write_json, sha256_file
from sevc.verification.on_demand_service import OnlineJob, build_source_bank
from sevc.verification.replay_coupled_probes import proof_component_hashes
from sevc.verification.paid_replay_service import StoredReplayProof


def measure_owner_cost(context, partition, clock, performance, config, root):
    from sevc.core.replay_environment import configure_strict_runtime, capture_environment, identity
    import json
    configure_strict_runtime()
    expected = json.loads(Path(config['engineering_environment_reference_path']).read_text())
    actual = capture_environment(config['datasets'], performance, config['profiles'][config['default_profile']]['gpu_uuid'])
    for key in ('gpu_uuid','gpu_model','python','torch','torchvision','cuda','cudnn','driver',
                'device','cpu_threads','interop_threads','deterministic','cudnn_benchmark',
                'cudnn_deterministic','matmul_tf32','cudnn_tf32'):
        if actual.get(key) != expected.get(key):
            raise ValueError('engineering environment mismatch: '+key)
    context.replay_environment_id = identity(actual)
    folder = Path(root) / context.dataset
    folder.mkdir()
    write_json(folder/'environment.json', actual)
    scratch = folder / 'scratch'
    scratch.mkdir()
    sources = scratch / 'sources'
    sources.mkdir()
    seed = config['engineering_seeds'][context.dataset]
    records = []
    clock.context = {'dataset': context.dataset, 'variant': 'shared-input'}
    began = time.monotonic()
    bank, recipes, _ = build_source_bank(context, seed=seed,
        namespace='owner-cost-repair-known-seed-v1', partition=partition,
        steps=4, batch_size=32, invalid_count=0, clock=clock,
        scratch_dir=sources, access_profile='authenticated-mmap', capture_compact_headers=True)
    source_seconds = time.monotonic()-began
    header_charge = sum(s.compact_header_seconds for s in bank)
    write_json(folder/'source-identities.json', {'sources':recipes,
        'shared_source_wall_seconds':source_seconds,'owner_header_capture_seconds':header_charge})
    # This secret is for a known engineering fixture only, not an unseen science draw.
    from sevc.verification.reference_acquisition import commitment
    secret = commitment(['owner-cost-engineering', context.dataset, seed])
    targets = None
    public_identity = None
    final_report = None
    result = {'dataset':context.dataset, 'seed':seed, 'steps':4, 'batch_size':32,
        'new_independent_scientific_blocks':0,'scope':'engineering_fixed_order_same_source',
        'source_seconds':source_seconds,'header_capture_owner_charge':header_charge,'rows':[],
        'equivalence_passed':False,'completed':False}
    for variant in ('before-R', 'before-O', 'after-R', 'after-O'):
        repaired = variant.startswith('after')
        direct = variant.endswith('O')
        perf = {**performance, 'compact_source_headers':repaired,
                'resident_delivery_bytes':config['resident_delivery_bytes'] if repaired else 0}
        clock.context = {'dataset':context.dataset, 'variant':variant, 'phase_scope':'online'}
        delivery = scratch / variant
        delivery.mkdir()
        started = time.monotonic()
        job = OnlineJob(bank=bank, context=context, clock=clock, performance=perf,
            job_id='engineering-paired-job', method_key=('owner-direct-v2' if direct else
                'rcmp-opaque-gradient-continuation-complete-v1'), role_secret=secret,
            emit=records.append, paired_production=targets if direct else None,
            delivery_scratch_dir=delivery)
        prep = time.monotonic()-started
        if not job.issued:
            raise ValueError('known valid engineering source failed acquisition')
        if targets is None:
            targets = job.production_source_ids
        if job.production_source_ids != targets:
            raise ValueError('paired production targets drifted')
        if direct:
            started = time.monotonic()
            verdicts = [job.references.acquire(sid) for sid in targets]
            admission = time.monotonic()-started
            if not all(v['passed'] for v in verdicts):
                raise ValueError('direct reference failed')
            calibration = 0.0  # all 32 production references already acquired and charged
            verifier = 0.0
            task_identity = None
        else:
            task_identity = [{'task_id':t.task_id,'envelope':t.envelope,'descriptor':t.descriptor}
                             for t in job.tasks]
            if public_identity is None:
                public_identity = task_identity
            elif task_identity != public_identity:
                raise ValueError('repair changed public task identity')
            job.begin_service_epoch(['paired-honest'])
            report, settlement, details = job.serve(behavior='honest', seed=seed,
                assignment_id='paired-honest', audit_secret=secret, trainer_cache={},
                commits=records.append, verifier_id='v0', job_id=job.job_id)
            if settlement.status != 'PASS':
                raise ValueError('honest settlement changed')
            report_identity = asdict(report)
            if final_report is None:
                final_report = report_identity
            elif report_identity != final_report:
                raise ValueError('repair changed committed report')
            admission = dict(settlement.diagnostics)['owner_admission_seconds']
            verifier = settlement.verifier_cost
            job.close_service_epoch()
            clock.context['phase_scope'] = 'post-service-calibration'
            started = time.monotonic()
            verdicts = [job.references.acquire(sid) for sid in targets]
            calibration = time.monotonic()-started
            if not all(v['passed'] for v in verdicts):
                raise ValueError('production calibration changed')
            # Outside timing: ensure delivered tensor identities, including resident
            # candidates, still match their committed content after actor execution.
            clock.context['phase_scope'] = 'engineering-equivalence-audit'
            for task in job.tasks:
                proof = task.proof.load() if isinstance(task.proof, StoredReplayProof) else task.proof
                if proof_component_hashes(proof) != task.envelope['wrapped_proof_identity']:
                    raise ValueError('delivered tensor changed or compact header identity is false')
            del proof
        extra = header_charge if repaired and not direct else 0.0
        row = {'variant':variant, 'preparation_seconds':prep,
            'owner_admission_or_direct_seconds':admission,'owner_header_capture_charge':extra,
            'owner_online_seconds':prep+admission+extra,
            'calibration_seconds':calibration,'owner_including_calibration_seconds':prep+admission+extra+calibration,
            'verifier_seconds':verifier,'service_fee_excluded':True,
            'public_tasks':task_identity,'task_compile_details':job.task_rows,
            'scratch_bytes':sum(p.stat().st_size for p in delivery.rglob('*') if p.is_file())}
        result['rows'].append(row)
        write_json(folder/'paired-results.json', result)
        del job
        context.close_task_lanes()
        gc.collect()
        # Preserve hashes first; only this test's disposable derivatives are removed.
        ledger = [{'path':str(p),'sha256':sha256_file(p),'bytes':p.stat().st_size}
                  for p in delivery.rglob('*') if p.is_file()]
        write_json(folder/(variant+'-scratch-seal.json'), ledger)
        for item in ledger:
            Path(item['path']).unlink()
        print({'owner_cost_pair_complete':context.dataset,'variant':variant,
               'owner_seconds':row['owner_online_seconds']}, flush=True)
    by = {r['variant']:r for r in result['rows']}
    result.update(equivalence_passed=True, completed=True,
        before_ratio=by['before-R']['owner_online_seconds']/by['before-O']['owner_online_seconds'],
        after_ratio=by['after-R']['owner_online_seconds']/by['after-O']['owner_online_seconds'],
        after_including_calibration_ratio=by['after-R']['owner_including_calibration_seconds']/by['after-O']['owner_including_calibration_seconds'])
    result['online_cost_gate'] = result['after_ratio'] <= .8
    result['including_calibration_cost_gate'] = result['after_including_calibration_ratio'] <= .8
    write_json(folder/'paired-results.json',result)
    write_json(folder/'events.json',records)
    ledger = [{'path':str(p),'sha256':sha256_file(p),'bytes':p.stat().st_size}
              for p in sources.glob('*.pt')]
    write_json(folder/'source-file-seal.json',ledger)
    del bank
    gc.collect()
    for item in ledger:
        Path(item['path']).unlink()


def measure_restored_owner_pair(study, context):
    """Affected owner timing only; no new verifier decisions or evidence rewrites."""
    destination=study.root/'owner-cost-repair-pairs.json'
    if destination.exists():
        return
    from sevc.experiments.submission_continuation import restore_group
    from sevc.incentives.verifier_protocol import CommittedVerifierReport
    candidates=[u for u in study.results.values() if u.get('dataset')=='cifar10'
        and u.get('invalid')==0 and u.get('package')=='M1' and u.get('behavior')=='honest'
        and u.get('method')==study.science['methods']['R']]
    if len(candidates)!=1:
        raise ValueError('owner repair requires the unique frozen predecessor R service')
    previous=candidates[0]
    uid=previous['unit_id'];group_id=previous['source_group']
    study.clock.context={'dataset':'cifar10','phase_scope':'owner-repair-migration'}
    bank,_,_,_,_=restore_group(study,group_id)
    secret=study.secret(group_id,'roles')
    phase_rows=[]
    values=[]
    for method in ('R','O'):
        study.clock.context={'dataset':'cifar10','phase_scope':'owner-repair-paired-measurement','method':method}
        delivery=Path(study.config['scratch_parent'])/'owner-repair-pair'/method
        delivery.mkdir(parents=True,exist_ok=False)
        started=time.monotonic()
        job=OnlineJob(bank=bank,context=context,clock=study.clock,performance=study.performance,
            job_id=uid+'-job',method_key=study.science['methods'][method],role_secret=secret,
            emit=phase_rows.append,delivery_scratch_dir=delivery,
            paired_production=previous['production_source_ids'] if method=='O' else None)
        preparation=time.monotonic()-started
        if not job.issued or list(job.production_source_ids)!=previous['production_source_ids']:
            raise ValueError('owner repair changed source acquisition')
        if method=='R':
            old={t['task_id']:t['wrapped_hashes'] for t in previous['tasks']}
            new={t['task_id']:t['wrapped_hashes'] for t in job.task_rows}
            if new!=old:
                raise ValueError('owner repair changed committed task content')
            a=previous['assignments'][0]
            report=CommittedVerifierReport(**a['report'])
            settled,admission=study.clock.call('owner-repair-fixed-report-settlement','owner',
                job._admit,report,a['execution'],a['assignment_id'],study.secret(a['assignment_id'],'audits'),
                a['settlement']['verifier_cost'])
            if (settled.status,settled.service_fee,settled.slashed_bond)!=(
                    a['settlement']['status'],a['settlement']['service_fee'],a['settlement']['slashed_bond']):
                raise ValueError('owner repair altered frozen-report settlement')
        else:
            started=time.monotonic()
            results=[job.references.acquire(sid) for sid in job.production_source_ids]
            admission=time.monotonic()-started
            if not all(r['passed'] for r in results):
                raise ValueError('owner repair direct comparison changed verdict')
        values.append({'method':method,'preparation_wall_seconds':preparation,
            'metadata_charge_seconds':job.header_capture_charge,'settlement_or_direct_seconds':admission,
            'owner_seconds':preparation+admission+job.header_capture_charge})
        del job
        context.close_task_lanes()
        gc.collect()
        ledger=[{'path':str(p),'sha256':sha256_file(p),'bytes':p.stat().st_size}
                for p in delivery.rglob('*') if p.is_file()]
        write_json(study.root/('owner-repair-'+method+'-scratch-seal.json'),ledger)
        for item in ledger:
            Path(item['path']).unlink()
    write_json(destination,{'dataset':'cifar10','invalid':0,'anchor':0,'steps':4,'batch_size':32,
        'predecessor_unit_id':uid,'source_group':group_id,'source_environment_id':previous['replay_environment_id'],
        'measurement_environment_id':context.replay_environment_id,'owner_cost_revision':study.config['owner_cost_revision'],
        'new_verifier_assignments':0,'new_independent_blocks':0,'fixed_report_reused':True,
        'old_costs_preserved':True,'rows':values,'R_O':values[0]['owner_seconds']/values[1]['owner_seconds'],
        'cost_gate':values[0]['owner_seconds']/values[1]['owner_seconds']<=.8,
        'scope':'paired repaired owner stages, not a replacement for historical assignment costs',
        'events':phase_rows})


def measure_f_readiness_repair(context, partition, clock, performance, config, root):
    """Fixed development cells; all training/replay/settlement reuse domain APIs."""
    import json
    import torch
    from types import SimpleNamespace
    from sevc.core.replay_environment import configure_strict_runtime, capture_environment, identity
    from sevc.core.scratch_cleanup import remove_owned_scratch
    from sevc.verification.on_demand_service import complete_calibration_observation, replay_owner_proof
    from sevc.verification.reference_acquisition import commitment
    from sevc.reputation.calibration import PaidCalibrationLedger
    from sevc.verification.replay_coupled_probes import derive_int
    configure_strict_runtime()
    actual = capture_environment(config['datasets'], performance,
        config['profiles'][config['default_profile']]['gpu_uuid'])
    expected = json.loads(Path(config['engineering_environment_reference_path']).read_text())
    for key in ('gpu_uuid','gpu_model','python','torch','torchvision','cuda','cudnn','driver',
                'device','cpu_threads','interop_threads','deterministic','cudnn_benchmark',
                'cudnn_deterministic','matmul_tf32','cudnn_tf32'):
        if actual.get(key) != expected.get(key):
            raise ValueError('repair environment mismatch: '+key)
    context.replay_environment_id = identity(actual)
    folder=Path(root)/context.dataset;folder.mkdir()
    write_json(folder/'environment.json',actual)
    spec=config['repair_protocol'];seed=config['engineering_seeds'][context.dataset]
    perf={**performance,'compact_source_headers':True,'resident_delivery_bytes':config['resident_delivery_bytes']}
    pending=[]
    def emit(row):
        with (folder/'events.jsonl').open('a') as stream:
            stream.write(json.dumps(row,sort_keys=True)+'\n')
    def cleanup(path):
        gc.collect();context.close_task_lanes()
        if not remove_owned_scratch(path,emit):pending.append(path)
    methods={'old-R':'rcmp-opaque-gradient-continuation-complete-v1',
             'new-R':'rcmp-opaque-gradient-continuation-gpu-v1',
             'new-G':'hidden-gold-gradient-continuation-gpu-v1','O':'owner-direct-v2'}
    for invalid in spec['invalid_counts']:
        cell=folder/f'invalid-{invalid}';cell.mkdir();scratch=cell/'scratch';scratch.mkdir()
        sources=scratch/'sources';sources.mkdir()
        namespace=spec['source_namespace']+f'-invalid-{invalid}'
        clock.context={'dataset':context.dataset,'invalid':invalid,'phase_scope':'shared-input'}
        bank,recipes,_=build_source_bank(context,seed=seed,namespace=namespace,partition=partition,
            steps=spec['steps'],batch_size=spec['batch_size'],invalid_count=invalid,clock=clock,
            scratch_dir=sources,access_profile='authenticated-mmap',capture_compact_headers=True)
        write_json(cell/'sources.json',recipes)
        secret=commitment([namespace,context.dataset,seed,'roles']);job_id=commitment([namespace,context.dataset,'job'])
        target_job,selection_seconds=clock.call('pair-target-selection','offline',OnlineJob,
            bank=bank,context=context,clock=clock,performance=perf,job_id=job_id,
            method_key=methods['new-R'],role_secret=secret,emit=emit,prepare_only=True)
        if not target_job.issued:raise ValueError('source acquisition failed; preserve cell')
        targets=target_job.production_source_ids;del target_job
        result={'dataset':context.dataset,'invalid':invalid,'seed':seed,'steps':spec['steps'],
            'batch_size':spec['batch_size'],'scope':'development_not_independent_parameter_evidence',
            'pair_selection_seconds_separate':selection_seconds,'rows':[],
            'production_ids':targets,'completed':False,'fees_excluded_from_owner_work':True}
        order=('old-R','new-R','new-G','O') if invalid==0 else ('new-R','old-R','O','new-G')
        for variant in order:
            clock.context={'dataset':context.dataset,'invalid':invalid,'variant':variant,'phase_scope':'online'}
            delivery=scratch/variant;delivery.mkdir()
            def gold_factory():
                gold_dir=delivery/'gold';gold_dir.mkdir()
                gold,gold_rec,_=build_source_bank(context,seed=seed,namespace=namespace+'-gold',partition=partition,
                    steps=spec['steps'],batch_size=spec['batch_size'],invalid_count=0,count=8,source_offset=40,
                    clock=clock,role='owner',scratch_dir=gold_dir,access_profile='authenticated-mmap',capture_compact_headers=True)
                write_json(cell/'gold-sources.json',gold_rec)
                return gold
            started=time.monotonic()
            job=OnlineJob(bank=bank,context=context,clock=clock,performance=perf,job_id=job_id,
                method_key=methods[variant],role_secret=secret,emit=emit,
                paired_production=targets if variant in ('O','new-G') else None,
                gold_factory=gold_factory if variant=='new-G' else None,delivery_scratch_dir=delivery)
            preparation=time.monotonic()-started
            if not job.issued or job.production_source_ids != targets:raise ValueError('paired sources drifted')
            row={'variant':variant,'method':methods[variant],'preparation_seconds':preparation,
                'header_charge_seconds':job.header_capture_charge,'tasks':job.task_rows,'services':[]}
            if variant=='O':
                started=time.monotonic();truth={sid:job.references.acquire(sid) for sid in targets}
                row['owner_online_seconds']=preparation+time.monotonic()-started
                row['production_reference_receipts']=truth
            else:
                behaviors=list(spec['strategies']) if variant=='new-R' else ['honest']
                services=[(b,'v0' if b=='honest' else 'deviation-'+b) for b in behaviors]
                if variant=='new-R':services += [('honest','v'+str(i)) for i in range(1,spec['identity_count'])]
                aids=[f'{variant}-{i}' for i in range(len(services))]
                job.begin_service_epoch(aids)
                reports=[]
                for aid,(behavior,vid) in zip(aids,services):
                    clock.context.update(assignment_id=aid,verifier_id=vid)
                    report,settled,details=job.serve(behavior=behavior,seed=derive_int(seed,invalid,behavior,vid)%2**32,
                        assignment_id=aid,audit_secret=commitment([secret,aid,'audits']),trainer_cache={},
                        commits=emit,verifier_id=vid,job_id=job_id)
                    entry={'assignment_id':aid,'verifier_id':vid,'behavior':behavior,'report':asdict(report),
                        'settlement':asdict(settled),'execution':details,
                        'observed_utility':settled.service_fee-settled.slashed_bond-.01*settled.verifier_cost}
                    row['services'].append(entry);reports.append((aid,behavior,vid,report,settled))
                    write_json(cell/(variant+'-partial.json'),row)
                row['owner_online_seconds']=preparation+job.header_capture_charge+dict(reports[0][4].diagnostics)['owner_admission_seconds']
                row['all_honest_admitted']=all(s.status=='PASS' for _,b,_,_,s in reports if b=='honest')
                job.close_service_epoch()
                # Production references become available only after the epoch closes.
                clock.context.pop('assignment_id',None);clock.context.pop('verifier_id',None)
                clock.context['phase_scope']='post-service-calibration'
                started=time.monotonic();truth={sid:job.references.acquire(sid) for sid in targets}
                row['post_service_owner_seconds']=time.monotonic()-started
                row['production_reference_receipts']=truth
                row['calibration']=[]
                for aid,behavior,vid,report,settled in reports:
                    values=dict(zip(report.ordered_segment_ids,report.verdicts))
                    wrong=[tid for tid,sid in job.production.items() if values.get(tid)!=truth[sid]['passed']]
                    if wrong and settled.accepted_report:
                        witness_dir=cell/'witnesses';witness_dir.mkdir(exist_ok=True)
                        for task in job.tasks:
                            if task.task_id in wrong:
                                proof=task.proof.load() if isinstance(task.proof,StoredReplayProof) else task.proof
                                path=witness_dir/(task.task_id+'.pt')
                                if not path.exists():torch.save(proof,path)
                                emit({'event':'ADMITTED_WRONG_REPORT_WITNESS','assignment_id':aid,
                                      'path':str(path),'sha256':sha256_file(path)});del proof
                    if variant=='new-R' and behavior=='honest':
                        ledger=PaidCalibrationLedger(vid,budget=2.5,fee=2.5)
                        ledger.reserve(aid,commitment([namespace,seed,'development']))
                        complete_calibration_observation(ledger=ledger,assignment_id=aid,report=report,
                            settlement=settled,production=job.production,references=job.references,clock=clock)
                        row['calibration'].append({'observations':ledger.observations,
                            'summary':ledger.summary(alpha=.05/54,independent_stationary_blocks=False),
                            'qualified':ledger.production_admissible(minimum_correctness=.9,minimum_availability=.9,
                                alpha=.05/54,independent_stationary_blocks=False)})
                # Domain hashes checked outside timed services; don't overwrite raw results.
                for task in job.tasks:
                    proof=task.proof.load() if isinstance(task.proof,StoredReplayProof) else task.proof
                    if proof_component_hashes(proof)!=task.envelope['wrapped_proof_identity']:
                        raise ValueError('delivered tensor identity changed')
                del proof,task,report,settled,details,reports
            result['rows'].append(row);write_json(cell/'paired-results.json',result)
            del job;cleanup(delivery)
            print({'repair_cell_progress':context.dataset,'invalid':invalid,'variant':variant,
                   'owner_seconds':row['owner_online_seconds']},flush=True)
        by={row['variant']:row for row in result['rows']}
        result.update(completed=True,old_R_O=by['old-R']['owner_online_seconds']/by['O']['owner_online_seconds'],
            new_R_O=by['new-R']['owner_online_seconds']/by['O']['owner_online_seconds'],
            observed_owner_reduction=by['new-R']['owner_online_seconds']<by['O']['owner_online_seconds'])
        result['diagnostic_20_percent_gate']=result['new_R_O']<=.8
        result['formal_sign_test_pass']=False
        write_json(cell/'paired-results.json',result)
        del bank;cleanup(scratch)
    # Reuse the canonical trajectory assembly; this is not a second train loop.
    from sevc.experiments.scoped_five_rq_units import ScopedStudy
    from sevc.verification.replay_coupled_probes import compile_canonical_replay_task,ATOM_KEYS
    from sevc.verification.paid_replay_service import VERSION,WRAPPER_DOMAIN
    (folder/'milestones').mkdir()
    adapter=SimpleNamespace(clock=clock,config=config,root=folder,events=emit)
    trajectory=ScopedStudy.trajectory(adapter,context,derive_int(seed,'target-trajectory')%2**32,partition)
    for anchor in spec['target_anchors']:
        state=torch.load(trajectory/f'step-{anchor}.pt',weights_only=False,map_location='cpu')
        target=folder/f'target-{anchor}';target.mkdir();scratch=target/'scratch';scratch.mkdir()
        bank,recipes,_=build_source_bank(context,seed=seed,namespace=spec['source_namespace']+f'-target-{anchor}',
            partition=partition,steps=spec['target_steps'],batch_size=spec['batch_size'],invalid_count=0,
            count=1,clock=clock,initial_state=state['model'],initial_momentum=state['momentum'],
            initial_rng={'cpu_rng':state['cpu_rng'],'cuda_rng':state['cuda_rng']},scratch_dir=scratch,
            access_profile='authenticated-mmap',capture_compact_headers=True)
        source=bank[0];proof=source.proof
        reference,seconds=replay_owner_proof(proof,context=context,clock=clock,performance=perf,phase='target-reference')
        rows=[];model=context.factory().cpu()
        for atom in ATOM_KEYS:
            bundle=compile_canonical_replay_task(proof,model,context.build_key,source_id=recipes[0]['source_id'],
                source_commitment=source.hashes['proof_sha256'],post_commit_seed=seed,role='challenge',atom_key=atom,
                permutation_seed=seed,protocol_version=VERSION,wrapper_seed_domain=WRAPPER_DOMAIN,tamper_delta=4e-5,
                source_component_hashes=source.hashes,delivery_profile='compact',identity_profile='fused',
                schema_validator_profile='shared',validated_source_verdict=reference['passed'],
                mutation_profile='gradient-continuation-v3',wrapper_profile='identity',challenge_device=context.device.name)
            verdict,cost=replay_owner_proof(bundle.canonical_candidate,context=context,clock=clock,performance=perf,phase='target-challenge')
            rows.append({'atom':atom,'verdict':verdict,'seconds':cost,'hashes':bundle.component_hashes,'compile_seconds':bundle.compile_seconds})
            if verdict['passed']:
                torch.save(bundle.canonical_candidate,target/(atom+'-unexpected-pass.pt'))
            del bundle
        write_json(target/'results.json',{'anchor':anchor,'steps':spec['target_steps'],'recipes':recipes,
            'reference':reference,'reference_seconds':seconds,'challenges':rows,
            'correctness_pass':reference['passed'] and all(not x['verdict']['passed'] for x in rows),
            'scope':'one-source engineering bridge; no full-matrix or statistical transfer claim'})
        if not reference['passed']:torch.save(proof,target/'unexpected-source-failure.pt')
        del proof,source,bank,state,model;cleanup(scratch)
    # Keep reconstructible order and measured prefix costs after tensor cleanup.
    import shutil
    retained=folder/'trajectory-metadata';retained.mkdir()
    for item in trajectory.glob('*.json'):
        shutil.copyfile(item,retained/item.name)
    cleanup(trajectory)
    for path in pending[:]:
        if remove_owned_scratch(path,emit):pending.remove(path)
    write_json(folder/'completion.json',{'completed':True,'remaining_scratch':list(map(str,pending)),
        'independent_parameter_blocks':0,'F_started':False})
