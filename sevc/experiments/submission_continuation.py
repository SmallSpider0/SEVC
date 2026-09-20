"""Hash-bound restoration of canonical tiny state; no measurement implementation."""
import json
import shutil
from pathlib import Path
from sevc.core.artifacts import sha256_file, write_json
from sevc.verification.on_demand_service import Source
from sevc.reputation.calibration import PaidCalibrationLedger


def checked_manifest(path):
    manifest=json.loads(Path(path).read_text())
    if not manifest.get('sealed') or (manifest.get('answers_published') and not manifest.get('closed_datasets')):
        raise ValueError('continuation requires sealed undisclosed predecessor')
    for row in manifest['files']:
        if sha256_file(Path(row['path']))!=row['sha256']:
            raise ValueError('predecessor hash mismatch: '+row['path'])
    validate_closed_datasets(manifest)
    return manifest


def source_capabilities(group):
    bank=[]
    for row in group['row']['sources']:
        hashes={k:row[k] for k in ('proof_sha256','checkpoint_sha256','optimizer_sha256')}
        recipe={k:v for k,v in row.items() if k not in {*hashes,'trainer_mutation','scratch_sha256'}}
        path=Path(group['source_directory'])/f"source-{row['source_index']}.pt"
        from sevc.core.tensor_storage import AuthenticatedTensorFile
        reader = AuthenticatedTensorFile.inspect_owned(path)
        if reader.digest != row['scratch_sha256']:
            raise ValueError('source payload hash mismatch')
        bank.append(Source(None,hashes,recipe,str(path),row['scratch_sha256'],'authenticated-mmap',reader))
    return tuple(bank)


def retained_log_row(row, completed, all_ids):
    """Drop partial-unit rows only from the effective view; predecessor is immutable."""
    if 'unit_id' in row:
        return row['unit_id'] in completed
    for key in ('assignment_id','job_id'):
        value=row.get(key,'')
        uid=value[:64]
        if uid in all_ids:
            return uid in completed
    return row.get('event') not in {'reference-answers-published'}


def production_targets(results, group_id):
    """Restore the already issued paired task identities without acquiring references."""
    targets={tuple(r['production_source_ids']) for r in results
             if r.get('source_group')==group_id and r.get('production_source_ids')}
    if len(targets)>1:
        raise ValueError('predecessor paired production identity disagreement')
    return next(iter(targets),None)


def restore(study):
    path=study.config.get('continuation_manifest_path')
    study.restored_groups={}
    study.closed_datasets=set()
    study.restored_bank_cache={}
    if not path:return
    manifest=checked_manifest(path)
    if sha256_file(Path(path))!=study.lock['continuation_manifest_sha256']:
        raise ValueError('continuation manifest identity mismatch')
    root=Path(manifest['predecessor_root'])
    study.closed_datasets=set(manifest.get('closed_datasets',[]))
    for name in ['owner-cost-repair-pairs.json','owner-repair-R-scratch-seal.json','owner-repair-O-scratch-seal.json','calibration-ledger.jsonl']:
        if (root/name).exists():
            shutil.copyfile(root/name,study.root/name)
    if (root/'native-witnesses').exists():
        shutil.copytree(root/'native-witnesses',study.root/'native-witnesses')
    all_ids={u['unit_id'] for u in study.all_units}
    completed=set(manifest['completed_unit_ids'])
    if not completed.issubset(all_ids):raise ValueError('unregistered predecessor unit')
    from sevc.core.evidence_rows import execution_identity_rows
    from sevc.core.replay_environment import require_continuation
    require_continuation(manifest,
        execution_identity_rows([json.loads((root/'units'/f'{uid}.json').read_text()) for uid in sorted(completed)],study.all_units),
        study.config.get('_runtime_replay_environment_id'), study.config.get('_runtime_replay_environment_bridge'))
    for uid in sorted(completed):
        source=root/'units'/f'{uid}.json';row=json.loads(source.read_text())
        expected=next(u for u in study.all_units if u['unit_id']==uid)
        if any(row[k]!=v for k,v in expected.items()):raise ValueError('predecessor unit identity drift')
        shutil.copyfile(source,study.root/'units'/source.name)
        study.results[uid]=row
        if study.lock['resource_budget']['unit_wall_seconds'] is not None and row.get('wall_seconds',0)>study.lock['resource_budget']['unit_wall_seconds']:
            study.unit_budget_exceeded=True
    counts={}
    for name,writer in [('phase-timing.jsonl',study.clock.emit),
        ('assignment-audit.jsonl',study.records),('source-task-identities.jsonl',study.sources),
        ('report-lifecycle.jsonl',study.commits)]:
        # RoleClock exposes emit via its configured append-only callback.
        kept=dropped=0
        for line in (root/name).read_text().splitlines():
            if not line:continue
            row=json.loads(line)
            if retained_log_row(row,completed,all_ids) or (study.closed_datasets and row.get('event')=='reference-answers-published'):
                writer(row);kept+=1
                if row.get('event')=='job-delivery-ready':
                    for task in row['tasks']:
                        if task.get('probe_answer') is not None:
                            study.disclosure_answers[task['task_id']]=task['probe_answer']
            else:dropped+=1
        counts[name]={'retained':kept,'partial_rows_kept_only_in_predecessor':dropped}
    study.restored_groups=manifest['source_groups']
    for unit in study.all_units:
        if unit['unit_id'] in completed or unit.get('anchor',0):continue
        for group in study.restored_groups.values():
            row=group['row']
            if (group['source_device']!=study.profile['device'] and
                all(unit.get(k)==row.get(k) for k in ('dataset','seed','steps','batch_size','invalid')) and unit['package']!='DP'):
                raise ValueError('cross-backend continuation permits only remaining native DP units in a reused source group')
    for group in study.restored_groups.values():
        dataset=group['row']['dataset']
        study.base_costs[dataset+'-bytes']=sum((Path(group['source_directory'])/
            f"source-{r['source_index']}.pt").stat().st_size for r in group['row']['sources'])
    for uid in sorted(completed):
        row=study.results[uid]
        tasks={t['task_id']:t for t in row.get('tasks',[])}
        for t in tasks.values():
            if t.get('role')!='production' and t.get('probe_answer') is not None:
                study.disclosure_answers[t['task_id']]=t['probe_answer']
        for a in row.get('assignments',[]):
            aid=a['assignment_id']
            if study.disclosure_epoch.covers([aid]):
                study.disclosure_epoch.begin(aid)
                study.disclosure_epoch.finish(aid,a['settlement']['status'])
        if row.get('anchor',0)==0 and row.get('issued'):
            dataset=row['dataset']
            study.base_costs[dataset]=max(study.base_costs.get(dataset,0),
                row['shared_trainer_prefix_seconds']+row['online_suffix_seconds'])
        if row.get('calibration') and row.get('issued') and row['dataset'] not in study.closed_datasets:
            group=study.restored_groups[row['source_group']]
            bank=source_capabilities(group)
            for i in range(9):
                vid=f'v{i}';ledger=PaidCalibrationLedger(vid,22.5,2.5,20.)
                ledger.reserve(uid+'-'+vid,f"{row['dataset']}-block0-{row['seed']}")
                study.calibration[row['dataset'],vid]=ledger
            study.calibration_jobs.append({'dataset':row['dataset'],'unit':row,
                'production':{t['task_id']:t['source_sha256'] for t in tasks.values() if t['role']=='production'},
                'sources':{s.hashes['proof_sha256']:s for s in bank},
                'assignments':row['assignments'],'source_device':group['source_device'],
                'replay_environment_id':row.get('replay_environment_id')})
    study.core_counterexamples.extend(manifest['core_counterexamples'])
    write_json(study.root/'continuation-receipt.json',{'manifest_sha256':sha256_file(Path(path)),
        'predecessor_root':str(root),'completed_units_reused_byte_exact':sorted(completed),
        'source_regeneration':False,'prior_attempt_wall_seconds':study.lock['prior_attempt_wall_seconds'],
        'effective_log_selection':counts,'source_device_map':{k:v['source_device'] for k,v in study.restored_groups.items()},
        'prior_partial_phase_costs':manifest.get('prior_partial_phase_costs', {}),
        'predecessor_source_hashes':[r['proof_sha256'] for g in study.restored_groups.values()
            if g['source_device']!=study.profile['device'] for r in g['row']['sources']],
        'closed_datasets_reused':sorted(study.closed_datasets),
        'environment_bridge':study.config.get('_runtime_replay_environment_bridge'),
        'owner_cost_revision':study.config.get('owner_cost_revision'),
        'old_costs_reclassified_as_new':False,
        'absolute_costs_across_devices_comparable':False,'candidate_blocks':3})


def restore_group(study, group_id):
    group=study.restored_groups.get(group_id)
    if group is None:return None
    row=group['row']
    if group_id in study.restored_bank_cache:
        bank=study.restored_bank_cache[group_id]
    else:
        if getattr(study,'performance',{}).get('compact_source_headers'):
            bank,_=study.clock.call('continuation_source_authentication','owner',source_capabilities,group)
            bank=capture_restored_headers(bank,study.clock)
        else:
            bank=source_capabilities(group)
        study.restored_bank_cache[group_id]=bank
    cache={r['proof_sha256']:r['trainer_mutation'] is None for r in row['sources']}
    return bank,row['sources'],cache,row['shared_trainer_prefix_seconds'],row['shared_trainer_prefix_cpu_seconds']


def capture_restored_headers(bank, clock):
    """Migration decoding is startup work; capture cost remains per-job visible."""
    from dataclasses import replace
    from sevc.verification.replay_coupled_probes import CompactProofHeader
    result=[]
    for source in bank:
        proof,_=clock.call('continuation_header_source_decode','owner',lambda:source.proof)
        header,seconds=clock.call('continuation_header_capture','owner',
            CompactProofHeader.capture,proof,source.hashes['proof_sha256'])
        result.append(replace(source,compact_header=header,compact_header_seconds=seconds))
    return tuple(result)


def validate_closed_datasets(manifest):
    """Only fully closed services and durable calibration may cross disclosure."""
    closed=set(manifest.get('closed_datasets',[]))
    if not closed:
        return
    root=Path(manifest['predecessor_root'])
    lock=json.loads((root/'protocol-lock.json').read_text())
    done=set(manifest['completed_unit_ids'])
    calibration=[json.loads(s) for s in (root/'calibration-ledger.jsonl').read_text().splitlines() if s]
    publications=[json.loads(s) for s in (root/'assignment-audit.jsonl').read_text().splitlines()
                  if s and json.loads(s).get('event')=='reference-answers-published']
    if publications != manifest.get('published_disclosures') or len(publications)!=len(closed):
        raise ValueError('closed dataset disclosure identity mismatch')
    if not closed <= set(lock['dataset_order']):
        raise ValueError('unregistered closed dataset')
    for dataset in closed:
        required={u['unit_id'] for u in lock['expanded_units'] if u['dataset']==dataset}
        rows=[r for r in calibration if r['dataset']==dataset]
        if not required or not required<=done or len(rows)!=9 or {r['verifier_id'] for r in rows}!={f'v{i}' for i in range(9)}:
            raise ValueError('disclosed dataset lacks complete frozen cells or calibration')
