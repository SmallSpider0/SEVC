"""Hash-bound primary reuse for SE-COST; the canonical runner executes the rest."""
import hashlib
import json
from pathlib import Path
import shutil

from sevc.core.artifacts import sha256_file, write_json


def _path(root, name):
    path = root / name
    if Path(name).is_absolute() or '..' in Path(name).parts or path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError('unsafe recovery path')
    return path


def recovery_spec(config):
    lock = Path(config['protocol_lock_path'])
    if sha256_file(lock) != config['protocol_lock_sha256']:
        raise ValueError('recovery protocol lock drift')
    return json.loads(lock.read_text()).get('primary_recovery')


def validate_recovery(config, spec):
    """Validate before any reuse; missing evidence never falls back to rerun."""
    from sevc.experiments.tdsc_five_rq_evidence import scientific_configuration_sha256
    from sevc.evaluation.f_fixed_design import _inventory
    root = Path(spec['source_root'])
    manifest_path = Path(spec['manifest_path'])
    if sha256_file(manifest_path) != spec['manifest_sha256']:
        raise ValueError('recovery manifest drift')
    manifest = json.loads(manifest_path.read_text())
    primary = [u for u in config['fixed_design_units'] if u['phase'] == 'primary']
    if len(primary) != 192 or manifest['unit_ids'] != [u['unit_id'] for u in primary]:
        raise ValueError('primary recovery matrix mismatch')
    for name, entry in manifest['files'].items():
        path = _path(root, name)
        if path.stat().st_size != entry['bytes'] or sha256_file(path) != entry['sha256']:
            raise ValueError('recovery file drift: '+name)
    required = {'provenance.json','stream-commitments.json','expanded-units.json','replay-environment-lock.json','stage-seals/primary-all-datasets.json'}
    for unit in primary:
        required.add('units/'+unit['unit_id']+'.json')
        if unit['arm'].startswith('R-'):
            required.add('equivalence/'+unit['unit_id']+'.json')
    for dataset in config['datasets']:
        name = 'stage-seals/primary-'+dataset+'.json'; required.add(name)
        seal = json.loads(_path(root,name).read_text())
        if seal['phase'] != 'primary' or seal['dataset'] != dataset or seal['completed_units'] != 64:
            raise ValueError('incomplete primary seal')
        required.update(seal['files'])
        if any(manifest['files'].get(n, {}).get('sha256') != h for n,h in seal['files'].items()):
            raise ValueError('recovery seal hash mismatch')
    if not required <= manifest['files'].keys():
        raise ValueError('recovery manifest omits sealed evidence')
    all_seal=json.loads((root/'stage-seals/primary-all-datasets.json').read_text())
    if len(all_seal['seals']) != 3 or any(sha256_file(root/'stage-seals'/n) != h for n,h in all_seal['seals'].items()):
        raise ValueError('three-dataset seal mismatch')
    provenance = json.loads((root/'provenance.json').read_text())
    if (provenance['snapshot_id'] != spec['source_snapshot_id'] or
            scientific_configuration_sha256(provenance['configuration']) != scientific_configuration_sha256(config)):
        raise ValueError('recovery scientific/producer identity mismatch')
    _, rows, errors = _inventory(root, primary)
    if errors:raise ValueError('primary recovery inventory: '+str(errors[:3]))
    for name, entry in manifest['prefixes'].items():
        path = _path(root,name)
        with path.open('rb') as stream:
            data = stream.read(entry['bytes'])
        if len(data) != entry['bytes'] or hashlib.sha256(data).hexdigest() != entry['sha256']:
            raise ValueError('recovery log prefix drift: '+name)
    final_seal=json.loads((root/'stage-seals/primary-mnist.json').read_text())
    if manifest['prefixes'] != final_seal['shared_log_prefixes']:
        raise ValueError('recovery log cutoff differs from final primary seal')
    return root, manifest, rows


def import_primary(study):
    spec = recovery_spec(study.config)
    if spec is None:return False
    root, manifest, rows = validate_recovery(study.config, spec)
    if json.loads((study.root/'stream-commitments.json').read_text()) != json.loads((root/'stream-commitments.json').read_text()):
        raise ValueError('recovery private stream mismatch')
    capsule = study.root/'reused-primary'; capsule.mkdir()
    ledger = []
    for name, entry in manifest['files'].items():
        # Preserve old seals/provenance with their original producer; never relabel.
        dst = study.root/name if Path(name).parts[0] in {'units','reference-audits','equivalence'} else capsule/name
        dst.parent.mkdir(parents=True,exist_ok=True)
        with _path(root,name).open('rb') as src, dst.open('xb') as out:
            shutil.copyfileobj(src,out)
        if sha256_file(dst) != entry['sha256']:raise ValueError('recovery copy drift')
        ledger.append({'source':str(root/name),'destination':str(dst.relative_to(study.root)),**entry,'method':'verified-copy'})
    for name,entry in manifest['prefixes'].items():
        with (root/name).open('rb') as stream:data=stream.read(entry['bytes'])
        (capsule/name).write_bytes(data)
        if name == 'phase-timing.jsonl':
            # Seed the newly opened canonical log before its first event. Other old
            # lifecycle logs stay in the capsule, so cleanup never owns old paths.
            handle=study.clock.emit.handle
            if handle.tell()!=0:raise ValueError('phase log already started')
            handle.write(data.decode('utf-8'));handle.flush()
    for row in rows:study.results[row['unit_id']]=row
    write_json(study.root/'primary-reuse-ledger.json',{'source_snapshot_id':spec['source_snapshot_id'],
        'source_root':str(root),'manifest_sha256':spec['manifest_sha256'],'files':ledger,
        'prefixes':manifest['prefixes'],'reused_units':192,'recomputed_units':72,
        'excluded_unsealed_target_units':2,'process_windows_are_separate':True})
    return True
