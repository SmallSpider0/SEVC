"""Resident component timing assembled from the shared source/compiler/replayer."""
from __future__ import annotations

import gc
import json
import statistics
import time
from pathlib import Path

from sevc.core.artifacts import sha256_file, write_json
from sevc.core.role_accounting import RoleClock
from sevc.verification.on_demand_service import build_source_bank, replay_owner_proof
from sevc.verification.paid_replay_service import VERSION, WRAPPER_DOMAIN
from sevc.verification.replay_coupled_probes import (
    ATOM_KEYS, compile_canonical_replay_task, proof_component_hashes,
)

CHANGE = 'experiment-tdsc-probe-compute-tiny-v1'


def validate_activation(config, profile, output_root):
    lock = json.loads(Path(config['protocol_lock_path']).read_text())
    expected = json.loads(Path(lock['config_path']).read_text())
    actual = {k: v for k, v in config.items() if not k.startswith('_runtime')}
    if (actual != expected or sha256_file(Path(lock['config_path'])) != lock['config_sha256']
            or not lock['ready'] or 'Ready: YES' not in Path(config['review_path']).read_text()
            or config['change_id'] != CHANGE or profile['formal']
            or profile['device'] != 'cpu' or not profile['equivalence_only']
            or config['formal_execution_authorized']
            or set(config['datasets']) != {'mnist', 'cifar10', 'cifar100'}
            or str(output_root) != lock['output_root']):
        raise PermissionError('resident tiny frozen identity mismatch')
    if output_root.exists():
        raise FileExistsError('resident tiny preserves existing output')
    for row in lock['source_files'] + lock['input_files']:
        if sha256_file(Path(row['path'])) != row['sha256']:
            raise PermissionError('resident tiny source/input hash drift')


def summarize(rows, spec):
    """Ratio of role-weighted source medians; repetitions are not new sources."""
    measured = [r for r in rows if not r['warmup']]
    sources = []
    for index in range(spec['source_count']):
        group = [r for r in measured if r['source_index'] == index]
        if len(group) != spec['repeats'] or any(not r['valid'] for r in group):
            raise ValueError('incomplete or invalid resident measurements')
        keys = ('direct_seconds', 'construction_seconds', 'source_validation_seconds',
                'challenge_validation_seconds')
        medians = {k: statistics.median(r[k] for r in group) for k in keys}
        sources.append({'source_index': index, 'role': group[0]['role'], **medians})
    totals = {k: sum(s[k] for s in sources) for k in keys}
    probe = sum(totals[k] for k in keys if k != 'direct_seconds')
    same = probe / totals['direct_seconds']
    normalized = same * spec['normalization_probes'] / spec['normalization_production']
    return {'source_medians': sources, 'totals_seconds': totals,
            'probe_total_seconds': probe, 'same_count_probe_direct_ratio': same,
            'normalized_8_probe_32_replay_ratio': normalized,
            'normalized_saving_fraction': 1 - normalized,
            'normalized_32_replay_seconds': totals['direct_seconds'] * 4,
            'normalization_is_extrapolation': True, 'formal_scientific_evidence': False}


def measure_components(context, partition, outer_clock, performance, config, root):
    spec = config['probe_compute']
    folder = Path(root) / context.dataset
    folder.mkdir()
    rows, sources, pending = [], [], []
    clock = RoleClock(context.device.synchronize, pending.append)
    schema = context.factory().cpu()
    started = config.setdefault('_runtime_probe_started', time.monotonic())
    for index in range(spec['source_count']):
        if time.monotonic() - started > spec['wall_budget_seconds']:
            raise TimeoutError('frozen tiny resource boundary reached')
        clock.context = {'dataset': context.dataset, 'source_index': index,
                         'phase_scope': 'excluded-source-creation'}
        bank, recipes, _ = build_source_bank(context, seed=spec['seed'],
            namespace=CHANGE, partition=partition, steps=spec['steps'],
            batch_size=spec['batch_size'], invalid_count=0, count=1,
            source_offset=index, clock=clock, scratch_dir=None)
        source = bank[0]
        if source.path is not None:
            raise ValueError('resident measurement may not read source files')
        proof = source.proof
        sources.extend(recipes)
        role = 'control' if index < 4 else 'challenge'
        atom = None if role == 'control' else ATOM_KEYS[index - 4]
        for repetition in range(-spec['warmups'], spec['repeats']):
            clock.context = {'dataset': context.dataset, 'source_index': index,
                             'repetition': repetition, 'role_kind': role,
                             'phase_scope': 'warmup' if repetition < 0 else 'component-timing'}
            row = {'source_index': index, 'role': role, 'repetition': repetition,
                   'warmup': repetition < 0, 'source_sha256': source.hashes['proof_sha256']}

            def direct():
                verdict, seconds = replay_owner_proof(proof, context=context, clock=clock,
                    performance=performance, phase='direct_replay')
                row.update(direct_seconds=seconds, direct_verdict=verdict)

            def probe():
                verdict, seconds = replay_owner_proof(proof, context=context, clock=clock,
                    performance=performance, phase='probe_source_validation')
                if not verdict['passed']:
                    raise ValueError('accepting source could not be validated')
                bundle, compile_seconds = clock.call('probe_construction', 'owner',
                    compile_canonical_replay_task, proof, schema, context.build_key,
                    source_id=source.recipe['source_id'], source_commitment=source.hashes['proof_sha256'],
                    post_commit_seed=spec['seed'] + index, role=role, atom_key=atom,
                    permutation_seed=spec['seed'] + index, protocol_version=VERSION,
                    wrapper_seed_domain=WRAPPER_DOMAIN, tamper_delta=4e-5,
                    source_component_hashes=source.hashes, delivery_profile='compact',
                    identity_profile='fused', schema_validator_profile='shared',
                    validated_source_verdict=True, final_two_checkpoints=True,
                    mutation_profile='gradient-continuation-v3', wrapper_profile='identity',
                    challenge_device=context.device.name)
                validation_seconds, challenge = 0.0, None
                if atom is not None:
                    challenge, validation_seconds = replay_owner_proof(bundle.canonical_candidate,
                        context=context, clock=clock, performance=performance,
                        phase='probe_challenge_validation')
                    if challenge['passed']:
                        raise ValueError('registered challenge unexpectedly passed')
                row.update(construction_seconds=compile_seconds, source_validation_seconds=seconds,
                    challenge_validation_seconds=validation_seconds, source_verdict=verdict,
                    challenge_verdict=challenge, compile_details=bundle.compile_seconds,
                    candidate_sha256=bundle.component_hashes['wrapped']['proof_sha256'])

            order = (direct, probe) if (index + repetition) % 2 == 0 else (probe, direct)
            for operation in order:
                operation()
            row['valid'] = row['direct_verdict']['passed'] and row['source_verdict']['passed']
            rows.append(row)
        if proof_component_hashes(proof) != source.hashes:
            raise ValueError('resident source mutated during paired measurement')
        # All artifact writes and cleanup are outside the measured component calls.
        write_json(folder / 'measurements.json', rows)
        write_json(folder / 'sources.json', sources)
        for event in pending:
            outer_clock.emit(event)
        pending.clear()
        print(json.dumps({'resident_component_complete': context.dataset, 'source': index}), flush=True)
        del direct, probe, order, operation, proof, source, bank
        context.close_task_lanes()
        gc.collect()
    summary = summarize(rows, spec)
    import resource
    summary['process_peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    write_json(folder / 'summary.json', summary)
    print(json.dumps({'resident_dataset_complete': context.dataset,
                      'normalized_ratio': summary['normalized_8_probe_32_replay_ratio']}), flush=True)
