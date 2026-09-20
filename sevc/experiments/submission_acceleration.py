"""Engineering-only scheduling assessment through the one experiment runner."""
import json,time
from pathlib import Path
from sevc.core.artifacts import write_json,sha256_file
from sevc.verification.reference_acquisition import commitment
from sevc.verification.paid_replay_service import verify_fixture_equivalence


def validate_engineering(config,profile,output_root):
    owner_cost = config.get('engineering_check') in {'owner-cost-repair-v1','f-readiness-repair-v1'}
    owner_repair = config.get('engineering_check') == 'owner-replay-repair-v1' or owner_cost
    expected_lanes = [1] if owner_repair else [1,4,8]
    if (profile.get('formal') is not False or not profile.get('equivalence_only') or
        config.get('formal_execution_authorized') or config.get('engineering_lanes')!=expected_lanes):
        raise PermissionError('engineering-only frozen profile required')
    if owner_repair and (set(config['datasets']) != {'mnist','cifar10','cifar100'} or
        (not owner_cost and any(profile['sample_ranges'][d] != [0,32] for d in config['datasets']))):
        raise PermissionError('owner repair requires all three engineering partitions')
    lock=json.loads(Path(config['engineering_lock_path']).read_text())
    if lock['scope']!='engineering_equivalence_only' or not lock['ready']:
        raise PermissionError('engineering review absent')
    projection={k:v for k,v in config.items() if not k.startswith('_runtime_')}
    if commitment(projection)!=lock['config_commitment'] or str(output_root)!=lock['output_root']:
        raise PermissionError('engineering identity drift')
    if owner_repair and not lock.get('input_files'):
        raise PermissionError('owner repair input hashes absent')
    for row in lock['source_files'] + lock.get('input_files', []):
        if sha256_file(Path(row['path']))!=row['sha256']:raise PermissionError('source identity drift')


def measure_equivalence(context,partition,clock,performance,config,root):
    if config.get('engineering_check') == 'f-readiness-repair-v1':
        from sevc.experiments.owner_cost_repair import measure_f_readiness_repair
        return measure_f_readiness_repair(context,partition,clock,performance,config,root)
    if config.get('engineering_check') in {'owner-cost-repair-v1','f-readiness-repair-v1'}:
        from sevc.experiments.owner_cost_repair import measure_owner_cost
        return measure_owner_cost(context, partition, clock, performance, config, root)
    if config.get('engineering_check') == 'owner-replay-repair-v1':
        return measure_owner_repair(context, partition, clock, performance, root)
    for width in config['engineering_lanes']:
        context.close_task_lanes()
        clock.context={'dataset':context.dataset,'engineering_lane_width':width}
        start=time.monotonic()
        result=verify_fixture_equivalence(context,partition,clock,
            {**performance,'task_lanes':width,'reuse_replay_model':True})
        context.device.synchronize()
        result.update(lanes=width,wall_seconds=time.monotonic()-start,
            new_candidate_blocks=0,scope='known engineering construction, not candidate scientific evidence')
        write_json(Path(root)/f'equivalence-{context.dataset}-lanes{width}.json',result)
        print(json.dumps({'engineering_complete':context.dataset,'lanes':width,'seconds':result['wall_seconds']}),flush=True)


def measure_owner_repair(context, partition, clock, performance, root):
    """Known constructions through canonical training/compile/replay interfaces.

    This tests backend equivalence, not a candidate block or owner/direct ratio.
    The legacy policy is an explicit call configuration, not a second verifier.
    """
    import torch
    from sevc.verification.paid_replay_service import source_recipe, VERSION, WRAPPER_DOMAIN
    from sevc.training.replay_sources import materialize_short_source
    from sevc.verification.replay_coupled_probes import (
        ATOM_KEYS, compile_canonical_replay_task, proof_component_hashes,
    )
    from sevc.verification.on_demand_service import replay_owner_proof

    recipe = source_recipe(context, 2026098000, 0, partition, 'engineering-equivalence')
    source = materialize_short_source(context, recipe)
    source_hashes = proof_component_hashes(source)
    from sevc.core.replay_environment import bind_source, require_source
    environment_id = getattr(context, 'replay_environment_id', None)
    source_binding = bind_source(source_hashes['proof_sha256'], environment_id)
    require_source({**source_hashes, **source_binding}, environment_id)
    folder = Path(root) / ('owner-repair-' + context.dataset)
    folder.mkdir(exist_ok=False)
    torch.save(source, folder / 'source.pt')
    result = {'dataset':context.dataset, 'device':context.device.name, 'recipe':recipe,
        'source_sha256':source_hashes['proof_sha256'], 'source_binding':source_binding,
        'source_file_sha256':sha256_file(folder/'source.pt'), 'checks':[],
        'new_candidate_blocks':0, 'scientifically_eligible':False, 'passed':False,
        'timing_scope':'single fixed-order engineering observation, not R/O or a speedup estimate'}
    write_json(folder/'equivalence.json', result)
    schema = context.factory().cpu()
    for atom in (None, *ATOM_KEYS, None):
        proof = source
        if atom is not None:
            bundle = compile_canonical_replay_task(source, schema, context.build_key,
                source_id=recipe['source_id'], source_commitment=source_hashes['proof_sha256'],
                post_commit_seed=19, role='challenge', atom_key=atom, permutation_seed=20,
                protocol_version=VERSION, wrapper_seed_domain=WRAPPER_DOMAIN, tamper_delta=4e-5,
                source_component_hashes=source_hashes, delivery_profile='compact',
                identity_profile='fused', schema_validator_profile='shared',
                validated_source_verdict=True, final_two_checkpoints=True,
                mutation_profile='gradient-continuation-v3', wrapper_profile='identity')
            proof = bundle.canonical_candidate
        before = proof_component_hashes(proof)
        phase = 'source_reference_replay' if atom is None else 'challenge_full_replay_validation'
        receipts = []
        policies = (
            ('legacy', {**performance, 'comparison_device':'cpu', 'reuse_replay_model':False}),
            ('repaired', performance),
        ) if environment_id is None else (
            ('server-first', performance), ('server-repeat', performance))
        for label, perf in policies:
            clock.context = {'dataset':context.dataset, 'phase_scope':'engineering',
                'variant':label, 'atom':atom, 'check_index':len(result['checks'])}
            verdict, seconds = replay_owner_proof(proof, context=context, clock=clock,
                performance=perf, phase=phase)
            receipts.append({'variant':label, 'verdict':verdict, 'seconds':seconds})
        intact = before == proof_component_hashes(proof)
        exact = receipts[0]['verdict'] == receipts[1]['verdict']
        known = all(r['verdict']['passed'] == (atom is None) for r in receipts)
        result['checks'].append({'atom':atom, 'proof_sha256':before['proof_sha256'],
            'receipts':receipts, 'proof_unchanged':intact, 'verdict_exact':exact,
            'known_construction_correct':known})
        write_json(folder/'equivalence.json', result)
        if not (intact and exact and known):
            raise ValueError('owner repair changed proof/verdict or failed known construction')
        if atom is not None:
            del bundle
    result['passed'] = True
    write_json(folder/'equivalence.json', result)
    print(json.dumps({'owner_repair_equivalence':context.dataset, 'passed':True,
                      'checks':len(result['checks'])}), flush=True)
