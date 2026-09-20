"""Native DePoL report production using the canonical training engine.

The local adaptation uses CPU estimator replicas; no heterogeneous GPU claim.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import random
from pathlib import Path
import numpy as np
import torch
from sevc.training import produce_worker_update, WorkerBehavior
from sevc.verification.reference_acquisition import commitment
from sevc.committee.depol_arbitration import NativeCommitments, arbitrate_digests, arbitrate_distances

KEY = 'depol-local-verification-arbitration-v1'


@dataclass(frozen=True)
class NativeReplayInput:
    initial_state: dict
    batches: tuple
    momentum_state: dict
    learning_rate: float
    momentum: float


def replay_input(proof):
    # No expected checkpoints, owner labels, transcript, or other party's output.
    return NativeReplayInput(proof.initial_state, proof.batches,
                             proof.optimizer_initial_state or {}, proof.learning_rate, proof.momentum)


def interval_input(proof, step):
    """One submitted checkpoint interval; expected endpoint is not in actor view."""
    if step is None:
        return replay_input(proof), proof.checkpoints[-1]
    if len(proof.batches) != len(proof.checkpoints) or len(proof.optimizer_checkpoints) != len(proof.checkpoints):
        raise ValueError('per-step checkpoint/optimizer alignment required')
    return NativeReplayInput(
        proof.initial_state if step == 0 else proof.checkpoints[step-1],
        (proof.batches[step],),
        (proof.optimizer_initial_state or {}) if step == 0 else proof.optimizer_checkpoints[step-1],
        proof.learning_rate, proof.momentum), proof.checkpoints[step]


def replay_native(view, factory, device, *, prefix=False):
    model = factory().to(device)
    model.load_state_dict(view.initial_state)
    result = produce_worker_update(model, factory, view.batches[:1] if prefix else view.batches,
        WorkerBehavior.NORMAL, device=device, learning_rate=view.learning_rate,
        momentum=view.momentum, initial_momentum_state=view.momentum_state)
    return {k: v.detach().cpu().clone() for k,v in result.model.state_dict().items()}


def state_distance(left, right):
    if set(left) != set(right) or any(left[k].shape != right[k].shape for k in left):
        raise ValueError('native checkpoint schema mismatch')
    return math.sqrt(sum(float(torch.sum((left[k].double()-right[k].double())**2)) for k in sorted(left)))


def lsh_digest(state, spec, seed):
    """Euclidean p-stable LSH: independent Gaussian projections + uniform offsets."""
    rng = np.random.default_rng(seed)
    projections = np.zeros(spec['hashes'])
    for key in sorted(state):
        flat = state[key].detach().cpu().numpy().reshape(-1)
        for start in range(0, len(flat), 65536):
            chunk = flat[start:start+65536].astype(np.float64)
            projections += rng.standard_normal((spec['hashes'],len(chunk))) @ chunk
    offsets = rng.uniform(0, spec['width'], spec['hashes'])
    return np.floor((projections+offsets)/spec['width']).astype(np.int64).tolist()


def execute_native_group(*, sources, estimator, factory, device, spec, seed, behavior, folder, clock):
    """Actual native calculations, with all report payloads hidden until commitment.

Each source is a checkpoint interval. All 32 production intervals are sampled,
so compared production coverage is exact; no RCMP probes are delivered to DP.
"""
    granularity = spec.get('interval_granularity', 'segment-endpoint')
    if granularity not in ('segment-endpoint', 'per-step'):
        raise ValueError('unknown native interval granularity')
    per_step = granularity == 'per-step'
    if per_step and behavior not in ('honest','constant-accept','constant-reject','uniform-k32','uniform-k39'):
        raise ValueError('unregistered per-step native action')
    intervals = [(i, step) for i, source in enumerate(sources)
                 for step in (range(len(source.proof.checkpoints)) if per_step else (None,))]
    folder=Path(folder); folder.mkdir()
    estimator_proof=estimator.proof
    estimates=[]; estimator_values=[]
    for worker in range(spec['estimator_workers']):
        errors=[]
        for i in range(spec['estimator_replicas']):
            for step in (range(len(estimator_proof.checkpoints)) if per_step else (None,)):
                view, target = interval_input(estimator_proof, step)
                state,_=clock.call('depol-estimator-replay','owner',replay_native,view,factory,device)
                error,_=clock.call('depol-estimator-distance','owner',state_distance,state,target)
                errors.append(error)
                del state, view, target
        estimates.append(errors)
        estimator_values.append(float(np.max(errors)+np.std(errors)))
    estimator_ledger=NativeCommitments([f'e{i}' for i in range(len(estimator_values))])
    for i,value in enumerate(estimator_values):estimator_ledger.commit(f'e{i}',commitment([f'e{i}',value]))
    for i,value in enumerate(estimator_values):estimator_ledger.reveal(f'e{i}',value)
    err=float(np.median(estimator_values))
    epsilon=spec['epsilon_floor']+spec['epsilon_multiplier']*err
    lsh={**spec['lsh'],'width':max(spec['lsh']['minimum_width'],spec['lsh']['epsilon_width_multiplier']*epsilon)}
    ids=list(range(len(intervals))); random.Random(seed).shuffle(ids)
    quotas = {'uniform-k32': len(ids)*25//32, 'uniform-k39': len(ids)*31//32} if per_step else {'uniform-k32':25,'uniform-k39':31}
    if per_step and len(ids) != 128:
        raise ValueError('per-step native requires 32 four-step sources')
    selected=set(random.Random(seed+1).sample(ids, quotas.get(behavior,len(ids))))
    workers=[f'worker{i}' for i in range(4)];random.Random(seed+2).shuffle(workers)
    trainer_report=[]; reports=[[],[],[]]; paths=[[],[],[],[]]; transcript=[]
    # Trainer commits the full checkpoint vector before group sampling is disclosed.
    for i,(source_index, step) in enumerate(intervals):
        proof = sources[source_index].proof
        _, state = interval_input(proof, step)
        value,_=clock.call('depol-trainer-lsh','trainer',lsh_digest,state,lsh,seed+i+11)
        trainer_report.append(value)
        del proof, state
    trainer_commitment=commitment(['trainer',trainer_report])
    transcript.append({'phase':'GROUPING','trainer':workers[0],'verifiers':workers[1:],'randomness':'private local fixture; distributed MPRNG not deployed'})
    transcript.append({'phase':'TRAINER_COMMITTED_BEFORE_SAMPLING','commitment':trainer_commitment})
    transcript.append({'phase':'GROUP_SAMPLING','sampled_interval_indices':ids,'coverage':len(ids)})
    for i in ids:
        source_index, step = intervals[i]
        proof=sources[source_index].proof
        view, target = interval_input(proof, step)
        path=folder/f'trainer-{i}.pt';torch.save(target,path);paths[0].append(str(path))
        for v in range(3):
            action=behavior if v==0 else 'honest'
            if action in ('constant-accept','constant-reject') or action.startswith('uniform-k') and i not in selected:
                state={k:(torch.zeros_like(x) if action=='constant-reject' else x.clone()) for k,x in view.initial_state.items()}
                executed=0
            else:
                state,_=clock.call('depol-native-recompute','verifier',replay_native,view,factory,device,
                                   prefix=action=='prefix-one-step-shortcut')
                executed=1 if action=='prefix-one-step-shortcut' else len(view.batches)
            value,_=clock.call('depol-verifier-lsh','verifier',lsh_digest,state,lsh,seed+i+11)
            reports[v].append(value)
            path=folder/f'v{v}-{i}.pt';clock.call('depol-checkpoint-store','verifier',torch.save,state,path);paths[v+1].append(str(path))
            transcript.append({'phase':'NATIVE_WORK','party':f'v{v}','interval':i,'training_steps':executed,
                               'input_fields':list(NativeReplayInput.__dataclass_fields__), 'action':action})
            del state
        del proof,view,target
    ordered_trainer=[trainer_report[i] for i in ids]
    # The initially committed vector is checked in original source order, then sampled order.
    ledger=NativeCommitments(['trainer','v0','v1','v2'])
    ledger.commit('trainer',trainer_commitment)
    for v in range(3):ledger.commit(f'v{v}',commitment([f'v{v}',reports[v]]))
    ledger.reveal('trainer',trainer_report)
    for v in range(3):ledger.reveal(f'v{v}',reports[v])
    fast=arbitrate_digests(ordered_trainer,reports)
    slow=None;slow_ledger=None
    # Any dissenting/misjudged party can request cross verification, never skipped.
    if not fast['trainer_verdict'] or not all(fast['verifier_reward_eligible']):
        def distances():
            matrices=[[] for _ in range(3)]
            for n in range(len(ids)):
                states=[torch.load(p[n],map_location='cpu',weights_only=False) for p in paths]
                for v in range(3):matrices[v].append([state_distance(states[v+1],other) for other in states])
            return matrices
        matrices,slow_seconds=clock.call('depol-slow-path-distance','verifier',distances)
        slow_ledger=NativeCommitments(['v0','v1','v2'])
        for v in range(3):slow_ledger.commit(f'v{v}',commitment([f'v{v}',matrices[v]]))
        for v in range(3):slow_ledger.reveal(f'v{v}',matrices[v])
        slow=arbitrate_distances(matrices,epsilon)
        slow.update(distance_matrices=matrices,actual_seconds=slow_seconds)
    result = {'label':'DePoL verification/arbitration subprotocol (local adaptation)',
        'behavior':behavior,'native_action_mapping':{'constant-accept':'fabricate initial checkpoint','constant-reject':'fabricate zero checkpoint','uniform-k32':'25/32 actual intervals; unchecked initial checkpoint','uniform-k39':'31/32 actual intervals; unchecked initial checkpoint','prefix-one-step-shortcut':'one-step native checkpoint; no final checkpoint input'},
        'estimator_errors':estimates,'estimator_values':estimator_values,'estimator_commitment_events':estimator_ledger.events,'epsilon':epsilon,'lsh':lsh,
        'native_commitment_events':ledger.events,'protocol_events':transcript,
        'slow_commitment_events':None if slow_ledger is None else slow_ledger.events,
        'fast':fast,'slow':slow,'final':slow or fast,'production_count':len(sources),
        'numeric_payment':None,'numeric_payment_status':'NOT_DEFINED','communication_cost':'NOT_MEASURED',
        'heterogeneous_gpu':'NOT_VALIDATED','independent_block_count':0}

    if per_step:
        result.update(interval_granularity=granularity, interval_count=len(intervals),
            estimator_interval_count=len(estimator_proof.checkpoints),
            interval_mapping=[{'interval': i, 'source_index': si, 'checkpoint_index': step,
                'input_checkpoint_index': step-1, 'optimizer_input_checkpoint_index': step-1,
                'source_proof_sha256': sources[si].hashes['proof_sha256']}
                for i, (si, step) in enumerate(intervals)],
            deviator_selected_intervals=sorted(selected),
            native_action_mapping={'constant-accept':'fabricate initial checkpoint',
                'constant-reject':'fabricate zero checkpoint',
                'uniform-k32':'100/128 actual intervals; unchecked initial checkpoint',
                'uniform-k39':'124/128 actual intervals; unchecked initial checkpoint',
                'prefix-one-step-shortcut':'NOT_APPLICABLE: per-step interval equals honest replay'})
    return result
