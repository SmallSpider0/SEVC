"""Reuse immutable shared trainer inputs, never verifier or owner answers."""
from sevc.verification.on_demand_service import build_source_bank


def derive_one_invalid_bank(context, *, valid_bank, valid_rows, valid_cache, **build_args):
    if len(valid_bank)!=40 or len(valid_rows)!=40:
        raise ValueError('source variant needs complete clean bank')
    if any(row.get('trainer_mutation') is not None for row in valid_rows):
        raise ValueError('donor bank is not the clean paired construction')
    for i,(source,row) in enumerate(zip(valid_bank,valid_rows)):
        if (row['source_index']!=i or source.recipe['source_index']!=i
                or row['proof_sha256']!=source.hashes['proof_sha256']):
            raise ValueError('donor source identity mismatch')
        for key in ('namespace','steps','batch_size'):
            if source.recipe[key]!=build_args[key]:raise ValueError('source group identity drift')
        if source.recipe['block_seed']!=build_args['seed'] or source.recipe['dataset']!=context.dataset:
            raise ValueError('source dataset/seed drift')
    changed,changed_rows,_=build_source_bank(context,invalid_count=1,count=1,source_offset=24,**build_args)
    from sevc.verification.replay_coupled_probes import tensor_state_sha256
    before=valid_bank[24].proof
    after=changed[0].proof
    def input_identity(proof):
        batches={str(i)+'-'+str(j):tensor for i,batch in enumerate(proof.batches) for j,tensor in enumerate(batch)}
        return (tensor_state_sha256([proof.initial_state,proof.optimizer_initial_state or {},batches]),
                proof.rng_state_sha256,proof.data_order_sha256,proof.learning_rate,proof.momentum)
    if input_identity(before)!=input_identity(after):
        raise ValueError('paired source initial state, RNG or input content drift')
    del before,after
    for key in ('source_id','source_index','source_seed','sample_indices','sample_labels','steps','batch_size','dataset','block_seed','namespace'):
        if changed_rows[0][key]!=valid_rows[24][key]:raise ValueError('paired source recipe drift:'+key)
    bank=list(valid_bank);rows=list(valid_rows);cache=dict(valid_cache)
    cache.pop(valid_bank[24].hashes['proof_sha256'])
    bank[24]=changed[0];rows[24]=changed_rows[0];cache[changed[0].hashes['proof_sha256']]=False
    return tuple(bank),rows,cache
