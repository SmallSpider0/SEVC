from __future__ import annotations

import torch

from sevc.models import build_model
from sevc.training import WorkerBehavior, produce_worker_update, verify_replay_proof
from sevc.verification.replay_coupled_probes import (
    ATOM_KEYS,
    assign_block_roles,
    compile_replay_task,
    mutate_replay_proof,
    state_complete_proof_sha256,
    validate_public_schema,
)


def _proof():
    torch.manual_seed(17)
    factory = lambda: build_model("small-mlp", class_count=10)
    batches = tuple(
        (torch.randn(2, 1, 28, 28), torch.tensor([index, index + 1]))
        for index in range(4)
    )
    update = produce_worker_update(
        factory(), factory, batches, WorkerBehavior.NORMAL, device="cpu",
        learning_rate=0.01, momentum=0.9, local_epochs=1, max_batches=4,
        capture_replay=True,
    )
    assert update.replay_proof is not None
    return update.replay_proof, factory


def test_block_roles_are_exact_and_probe_bases_replace_production() -> None:
    source_ids = [f"{index:064x}" for index in range(40)]
    roles = assign_block_roles(source_ids, post_commit_seed=999)
    assert sum(value[0] == "production" for value in roles.values()) == 32
    assert sum(value[0] == "control" for value in roles.values()) == 4
    challenges = [value[1] for value in roles.values() if value[0] == "challenge"]
    assert sorted(challenges) == sorted(ATOM_KEYS)


def test_registered_mutation_is_recomputable_and_breaks_full_replay() -> None:
    proof, factory = _proof()
    mutated, detail = mutate_replay_proof(
        proof, source_id="a" * 64, post_commit_seed=7,
        atom_key="cp3-positive", magnitude=4e-5,
    )
    assert detail["checkpoint_index_zero_based"] == 2
    assert detail["observed_delta"] > 1e-5
    assert not verify_replay_proof(mutated, factory, device="cpu", tolerance=1e-5)["passed"]
    assert state_complete_proof_sha256(mutated) != state_complete_proof_sha256(proof)


def test_public_task_has_one_schema_without_role_or_truth() -> None:
    proof, factory = _proof()
    source_hash = state_complete_proof_sha256(proof)
    task = compile_replay_task(
        proof, factory(), "small-mlp", source_id="b" * 64,
        source_commitment=source_hash, post_commit_seed=8, role="control",
        atom_key=None, permutation_seed=9, replay_tolerance=1e-5,
        tamper_delta=4e-5,
    )
    public = task.public.to_dict()
    audit = validate_public_schema([public])
    assert audit["schema_violation_count"] == 0
    assert audit["role_field_violation_count"] == 0
    assert task.sealed.role == "control"
    assert task.certificate.keys() >= {
        "provenance", "state_reference", "action_separation",
        "recognizability", "attack_coupling",
    }
