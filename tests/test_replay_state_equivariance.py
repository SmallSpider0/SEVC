from __future__ import annotations

from io import BytesIO

import torch

from sevc.models import build_model
from sevc.training import ReplayProof, WorkerBehavior, produce_worker_update, verify_replay_proof
from sevc.training.replay_state import (
    build_replay_state_permutation,
    inverse_replay_state_permutation,
    replay_state_inventory,
    transform_optimizer_momentum,
    transform_replay_state,
)
from sevc.verification.replay_coupled_probes import wrap_replay_proof


def _proof() -> ReplayProof:
    torch.manual_seed(11)
    factory = lambda: build_model("small-mlp", class_count=10)
    batches = tuple(
        (torch.randn(2, 1, 28, 28), torch.tensor([index, index + 1]))
        for index in range(4)
    )
    update = produce_worker_update(
        factory(),
        factory,
        batches,
        WorkerBehavior.NORMAL,
        device="cpu",
        learning_rate=0.01,
        momentum=0.9,
        local_epochs=1,
        max_batches=4,
        capture_replay=True,
    )
    assert update.replay_proof is not None
    return update.replay_proof


def test_state_complete_mlp_inverse_and_replay_equivariance() -> None:
    proof = _proof()
    factory = lambda: build_model("small-mlp", class_count=10)
    wrapped, plan = wrap_replay_proof(
        proof, factory(), "small-mlp", permutation_seed=2101
    )
    inverse = inverse_replay_state_permutation(plan)
    assert all(
        torch.equal(value, proof.initial_state[key])
        for key, value in transform_replay_state(wrapped.initial_state, inverse).items()
    )
    assert proof.optimizer_initial_state is not None
    assert wrapped.optimizer_initial_state is not None
    assert transform_optimizer_momentum(wrapped.optimizer_initial_state, inverse) == {}
    result = verify_replay_proof(wrapped, factory, device="cpu", tolerance=1e-6)
    assert result["passed"]
    assert result["state_complete"]


def test_resnet_inventory_covers_parameters_buffers_and_momentum_shapes() -> None:
    model = build_model("resnet18", class_count=10)
    plan = build_replay_state_permutation(model, "resnet18", seed=2101)
    inventory = replay_state_inventory(model, plan)
    assert inventory["parameter_key_count"] == 62
    assert inventory["buffer_key_count"] == 60
    assert inventory["unsupported_layer_count"] == 0
    assert not inventory["unmapped_parameter_keys"]
    assert not inventory["unmapped_buffer_keys"]


def test_legacy_proof_remains_valid_but_missing_v2_state_fails_closed() -> None:
    proof = _proof()
    factory = lambda: build_model("small-mlp", class_count=10)
    legacy = ReplayProof(
        initial_state=proof.initial_state,
        batches=proof.batches,
        checkpoints=proof.checkpoints,
        learning_rate=proof.learning_rate,
        momentum=proof.momentum,
    )
    assert verify_replay_proof(legacy, factory, device="cpu", tolerance=1e-6)["passed"]
    missing = ReplayProof(
        initial_state=proof.initial_state,
        batches=proof.batches,
        checkpoints=proof.checkpoints,
        learning_rate=proof.learning_rate,
        momentum=proof.momentum,
        schema_version="sevc-state-complete-replay-proof-v2",
    )
    assert not verify_replay_proof(missing, factory, device="cpu", tolerance=1e-6)["passed"]


def test_state_complete_replay_proof_serialization_round_trip() -> None:
    proof = _proof()
    handle = BytesIO()
    torch.save(proof, handle)
    handle.seek(0)
    restored = torch.load(handle, weights_only=False)
    assert restored.schema_version == proof.schema_version
    assert restored.data_order_sha256 == proof.data_order_sha256
    assert restored.optimizer_initial_state == proof.optimizer_initial_state
    assert len(restored.optimizer_checkpoints) == 4
