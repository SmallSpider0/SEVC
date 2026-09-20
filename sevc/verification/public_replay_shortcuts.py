"""Ordinary public-view diagnostics and opaque owner binding.

These checks do not receive a trainer cache, task roles or reference answers.
They are attack modules, not substitutes for the canonical replay verifier.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import hmac

from sevc.core.artifacts import canonical_json_text


def opaque_source_handle(secret: str, job_id: str, source_commitment: str) -> str:
    message = canonical_json_text(["opaque-source-binding-v1", job_id, source_commitment])
    return hmac.new(secret.encode(), message.encode(), hashlib.sha256).hexdigest()


def gradient_continuation_challenge(proof, model, *, checkpoint_index, scale, device="cpu"):
    """Build an invalid trajectory using the sole trainer, retaining public batches.

    This candidate is restricted to the deterministic MLP/ResNet fixture models.
    Prefix tensors are borrowed read-only; the trainer owns the new suffix.
    """
    import copy
    from sevc.training import produce_worker_update, WorkerBehavior
    if not 0 <= checkpoint_index < len(proof.checkpoints) or not 0 < scale < 1:
        raise ValueError("invalid continuation challenge parameters")
    initial = proof.initial_state if checkpoint_index == 0 else proof.checkpoints[checkpoint_index - 1]
    momentum = proof.optimizer_initial_state if checkpoint_index == 0 else proof.optimizer_checkpoints[checkpoint_index - 1]
    seed_model = copy.deepcopy(model).cpu()
    seed_model.load_state_dict(initial, strict=True)
    update = produce_worker_update(seed_model, lambda: copy.deepcopy(seed_model),
        proof.batches[checkpoint_index:], WorkerBehavior.NORMAL, device=device,
        learning_rate=proof.learning_rate, momentum=proof.momentum,
        capture_replay=True, initial_momentum_state=momentum,
        gradient_scale_by_step={0: scale})
    suffix = update.replay_proof
    if suffix is None:
        raise RuntimeError("shared trainer did not capture continuation")
    return replace(proof,
        checkpoints=proof.checkpoints[:checkpoint_index] + suffix.checkpoints,
        optimizer_checkpoints=proof.optimizer_checkpoints[:checkpoint_index] + suffix.optimizer_checkpoints), {
            "profile": "gradient-continuation-v3", "checkpoint_index_zero_based": checkpoint_index,
            "gradient_scale": scale, "owner_training_steps": len(suffix.checkpoints)}


def sgd_public_consistency(proof, *, tolerance: float = 1e-5) -> dict:
    """Check public SGD update identities, without forward/backward calls."""
    import torch
    previous = proof.initial_state
    worst = 0.0
    checked = 0
    if len(proof.optimizer_checkpoints) != len(proof.checkpoints):
        return {"passed": False, "max_residual": None, "checked_tensors": 0}
    with torch.no_grad():
        for checkpoint, momentum in zip(proof.checkpoints, proof.optimizer_checkpoints):
            for key, buffer in momentum.items():
                # BatchNorm buffers are not SGD parameters and have no momentum.
                expected = previous[key].add(buffer, alpha=-proof.learning_rate)
                residual = float((checkpoint[key] - expected).abs().max())
                if not torch.isfinite(torch.tensor(residual)):
                    return {"passed": False, "max_residual": None, "checked_tensors": checked}
                worst = max(worst, residual)
                checked += 1
            previous = checkpoint
    return {"passed": checked > 0 and worst <= tolerance,
            "max_residual": worst, "checked_tensors": checked}


def make_sgd_coherent_forgery(proof, mutation):
    """Repair the public update identity of a trainer's last-checkpoint forgery.

    This is an independently forged input, not information shared with an actor.
    The shared replay verifier must still reject its model/gradient mismatch.
    """
    index = mutation["checkpoint_index_zero_based"]
    if index != len(proof.checkpoints) - 1:
        raise ValueError("this registered trainer forgery only changes the final checkpoint")
    return preserve_sgd_update_identity(proof, mutation)


def preserve_sgd_update_identity(proof, mutation):
    """Maintain the affected SGD identities after one checkpoint coordinate changes."""
    index = mutation["checkpoint_index_zero_based"]
    key = mutation["tensor_key"]
    if key not in proof.optimizer_checkpoints[index]:
        return proof
    previous = proof.initial_state if index == 0 else proof.checkpoints[index - 1]
    states = list(proof.optimizer_checkpoints)
    states[index] = dict(states[index])
    # Change only the affected coordinate, preserving all other public gradients.
    buffer = states[index][key].clone()
    coordinate = mutation["flat_coordinate"]
    buffer.reshape(-1)[coordinate] = (
        previous[key].reshape(-1)[coordinate] - proof.checkpoints[index][key].reshape(-1)[coordinate]
    ) / proof.learning_rate
    states[index][key] = buffer
    if index + 1 < len(states):
        states[index + 1] = dict(states[index + 1])
        following = states[index + 1][key].clone()
        following.reshape(-1)[coordinate] = (
            proof.checkpoints[index][key].reshape(-1)[coordinate]
            - proof.checkpoints[index + 1][key].reshape(-1)[coordinate]
        ) / proof.learning_rate
        states[index + 1][key] = following
    return replace(proof, optimizer_checkpoints=tuple(states))


def softmax_bias_gradient_consistency(proof, *, tolerance=1e-5):
    """Public cross-entropy output-bias gradient sum; no model execution."""
    keys = [k for k in ("network.5.bias", "fc.bias") if k in proof.initial_state]
    import torch
    if len(keys) != 1 or proof.criterion_key != "torch.nn.CrossEntropyLoss":
        return {"passed": True, "applicable": False, "max_residual": None}
    key = keys[0]
    previous = proof.optimizer_initial_state.get(key)
    worst = 0.0
    for checkpoint in proof.optimizer_checkpoints:
        current = checkpoint[key]
        gradient = current if previous is None else current - proof.momentum * previous
        worst = max(worst, abs(float(gradient.to(torch.float64).sum())))
        previous = current
    return {"passed": worst <= tolerance, "applicable": True, "max_residual": worst}


def low_rank_gradient_consistency(proof, *, tolerance=1e-5):
    """Necessary rank bound for public minibatch gradients of linear layers."""
    import torch
    previous = proof.optimizer_initial_state
    worst, checked = 0.0, 0
    projections = {}
    for step, current in enumerate(proof.optimizer_checkpoints):
        batch_size = int(proof.batches[step][0].shape[0])
        size = batch_size + 2
        for key, buffer in current.items():
            if buffer.ndim != 2 or min(buffer.shape) <= batch_size:
                continue
            if (key, size) not in projections:
                generator = torch.Generator().manual_seed(910)
                left = torch.randint(0, 2, (size, buffer.shape[0]), generator=generator).float() * 2 - 1
                right = torch.randint(0, 2, (buffer.shape[1], size), generator=generator).float() * 2 - 1
                projections[key, size] = left, right
            left, right = projections[key, size]
            gradient = buffer if key not in previous else buffer - proof.momentum * previous[key]
            singular = torch.linalg.svdvals(left @ gradient.float() @ right)
            worst = max(worst, float(singular[batch_size]))
            checked += 1
        previous = current
    return {"passed": worst <= tolerance, "checked_tensors": checked, "max_residual": worst}
