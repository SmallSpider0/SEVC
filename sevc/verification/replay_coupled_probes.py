"""Canonical compiler for replay-coupled C-TIV tasks and certificates."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import time
from typing import Any, Callable, Iterable, Mapping, Sequence

import torch
import torch.nn as nn

from sevc.training import ReplayProof
from sevc.training.replay_state import (
    ReplayStatePermutation,
    build_replay_state_permutation,
    inverse_replay_state_permutation,
    replay_state_permutation_from_descriptor,
    transform_optimizer_momentum,
    transform_replay_state,
)


PROTOCOL_VERSION = "tdsc-ctiv-replay-coupled-probes-v1"
TASK_SCHEMA = "sevc-rcmp-public-replay-task-v1"
T2_PROTOCOL_VERSION = "tdsc-ctiv-canonical-probe-certificate-v2"
T2_TASK_SCHEMA = "sevc-rcmp-public-replay-envelope-v2"
T2_COMPACT_TASK_SCHEMA = "sevc-rcmp-compact-public-replay-envelope-v1"
T2_CERTIFICATE_SCHEMA = "sevc-rcmp-five-field-certificate-v2"
WRAPPER_DESCRIPTOR_SCHEMA = "sevc-canonical-replay-wrapper-descriptor-v2"
ATOM_KEYS = (
    "cp3-positive",
    "cp3-negative",
    "cp4-positive",
    "cp4-negative",
)

_SCHEMA_VALIDATOR_CACHE: dict[str, nn.Module] = {}


def shared_schema_validator(
    model_key: str, factory: Callable[[], nn.Module]
) -> nn.Module:
    """Return one RNG-isolated model used only for static schema validation."""

    key = model_key.lower()
    if key not in {"small-mlp", "resnet18"}:
        raise ValueError(f"unsupported shared schema validator: {model_key}")
    validator = _SCHEMA_VALIDATOR_CACHE.get(key)
    if validator is None:
        with torch.random.fork_rng(devices=[]):
            validator = factory().cpu()
        validator.eval()
        validator.requires_grad_(False)
        _SCHEMA_VALIDATOR_CACHE[key] = validator
    return validator


def shared_schema_validator_cache_info() -> dict[str, object]:
    """Expose machine-checkable cache state without exposing model tensors."""

    return {
        "instance_count": len(_SCHEMA_VALIDATOR_CACHE),
        "keys": sorted(_SCHEMA_VALIDATOR_CACHE),
        "all_eval": all(not model.training for model in _SCHEMA_VALIDATOR_CACHE.values()),
        "all_grad_disabled": all(
            not parameter.requires_grad
            for model in _SCHEMA_VALIDATOR_CACHE.values()
            for parameter in model.parameters()
        ),
    }


def _clear_shared_schema_validator_cache_for_tests() -> None:
    """Reset process-local cache for focused isolation tests only."""

    _SCHEMA_VALIDATOR_CACHE.clear()


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def derive_int(*parts: object) -> int:
    encoded = "|".join(str(value) for value in parts)
    return int.from_bytes(hashlib.sha256(encoded.encode("utf-8")).digest()[:8], "big")


def tensor_state_sha256(states: Sequence[Mapping[str, torch.Tensor]]) -> str:
    digest = hashlib.sha256(b"sevc-rcmp-tensor-state-v1")
    for state_index, state in enumerate(states):
        digest.update(str(state_index).encode("utf-8"))
        for key in sorted(state):
            value = state[key].detach().cpu().contiguous()
            digest.update(key.encode("utf-8"))
            digest.update(str(value.dtype).encode("utf-8"))
            digest.update(str(tuple(value.shape)).encode("utf-8"))
            digest.update(memoryview(value.numpy()).cast("B"))
    return digest.hexdigest()


def state_complete_proof_sha256(proof: ReplayProof) -> str:
    digest = hashlib.sha256(b"sevc-state-complete-replay-proof-v2")
    digest.update(proof.schema_version.encode("utf-8"))
    digest.update(str(float(proof.learning_rate)).encode("utf-8"))
    digest.update(str(float(proof.momentum)).encode("utf-8"))
    digest.update((proof.rng_state_sha256 or "").encode("utf-8"))
    digest.update((proof.data_order_sha256 or "").encode("utf-8"))
    digest.update((proof.criterion_key or "").encode("utf-8"))
    digest.update(tensor_state_sha256((proof.initial_state, *proof.checkpoints)).encode("utf-8"))
    optimizer_states = (
        *((proof.optimizer_initial_state or {}),),
        *proof.optimizer_checkpoints,
    )
    digest.update(tensor_state_sha256(optimizer_states).encode("utf-8"))
    for data, target in proof.batches:
        for value in (data, target):
            tensor = value.detach().cpu().contiguous()
            digest.update(str(tensor.dtype).encode("utf-8"))
            digest.update(str(tuple(tensor.shape)).encode("utf-8"))
            digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


@dataclass
class _ProofComponentHashAccumulator:
    """Accumulate the public hash law while transform tensors are produced."""

    model_complete_digest: Any = field(
        default_factory=lambda: hashlib.sha256(b"sevc-rcmp-tensor-state-v1")
    )
    checkpoint_digest: Any = field(
        default_factory=lambda: hashlib.sha256(b"sevc-rcmp-tensor-state-v1")
    )
    optimizer_digest: Any = field(
        default_factory=lambda: hashlib.sha256(b"sevc-rcmp-tensor-state-v1")
    )
    include_checkpoint_digest: bool = True
    tensor_count: int = 0
    tensor_payload_bytes: int = 0
    proof_digest: Any | None = None

    @staticmethod
    def _tensor(value: torch.Tensor) -> torch.Tensor:
        return value.detach().cpu().contiguous()

    @staticmethod
    def _payload(value: torch.Tensor) -> memoryview:
        return memoryview(value.numpy()).cast("B")

    def start_model_state(self, state_index: int) -> None:
        self.model_complete_digest.update(str(state_index).encode("utf-8"))
        if state_index and self.include_checkpoint_digest:
            self.checkpoint_digest.update(str(state_index - 1).encode("utf-8"))

    def observe_model_tensor(
        self, state_index: int, key: str, value: torch.Tensor
    ) -> None:
        tensor = self._tensor(value)
        metadata = (
            key.encode("utf-8"),
            str(tensor.dtype).encode("utf-8"),
            str(tuple(tensor.shape)).encode("utf-8"),
        )
        for item in metadata:
            self.model_complete_digest.update(item)
            if state_index and self.include_checkpoint_digest:
                self.checkpoint_digest.update(item)
        payload = self._payload(tensor)
        self.model_complete_digest.update(payload)
        if state_index and self.include_checkpoint_digest:
            self.checkpoint_digest.update(payload)
        self.tensor_count += 1
        self.tensor_payload_bytes += int(tensor.numel()) * int(tensor.element_size())

    def start_optimizer_state(self, state_index: int) -> None:
        self.optimizer_digest.update(str(state_index).encode("utf-8"))

    def observe_optimizer_tensor(self, key: str, value: torch.Tensor) -> None:
        tensor = self._tensor(value)
        self.optimizer_digest.update(key.encode("utf-8"))
        self.optimizer_digest.update(str(tensor.dtype).encode("utf-8"))
        self.optimizer_digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        self.optimizer_digest.update(self._payload(tensor))
        self.tensor_count += 1
        self.tensor_payload_bytes += int(tensor.numel()) * int(tensor.element_size())

    def prepare_proof(self, proof: ReplayProof) -> None:
        digest = hashlib.sha256(b"sevc-state-complete-replay-proof-v2")
        digest.update(proof.schema_version.encode("utf-8"))
        digest.update(str(float(proof.learning_rate)).encode("utf-8"))
        digest.update(str(float(proof.momentum)).encode("utf-8"))
        digest.update((proof.rng_state_sha256 or "").encode("utf-8"))
        digest.update((proof.data_order_sha256 or "").encode("utf-8"))
        digest.update((proof.criterion_key or "").encode("utf-8"))
        digest.update(self.model_complete_digest.hexdigest().encode("utf-8"))
        digest.update(self.optimizer_digest.hexdigest().encode("utf-8"))
        self.proof_digest = digest

    def observe_batch_tensor(self, value: torch.Tensor) -> None:
        if self.proof_digest is None:
            raise RuntimeError("proof hash accumulator was not prepared")
        tensor = self._tensor(value)
        self.proof_digest.update(str(tensor.dtype).encode("utf-8"))
        self.proof_digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        self.proof_digest.update(self._payload(tensor))
        self.tensor_count += 1
        self.tensor_payload_bytes += int(tensor.numel()) * int(tensor.element_size())

    def finish(self) -> tuple[dict[str, str], tuple[int, int]]:
        if self.proof_digest is None:
            raise RuntimeError("proof hash accumulator was not prepared")
        return (
            {
                "proof_sha256": self.proof_digest.hexdigest(),
                **({"checkpoint_sha256": self.checkpoint_digest.hexdigest()}
                   if self.include_checkpoint_digest else {}),
                "optimizer_sha256": self.optimizer_digest.hexdigest(),
            },
            (self.tensor_count, self.tensor_payload_bytes),
        )


def _finish_proof_hash_accumulators(
    proof_and_accumulators: Sequence[tuple[ReplayProof, _ProofComponentHashAccumulator]],
) -> list[tuple[dict[str, str], tuple[int, int]]]:
    """Hash one shared batch stream for all proof identities in the pass."""

    if not proof_and_accumulators:
        return []
    reference_batches = proof_and_accumulators[0][0].batches
    if any(proof.batches is not reference_batches for proof, _ in proof_and_accumulators[1:]):
        raise ValueError("fused proof identities require one shared immutable batch tuple")
    for proof, accumulator in proof_and_accumulators:
        accumulator.prepare_proof(proof)
    for data, target in reference_batches:
        for value in (data, target):
            tensor = value.detach().cpu().contiguous()
            for _, accumulator in proof_and_accumulators:
                accumulator.observe_batch_tensor(tensor)
    return [accumulator.finish() for _, accumulator in proof_and_accumulators]


def proof_component_hashes(proof: ReplayProof) -> dict[str, str]:
    """Return the three public state-complete identities used by T1/T2."""

    hashes, _, _ = _profile_public_replay_proof(proof)
    return hashes


def _profile_public_replay_proof(
    proof: ReplayProof,
) -> tuple[dict[str, str], tuple[int, int], int]:
    """Hash and inventory one proof in one explicit tensor traversal.

    The compact delivery compiler calls this once for each newly materialized
    proof.  Returning the scan count makes the no-hidden-rescan contract
    machine-checkable without changing the public hash identities.
    """

    def update_state(
        digest: "hashlib._Hash",
        state_index: int,
        state: Mapping[str, torch.Tensor],
    ) -> None:
        digest.update(str(state_index).encode("utf-8"))
        for key in sorted(state):
            value = state[key].detach().cpu().contiguous()
            digest.update(key.encode("utf-8"))
            digest.update(str(value.dtype).encode("utf-8"))
            digest.update(str(tuple(value.shape)).encode("utf-8"))
            digest.update(memoryview(value.numpy()).cast("B"))

    model_complete_digest = hashlib.sha256(b"sevc-rcmp-tensor-state-v1")
    checkpoint_digest = hashlib.sha256(b"sevc-rcmp-tensor-state-v1")
    tensor_count = 0
    tensor_payload_bytes = 0
    for state_index, state in enumerate((proof.initial_state, *proof.checkpoints)):
        update_state(model_complete_digest, state_index, state)
        tensor_count += len(state)
        tensor_payload_bytes += sum(
            int(value.numel()) * int(value.element_size()) for value in state.values()
        )
        if state_index:
            update_state(checkpoint_digest, state_index - 1, state)
    optimizer_digest = hashlib.sha256(b"sevc-rcmp-tensor-state-v1")
    for state_index, state in enumerate(
        (proof.optimizer_initial_state or {}, *proof.optimizer_checkpoints)
    ):
        update_state(optimizer_digest, state_index, state)
        tensor_count += len(state)
        tensor_payload_bytes += sum(
            int(value.numel()) * int(value.element_size()) for value in state.values()
        )
    model_complete_sha256 = model_complete_digest.hexdigest()
    checkpoint_sha256 = checkpoint_digest.hexdigest()
    optimizer_sha256 = optimizer_digest.hexdigest()
    proof_digest = hashlib.sha256(b"sevc-state-complete-replay-proof-v2")
    proof_digest.update(proof.schema_version.encode("utf-8"))
    proof_digest.update(str(float(proof.learning_rate)).encode("utf-8"))
    proof_digest.update(str(float(proof.momentum)).encode("utf-8"))
    proof_digest.update((proof.rng_state_sha256 or "").encode("utf-8"))
    proof_digest.update((proof.data_order_sha256 or "").encode("utf-8"))
    proof_digest.update((proof.criterion_key or "").encode("utf-8"))
    proof_digest.update(model_complete_sha256.encode("utf-8"))
    proof_digest.update(optimizer_sha256.encode("utf-8"))
    for data, target in proof.batches:
        for value in (data, target):
            tensor = value.detach().cpu().contiguous()
            tensor_count += 1
            tensor_payload_bytes += int(tensor.numel()) * int(tensor.element_size())
            proof_digest.update(str(tensor.dtype).encode("utf-8"))
            proof_digest.update(str(tuple(tensor.shape)).encode("utf-8"))
            proof_digest.update(memoryview(tensor.numpy()).cast("B"))
    return (
        {
            "proof_sha256": proof_digest.hexdigest(),
            "checkpoint_sha256": checkpoint_sha256,
            "optimizer_sha256": optimizer_sha256,
        },
        (tensor_count, tensor_payload_bytes),
        1,
    )


def _transform_replay_proof_with_fused_hashes(
    proof: ReplayProof,
    plan: ReplayStatePermutation,
    *,
    include_input_identity: bool,
    transform_device: str | None = None,
    include_checkpoint_digest: bool = True,
) -> tuple[
    ReplayProof,
    dict[str, str] | None,
    dict[str, str],
    tuple[int, int],
    dict[str, float],
]:
    """Transform a proof and hash input/output tensors in the same loops."""

    if proof.optimizer_initial_state is None:
        raise ValueError("state-complete replay proof lacks optimizer start state")
    input_accumulator = _ProofComponentHashAccumulator() if include_input_identity else None
    output_accumulator = _ProofComponentHashAccumulator(include_checkpoint_digest=include_checkpoint_digest)

    def transform_model_state(
        state_index: int, state: Mapping[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        if input_accumulator is not None:
            input_accumulator.start_model_state(state_index)
        output_accumulator.start_model_state(state_index)

        def observe(key: str, before: torch.Tensor, after: torch.Tensor) -> None:
            if input_accumulator is not None:
                input_accumulator.observe_model_tensor(state_index, key, before)
            output_accumulator.observe_model_tensor(state_index, key, after)

        return transform_replay_state(state, plan, tensor_observer=observe, transform_device=transform_device)

    def transform_optimizer_state(
        state_index: int, state: Mapping[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        if input_accumulator is not None:
            input_accumulator.start_optimizer_state(state_index)
        output_accumulator.start_optimizer_state(state_index)

        def observe(key: str, before: torch.Tensor, after: torch.Tensor) -> None:
            if input_accumulator is not None:
                input_accumulator.observe_optimizer_tensor(key, before)
            output_accumulator.observe_optimizer_tensor(key, after)

        return transform_optimizer_momentum(state, plan, tensor_observer=observe, transform_device=transform_device)

    model_started = time.perf_counter()
    initial_state = transform_model_state(0, proof.initial_state)
    checkpoints = tuple(
        transform_model_state(state_index, state)
        for state_index, state in enumerate(proof.checkpoints, start=1)
    )
    model_seconds = time.perf_counter() - model_started
    batch_started = time.perf_counter()
    batches = proof.batches
    batch_seconds = time.perf_counter() - batch_started
    optimizer_started = time.perf_counter()
    optimizer_initial_state = transform_optimizer_state(
        0, proof.optimizer_initial_state
    )
    optimizer_checkpoints = tuple(
        transform_optimizer_state(state_index, state)
        for state_index, state in enumerate(proof.optimizer_checkpoints, start=1)
    )
    optimizer_seconds = time.perf_counter() - optimizer_started
    transformed = ReplayProof(
        initial_state=initial_state,
        batches=batches,
        checkpoints=checkpoints,
        learning_rate=proof.learning_rate,
        momentum=proof.momentum,
        schema_version=proof.schema_version,
        optimizer_initial_state=optimizer_initial_state,
        optimizer_checkpoints=optimizer_checkpoints,
        rng_state_sha256=proof.rng_state_sha256,
        data_order_sha256=proof.data_order_sha256,
        criterion_key=proof.criterion_key,
    )
    identity_started = time.perf_counter()
    accumulator_rows: list[tuple[ReplayProof, _ProofComponentHashAccumulator]] = []
    if input_accumulator is not None:
        accumulator_rows.append((proof, input_accumulator))
    accumulator_rows.append((transformed, output_accumulator))
    identities = _finish_proof_hash_accumulators(accumulator_rows)
    identity_seconds = time.perf_counter() - identity_started
    if input_accumulator is None:
        input_hashes = None
        output_hashes, output_metrics = identities[0]
    else:
        input_hashes = identities[0][0]
        output_hashes, output_metrics = identities[1]
    return transformed, input_hashes, output_hashes, output_metrics, {
        "model_transform": model_seconds,
        "optimizer_transform": optimizer_seconds,
        "batch_copy_or_share": batch_seconds,
        "fused_batch_identity": identity_seconds,
    }


def _copy_batches(
    batches: Sequence[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    return tuple(
        (data.detach().clone(), target.detach().clone()) for data, target in batches
    )


def clone_replay_proof(
    proof: ReplayProof,
    *,
    checkpoints: Sequence[Mapping[str, torch.Tensor]] | None = None,
    optimizer_checkpoints: Sequence[Mapping[str, torch.Tensor]] | None = None,
    batches: Sequence[tuple[torch.Tensor, torch.Tensor]] | None = None,
    data_order_sha256: str | None = None,
) -> ReplayProof:
    return ReplayProof(
        initial_state={key: value.detach().clone() for key, value in proof.initial_state.items()},
        batches=_copy_batches(proof.batches if batches is None else batches),
        checkpoints=tuple(
            {key: value.detach().clone() for key, value in state.items()}
            for state in (proof.checkpoints if checkpoints is None else checkpoints)
        ),
        learning_rate=proof.learning_rate,
        momentum=proof.momentum,
        schema_version=proof.schema_version,
        optimizer_initial_state=(
            None
            if proof.optimizer_initial_state is None
            else {
                key: value.detach().clone()
                for key, value in proof.optimizer_initial_state.items()
            }
        ),
        optimizer_checkpoints=tuple(
            {key: value.detach().clone() for key, value in state.items()}
            for state in (
                proof.optimizer_checkpoints
                if optimizer_checkpoints is None
                else optimizer_checkpoints
            )
        ),
        rng_state_sha256=proof.rng_state_sha256,
        data_order_sha256=(
            proof.data_order_sha256 if data_order_sha256 is None else data_order_sha256
        ),
        criterion_key=proof.criterion_key,
    )


def _wrap_replay_proof_profiled(
    proof: ReplayProof,
    model: nn.Module,
    model_key: str,
    *,
    permutation_seed: int,
) -> tuple[ReplayProof, ReplayStatePermutation, dict[str, float]]:
    if proof.schema_version != "sevc-state-complete-replay-proof-v2":
        raise ValueError("RCMP wrapper requires a state-complete replay proof")
    if proof.optimizer_initial_state is None:
        raise ValueError("state-complete replay proof lacks optimizer start state")
    plan_started = time.perf_counter()
    plan = build_replay_state_permutation(model, model_key, seed=permutation_seed)
    plan_seconds = time.perf_counter() - plan_started
    model_started = time.perf_counter()
    initial_state = transform_replay_state(proof.initial_state, plan)
    checkpoints = tuple(
        transform_replay_state(state, plan) for state in proof.checkpoints
    )
    model_seconds = time.perf_counter() - model_started
    batch_started = time.perf_counter()
    # Replay verification is read-only.  Keep the immutable batch tuple shared
    # instead of duplicating payload tensors for every delivered task.
    batches = tuple(proof.batches)
    batch_seconds = time.perf_counter() - batch_started
    optimizer_started = time.perf_counter()
    optimizer_initial_state = transform_optimizer_momentum(
        proof.optimizer_initial_state, plan
    )
    optimizer_checkpoints = tuple(
        transform_optimizer_momentum(state, plan)
        for state in proof.optimizer_checkpoints
    )
    optimizer_seconds = time.perf_counter() - optimizer_started
    wrapped = ReplayProof(
        initial_state=initial_state,
        batches=batches,
        checkpoints=checkpoints,
        learning_rate=proof.learning_rate,
        momentum=proof.momentum,
        schema_version=proof.schema_version,
        optimizer_initial_state=optimizer_initial_state,
        optimizer_checkpoints=optimizer_checkpoints,
        rng_state_sha256=proof.rng_state_sha256,
        data_order_sha256=proof.data_order_sha256,
        criterion_key=proof.criterion_key,
    )
    return wrapped, plan, {
        "permutation_plan": plan_seconds,
        "model_transform": model_seconds,
        "optimizer_transform": optimizer_seconds,
        "batch_copy_or_share": batch_seconds,
    }


def wrap_replay_proof(
    proof: ReplayProof,
    model: nn.Module,
    model_key: str,
    *,
    permutation_seed: int,
) -> tuple[ReplayProof, ReplayStatePermutation]:
    wrapped, plan, _ = _wrap_replay_proof_profiled(
        proof,
        model,
        model_key,
        permutation_seed=permutation_seed,
    )
    return wrapped, plan


def build_wrapper_descriptor(
    plan: ReplayStatePermutation,
    *,
    protocol_version: str,
    wrapper_seed_domain: str,
) -> dict[str, object]:
    """Build the role-neutral public descriptor authenticated by the verifier."""

    body: dict[str, object] = {
        "schema_version": WRAPPER_DESCRIPTOR_SCHEMA,
        "protocol_version": str(protocol_version),
        "wrapper_seed_domain": str(wrapper_seed_domain),
        "permutation": plan.to_dict(),
    }
    return {
        **body,
        "descriptor_commitment": sha256_text(canonical_json(body)),
    }


def verify_wrapper_descriptor(
    model: nn.Module,
    descriptor: Mapping[str, Any],
    *,
    protocol_version: str,
    wrapper_seed_domain: str,
) -> ReplayStatePermutation:
    """Authenticate a descriptor and reconstruct its forward permutation."""

    expected_fields = {
        "schema_version",
        "protocol_version",
        "wrapper_seed_domain",
        "permutation",
        "descriptor_commitment",
    }
    if set(descriptor) != expected_fields:
        raise ValueError("wrapper descriptor public schema mismatch")
    if descriptor["schema_version"] != WRAPPER_DESCRIPTOR_SCHEMA:
        raise ValueError("wrapper descriptor version mismatch")
    if descriptor["protocol_version"] != protocol_version:
        raise ValueError("wrapper descriptor protocol mismatch")
    if descriptor["wrapper_seed_domain"] != wrapper_seed_domain:
        raise ValueError("wrapper descriptor seed domain mismatch")
    body = {
        key: descriptor[key]
        for key in (
            "schema_version",
            "protocol_version",
            "wrapper_seed_domain",
            "permutation",
        )
    }
    if descriptor["descriptor_commitment"] != sha256_text(canonical_json(body)):
        raise ValueError("wrapper descriptor commitment mismatch")
    raw_permutation = descriptor["permutation"]
    if not isinstance(raw_permutation, Mapping):
        raise ValueError("wrapper descriptor permutation must be an object")
    return replay_state_permutation_from_descriptor(model, raw_permutation)


def canonicalize_replay_proof(
    proof: ReplayProof,
    model: nn.Module,
    descriptor: Mapping[str, Any],
    *,
    protocol_version: str,
    wrapper_seed_domain: str,
) -> tuple[ReplayProof, ReplayStatePermutation]:
    """Invert a delivered proof before invoking the shared replay verifier."""

    if proof.schema_version != "sevc-state-complete-replay-proof-v2":
        raise ValueError("canonical replay requires a state-complete proof")
    if proof.optimizer_initial_state is None:
        raise ValueError("canonical replay proof lacks optimizer start state")
    forward = verify_wrapper_descriptor(
        model,
        descriptor,
        protocol_version=protocol_version,
        wrapper_seed_domain=wrapper_seed_domain,
    )
    inverse = inverse_replay_state_permutation(forward)
    canonical = ReplayProof(
        initial_state=transform_replay_state(proof.initial_state, inverse),
        batches=tuple(proof.batches),
        checkpoints=tuple(
            transform_replay_state(state, inverse) for state in proof.checkpoints
        ),
        learning_rate=proof.learning_rate,
        momentum=proof.momentum,
        schema_version=proof.schema_version,
        optimizer_initial_state=transform_optimizer_momentum(
            proof.optimizer_initial_state, inverse
        ),
        optimizer_checkpoints=tuple(
            transform_optimizer_momentum(state, inverse)
            for state in proof.optimizer_checkpoints
        ),
        rng_state_sha256=proof.rng_state_sha256,
        data_order_sha256=proof.data_order_sha256,
        criterion_key=proof.criterion_key,
    )
    return canonical, forward


def canonicalize_replay_proof_with_identity(
    proof: ReplayProof,
    model: nn.Module,
    descriptor: Mapping[str, Any],
    *,
    protocol_version: str,
    wrapper_seed_domain: str,
    transform_device: str | None = None,
    include_checkpoint_digest: bool = True,
) -> tuple[ReplayProof, ReplayStatePermutation, dict[str, str], dict[str, float]]:
    """Canonicalize and bind output bytes without a post-transform proof scan."""

    if proof.schema_version != "sevc-state-complete-replay-proof-v2":
        raise ValueError("canonical replay requires a state-complete proof")
    forward = verify_wrapper_descriptor(
        model,
        descriptor,
        protocol_version=protocol_version,
        wrapper_seed_domain=wrapper_seed_domain,
    )
    inverse = inverse_replay_state_permutation(forward)
    canonical, _, hashes, _, timings = _transform_replay_proof_with_fused_hashes(
        proof,
        inverse,
        include_input_identity=False,
        transform_device=transform_device,
        include_checkpoint_digest=include_checkpoint_digest,
    )
    return canonical, forward, hashes, timings


def _atom_parts(atom_key: str) -> tuple[int, int]:
    if atom_key not in ATOM_KEYS:
        raise ValueError(f"unknown RCMP mutation atom: {atom_key}")
    checkpoint = 2 if atom_key.startswith("cp3") else 3
    sign = 1 if atom_key.endswith("positive") else -1
    return checkpoint, sign


def mutate_replay_proof(
    proof: ReplayProof,
    *,
    source_id: str,
    post_commit_seed: int,
    atom_key: str,
    magnitude: float,
    protocol_version: str = PROTOCOL_VERSION,
    final_two_checkpoints: bool = False,
) -> tuple[ReplayProof, dict[str, object]]:
    checkpoint_index, sign = _atom_parts(atom_key)
    if not math.isfinite(float(magnitude)) or float(magnitude) <= 0.0:
        raise ValueError("registered suffix mutation magnitude must be finite and positive")
    if final_two_checkpoints:
        if len(proof.checkpoints) < 2:
            raise ValueError("suffix mutation needs at least two checkpoints")
        checkpoint_index = len(proof.checkpoints) - 2 + (checkpoint_index - 2)
    elif len(proof.checkpoints) != 4:
        raise ValueError("registered suffix mutation requires four checkpoints")
    # Copy only the checkpoint mapping and tensor that carry the registered
    # mutation.  Every other replay component remains a read-only reference.
    checkpoints = list(proof.checkpoints)
    target = dict(checkpoints[checkpoint_index])
    floating_keys = sorted(
        key for key, value in target.items() if value.is_floating_point() and value.numel()
    )
    if not floating_keys:
        raise ValueError("registered suffix mutation found no floating state")
    selector = derive_int(
        protocol_version,
        "suffix-checkpoint-tamper",
        source_id,
        post_commit_seed,
        atom_key,
    )
    tensor_key = floating_keys[selector % len(floating_keys)]
    coordinate = (
        (selector // len(floating_keys)) % target[tensor_key].numel()
    )
    precision_safe_attempt_count = 1
    precision_safe_protocol = (
        protocol_version == "tdsc-ctiv-canonical-probe-certificate-v4"
    )
    if precision_safe_protocol:
        # A fixed 4e-5 perturbation can round to zero on large float32 state
        # values (notably ResNet BatchNorm running variances). Select a
        # coordinate prospectively from the same public state using only the
        # frozen PRF inputs, and require that dtype rounding retains at least
        # 75% of the requested magnitude. This changes neither the requested
        # delta nor any replay/gate threshold.
        selected: tuple[str, int] | None = None
        for attempt in range(4096):
            candidate_selector = derive_int(
                protocol_version,
                "suffix-checkpoint-tamper-precision-safe-coordinate",
                source_id,
                post_commit_seed,
                atom_key,
                attempt,
            )
            candidate_key = floating_keys[
                candidate_selector % len(floating_keys)
            ]
            candidate_tensor = target[candidate_key].detach().reshape(-1)
            candidate_coordinate = (
                candidate_selector // len(floating_keys)
            ) % candidate_tensor.numel()
            candidate_before = candidate_tensor[candidate_coordinate]
            candidate_after = candidate_before + torch.as_tensor(
                sign * float(magnitude), dtype=candidate_before.dtype
            )
            candidate_observed = float(
                candidate_after.to(torch.float64)
                - candidate_before.to(torch.float64)
            )
            if math.isfinite(candidate_observed) and abs(candidate_observed) >= (
                0.75 * float(magnitude)
            ):
                selected = (candidate_key, int(candidate_coordinate))
                precision_safe_attempt_count = attempt + 1
                break
        if selected is None:
            raise ValueError(
                "registered suffix mutation found no precision-safe coordinate"
            )
        tensor_key, coordinate = selected
    tensor = target[tensor_key].detach().clone()
    flat = tensor.reshape(-1)
    before = float(flat[coordinate].to(torch.float64))
    flat[coordinate] = flat[coordinate] + torch.as_tensor(
        sign * float(magnitude), dtype=flat.dtype
    )
    after = float(flat[coordinate].to(torch.float64))
    target[tensor_key] = tensor
    checkpoints[checkpoint_index] = target
    mutated = ReplayProof(
        initial_state=proof.initial_state,
        batches=proof.batches,
        checkpoints=tuple(checkpoints),
        learning_rate=proof.learning_rate,
        momentum=proof.momentum,
        schema_version=proof.schema_version,
        optimizer_initial_state=proof.optimizer_initial_state,
        optimizer_checkpoints=proof.optimizer_checkpoints,
        rng_state_sha256=proof.rng_state_sha256,
        data_order_sha256=proof.data_order_sha256,
        criterion_key=proof.criterion_key,
    )
    return mutated, {
        "atom_key": atom_key,
        "checkpoint_index_zero_based": checkpoint_index,
        "tensor_key": tensor_key,
        "flat_coordinate": int(coordinate),
        "requested_delta": sign * float(magnitude),
        "observed_delta": after - before,
        "precision_safe_coordinate_selection": precision_safe_protocol,
        "precision_safe_attempt_count": precision_safe_attempt_count,
        "prf_sha256": sha256_text(
            f"{protocol_version}|suffix-checkpoint-tamper|{source_id}|{post_commit_seed}|{atom_key}"
        ),
    }


def slice_replay_proof(proof: ReplayProof, transitions: int) -> ReplayProof:
    if not 0 < transitions <= len(proof.batches):
        raise ValueError("invalid replay prefix length")
    batches = proof.batches[:transitions]
    digest = hashlib.sha256(b"sevc-replay-data-order-v1")
    for index, (data, target) in enumerate(batches):
        digest.update(str(index).encode("utf-8"))
        for value in (data, target):
            tensor = value.detach().cpu().contiguous()
            digest.update(str(tensor.dtype).encode("utf-8"))
            digest.update(str(tuple(tensor.shape)).encode("utf-8"))
            digest.update(tensor.numpy().tobytes())
    return clone_replay_proof(
        proof,
        batches=batches,
        checkpoints=proof.checkpoints[:transitions],
        optimizer_checkpoints=proof.optimizer_checkpoints[:transitions],
        data_order_sha256=digest.hexdigest(),
    )


@dataclass(frozen=True)
class PublicReplayTask:
    schema_version: str
    protocol_version: str
    task_id: str
    source_commitment: str
    wrapper_id: str
    permutation_id: str
    report_nonce_sha256: str
    checkpoint_count: int
    batch_count: int
    tensor_count: int
    tensor_payload_bytes: int
    model_code: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SealedReplayTruth:
    task_id: str
    source_id: str
    role: str
    expected_verdict: bool
    atom_key: str | None
    mutation: dict[str, object] | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class CompiledReplayTask:
    public: PublicReplayTask
    sealed: SealedReplayTruth
    proof: ReplayProof
    certificate: dict[str, object]


def public_payload_metrics(proof: ReplayProof) -> tuple[int, int]:
    _, metrics, _ = _profile_public_replay_proof(proof)
    return metrics


@dataclass
class PublicReplayTaskView:
    """Read-only selector view over compact bytes and the complete public proof."""

    compact_envelope_bytes: bytes
    public_proof: ReplayProof
    protocol_version: str
    task_id: str
    _public_state_features: dict[str, object] | None = field(
        default=None, init=False, repr=False
    )
    proof_feature_full_scan_count: int = field(default=0, init=False)

    def public_state_features(self) -> dict[str, object]:
        if self._public_state_features is None:
            self._public_state_features = public_state_feature_payload(
                self.public_proof,
                task_id=self.task_id,
                protocol_version=self.protocol_version,
            )
            self.proof_feature_full_scan_count += 1
        return self._public_state_features


def _tensor_summary(values: Iterable[torch.Tensor]) -> dict[str, float | int]:
    """Compute deterministic public summaries without materializing one giant vector."""

    element_count = 0
    payload_bytes = 0
    finite_count = 0
    zero_count = 0
    total = 0.0
    total_square = 0.0
    l1 = 0.0
    linf = 0.0
    tensor_count = 0
    for value in values:
        tensor_count += 1
        tensor = value.detach().cpu().contiguous()
        element_count += int(tensor.numel())
        payload_bytes += int(tensor.numel()) * int(tensor.element_size())
        if not tensor.numel():
            continue
        numeric = tensor.to(torch.float64).reshape(-1)
        finite = torch.isfinite(numeric)
        if bool(finite.all()):
            finite_values = numeric
            finite_count += int(numeric.numel())
        else:
            finite_values = numeric[finite]
            finite_count += int(finite.sum())
        zero_count += int((finite_values == 0).sum())
        if finite_values.numel():
            total += float(finite_values.sum())
            squares = torch.square(finite_values)
            absolute = torch.abs(finite_values)
            total_square += float(squares.sum())
            l1 += float(absolute.sum())
            linf = max(linf, float(absolute.max()))
    mean = total / finite_count if finite_count else 0.0
    variance = max(0.0, total_square / finite_count - mean * mean) if finite_count else 0.0
    return {
        "tensor_count": tensor_count,
        "element_count": element_count,
        "payload_bytes": payload_bytes,
        "finite_fraction": finite_count / element_count if element_count else 1.0,
        "zero_fraction": zero_count / finite_count if finite_count else 0.0,
        "mean": mean,
        "std": math.sqrt(variance),
        "l1": l1,
        "l2": math.sqrt(max(0.0, total_square)),
        "linf": linf,
    }


def _state_values(states: Sequence[Mapping[str, torch.Tensor]]) -> list[torch.Tensor]:
    return [state[key] for state in states for key in sorted(state)]


def _adjacent_state_deltas(
    states: Sequence[Mapping[str, torch.Tensor]],
) -> Iterable[torch.Tensor]:
    for left, right in zip(states, states[1:]):
        for key in sorted(set(left) | set(right)):
            left_value = left.get(key)
            right_value = right.get(key)
            if left_value is None:
                assert right_value is not None
                left_value = torch.zeros_like(right_value)
            if right_value is None:
                right_value = torch.zeros_like(left_value)
            if left_value.shape != right_value.shape:
                raise ValueError("adjacent public state shape drift")
            yield (
                right_value.detach().cpu().to(torch.float64)
                - left_value.detach().cpu().to(torch.float64)
            )


def _public_state_feature_payload_profiled(
    proof: ReplayProof,
    *,
    task_id: str,
    protocol_version: str,
    sketch_size: int = 64,
) -> tuple[dict[str, object], dict[str, float]]:
    """Build the frozen public state summaries and task-selected sketch.

    These values are part of the immutable pre-effort envelope.  They expose
    only deterministic functions of the delivered wrapped proof, never sealed
    role, truth, atom, or replay residuals.
    """

    if proof.optimizer_initial_state is None:
        raise ValueError("public T2 envelope requires state-complete optimizer state")
    model_states = (proof.initial_state, *proof.checkpoints)
    optimizer_states = (proof.optimizer_initial_state, *proof.optimizer_checkpoints)
    batch_values = [value for batch in proof.batches for value in batch]
    adjacent_started = time.perf_counter()
    model_adjacent_deltas = _adjacent_state_deltas(model_states)
    optimizer_adjacent_deltas = _adjacent_state_deltas(optimizer_states)
    adjacent_seconds = time.perf_counter() - adjacent_started
    groups: dict[str, Iterable[torch.Tensor]] = {
        "model_initial": _state_values((proof.initial_state,)),
        "model_checkpoints": _state_values(proof.checkpoints),
        "optimizer_initial": _state_values((proof.optimizer_initial_state,)),
        "optimizer_checkpoints": _state_values(proof.optimizer_checkpoints),
        "batches": batch_values,
        "model_adjacent_deltas": model_adjacent_deltas,
        "optimizer_adjacent_deltas": optimizer_adjacent_deltas,
    }
    summary_started = time.perf_counter()
    summaries = {name: _tensor_summary(values) for name, values in groups.items()}
    summary_seconds = time.perf_counter() - summary_started
    ordered_tensors: list[tuple[str, torch.Tensor]] = []
    for group_name in (
        "model_initial",
        "model_checkpoints",
        "optimizer_initial",
        "optimizer_checkpoints",
        "batches",
    ):
        for index, value in enumerate(groups[group_name]):
            if value.numel():
                ordered_tensors.append((f"{group_name}:{index}", value))
    if not ordered_tensors:
        raise ValueError("public proof has no sketchable tensor")
    sketch_started = time.perf_counter()
    sketch = []
    for index in range(int(sketch_size)):
        selector = derive_int(protocol_version, "public-sketch", task_id, index)
        tensor_name, tensor = ordered_tensors[selector % len(ordered_tensors)]
        flat = tensor.detach().cpu().contiguous().reshape(-1)
        coordinate = (selector // len(ordered_tensors)) % int(flat.numel())
        sketch.append(
            {
                "slot": index,
                "tensor_sha256": sha256_text(tensor_name),
                "coordinate": int(coordinate),
                "value": float(flat[coordinate].to(torch.float64)),
            }
        )
    payload = {
        "schema_version": "sevc-rcmp-public-state-features-v2",
        "summaries": summaries,
        "task_selected_coordinate_sketch": sketch,
    }
    return payload, {
        "adjacent_delta": adjacent_seconds,
        "state_summary": summary_seconds,
        "sketch": time.perf_counter() - sketch_started,
    }


def public_state_feature_payload(
    proof: ReplayProof,
    *,
    task_id: str,
    protocol_version: str,
    sketch_size: int = 64,
) -> dict[str, object]:
    payload, _ = _public_state_feature_payload_profiled(
        proof,
        task_id=task_id,
        protocol_version=protocol_version,
        sketch_size=sketch_size,
    )
    return payload


def _build_public_replay_envelope_profiled(
    proof: ReplayProof,
    descriptor: Mapping[str, Any],
    *,
    protocol_version: str,
    task_id: str,
    source_commitment: str,
    report_nonce_sha256: str,
    model_key: str,
    delivery_profile: str = "full",
    wrapped_component_hashes: Mapping[str, str] | None = None,
    wrapped_payload_metrics: tuple[int, int] | None = None,
) -> tuple[dict[str, object], dict[str, float], dict[str, str]]:
    """Build the one role-neutral T2 public envelope schema."""

    identity_started = time.perf_counter()
    if delivery_profile not in {"full", "compact"}:
        raise ValueError(f"unknown T2 delivery profile: {delivery_profile}")
    if wrapped_component_hashes is None or wrapped_payload_metrics is None:
        hashes, metrics, _ = _profile_public_replay_proof(proof)
        tensor_count, tensor_payload_bytes = metrics
    else:
        hashes = dict(wrapped_component_hashes)
        tensor_count, tensor_payload_bytes = wrapped_payload_metrics
    identity_seconds = time.perf_counter() - identity_started
    envelope: dict[str, object] = {
        "schema_version": (
            T2_TASK_SCHEMA if delivery_profile == "full" else T2_COMPACT_TASK_SCHEMA
        ),
        "protocol_version": protocol_version,
        "task_id": task_id,
        "source_commitment": source_commitment,
        "wrapper_id": "coordinate-channel-permutation-v1",
        "wrapper_descriptor": dict(descriptor),
        "permutation_id": str(descriptor["permutation"]["permutation_id"]),
        "report_nonce_sha256": report_nonce_sha256,
        "checkpoint_count": len(proof.checkpoints),
        "batch_count": len(proof.batches),
        "tensor_count": tensor_count,
        "tensor_payload_bytes": tensor_payload_bytes,
        "model_code": 1 if model_key == "small-mlp" else 2,
        "wrapped_proof_identity": hashes,
    }
    feature_seconds = {"adjacent_delta": 0.0, "state_summary": 0.0, "sketch": 0.0}
    if delivery_profile == "full":
        features, feature_seconds = _public_state_feature_payload_profiled(
            proof,
            task_id=task_id,
            protocol_version=protocol_version,
        )
        envelope["public_state_features"] = features
    else:
        envelope["public_proof_content_address"] = str(hashes["proof_sha256"])
    return (
        envelope,
        {
            "public_metrics_and_wrapped_identity": identity_seconds,
            **feature_seconds,
        },
        hashes,
    )


def build_public_replay_envelope(
    proof: ReplayProof,
    descriptor: Mapping[str, Any],
    *,
    protocol_version: str,
    task_id: str,
    source_commitment: str,
    report_nonce_sha256: str,
    model_key: str,
    delivery_profile: str = "full",
) -> dict[str, object]:
    envelope, _, _ = _build_public_replay_envelope_profiled(
        proof,
        descriptor,
        protocol_version=protocol_version,
        task_id=task_id,
        source_commitment=source_commitment,
        report_nonce_sha256=report_nonce_sha256,
        model_key=model_key,
        delivery_profile=delivery_profile,
    )
    return envelope


def serialize_public_replay_envelope(envelope: Mapping[str, Any]) -> bytes:
    return canonical_json(dict(envelope)).encode("utf-8")


@dataclass(frozen=True)
class CanonicalReplayTaskBundle:
    public_envelope: dict[str, object]
    public_envelope_bytes: bytes
    descriptor: dict[str, object]
    sealed: SealedReplayTruth
    canonical_candidate: ReplayProof
    wrapped_proof: ReplayProof
    mutation: dict[str, object] | None
    component_hashes: dict[str, dict[str, str]]
    compile_seconds: dict[str, float]
    public_view: PublicReplayTaskView | None = None
    identity_profile: str = "reference"
    candidate_identity_basis: str = "compile_input_hash"
    schema_validator: nn.Module | None = None
    schema_validator_profile: str = "fresh"


def compile_canonical_replay_task(
    proof: ReplayProof,
    model: nn.Module,
    model_key: str,
    *,
    source_id: str,
    source_commitment: str,
    post_commit_seed: int,
    role: str,
    atom_key: str | None,
    permutation_seed: int,
    protocol_version: str,
    wrapper_seed_domain: str,
    tamper_delta: float,
    source_component_hashes: Mapping[str, str] | None = None,
    delivery_profile: str = "full",
    identity_profile: str = "reference",
    schema_validator_profile: str = "fresh",
    validated_source_verdict: bool | None = None,
    final_two_checkpoints: bool = False,
    transform_device: str | None = None,
) -> CanonicalReplayTaskBundle:
    """Compile one T2 task through the shared canonical mutation/wrapper seam."""

    if role not in {"production", "control", "challenge"}:
        raise ValueError(f"unknown sealed RCMP role: {role}")
    if (role == "challenge") != (atom_key is not None):
        raise ValueError("challenge and mutation atom must appear together")
    if validated_source_verdict is not None and type(validated_source_verdict) is not bool:
        raise TypeError("validated source verdict must be a replay result boolean")
    if role != "production" and validated_source_verdict is False:
        raise ValueError("certified probes require a replay-validated accepting source")
    started = time.perf_counter()
    candidate = proof
    mutation = None
    mutation_seconds = 0.0
    if atom_key is not None:
        mutation_started = time.perf_counter()
        candidate, mutation = mutate_replay_proof(
            proof,
            source_id=source_id,
            post_commit_seed=post_commit_seed,
            atom_key=atom_key,
            magnitude=tamper_delta,
            protocol_version=protocol_version,
            final_two_checkpoints=final_two_checkpoints,
        )
        mutation_seconds = time.perf_counter() - mutation_started
    if identity_profile not in {"reference", "fused", "deferred", "separated-reuse"}:
        raise ValueError(f"unknown proof identity profile: {identity_profile}")
    if schema_validator_profile not in {"fresh", "shared"}:
        raise ValueError(
            f"unknown schema validator profile: {schema_validator_profile}"
        )
    wrapper_started = time.perf_counter()
    fused_candidate_hashes = None
    fused_wrapped_hashes = None
    fused_wrapped_metrics = None
    if identity_profile in {"fused", "deferred"}:
        plan_started = time.perf_counter()
        plan = build_replay_state_permutation(model, model_key, seed=permutation_seed)
        plan_seconds = time.perf_counter() - plan_started
        wrapped, fused_candidate_hashes, fused_wrapped_hashes, fused_wrapped_metrics, wrap_seconds = (
            _transform_replay_proof_with_fused_hashes(
                candidate,
                plan,
                transform_device=transform_device,
                include_input_identity=(
                    atom_key is not None and identity_profile == "fused"
                ),
            )
        )
        wrap_seconds["permutation_plan"] = plan_seconds
    else:
        wrapped, plan, wrap_seconds = _wrap_replay_proof_profiled(
            candidate,
            model,
            model_key,
            permutation_seed=permutation_seed,
        )
    descriptor_started = time.perf_counter()
    descriptor = build_wrapper_descriptor(
        plan,
        protocol_version=protocol_version,
        wrapper_seed_domain=wrapper_seed_domain,
    )
    descriptor_seconds = time.perf_counter() - descriptor_started
    wrapper_seconds = time.perf_counter() - wrapper_started
    task_identity_started = time.perf_counter()
    task_id = sha256_text(
        canonical_json(
            {
                "protocol_version": protocol_version,
                "source_commitment": source_commitment,
                "permutation_id": plan.permutation_id,
            }
        )
    )
    report_nonce = sha256_text(
        f"{protocol_version}|report-nonce|{task_id}|{source_id}"
    )
    task_identity_seconds = time.perf_counter() - task_identity_started
    component_hash_started = time.perf_counter()
    source_hashes = (
        dict(source_component_hashes)
        if source_component_hashes is not None
        else _profile_public_replay_proof(proof)[0]
    )
    if candidate is proof:
        candidate_hashes = source_hashes
        candidate_scan_count = 0
    elif fused_candidate_hashes is not None:
        candidate_hashes = fused_candidate_hashes
        candidate_scan_count = 0
    elif identity_profile == "deferred":
        candidate_hashes = None
        candidate_scan_count = 0
    else:
        candidate_hashes, _, candidate_scan_count = _profile_public_replay_proof(
            candidate
        )
    if fused_wrapped_hashes is not None and fused_wrapped_metrics is not None:
        wrapped_hashes = fused_wrapped_hashes
        wrapped_metrics = fused_wrapped_metrics
        wrapped_scan_count = 0
    else:
        wrapped_hashes, wrapped_metrics, wrapped_scan_count = _profile_public_replay_proof(
            wrapped
        )
    component_hashes = {
        "source": source_hashes,
        "wrapped": wrapped_hashes,
    }
    if candidate_hashes is not None:
        component_hashes["candidate"] = candidate_hashes
    component_hash_seconds = time.perf_counter() - component_hash_started
    serialization_started = time.perf_counter()
    public, public_seconds, wrapped_hashes = _build_public_replay_envelope_profiled(
        wrapped,
        descriptor,
        protocol_version=protocol_version,
        task_id=task_id,
        source_commitment=source_commitment,
        report_nonce_sha256=report_nonce,
        model_key=model_key,
        delivery_profile=delivery_profile,
        wrapped_component_hashes=wrapped_hashes,
        wrapped_payload_metrics=wrapped_metrics,
    )
    serialized = serialize_public_replay_envelope(public)
    serialization_seconds = time.perf_counter() - serialization_started
    expected = (
        validated_source_verdict
        if role == "production" and validated_source_verdict is not None
        else role != "challenge"
    )
    sealed = SealedReplayTruth(
        task_id=task_id,
        source_id=source_id,
        role=role,
        expected_verdict=expected,
        atom_key=atom_key,
        mutation=mutation,
    )
    public_view = (
        PublicReplayTaskView(
            compact_envelope_bytes=serialized,
            public_proof=wrapped,
            protocol_version=protocol_version,
            task_id=task_id,
        )
        if delivery_profile == "compact"
        else None
    )
    return CanonicalReplayTaskBundle(
        public_envelope=public,
        public_envelope_bytes=serialized,
        descriptor=descriptor,
        sealed=sealed,
        canonical_candidate=candidate,
        wrapped_proof=wrapped,
        mutation=mutation,
        component_hashes=component_hashes,
        public_view=public_view,
        identity_profile=identity_profile,
        schema_validator=(model if schema_validator_profile == "shared" else None),
        schema_validator_profile=schema_validator_profile,
        candidate_identity_basis=(
            "source_identity_reuse"
            if candidate is proof
            else (
                "verified_canonical_roundtrip_deferred"
                if identity_profile == "deferred"
                else "compile_input_hash"
            )
        ),
        compile_seconds={
            "mutation": mutation_seconds,
            "wrapper_and_descriptor": wrapper_seconds,
            "permutation_plan": wrap_seconds["permutation_plan"],
            "model_transform": wrap_seconds["model_transform"],
            "optimizer_transform": wrap_seconds["optimizer_transform"],
            "batch_copy_or_share": wrap_seconds["batch_copy_or_share"],
            "fused_batch_identity": wrap_seconds.get("fused_batch_identity", 0.0),
            "descriptor": descriptor_seconds,
            "task_and_nonce_identity": task_identity_seconds,
            "public_metrics_and_wrapped_identity": public_seconds[
                "public_metrics_and_wrapped_identity"
            ],
            "state_summary": public_seconds["state_summary"],
            "adjacent_delta": public_seconds["adjacent_delta"],
            "sketch": public_seconds["sketch"],
            "serialization_and_public_features": serialization_seconds,
            "source_candidate_wrapped_hashes": component_hash_seconds,
            "owner_derived_summary_full_scans": (
                0.0 if delivery_profile == "compact" else 1.0
            ),
            "owner_component_hash_full_scans_max_per_materialized_proof": float(
                max(candidate_scan_count, wrapped_scan_count)
            ),
            "source_identity_extra_full_scans": float(
                0 if source_component_hashes is not None else 1
            ),
            "candidate_identity_extra_full_scans": float(candidate_scan_count),
            "wrapped_identity_extra_full_scans": float(wrapped_scan_count),
            "total": time.perf_counter() - started,
            "pure_transformation_measured": float(identity_profile == "separated-reuse"),
        },
    )


FIVE_CERTIFICATE_FIELDS = (
    "provenance",
    "state_reference",
    "action_separation",
    "recognizability",
    "attack_coupling",
)


def assemble_five_field_certificate(
    fields: Mapping[str, Mapping[str, Any]],
) -> dict[str, object]:
    """Hash-bind measured raw evidence into the frozen five-field schema."""

    if set(fields) != set(FIVE_CERTIFICATE_FIELDS):
        raise ValueError("five-field certificate schema mismatch")
    payload = {name: dict(fields[name]) for name in FIVE_CERTIFICATE_FIELDS}
    for name, value in payload.items():
        if not value or not any("sha256" in key for key in value):
            raise ValueError(f"certificate field lacks raw hash binding: {name}")
    return {
        "schema_version": T2_CERTIFICATE_SCHEMA,
        **payload,
        "certificate_sha256": sha256_text(canonical_json(payload)),
    }


def build_mutation_invalidity_witness(
    source: ReplayProof,
    mutated: ReplayProof,
    mutation: Mapping[str, Any] | None,
    *,
    source_id: str,
    source_commitment: str,
    source_proof_sha256: str,
    mutated_proof_sha256: str,
    descriptor: Mapping[str, Any],
    replay_tolerance: float,
    source_replay_valid: bool,
) -> dict[str, object]:
    """Build the constant-size witness for a registered suffix mutation."""

    if mutation is None:
        raise ValueError("mutation-invalidity witness requires a mutation")
    checkpoint_index = int(mutation["checkpoint_index_zero_based"])
    tensor_key = str(mutation["tensor_key"])
    coordinate = int(mutation["flat_coordinate"])
    if not 0 <= checkpoint_index < len(source.checkpoints):
        raise ValueError("witness checkpoint is outside the source proof")
    if tensor_key not in source.checkpoints[checkpoint_index]:
        raise ValueError("witness tensor is absent from the source proof")
    before_tensor = source.checkpoints[checkpoint_index][tensor_key]
    after_tensor = mutated.checkpoints[checkpoint_index][tensor_key]
    if before_tensor.shape != after_tensor.shape or not 0 <= coordinate < before_tensor.numel():
        raise ValueError("witness coordinate is outside the registered tensor")
    before = float(before_tensor.detach().reshape(-1)[coordinate].to(torch.float64))
    after = float(after_tensor.detach().reshape(-1)[coordinate].to(torch.float64))
    observed = after - before
    raw_permutation = descriptor.get("permutation")
    if not isinstance(raw_permutation, Mapping):
        raise ValueError("witness descriptor lacks a permutation")
    raw_maps = raw_permutation.get("channel_maps")
    if not isinstance(raw_maps, Mapping) or not raw_maps:
        raise ValueError("witness permutation lacks channel maps")
    permutation_isometry = all(
        isinstance(values, (list, tuple))
        and sorted(int(value) for value in values) == list(range(len(values)))
        for values in raw_maps.values()
    )
    tolerance = float(replay_tolerance)
    payload: dict[str, object] = {
        "schema_version": "sevc-rcmp-mutation-invalidity-witness-v1",
        "source_id": str(source_id),
        "source_commitment": str(source_commitment),
        "source_proof_sha256": str(source_proof_sha256),
        "mutated_proof_sha256": str(mutated_proof_sha256),
        "mutation_prf_sha256": str(mutation["prf_sha256"]),
        "atom_key": str(mutation["atom_key"]),
        "checkpoint_index_zero_based": checkpoint_index,
        "tensor_key": tensor_key,
        "flat_coordinate": coordinate,
        "before": before,
        "after": after,
        "requested_delta": float(mutation["requested_delta"]),
        "observed_delta": observed,
        "absolute_observed_delta": abs(observed),
        "replay_tolerance": tolerance,
        "guaranteed_mutated_residual_lower_bound": abs(observed) - tolerance,
        "source_replay_valid": bool(source_replay_valid),
        "descriptor_commitment": str(descriptor["descriptor_commitment"]),
        "permutation_id": str(raw_permutation["permutation_id"]),
        "permutation_isometry": bool(permutation_isometry),
    }
    return {**payload, "witness_sha256": sha256_text(canonical_json(payload))}


def verify_mutation_invalidity_witness(witness: Mapping[str, Any]) -> bool:
    payload = dict(witness)
    witness_sha256 = payload.pop("witness_sha256", None)
    expected_fields = {
        "schema_version",
        "source_id",
        "source_commitment",
        "source_proof_sha256",
        "mutated_proof_sha256",
        "mutation_prf_sha256",
        "atom_key",
        "checkpoint_index_zero_based",
        "tensor_key",
        "flat_coordinate",
        "before",
        "after",
        "requested_delta",
        "observed_delta",
        "absolute_observed_delta",
        "replay_tolerance",
        "guaranteed_mutated_residual_lower_bound",
        "source_replay_valid",
        "descriptor_commitment",
        "permutation_id",
        "permutation_isometry",
    }
    if set(payload) != expected_fields:
        return False
    tolerance = float(payload["replay_tolerance"])
    observed = float(payload["after"]) - float(payload["before"])
    absolute = abs(observed)
    return bool(
        payload["schema_version"] == "sevc-rcmp-mutation-invalidity-witness-v1"
        and isinstance(witness_sha256, str)
        and witness_sha256 == sha256_text(canonical_json(payload))
        and payload["source_replay_valid"] is True
        and payload["permutation_isometry"] is True
        and tolerance > 0.0
        and math.isclose(float(payload["observed_delta"]), observed, rel_tol=0.0, abs_tol=0.0)
        and math.isclose(float(payload["absolute_observed_delta"]), absolute, rel_tol=0.0, abs_tol=0.0)
        and math.isclose(
            float(payload["guaranteed_mutated_residual_lower_bound"]),
            absolute - tolerance,
            rel_tol=0.0,
            abs_tol=0.0,
        )
        and absolute >= 3.0 * tolerance
        and absolute - tolerance > tolerance
    )


def compile_replay_task(
    proof: ReplayProof,
    model: nn.Module,
    model_key: str,
    *,
    source_id: str,
    source_commitment: str,
    post_commit_seed: int,
    role: str,
    atom_key: str | None,
    permutation_seed: int,
    replay_tolerance: float,
    tamper_delta: float,
) -> CompiledReplayTask:
    if role not in {"production", "control", "challenge"}:
        raise ValueError(f"unknown sealed RCMP role: {role}")
    if (role == "challenge") != (atom_key is not None):
        raise ValueError("challenge and mutation atom must appear together")
    candidate = proof
    mutation = None
    if atom_key is not None:
        candidate, mutation = mutate_replay_proof(
            candidate,
            source_id=source_id,
            post_commit_seed=post_commit_seed,
            atom_key=atom_key,
            magnitude=tamper_delta,
        )
    wrapped, plan = wrap_replay_proof(
        candidate, model, model_key, permutation_seed=permutation_seed
    )
    task_id = sha256_text(
        f"{PROTOCOL_VERSION}|task|{source_id}|{post_commit_seed}|{permutation_seed}"
    )
    nonce = sha256_text(
        f"{PROTOCOL_VERSION}|report-nonce|{source_id}|{post_commit_seed}|{permutation_seed}"
    )
    tensor_count, tensor_payload_bytes = public_payload_metrics(wrapped)
    expected = role != "challenge"
    public = PublicReplayTask(
        schema_version=TASK_SCHEMA,
        protocol_version=PROTOCOL_VERSION,
        task_id=task_id,
        source_commitment=source_commitment,
        wrapper_id="coordinate-channel-permutation-v1",
        permutation_id=plan.permutation_id,
        report_nonce_sha256=nonce,
        checkpoint_count=len(wrapped.checkpoints),
        batch_count=len(wrapped.batches),
        tensor_count=tensor_count,
        tensor_payload_bytes=tensor_payload_bytes,
        model_code=1 if model_key == "small-mlp" else 2,
    )
    sealed = SealedReplayTruth(
        task_id=task_id,
        source_id=source_id,
        role=role,
        expected_verdict=expected,
        atom_key=atom_key,
        mutation=mutation,
    )
    certificate = {
        "certificate_id": sha256_text(
            canonical_json(
                {
                    "source_commitment": source_commitment,
                    "task_id": task_id,
                    "permutation_id": plan.permutation_id,
                    "atom_key": atom_key,
                }
            )
        ),
        "protocol_version": PROTOCOL_VERSION,
        "provenance": {
            "source_id": source_id,
            "source_commitment": source_commitment,
            "post_commit_seed": int(post_commit_seed),
            "task_id": task_id,
        },
        "state_reference": {
            "schema_version": wrapped.schema_version,
            "state_complete": wrapped.optimizer_initial_state is not None,
            "permutation_id": plan.permutation_id,
            "replay_tolerance": float(replay_tolerance),
            "expected_verdict": expected,
        },
        "action_separation": {
            "actions": ["H", "L", "C+", "C-", "D"],
            "failure_threshold": 2,
            "measured_in_gate": True,
        },
        "recognizability": {
            "public_schema": TASK_SCHEMA,
            "role_field_present": False,
            "selector_family": "protocol/selector_family.json",
        },
        "attack_coupling": {
            "family": "suffix-checkpoint-tamper",
            "atom_key": atom_key,
            "mutation": mutation,
            "eta_cov_registered": 0.0,
            "open_world_mass": "UNKNOWN",
        },
    }
    return CompiledReplayTask(public=public, sealed=sealed, proof=wrapped, certificate=certificate)


def assign_block_roles(
    source_ids: Sequence[str], *, post_commit_seed: int,
    protocol_version: str = PROTOCOL_VERSION,
) -> dict[str, tuple[str, str | None]]:
    if len(source_ids) != 40 or len(set(source_ids)) != 40:
        raise ValueError("RCMP block requires 40 unique committed sources")
    ordered = sorted(
        source_ids,
        key=lambda source_id: sha256_text(
            f"{protocol_version}|probe-base|{post_commit_seed}|{source_id}"
        ),
    )
    probes = ordered[:8]
    controls = sorted(
        probes,
        key=lambda source_id: sha256_text(
            f"{protocol_version}|control-split|{post_commit_seed}|{source_id}"
        ),
    )[:4]
    challenges = [source_id for source_id in probes if source_id not in controls]
    challenges = sorted(
        challenges,
        key=lambda source_id: sha256_text(
            f"{protocol_version}|atom-order|{post_commit_seed}|{source_id}"
        ),
    )
    result = {source_id: ("production", None) for source_id in source_ids}
    for source_id in controls:
        result[source_id] = ("control", None)
    for source_id, atom_key in zip(challenges, ATOM_KEYS):
        result[source_id] = ("challenge", atom_key)
    return result


def validate_public_schema(rows: Sequence[Mapping[str, Any]]) -> dict[str, object]:
    expected = set(PublicReplayTask.__dataclass_fields__)
    forbidden = {"is_probe", "is_sentinel", "role", "truth", "expected_verdict"}
    schema_violations = 0
    role_violations = 0
    for row in rows:
        schema_violations += int(set(row) != expected)
        role_violations += int(bool(set(row) & forbidden))
    return {
        "schema_violation_count": schema_violations,
        "role_field_violation_count": role_violations,
        "public_fields": sorted(expected),
    }


__all__ = [
    "ATOM_KEYS",
    "FIVE_CERTIFICATE_FIELDS",
    "PROTOCOL_VERSION",
    "TASK_SCHEMA",
    "T2_CERTIFICATE_SCHEMA",
    "T2_PROTOCOL_VERSION",
    "T2_TASK_SCHEMA",
    "T2_COMPACT_TASK_SCHEMA",
    "CanonicalReplayTaskBundle",
    "WRAPPER_DESCRIPTOR_SCHEMA",
    "CompiledReplayTask",
    "PublicReplayTask",
    "SealedReplayTruth",
    "assign_block_roles",
    "assemble_five_field_certificate",
    "build_mutation_invalidity_witness",
    "build_public_replay_envelope",
    "build_wrapper_descriptor",
    "canonicalize_replay_proof",
    "canonical_json",
    "compile_canonical_replay_task",
    "compile_replay_task",
    "derive_int",
    "mutate_replay_proof",
    "sha256_text",
    "proof_component_hashes",
    "PublicReplayTaskView",
    "public_state_feature_payload",
    "serialize_public_replay_envelope",
    "shared_schema_validator",
    "shared_schema_validator_cache_info",
    "_clear_shared_schema_validator_cache_for_tests",
    "slice_replay_proof",
    "state_complete_proof_sha256",
    "tensor_state_sha256",
    "validate_public_schema",
    "verify_wrapper_descriptor",
    "verify_mutation_invalidity_witness",
    "wrap_replay_proof",
]


def verify_wrapped_replay_proof(proof, model, descriptor, model_factory, *,
                                protocol_version, wrapper_seed_domain, device,
                                tolerance, comparison_device="replay"):
    """Validate the wrapper and invert each state only when replay consumes it."""
    from sevc.training import verify_replay_proof
    if proof.schema_version != "sevc-state-complete-replay-proof-v2":
        raise ValueError("canonical replay requires a state-complete proof")
    if proof.optimizer_initial_state is None:
        raise ValueError("canonical replay proof lacks optimizer start state")
    forward = verify_wrapper_descriptor(model, descriptor,
        protocol_version=protocol_version, wrapper_seed_domain=wrapper_seed_domain)
    inverse = inverse_replay_state_permutation(forward)
    def restore(state, optimizer):
        transform = transform_optimizer_momentum if optimizer else transform_replay_state
        return transform(state, inverse, transform_device=device,
                         output_device=device if comparison_device == "replay" else "cpu")
    return verify_replay_proof(proof, model_factory, device=device, tolerance=tolerance,
        comparison_device=comparison_device, state_transform=restore)
