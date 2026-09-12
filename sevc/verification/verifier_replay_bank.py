"""Canonical replay microsegments and run-local observation caching.

The bank is evaluator-owned.  Behavior policies only select public task IDs;
they never receive cache keys, replay truth, or the proofs stored here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Protocol, Sequence, runtime_checkable

import torch
import torch.nn as nn

from sevc.core.artifacts import canonical_json_text, sha256_file, sha256_text
from sevc.incentives.verifier_protocol import (
    PublicReplaySegment,
    SealedSegmentTruth,
)
from sevc.training import ReplayProof


ModelFactory = Callable[[], nn.Module]


@dataclass(frozen=True)
class ReplayPayloadAccounting:
    """Exact uncompressed tensor and canonical-metadata accounting for one proof."""

    tensor_payload_bytes: int
    metadata_bytes: int
    total_evidence_bytes: int
    tensor_count: int
    state_tensor_count: int
    batch_tensor_count: int
    checkpoint_count: int
    batch_count: int
    metadata_sha256: str


@dataclass(frozen=True)
class DatasetReplayConstruction:
    """One live dataset-backed proof plus its immutable construction evidence."""

    proof: ReplayProof
    proof_sha256: str
    generation_seconds: float
    generation_clock_ns: int
    dataset_indices: tuple[int, ...]
    trainer_state_unchanged: bool
    payload: ReplayPayloadAccounting


def replay_proof_payload_accounting(
    proof: ReplayProof,
    *,
    metadata: Mapping[str, Any],
) -> ReplayPayloadAccounting:
    """Count raw tensor bytes and canonical JSON metadata without serialization guesses."""

    state_tensors = tuple(
        state[name]
        for state in (proof.initial_state, *proof.checkpoints)
        for name in sorted(state)
    )
    batch_tensors = tuple(value for batch in proof.batches for value in batch)
    tensors = state_tensors + batch_tensors
    tensor_payload_bytes = sum(
        int(value.numel()) * int(value.element_size()) for value in tensors
    )
    tensor_descriptors = [
        {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "numel": int(value.numel()),
            "element_size": int(value.element_size()),
        }
        for value in tensors
    ]
    canonical_metadata = canonical_json_text(
        {
            "schema_version": "sevc-replay-payload-metadata-v1",
            "learning_rate": float(proof.learning_rate),
            "momentum": float(proof.momentum),
            "checkpoint_count": len(proof.checkpoints),
            "batch_count": len(proof.batches),
            "tensor_descriptors": tensor_descriptors,
            "identity_metadata": dict(metadata),
        }
    )
    metadata_bytes = len(canonical_metadata.encode("utf-8"))
    return ReplayPayloadAccounting(
        tensor_payload_bytes=tensor_payload_bytes,
        metadata_bytes=metadata_bytes,
        total_evidence_bytes=tensor_payload_bytes + metadata_bytes,
        tensor_count=len(tensors),
        state_tensor_count=len(state_tensors),
        batch_tensor_count=len(batch_tensors),
        checkpoint_count=len(proof.checkpoints),
        batch_count=len(proof.batches),
        metadata_sha256=sha256_text(canonical_metadata),
    )


def construct_dataset_replay_proof(
    *,
    dataset: Any,
    dataset_indices: Sequence[int],
    model_factory: ModelFactory,
    device: str,
    seed: int,
    learning_rate: float,
    momentum: float,
    local_epochs: int,
    max_batches: int,
    identity_metadata: Mapping[str, Any],
) -> DatasetReplayConstruction:
    """Construct exactly one real-input proof through the canonical trainer.

    The caller owns the returned proof and must release it before constructing the
    next one.  This is the sole dataset-backed streaming extension of the shared
    replay bank; it deliberately does not cache proof tensors.
    """

    from sevc.core.runtime import set_global_seed
    from sevc.training import WorkerBehavior, produce_worker_update

    if not dataset_indices:
        raise ValueError("dataset replay construction requires at least one sample")
    runtime_seed = int(seed) & 0xFFFFFFFF
    set_global_seed(runtime_seed)
    if device == "mps":
        torch.mps.synchronize()
    started_ns = time.perf_counter_ns()
    examples = [dataset[int(index)] for index in dataset_indices]
    data = torch.stack([item[0] for item in examples])
    target = torch.as_tensor([int(item[1]) for item in examples], dtype=torch.long)
    global_model = model_factory().to(device)
    before = {
        key: value.detach().cpu().clone()
        for key, value in global_model.state_dict().items()
    }
    update = produce_worker_update(
        global_model,
        model_factory,
        ((data, target),),
        WorkerBehavior.NORMAL,
        device=device,
        learning_rate=float(learning_rate),
        momentum=float(momentum),
        local_epochs=int(local_epochs),
        max_batches=int(max_batches),
        capture_replay=True,
    )
    if device == "mps":
        torch.mps.synchronize()
    elapsed_ns = time.perf_counter_ns() - started_ns
    proof = update.replay_proof
    if proof is None:
        raise RuntimeError("canonical trainer did not emit a replay proof")
    unchanged = all(
        torch.equal(before[key], value.detach().cpu())
        for key, value in global_model.state_dict().items()
    )
    proof_identity = replay_proof_sha256(proof)
    accounting = replay_proof_payload_accounting(
        proof,
        metadata={
            **dict(identity_metadata),
            "dataset_indices": [int(value) for value in dataset_indices],
            "seed": int(seed),
            "proof_sha256": proof_identity,
        },
    )
    return DatasetReplayConstruction(
        proof=proof,
        proof_sha256=proof_identity,
        generation_seconds=elapsed_ns / 1_000_000_000.0,
        generation_clock_ns=elapsed_ns,
        dataset_indices=tuple(int(value) for value in dataset_indices),
        trainer_state_unchanged=unchanged,
        payload=accounting,
    )


def replay_proof_sha256(proof: ReplayProof) -> str:
    """Return the single canonical identity for replay proof contents."""

    digest = hashlib.sha256(b"sevc-replay-proof-v2")
    digest.update(str(float(proof.learning_rate)).encode("utf-8"))
    digest.update(str(float(proof.momentum)).encode("utf-8"))
    for state in (proof.initial_state, *proof.checkpoints):
        for name in sorted(state):
            tensor = state[name].detach().cpu().contiguous()
            digest.update(name.encode("utf-8"))
            digest.update(str(tensor.dtype).encode("utf-8"))
            digest.update(str(tuple(tensor.shape)).encode("utf-8"))
            digest.update(tensor.numpy().tobytes())
    for data, target in proof.batches:
        for tensor in (data, target):
            value = tensor.detach().cpu().contiguous()
            digest.update(str(value.dtype).encode("utf-8"))
            digest.update(str(tuple(value.shape)).encode("utf-8"))
            digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _proof_public_measurements(proof: ReplayProof) -> dict[str, float | int]:
    states = (proof.initial_state, *proof.checkpoints)
    tensors = tuple(state[name] for state in states for name in sorted(state))
    serialized_size = sum(
        int(value.numel() * value.element_size()) for value in tensors
    ) + sum(
        int(value.numel() * value.element_size())
        for batch in proof.batches
        for value in batch
    )
    initial_flat = torch.cat(
        [proof.initial_state[name].detach().cpu().reshape(-1).float() for name in sorted(proof.initial_state)]
    )
    final_state = proof.checkpoints[-1]
    final_flat = torch.cat(
        [final_state[name].detach().cpu().reshape(-1).float() for name in sorted(final_state)]
    )
    return {
        "serialized_size": serialized_size,
        "tensor_count": len(tensors),
        "checkpoint_count": len(proof.checkpoints),
        "batch_count": len(proof.batches),
        "initial_norm": float(torch.linalg.vector_norm(initial_flat)),
        "final_norm": float(torch.linalg.vector_norm(final_flat)),
        "delta_norm": float(torch.linalg.vector_norm(final_flat - initial_flat)),
    }


def execute_replay_proof(
    proof: ReplayProof,
    model_factory: ModelFactory,
    *,
    device: str,
    tolerance: float,
    seed: int,
) -> tuple[bool, float, dict[str, Any]]:
    """Execute the canonical ``trajectory-replay`` verifier once."""

    from sevc.verification.methods import VerificationContext, run_verification

    if device == "mps":
        torch.mps.synchronize()
    started_ns = time.perf_counter_ns()
    result = run_verification(
        "trajectory-replay",
        VerificationContext(
            global_model=model_factory(),
            local_models=(model_factory(),),
            evaluator=lambda model: (0.0, 0.0),
            ground_truth_malicious=(False,),
            seed=int(seed),
            replay_proofs=(proof,),
            replay_model_factory=model_factory,
            replay_device=device,
            replay_tolerance=float(tolerance),
        ),
    )
    if device == "mps":
        torch.mps.synchronize()
    elapsed = (time.perf_counter_ns() - started_ns) / 1_000_000_000.0
    detail = result.details["proof_results"][0]
    return (
        not bool(result.predicted_malicious[0]),
        float(elapsed),
        dict(detail) if detail is not None else {},
    )


@dataclass(frozen=True)
class ReplayBankEntry:
    source_segment_id: str
    proof_sha256: str
    proof: ReplayProof
    model_factory: ModelFactory
    public_segment: PublicReplaySegment
    sealed_truth: SealedSegmentTruth


@dataclass(frozen=True)
class ReplayObservation:
    cache_key: str
    proof_sha256: str
    execution_plan_key: str
    verdict: bool
    replay_seconds: float
    checkpoint_count: int
    batch_count: int
    cache_hit: bool


@runtime_checkable
class ReplayObservationProvider(Protocol):
    """The one observation interface shared by live and frozen banks."""

    def observe(
        self,
        proof_sha256: str,
        *,
        execution_plan_key: str,
        seed: int,
    ) -> ReplayObservation: ...


class ReplayObservationBank:
    """A run-local strict cache keyed by proof, plan, device, and source."""

    def __init__(
        self,
        *,
        device: str,
        tolerance: float,
        source_sha256: str,
        split: str,
    ) -> None:
        if split not in {
            "calibration",
            "evaluation",
            "confirmatory",
            "scale",
            "selection",
            "validation",
            "fixture",
        }:
            raise ValueError("replay bank split is invalid")
        if not source_sha256 or tolerance < 0:
            raise ValueError("replay bank identity is incomplete")
        self.device = str(device)
        self.tolerance = float(tolerance)
        self.source_sha256 = str(source_sha256)
        self.split = split
        self._entries_by_proof: dict[str, ReplayBankEntry] = {}
        self._entries_by_source: dict[str, ReplayBankEntry] = {}
        self._observations: dict[str, ReplayObservation] = {}
        self._hit_counts: dict[str, int] = {}
        self._miss_count = 0

    def register(self, entry: ReplayBankEntry) -> None:
        if entry.proof_sha256 != replay_proof_sha256(entry.proof):
            raise ValueError("replay bank proof identity drift")
        if entry.public_segment.payload_digest != entry.proof_sha256:
            raise ValueError("public payload identity differs from proof identity")
        if entry.sealed_truth.proof_sha256 not in {"", entry.proof_sha256}:
            raise ValueError("sealed proof identity drift")
        if entry.proof_sha256 in self._entries_by_proof:
            raise ValueError("duplicate proof identity in replay bank")
        if entry.source_segment_id in self._entries_by_source:
            raise ValueError("duplicate source segment identity in replay bank")
        self._entries_by_proof[entry.proof_sha256] = entry
        self._entries_by_source[entry.source_segment_id] = entry

    @property
    def proof_sha256s(self) -> tuple[str, ...]:
        return tuple(sorted(self._entries_by_proof))

    @property
    def public_segments(self) -> tuple[PublicReplaySegment, ...]:
        return tuple(
            self._entries_by_source[key].public_segment
            for key in sorted(self._entries_by_source)
        )

    @property
    def sealed_truths(self) -> tuple[SealedSegmentTruth, ...]:
        return tuple(
            self._entries_by_source[key].sealed_truth
            for key in sorted(self._entries_by_source)
        )

    def entry_for_payload(self, proof_sha256: str) -> ReplayBankEntry:
        try:
            return self._entries_by_proof[str(proof_sha256)]
        except KeyError as exc:
            raise KeyError(f"unknown replay proof identity: {proof_sha256}") from exc

    def entry_for_source(self, source_segment_id: str) -> ReplayBankEntry:
        try:
            return self._entries_by_source[str(source_segment_id)]
        except KeyError as exc:
            raise KeyError(f"unknown replay source segment: {source_segment_id}") from exc

    def cache_key(self, proof_sha256: str, execution_plan_key: str) -> str:
        payload = {
            "domain": "sevc-replay-observation-cache-v2",
            "proof_sha256": str(proof_sha256),
            "execution_plan_key": str(execution_plan_key),
            "device": self.device,
            "tolerance": self.tolerance,
            "source_sha256": self.source_sha256,
            "split": self.split,
        }
        return sha256_text(canonical_json_text(payload))

    def observe(
        self,
        proof_sha256: str,
        *,
        execution_plan_key: str,
        seed: int,
    ) -> ReplayObservation:
        if not execution_plan_key:
            raise ValueError("execution_plan_key must be non-empty")
        entry = self.entry_for_payload(proof_sha256)
        key = self.cache_key(proof_sha256, execution_plan_key)
        if key in self._observations:
            stored = self._observations[key]
            self._hit_counts[key] = self._hit_counts.get(key, 0) + 1
            return ReplayObservation(
                **{
                    **stored.__dict__,
                    "cache_hit": True,
                }
            )
        verdict, seconds, detail = execute_replay_proof(
            entry.proof,
            entry.model_factory,
            device=self.device,
            tolerance=self.tolerance,
            seed=seed,
        )
        observation = ReplayObservation(
            cache_key=key,
            proof_sha256=entry.proof_sha256,
            execution_plan_key=execution_plan_key,
            verdict=bool(verdict),
            replay_seconds=float(seconds),
            checkpoint_count=int(
                detail.get("checkpoint_count", len(entry.proof.checkpoints))
            ),
            batch_count=len(entry.proof.batches),
            cache_hit=False,
        )
        self._observations[key] = observation
        self._hit_counts[key] = 0
        self._miss_count += 1
        return observation

    def to_evidence(self) -> dict[str, Any]:
        rows = []
        for key in sorted(self._observations):
            item = self._observations[key]
            rows.append(
                {
                    "cache_key": key,
                    "proof_sha256": item.proof_sha256,
                    "execution_plan_key": item.execution_plan_key,
                    "device": self.device,
                    "tolerance": self.tolerance,
                    "source_sha256": self.source_sha256,
                    "split": self.split,
                    "verdict": item.verdict,
                    "replay_seconds": item.replay_seconds,
                    "checkpoint_count": item.checkpoint_count,
                    "batch_count": item.batch_count,
                    "miss_count": 1,
                    "hit_count": self._hit_counts.get(key, 0),
                }
            )
        return {
            "schema_version": "sevc-replay-observation-cache-v2",
            "split": self.split,
            "device": self.device,
            "tolerance": self.tolerance,
            "source_sha256": self.source_sha256,
            "registered_proof_count": len(self._entries_by_proof),
            "observation_count": len(self._observations),
            "cache_miss_count": self._miss_count,
            "cache_hit_count": sum(self._hit_counts.values()),
            "rows": rows,
            "cache_identity_sha256": sha256_text(canonical_json_text(rows)),
        }

    @property
    def live_replay_proof_count(self) -> int:
        return len(self._entries_by_proof)

    def characterize_all(self, *, seed: int) -> None:
        """Execute the complete block characterization before economic use."""

        truth_by_proof = {
            item.proof_sha256: item for item in self.sealed_truths
        }
        if set(truth_by_proof) != set(self.proof_sha256s):
            raise ValueError("sealed truth and proof identities differ")
        for index, proof_sha256 in enumerate(self.proof_sha256s):
            self.observe(
                proof_sha256,
                execution_plan_key="canonical-full-microsegment",
                seed=int(seed) + index,
            )
        sentinel_proofs = tuple(
            sorted(
                proof_sha256
                for proof_sha256, truth in truth_by_proof.items()
                if truth.is_sentinel
            )
        )
        for index, proof_sha256 in enumerate(sentinel_proofs):
            self.observe(
                proof_sha256,
                execution_plan_key="adjudication-rerun",
                seed=int(seed) + 1_000_000 + index,
            )

    def to_characterization_rows(self) -> tuple[dict[str, Any], ...]:
        """Return one canonical tensor-free row per registered proof."""

        rows = []
        observations_by_proof: dict[str, list[dict[str, Any]]] = {}
        for observation in self._observations.values():
            observations_by_proof.setdefault(
                observation.proof_sha256, []
            ).append(
                {
                    **asdict(observation),
                    "cache_hit": False,
                }
            )
        for source_segment_id in sorted(self._entries_by_source):
            entry = self._entries_by_source[source_segment_id]
            observations = tuple(
                sorted(
                    observations_by_proof.get(entry.proof_sha256, ()),
                    key=lambda item: str(item["execution_plan_key"]),
                )
            )
            required = {"canonical-full-microsegment"}
            if entry.sealed_truth.is_sentinel:
                required.add("adjudication-rerun")
            observed = {
                str(item["execution_plan_key"]) for item in observations
            }
            if observed != required:
                raise RuntimeError(
                    "incomplete replay characterization: "
                    f"proof={entry.proof_sha256}, expected={sorted(required)}, "
                    f"observed={sorted(observed)}"
                )
            rows.append(
                {
                    "schema_version": "sevc-replay-characterization-v3",
                    "split": self.split,
                    "device": self.device,
                    "tolerance": self.tolerance,
                    "source_sha256": self.source_sha256,
                    "source_segment_id": entry.source_segment_id,
                    "proof_sha256": entry.proof_sha256,
                    "public_segment": asdict(entry.public_segment),
                    "sealed_truth": asdict(entry.sealed_truth),
                    "observations": observations,
                }
            )
        return tuple(rows)

    def freeze_characterization(self) -> "FrozenReplayObservationBank":
        """Round-trip through canonical rows so proof tensors cannot survive."""

        rows = self.to_characterization_rows()
        canonical_rows = tuple(
            json.loads(canonical_json_text(row)) for row in rows
        )
        return FrozenReplayObservationBank.from_characterization_rows(
            canonical_rows,
            expected_device=self.device,
            expected_tolerance=self.tolerance,
            expected_source_sha256=self.source_sha256,
            expected_split=self.split,
        )


class FrozenReplayObservationBank:
    """Immutable tensor-free characterization consumed by economic cells."""

    def __init__(
        self,
        *,
        device: str,
        tolerance: float,
        source_sha256: str,
        split: str,
        public_segments: Sequence[PublicReplaySegment],
        sealed_truths: Sequence[SealedSegmentTruth],
        observations: Sequence[ReplayObservation],
        characterization_rows: Sequence[Mapping[str, Any]],
    ) -> None:
        self.device = str(device)
        self.tolerance = float(tolerance)
        self.source_sha256 = str(source_sha256)
        self.split = str(split)
        self._public_by_source = {
            item.segment_id: item for item in public_segments
        }
        self._truth_by_source = {
            item.segment_id: item for item in sealed_truths
        }
        if set(self._public_by_source) != set(self._truth_by_source):
            raise ValueError("frozen public/sealed source identities differ")
        self._observations = {item.cache_key: item for item in observations}
        if len(self._observations) != len(tuple(observations)):
            raise ValueError("duplicate frozen observation cache key")
        self._hit_counts = {key: 0 for key in self._observations}
        self._characterization_rows = tuple(
            dict(item) for item in characterization_rows
        )
        self.characterization_identity_sha256 = sha256_text(
            canonical_json_text(self._characterization_rows)
        )

    @classmethod
    def from_characterization_rows(
        cls,
        rows: Sequence[Mapping[str, Any]],
        *,
        expected_device: str,
        expected_tolerance: float,
        expected_source_sha256: str,
        expected_split: str,
    ) -> "FrozenReplayObservationBank":
        if not rows:
            raise ValueError("frozen replay characterization cannot be empty")
        public_segments = []
        sealed_truths = []
        observations = []
        canonical_rows = tuple(dict(item) for item in rows)
        for row in canonical_rows:
            if row.get("schema_version") != "sevc-replay-characterization-v3":
                raise ValueError("replay characterization schema drift")
            if str(row.get("device")) != str(expected_device):
                raise ValueError("replay characterization device drift")
            if float(row.get("tolerance", -1.0)) != float(expected_tolerance):
                raise ValueError("replay characterization tolerance drift")
            if str(row.get("source_sha256")) != str(expected_source_sha256):
                raise ValueError("replay characterization source drift")
            if str(row.get("split")) != str(expected_split):
                raise ValueError("replay characterization split drift")
            public_payload = dict(row["public_segment"])
            public_payload["public_features"] = tuple(
                float(value)
                for value in public_payload.get("public_features", ())
            )
            public = PublicReplaySegment(**public_payload)
            sealed = SealedSegmentTruth(**dict(row["sealed_truth"]))
            proof_sha256 = str(row["proof_sha256"])
            source_segment_id = str(row["source_segment_id"])
            if (
                public.segment_id != source_segment_id
                or sealed.segment_id != source_segment_id
                or public.payload_digest != proof_sha256
                or sealed.proof_sha256 != proof_sha256
            ):
                raise ValueError("replay characterization identity drift")
            public_segments.append(public)
            sealed_truths.append(sealed)
            required = {"canonical-full-microsegment"}
            if sealed.is_sentinel:
                required.add("adjudication-rerun")
            observed_plans = set()
            for raw_observation in row["observations"]:
                payload = dict(raw_observation)
                payload["cache_hit"] = False
                observation = ReplayObservation(**payload)
                expected_key = sha256_text(
                    canonical_json_text(
                        {
                            "domain": "sevc-replay-observation-cache-v2",
                            "proof_sha256": proof_sha256,
                            "execution_plan_key": observation.execution_plan_key,
                            "device": str(expected_device),
                            "tolerance": float(expected_tolerance),
                            "source_sha256": str(expected_source_sha256),
                            "split": str(expected_split),
                        }
                    )
                )
                if (
                    observation.proof_sha256 != proof_sha256
                    or observation.cache_key != expected_key
                ):
                    raise ValueError("frozen observation identity drift")
                observed_plans.add(observation.execution_plan_key)
                observations.append(observation)
            if observed_plans != required:
                raise ValueError("frozen observation plan coverage drift")
        if len({item.segment_id for item in public_segments}) != len(public_segments):
            raise ValueError("duplicate frozen source segment")
        return cls(
            device=expected_device,
            tolerance=expected_tolerance,
            source_sha256=expected_source_sha256,
            split=expected_split,
            public_segments=public_segments,
            sealed_truths=sealed_truths,
            observations=observations,
            characterization_rows=canonical_rows,
        )

    @property
    def proof_sha256s(self) -> tuple[str, ...]:
        return tuple(
            sorted(item.payload_digest for item in self._public_by_source.values())
        )

    @property
    def public_segments(self) -> tuple[PublicReplaySegment, ...]:
        return tuple(
            self._public_by_source[key] for key in sorted(self._public_by_source)
        )

    @property
    def sealed_truths(self) -> tuple[SealedSegmentTruth, ...]:
        return tuple(
            self._truth_by_source[key] for key in sorted(self._truth_by_source)
        )

    @property
    def live_replay_proof_count(self) -> int:
        return 0

    def cache_key(self, proof_sha256: str, execution_plan_key: str) -> str:
        return sha256_text(
            canonical_json_text(
                {
                    "domain": "sevc-replay-observation-cache-v2",
                    "proof_sha256": str(proof_sha256),
                    "execution_plan_key": str(execution_plan_key),
                    "device": self.device,
                    "tolerance": self.tolerance,
                    "source_sha256": self.source_sha256,
                    "split": self.split,
                }
            )
        )

    def observe(
        self,
        proof_sha256: str,
        *,
        execution_plan_key: str,
        seed: int,
    ) -> ReplayObservation:
        del seed
        key = self.cache_key(proof_sha256, execution_plan_key)
        try:
            stored = self._observations[key]
        except KeyError as exc:
            raise KeyError(
                "uncharacterized frozen replay observation: "
                f"proof={proof_sha256}, plan={execution_plan_key}"
            ) from exc
        self._hit_counts[key] += 1
        return ReplayObservation(
            **{
                **asdict(stored),
                "cache_hit": True,
            }
        )

    def to_characterization_rows(self) -> tuple[dict[str, Any], ...]:
        return tuple(dict(item) for item in self._characterization_rows)

    def to_evidence(self) -> dict[str, Any]:
        rows = []
        for key in sorted(self._observations):
            item = self._observations[key]
            rows.append(
                {
                    "cache_key": key,
                    "proof_sha256": item.proof_sha256,
                    "execution_plan_key": item.execution_plan_key,
                    "device": self.device,
                    "tolerance": self.tolerance,
                    "source_sha256": self.source_sha256,
                    "split": self.split,
                    "verdict": item.verdict,
                    "replay_seconds": item.replay_seconds,
                    "checkpoint_count": item.checkpoint_count,
                    "batch_count": item.batch_count,
                    "miss_count": 1,
                    "hit_count": self._hit_counts[key],
                }
            )
        return {
            "schema_version": "sevc-frozen-replay-observation-cache-v3",
            "split": self.split,
            "device": self.device,
            "tolerance": self.tolerance,
            "source_sha256": self.source_sha256,
            "registered_proof_count": len(self._public_by_source),
            "observation_count": len(rows),
            "cache_miss_count": len(rows),
            "cache_hit_count": sum(self._hit_counts.values()),
            "rows": rows,
            "cache_identity_sha256": sha256_text(canonical_json_text(rows)),
            "characterization_identity_sha256": self.characterization_identity_sha256,
            "live_replay_proof_count": 0,
        }


def split_replay_microsegments(proof: ReplayProof) -> tuple[ReplayProof, ...]:
    if not proof.batches or len(proof.batches) != len(proof.checkpoints):
        raise ValueError("canonical proof must have one checkpoint per batch")
    result = []
    for index, (batch, checkpoint) in enumerate(
        zip(proof.batches, proof.checkpoints)
    ):
        initial = proof.initial_state if index == 0 else proof.checkpoints[index - 1]
        result.append(
            ReplayProof(
                initial_state=initial,
                batches=(batch,),
                checkpoints=(checkpoint,),
                learning_rate=proof.learning_rate,
                momentum=proof.momentum,
            )
        )
    return tuple(result)


def build_replay_microsegment_bank(
    *,
    repo_root: Path,
    block_seed: int,
    split: str,
    base_seed: int,
    block_index: int,
    settings: Mapping[str, Any],
    device: str,
    production_count: int,
    sentinel_pool_count: int,
    trainer_truth_valid: bool,
) -> ReplayObservationBank:
    """Generate one independent block and register every unique microsegment."""

    import random

    from sevc.verification.verifier_task_policies import (
        produce_canonical_replay_microsegments,
        tamper_replay_proof,
    )

    total = int(production_count) + int(sentinel_pool_count)
    if production_count <= 0 or sentinel_pool_count <= 0:
        raise ValueError("production and sentinel counts must be positive")
    microsegments, factory = produce_canonical_replay_microsegments(
        int(block_seed), settings, device, count=total
    )
    if len(microsegments) != total:
        raise RuntimeError(
            f"generated {len(microsegments)} microsegments, expected {total}"
        )
    source_sha = sha256_file(repo_root / "sevc/verification/methods.py")
    bank = ReplayObservationBank(
        device=device,
        tolerance=float(settings["replay_tolerance"]),
        source_sha256=source_sha,
        split=split,
    )
    rng = random.Random(int(block_seed))
    roles = ["production"] * production_count + ["sentinel"] * sentinel_pool_count
    rng.shuffle(roles)
    majority_fraction = float(settings["production_majority_fraction"])
    if not 0.5 < majority_fraction <= 1.0:
        raise ValueError("production majority fraction must lie in (0.5, 1]")
    majority_count = round(production_count * majority_fraction)
    production_valid = (
        majority_count
        if trainer_truth_valid
        else production_count - majority_count
    )
    production_truths = [True] * production_valid + [False] * (
        production_count - production_valid
    )
    sentinel_truths = [True] * (sentinel_pool_count // 2) + [False] * (
        sentinel_pool_count - sentinel_pool_count // 2
    )
    rng.shuffle(production_truths)
    rng.shuffle(sentinel_truths)
    production_cursor = 0
    sentinel_cursor = 0
    for position, (role, canonical_proof) in enumerate(zip(roles, microsegments)):
        if role == "production":
            expected = production_truths[production_cursor]
            production_cursor += 1
        else:
            expected = sentinel_truths[sentinel_cursor]
            sentinel_cursor += 1
        proof = (
            canonical_proof
            if expected
            else tamper_replay_proof(
                canonical_proof,
                0,
                float(settings["checkpoint_tamper_delta"]),
            )
        )
        proof_hash = replay_proof_sha256(proof)
        source_segment_id = sha256_text(
            "|".join(
                (
                    "sevc-rc-phse-source-segment-v2",
                    split,
                    str(base_seed),
                    str(block_index),
                    str(position),
                )
            )
        )[:24]
        measurements = _proof_public_measurements(proof)
        hash_bucket = int(proof_hash[:8], 16) / float(0xFFFFFFFF)
        timing_bucket = int(
            sha256_text(f"pre-replay-timing|{source_segment_id}")[:8], 16
        ) / float(0xFFFFFFFF)
        public_features = (
            float(measurements["serialized_size"]),
            float(measurements["tensor_count"]),
            float(measurements["checkpoint_count"]),
            float(measurements["batch_count"]),
            float(measurements["initial_norm"]),
            float(measurements["final_norm"]),
            float(measurements["delta_norm"]),
            hash_bucket,
            1.0,
            timing_bucket,
        )
        public = PublicReplaySegment(
            segment_id=source_segment_id,
            payload_digest=proof_hash,
            public_features=public_features,
            public_nonce=sha256_text(f"source-nonce|{source_segment_id}")[:24],
            serialized_size=int(measurements["serialized_size"]),
            tensor_count=int(measurements["tensor_count"]),
            checkpoint_count=int(measurements["checkpoint_count"]),
            batch_count=int(measurements["batch_count"]),
        )
        sealed = SealedSegmentTruth(
            segment_id=source_segment_id,
            expected_verdict=bool(expected),
            is_sentinel=role == "sentinel",
            technical_failure=False,
            source_segment_id=source_segment_id,
            proof_sha256=proof_hash,
        )
        bank.register(
            ReplayBankEntry(
                source_segment_id=source_segment_id,
                proof_sha256=proof_hash,
                proof=proof,
                model_factory=factory,
                public_segment=public,
                sealed_truth=sealed,
            )
        )
    return bank


def assert_disjoint_banks(
    calibration: Sequence[Any],
    evaluation: Sequence[Any],
) -> None:
    def provider(value: Any) -> ReplayObservationProvider:
        candidate = getattr(value, "bank", value)
        if not isinstance(candidate, ReplayObservationProvider):
            raise TypeError("disjointness input lacks replay observation provider")
        return candidate

    left = {
        value
        for item in calibration
        for value in provider(item).proof_sha256s  # type: ignore[attr-defined]
    }
    right = {
        value
        for item in evaluation
        for value in provider(item).proof_sha256s  # type: ignore[attr-defined]
    }
    overlap = left & right
    if overlap:
        raise ValueError(f"calibration/evaluation proof overlap: {len(overlap)}")
