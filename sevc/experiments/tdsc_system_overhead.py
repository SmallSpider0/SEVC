"""Selected canonical scientific routines; deployment wrappers are omitted."""
from __future__ import annotations


import gc

import hashlib

from pathlib import Path

import time

from typing import Any, Mapping, Sequence

import torch

from sevc.core.artifacts import canonical_json_text, sha256_text

from sevc.core.runtime import set_global_seed

from sevc.evaluation.tdsc_system_overhead import PhaseLedger, measure_canonical_payload, verify_received_commitment, settlement_identity, write_canonical_payload

from sevc.training import verify_replay_proof

from sevc.verification.replay_coupled_probes import assemble_five_field_certificate, assign_block_roles, canonicalize_replay_proof, canonicalize_replay_proof_with_identity, compile_canonical_replay_task, proof_component_hashes, shared_schema_validator

WARMUP_REPETITION = 0


TIMED_REPETITIONS = (1, 2, 3)


def _row_sha256(row: Mapping[str, Any]) -> str:
    return sha256_text(canonical_json_text(dict(row)))


def _derive_role_seed(domain: str, *parts: object, bits: int = 63) -> int:
    material = "|".join((domain, *(str(part) for part in parts)))
    value = int.from_bytes(hashlib.sha256(material.encode("utf-8")).digest()[:8], "big")
    return value & ((1 << bits) - 1)


class _Device:
    """One device abstraction so that CUDA synchronization is explicit and recorded."""

    def __init__(self, name: str) -> None:
        self.name = str(name)
        self.is_cuda = self.name.startswith("cuda")
        self.is_mps = self.name == "mps"
        self.sync_calls = 0

    def synchronize(self) -> None:
        if self.is_cuda:
            torch.cuda.synchronize()
            self.sync_calls += 1
        elif self.is_mps:
            torch.mps.synchronize()
            self.sync_calls += 1

    def empty_cache(self) -> None:
        if self.is_cuda:
            torch.cuda.empty_cache()
        elif self.is_mps:
            torch.mps.empty_cache()

    def interval(self, function, *args, **kwargs) -> tuple[Any, float, dict[str, Any]]:
        """Time one interval with device synchronization on both sides."""

        self.synchronize()
        started = time.perf_counter_ns()
        result = function(*args, **kwargs)
        self.synchronize()
        ended = time.perf_counter_ns()
        evidence = {
            "clock": "time.perf_counter_ns",
            "synchronized_before": bool(self.is_cuda or self.is_mps),
            "synchronized_after": bool(self.is_cuda or self.is_mps),
            "device": self.name,
        }
        return result, (ended - started) / 1e9, evidence


from sevc.training.replay_sources import ReplayDatasetContext as _DatasetContext, batches_from_indices as _batches_from_indices, train_short_source

def _materialize_source(
    context: _DatasetContext, row: Mapping[str, Any]
) -> dict[str, Any]:
    """Recreate one frozen committed segment on this host, outside the science timers."""

    set_global_seed(int(row["source_seed"]))
    batches = _batches_from_indices(context, row["sample_indices"], row["sample_labels"])
    started = time.perf_counter_ns()
    proof = train_short_source(context, batches)
    context.device.synchronize()
    seconds = (time.perf_counter_ns() - started) / 1e9
    hashes = proof_component_hashes(proof)
    receipt = {
        "dataset": context.dataset,
        "block_seed": int(row["block_seed"]),
        "source_index": int(row["source_index"]),
        "source_id": str(row["source_id"]),
        "source_seed": source_seed,
        "sample_indices_sha256": str(row["sample_indices_sha256"]),
        "historical_proof_sha256_mps": str(row["proof_sha256"]),
        "rematerialized_proof_sha256": hashes["proof_sha256"],
        "rematerialized_checkpoint_sha256": hashes["checkpoint_sha256"],
        "rematerialized_optimizer_sha256": hashes["optimizer_sha256"],
        "cross_device_hash_equality_asserted": False,
        "rematerialization_seconds_excluded_from_science": seconds,
        "device": context.device.name,
    }
    return {"proof": proof, "component_hashes": hashes, "receipt": receipt}


def _schema_model(context: _DatasetContext, settings: Mapping[str, Any]):
    if str(settings["schema_validator_profile"]) == "shared":
        return shared_schema_validator(context.build_key, context.factory)
    return context.factory()


def _canonicalize_task(
    bundle: Any,
    schema_model: Any,
    *,
    identity_profile: str,
    protocol_version: str,
    wrapper_domain: str,
) -> tuple[Any, Mapping[str, str]]:
    """Undo the registered wrapper through the one canonical seam."""

    if identity_profile in {"fused", "deferred"}:
        canonical, _plan, canonical_hashes, _ = canonicalize_replay_proof_with_identity(
            bundle.wrapped_proof,
            schema_model,
            bundle.descriptor,
            protocol_version=protocol_version,
            wrapper_seed_domain=wrapper_domain,
        )
        bundle.component_hashes["canonical"] = canonical_hashes
        if "candidate" not in bundle.component_hashes:
            bundle.component_hashes["candidate"] = dict(canonical_hashes)
    else:
        canonical, _plan = canonicalize_replay_proof(
            bundle.wrapped_proof,
            schema_model,
            bundle.descriptor,
            protocol_version=protocol_version,
            wrapper_seed_domain=wrapper_domain,
        )
        canonical_hashes = proof_component_hashes(canonical)
        bundle.component_hashes["canonical"] = canonical_hashes
    return canonical, canonical_hashes


def _compile_probe(
    context: _DatasetContext,
    protocol: Mapping[str, Any],
    source: Mapping[str, Any],
    row: Mapping[str, Any],
    *,
    role: str,
    atom_key: str | None,
    role_seed: int,
    scratch: Path | None,
) -> dict[str, Any]:
    """Run the complete owner online path for one probe through the canonical seams."""

    started_ns = time.perf_counter_ns()
    settings = protocol["protocol"]
    optimized = settings.get("owner_execution_profile") == "canonical-candidate-reuse-v1"
    source_protocol = str(row["source_protocol_version"])
    wrapper_domain = f"{source_protocol}|wrapper"
    schema_model = _schema_model(context, settings)
    compiler_started_ns = time.perf_counter_ns()
    setup_seconds = (compiler_started_ns - started_ns) / 1e9
    bundle = compile_canonical_replay_task(
        source["proof"],
        schema_model,
        context.build_key,
        source_id=str(row["source_id"]),
        source_commitment=str(row["source_commitment"]),
        post_commit_seed=int(role_seed),
        role=role,
        atom_key=atom_key,
        permutation_seed=_derive_role_seed(
            wrapper_domain, int(row["block_seed"]), str(row["source_id"])
        ),
        protocol_version=source_protocol,
        wrapper_seed_domain=wrapper_domain,
        tamper_delta=float(settings["tamper_delta"]),
        source_component_hashes=source["component_hashes"],
        delivery_profile=str(settings["delivery_profile"]),
        identity_profile=str(settings["identity_profile"]),
        schema_validator_profile=str(settings["schema_validator_profile"]),
    )
    binding_started_ns = time.perf_counter_ns()
    compiler_interval_seconds = (binding_started_ns - compiler_started_ns) / 1e9
    if optimized:
        # The owner constructed this immutable candidate. Do not invert its own
        # wrapper again; external delivered proofs still use canonicalization.
        canonical = bundle.canonical_candidate
        canonical_hashes = bundle.component_hashes["candidate"]
        bundle.component_hashes["canonical"] = dict(canonical_hashes)
    else:
        canonical, canonical_hashes = _canonicalize_task(
            bundle, schema_model,
            identity_profile=str(settings["identity_profile"]),
            protocol_version=source_protocol, wrapper_domain=wrapper_domain,
        )
    payload_started_ns = time.perf_counter_ns()
    binding_seconds = (payload_started_ns - binding_started_ns) / 1e9
    cached_payload = source.get("canonical_payload_identity")
    if optimized and canonical is source["proof"] and cached_payload is not None:
        delivery_bytes, delivery_sha = cached_payload
    else:
        delivery_bytes, delivery_sha = measure_canonical_payload(canonical)
    payload_seconds = (time.perf_counter_ns() - payload_started_ns) / 1e9
    envelope_bytes = len(bundle.public_envelope_bytes)
    owner_io_seconds = 0.0
    physical_st_size = None
    if scratch is not None and not optimized:
        io_started = time.perf_counter_ns()
        length, digest, st_size = write_canonical_payload(
            canonical, scratch / f"{row['source_id']}.canonical"
        )
        owner_io_seconds = (time.perf_counter_ns() - io_started) / 1e9
        if length != delivery_bytes or digest != delivery_sha:
            raise ValueError("canonical delivery payload is not deterministic")
        physical_st_size = st_size
        (scratch / f"{row['source_id']}.canonical").unlink()
    certificate_started_ns = time.perf_counter_ns()
    certificate = assemble_five_field_certificate(
        {
            "provenance": {"source_commitment_sha256": str(row["source_commitment"])},
            "state_reference": {
                "canonical_proof_sha256": canonical_hashes["proof_sha256"]
            },
            "action_separation": {
                "descriptor_sha256": _row_sha256(bundle.descriptor)
            },
            "recognizability": {
                "public_envelope_sha256": hashlib.sha256(
                    bundle.public_envelope_bytes
                ).hexdigest()
            },
            "attack_coupling": {
                "mutation_sha256": (
                    _row_sha256(bundle.mutation)
                    if bundle.mutation is not None
                    else sha256_text("NO_MUTATION")
                )
            },
        }
    )
    certificate_seconds = (time.perf_counter_ns() - certificate_started_ns) / 1e9
    stage_seconds = None
    if optimized:
        part = bundle.compile_seconds
        stage_seconds = {
            "setup": setup_seconds,
            "transformation_only": part["mutation"] + part["model_transform"] + part["optimizer_transform"] + part["batch_copy_or_share"],
            "wrapper_descriptor": part["permutation_plan"] + part["descriptor"] + part["task_and_nonce_identity"],
            "identity_hashing": part["source_candidate_wrapped_hashes"],
            "envelope_serialization": part["serialization_and_public_features"],
            "canonical_binding": binding_seconds,
            "payload_encoding_hashing": payload_seconds,
            "certificate_assembly": certificate_seconds,
            "required_owner_io": 0.0,
        }
        compiler_parts = sum(stage_seconds[k] for k in ("transformation_only", "wrapper_descriptor", "identity_hashing", "envelope_serialization"))
        stage_seconds["compiler_residual"] = compiler_interval_seconds - compiler_parts
        if stage_seconds["compiler_residual"] < 0:
            raise ValueError("overlapping compiler stage timers")
    return {
        "stage_seconds": stage_seconds,
        "bundle": bundle,
        "canonical": canonical,
        "schema_model": schema_model,
        "source_protocol": source_protocol,
        "wrapper_domain": wrapper_domain,
        "canonical_hashes": canonical_hashes,
        "certificate": certificate,
        "delivery_bytes": delivery_bytes,
        "delivery_sha256": delivery_sha,
        "envelope_bytes": envelope_bytes,
        "physical_st_size": physical_st_size,
        "owner_io_seconds": owner_io_seconds,
        "compile_seconds": dict(bundle.compile_seconds),
    }


def _semantic_identity(
    compiled: Mapping[str, Any],
    *,
    role: str,
    atom_key: str | None,
    canonical_verdict: bool,
    candidate_verdict: bool,
    expected_verdict: bool,
) -> dict[str, Any]:
    bundle = compiled["bundle"]
    return {
        "task_id": bundle.sealed.task_id,
        "role": role,
        "atom_key": atom_key,
        "descriptor_sha256": _row_sha256(bundle.descriptor),
        "public_envelope_sha256": hashlib.sha256(
            bundle.public_envelope_bytes
        ).hexdigest(),
        "canonical_proof_sha256": compiled["canonical_hashes"]["proof_sha256"],
        "certificate_sha256": compiled["certificate"]["certificate_sha256"],
        "canonical_verdict": bool(canonical_verdict),
        "candidate_verdict": bool(candidate_verdict),
        "expected_verdict": bool(expected_verdict),
        "delivery_canonical_bytes": int(compiled["delivery_bytes"]),
        "delivery_canonical_sha256": compiled["delivery_sha256"],
    }


def _timed(
    phases: PhaseLedger,
    device: _Device,
    phase: str,
    function,
    /,
    *args: Any,
    **kwargs: Any,
) -> tuple[Any, float, dict[str, Any]]:
    """Measure one synchronized interval and record its enclosing phase."""

    phase_start_ns = time.perf_counter_ns()
    device.synchronize()
    started_ns = time.perf_counter_ns()
    result = function(*args, **kwargs)
    device.synchronize()
    ended_ns = time.perf_counter_ns()
    phases.record(phase, phase_start_ns, ended_ns)
    seconds = (ended_ns - started_ns) / 1e9
    evidence = {
        "clock": "time.perf_counter_ns",
        "synchronized_before": bool(device.is_cuda or device.is_mps),
        "synchronized_after": bool(device.is_cuda or device.is_mps),
        "device": device.name,
    }
    return result, seconds, evidence


def role_of_index(
    roles: Mapping[str, Sequence[Any]], row: Mapping[str, Any]
) -> str:
    return str(roles[str(row["source_id"])][0])


def _compile_first(block_ordinal: int, source_index: int, repetition: int) -> bool:
    return (int(block_ordinal) + int(source_index) + int(repetition)) % 2 == 0


def _measure_block(
    context: _DatasetContext,
    protocol: Mapping[str, Any],
    *,
    block_ordinal: int,
    block_lock: Mapping[str, Any],
    source_rows: Sequence[Mapping[str, Any]],
    sealed_roles: Mapping[str, tuple[str, str | None]],
    source_protocol_version: str,
    scratch: Path,
    cross_check_indices: tuple[int, int],
) -> dict[str, Any]:
    """Measure one frozen block and return every raw row it produced."""

    device = context.device
    settings = protocol["protocol"]
    tolerance = float(settings["replay_tolerance"])
    block_seed = int(block_lock["block_seed"])
    job_start_ns = time.perf_counter_ns()
    phases = PhaseLedger(job_start_ns)

    materialization_rows: list[dict[str, Any]] = []
    timing_rows: list[dict[str, Any]] = []
    byte_rows: list[dict[str, Any]] = []
    retention_rows: list[dict[str, Any]] = []
    identity_rows: list[dict[str, Any]] = []
    observer_rows: list[dict[str, Any]] = []
    temporary_peak_bytes = 0

    ordered = sorted(source_rows, key=lambda row: int(row["source_index"]))
    source_ids = [str(row["source_id"]) for row in ordered]

    setup_start_ns = time.perf_counter_ns()
    role_started_ns = time.perf_counter_ns()
    roles = assign_block_roles(
        source_ids,
        post_commit_seed=int(block_lock["role_seed"]),
        protocol_version=source_protocol_version,
    )
    role_assignment_seconds = (time.perf_counter_ns() - role_started_ns) / 1e9
    if {key: tuple(value) for key, value in roles.items()} != {
        key: tuple(value) for key, value in sealed_roles.items()
    }:
        raise ValueError(
            f"reconstructed roles differ from the sealed assignment: block {block_seed}"
        )
    logical = hashlib.sha256(
        canonical_json_text(
            [
                {
                    key: row[key]
                    for key in (
                        "source_index",
                        "source_id",
                        "source_seed",
                        "sample_indices",
                        "sample_labels",
                        "sample_indices_sha256",
                    )
                }
                for row in ordered
            ]
        ).encode("utf-8")
    ).hexdigest()
    if logical != str(block_lock["logical_identity_sha256"]):
        raise ValueError(f"block logical identity drift: block {block_seed}")
    chain_rows = [
        {
            key: row[key]
            for key in (
                "block_seed",
                "source_index",
                "source_id",
                "commitment_ordinal",
                "commitment_chain_sha256",
            )
            if key in row
        }
        for row in ordered
    ]
    chain_bytes, chain_sha = measure_canonical_payload(chain_rows)
    phases.record("block_setup", setup_start_ns, time.perf_counter_ns())

    retention_rows.append(
        {
            "dataset": context.dataset,
            "block_seed": block_seed,
            "source_index": -1,
            "role": "block",
            "retention_role": "checkpoint_chain",
            "canonical_bytes": int(chain_bytes),
            "canonical_sha256": chain_sha,
            "physical_st_size": None,
        }
    )
    probe_count = sum(
        1 for value in roles.values() if value[0] in {"control", "challenge"}
    )
    allocated_role_seconds = role_assignment_seconds / float(probe_count)

    for row in ordered:
        source_index = int(row["source_index"])
        source_id = str(row["source_id"])
        role, atom_key = roles[source_id]
        cross_check = source_index in cross_check_indices

        remat_start_ns = time.perf_counter_ns()
        source = _materialize_source(context, row)
        phases.record("source_rematerialization", remat_start_ns, time.perf_counter_ns())
        materialization_rows.append(source["receipt"])

        accounting_start_ns = time.perf_counter_ns()
        if cross_check and role_of_index(roles, row) == "production":
            commitment_path = scratch / f"{row['source_id']}.commitment"
            commitment_bytes, commitment_sha, commitment_st_size = (
                write_canonical_payload(source["proof"], commitment_path)
            )
            commitment_path.unlink()
        else:
            commitment_bytes, commitment_sha = measure_canonical_payload(source["proof"])
            commitment_st_size = None
        byte_rows.append(
            {
                "dataset": context.dataset,
                "block_seed": block_seed,
                "source_index": source_index,
                "role": role,
                "direction": "trainer_commitment_upload",
                "canonical_bytes": int(commitment_bytes),
                "canonical_sha256": commitment_sha,
                "physical_st_size": commitment_st_size,
            }
        )
        retention_rows.append(
            {
                "dataset": context.dataset,
                "block_seed": block_seed,
                "source_index": source_index,
                "role": role,
                "retention_role": "committed_segment",
                "canonical_bytes": int(commitment_bytes),
                "canonical_sha256": commitment_sha,
                "physical_st_size": commitment_st_size,
            }
        )
        source["canonical_payload_identity"] = (commitment_bytes, commitment_sha)
        phases.record("byte_accounting", accounting_start_ns, time.perf_counter_ns())

        if role == "production":
            compile_start_ns = time.perf_counter_ns()
            compiled = _compile_probe(
                context,
                protocol,
                source,
                {**row, "source_protocol_version": source_protocol_version},
                role=role,
                atom_key=None,
                role_seed=int(block_lock["role_seed"]),
                scratch=scratch if cross_check else None,
            )
            production_compile_seconds = (
                time.perf_counter_ns() - compile_start_ns
            ) / 1e9
            phases.record(
                "production_owner_compile", compile_start_ns, time.perf_counter_ns()
            )
            def canonical_replay():
                canonical_proof, _hashes = _canonicalize_task(
                    compiled["bundle"],
                    compiled["schema_model"],
                    identity_profile=str(settings["identity_profile"]),
                    protocol_version=str(compiled["source_protocol"]),
                    wrapper_domain=str(compiled["wrapper_domain"]),
                )
                return verify_replay_proof(
                    canonical_proof,
                    context.factory,
                    device=device.name,
                    tolerance=tolerance,
                )

            for repetition in (WARMUP_REPETITION, *TIMED_REPETITIONS):
                result, seconds, evidence = _timed(
                    phases, device, "production_replay", canonical_replay
                )
                timing_rows.append(
                    {
                        "dataset": context.dataset,
                        "block_seed": block_seed,
                        "block_ordinal": block_ordinal,
                        "source_index": source_index,
                        "source_id": source_id,
                        "role": role,
                        "repetition": repetition,
                        "phase": "production_replay",
                        "instrumentation_mode": "instrumented",
                        "retained": True,
                        "timed": repetition != WARMUP_REPETITION,
                        "order": "canonicalize_then_replay",
                        "seconds": float(seconds),
                        "verdict": bool(result["passed"]),
                        **evidence,
                    }
                )
                verdict = bool(result["passed"])
            byte_rows.append(
                {
                    "dataset": context.dataset,
                    "block_seed": block_seed,
                    "source_index": source_index,
                    "role": role,
                    "direction": "owner_to_verifier_production_delivery",
                    "canonical_bytes": int(compiled["delivery_bytes"]),
                    "canonical_sha256": compiled["delivery_sha256"],
                    "physical_st_size": compiled["physical_st_size"],
                }
            )
            identity_rows.append(
                {
                    "dataset": context.dataset,
                    "block_seed": block_seed,
                    "source_index": source_index,
                    "source_id": source_id,
                    "production_owner_compile_seconds_descriptive": (
                        production_compile_seconds
                    ),
                    **_semantic_identity(
                        compiled,
                        role=role,
                        atom_key=None,
                        canonical_verdict=verdict,
                        candidate_verdict=verdict,
                        expected_verdict=bool(compiled["bundle"].sealed.expected_verdict),
                    ),
                }
            )
            temporary_peak_bytes = max(
                temporary_peak_bytes,
                int(commitment_bytes) + int(compiled["delivery_bytes"]),
            )
            del compiled
        else:
            owner_seconds_by_repetition: dict[int, float] = {}
            matched_seconds_by_repetition: dict[int, float] = {}
            last_identity: dict[str, Any] | None = None
            last_compiled_bytes = 0
            last_delivery_sha = ""
            last_physical = None
            for repetition in (WARMUP_REPETITION, *TIMED_REPETITIONS):
                compile_first = _compile_first(block_ordinal, source_index, repetition)
                order = "compile_first" if compile_first else "replay_first"

                def run_compile():
                    return _compile_probe(
                        context,
                        protocol,
                        source,
                        {**row, "source_protocol_version": source_protocol_version},
                        role=role,
                        atom_key=atom_key,
                        role_seed=int(block_lock["role_seed"]),
                        scratch=(
                            scratch
                            if cross_check and repetition == TIMED_REPETITIONS[0]
                            else None
                        ),
                    )

                def run_matched():
                    return verify_replay_proof(
                        source["proof"],
                        context.factory,
                        device=device.name,
                        tolerance=tolerance,
                    )

                def timed_matched():
                    if "G3" in protocol.get("gated_endpoints", {}):
                        verified, seconds, evidence = _timed(
                            phases, device, "matched_replay_commitment_verification",
                            lambda: verify_received_commitment(source["proof"], commitment_sha),
                        )
                        timing_rows.append({
                            "dataset": context.dataset, "block_seed": block_seed,
                            "block_ordinal": block_ordinal, "source_index": source_index,
                            "source_id": source_id, "role": role, "repetition": repetition,
                            "phase": "matched_replay_commitment_verification",
                            "instrumentation_mode": "instrumented", "retained": True,
                            "timed": repetition != WARMUP_REPETITION, "order": order,
                            "seconds": seconds, "commitment_verified": True,
                            "expected_commitment_sha256": commitment_sha,
                            "received_commitment_sha256": verified, **evidence,
                        })
                    return _timed(phases, device, "matched_replay", run_matched)

                if compile_first:
                    compiled, compile_seconds, compile_evidence = _timed(
                        phases, device, "owner_compile", run_compile
                    )
                    matched, matched_seconds, matched_evidence = timed_matched()
                else:
                    matched, matched_seconds, matched_evidence = timed_matched()
                    compiled, compile_seconds, compile_evidence = _timed(
                        phases, device, "owner_compile", run_compile
                    )
                owner_all_in = float(compile_seconds) + allocated_role_seconds
                exclusive = compiled.get("stage_seconds")
                if exclusive is not None:
                    exclusive = dict(exclusive)
                    exclusive["outer_residual"] = float(compile_seconds) - sum(exclusive.values())
                    exclusive["role_assignment"] = allocated_role_seconds
                    if exclusive["outer_residual"] < 0:
                        raise ValueError("owner stage timers overlap")
                # Physical-size audit is not a dispatch operation. Keep it
                # outside the registered owner interval in the new profile.
                if exclusive is not None and cross_check and repetition == TIMED_REPETITIONS[0]:
                    audit_started = time.perf_counter_ns()
                    audit_path = scratch / f"{source_id}.canonical-audit"
                    n, h, size = write_canonical_payload(compiled["canonical"], audit_path)
                    if (n, h) != (compiled["delivery_bytes"], compiled["delivery_sha256"]):
                        raise ValueError("candidate canonical payload audit mismatch")
                    compiled["physical_st_size"] = size
                    audit_path.unlink()
                    phases.record("payload_physical_audit", audit_started, time.perf_counter_ns())
                verdict_start_ns = time.perf_counter_ns()
                canonical_result = verify_replay_proof(
                    compiled["canonical"],
                    context.factory,
                    device=device.name,
                    tolerance=tolerance,
                )
                device.synchronize()
                phases.record(
                    "probe_verdict_replay", verdict_start_ns, time.perf_counter_ns()
                )
                timing_rows.append(
                    {
                        "dataset": context.dataset,
                        "block_seed": block_seed,
                        "block_ordinal": block_ordinal,
                        "source_index": source_index,
                        "source_id": source_id,
                        "role": role,
                        "repetition": repetition,
                        "phase": "owner_compile",
                        "instrumentation_mode": "instrumented",
                        "retained": True,
                        "timed": repetition != WARMUP_REPETITION,
                        "order": order,
                        "seconds": float(owner_all_in),
                        "compile_seconds": float(compile_seconds),
                        "owner_io_seconds": float(compiled["owner_io_seconds"]),
                        "owner_accounting_version": "inclusive-interval-v2",
                        "allocated_role_assignment_seconds": allocated_role_seconds,
                        "decomposition": compiled["compile_seconds"],
                        "exclusive_stage_seconds": exclusive,
                        "transformation_only_measured": exclusive is not None,
                        **compile_evidence,
                    }
                )
                timing_rows.append(
                    {
                        "dataset": context.dataset,
                        "block_seed": block_seed,
                        "block_ordinal": block_ordinal,
                        "source_index": source_index,
                        "source_id": source_id,
                        "role": role,
                        "repetition": repetition,
                        "phase": "matched_replay",
                        "instrumentation_mode": "instrumented",
                        "retained": True,
                        "timed": repetition != WARMUP_REPETITION,
                        "order": order,
                        "seconds": float(matched_seconds),
                        "verdict": bool(matched["passed"]),
                        **matched_evidence,
                    }
                )
                if repetition != WARMUP_REPETITION:
                    owner_seconds_by_repetition[repetition] = float(owner_all_in)
                    matched_seconds_by_repetition[repetition] = float(matched_seconds)
                last_identity = _semantic_identity(
                    compiled,
                    role=role,
                    atom_key=atom_key,
                    canonical_verdict=bool(canonical_result["passed"]),
                    candidate_verdict=bool(canonical_result["passed"]),
                    expected_verdict=bool(compiled["bundle"].sealed.expected_verdict),
                )
                last_compiled_bytes = int(compiled["delivery_bytes"])
                last_delivery_sha = str(compiled["delivery_sha256"])
                if compiled["physical_st_size"] is not None:
                    last_physical = int(compiled["physical_st_size"])
                report = {
                    "task_id": compiled["bundle"].sealed.task_id,
                    "verdict": bool(canonical_result["passed"]),
                    "max_abs_difference": float(
                        canonical_result["max_abs_difference"]
                    ),
                    "tolerance": tolerance,
                }
                retention_start_ns = time.perf_counter_ns()
                report_bytes, report_sha = measure_canonical_payload(report)
                if repetition == TIMED_REPETITIONS[-1]:
                    byte_rows.append(
                        {
                            "dataset": context.dataset,
                            "block_seed": block_seed,
                            "source_index": source_index,
                            "role": role,
                            "direction": "verifier_to_owner_report_return",
                            "canonical_bytes": int(report_bytes),
                            "canonical_sha256": report_sha,
                            "physical_st_size": None,
                        }
                    )
                    retention_rows.append(
                        {
                            "dataset": context.dataset,
                            "block_seed": block_seed,
                            "source_index": source_index,
                            "role": role,
                            "retention_role": "report",
                            "canonical_bytes": int(report_bytes),
                            "canonical_sha256": report_sha,
                            "physical_st_size": None,
                        }
                    )
                    certificate_bytes, certificate_sha = measure_canonical_payload(
                        compiled["certificate"]
                    )
                    retention_rows.append(
                        {
                            "dataset": context.dataset,
                            "block_seed": block_seed,
                            "source_index": source_index,
                            "role": role,
                            "retention_role": "certificate",
                            "canonical_bytes": int(certificate_bytes),
                            "canonical_sha256": certificate_sha,
                            "physical_st_size": None,
                        }
                    )
                    reference_bytes, reference_sha = measure_canonical_payload(
                        {
                            "descriptor": compiled["bundle"].descriptor,
                            "mutation": compiled["bundle"].mutation,
                            "component_hashes": compiled["bundle"].component_hashes,
                        }
                    )
                    retention_rows.append(
                        {
                            "dataset": context.dataset,
                            "block_seed": block_seed,
                            "source_index": source_index,
                            "role": role,
                            "retention_role": "owner_reference_material",
                            "canonical_bytes": int(reference_bytes),
                            "canonical_sha256": reference_sha,
                            "physical_st_size": None,
                        }
                    )
                temporary_peak_bytes = max(
                    temporary_peak_bytes,
                    int(commitment_bytes) + 2 * int(compiled["delivery_bytes"]),
                )
                phases.record(
                    "byte_accounting", retention_start_ns, time.perf_counter_ns()
                )
                del compiled
            byte_rows.append(
                {
                    "dataset": context.dataset,
                    "block_seed": block_seed,
                    "source_index": source_index,
                    "role": role,
                    "direction": "owner_to_verifier_probe_delivery",
                    "canonical_bytes": last_compiled_bytes,
                    "canonical_sha256": last_delivery_sha,
                    "physical_st_size": last_physical,
                }
            )
            identity_rows.append(
                {
                    "dataset": context.dataset,
                    "block_seed": block_seed,
                    "source_index": source_index,
                    "source_id": source_id,
                    "owner_all_in_seconds_by_repetition": owner_seconds_by_repetition,
                    "matched_replay_seconds_by_repetition": matched_seconds_by_repetition,
                    **(last_identity or {}),
                }
            )
        del source
        gc.collect()
        device.empty_cache()
    return {
        "block_seed": block_seed,
        "block_ordinal": block_ordinal,
        "phases": phases,
        "job_start_ns": job_start_ns,
        "role_assignment_seconds": role_assignment_seconds,
        "allocated_role_assignment_seconds": allocated_role_seconds,
        "materialization_rows": materialization_rows,
        "timing_rows": timing_rows,
        "byte_rows": byte_rows,
        "retention_rows": retention_rows,
        "identity_rows": identity_rows,
        "observer_rows": observer_rows,
        "temporary_peak_bytes": temporary_peak_bytes,
        "roles": {key: list(value) for key, value in roles.items()},
    }


def _uninstrumented_block(
    context: _DatasetContext,
    protocol: Mapping[str, Any],
    *,
    block_lock: Mapping[str, Any],
    source_rows: Sequence[Mapping[str, Any]],
    roles: Mapping[str, Sequence[Any]],
    source_protocol_version: str,
    materialization_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Repeat the same block without instrumentation and return its semantic identities."""

    device = context.device
    tolerance = float(protocol["protocol"]["replay_tolerance"])
    block_seed = int(block_lock["block_seed"])
    materialized = {
        int(row["source_index"]): str(row["rematerialized_proof_sha256"])
        for row in materialization_rows
    }
    rows: list[dict[str, Any]] = []
    for row in sorted(source_rows, key=lambda value: int(value["source_index"])):
        source_index = int(row["source_index"])
        source_id = str(row["source_id"])
        role, atom_key = roles[source_id]
        source = _materialize_source(context, row)
        if source["component_hashes"]["proof_sha256"] != materialized[source_index]:
            raise ValueError(
                "uninstrumented rematerialization differs from the instrumented input: "
                f"block {block_seed} source {source_index}"
            )
        observer_protocol = protocol
        if protocol["protocol"].get("owner_execution_profile") == "canonical-candidate-reuse-v1":
            observer_protocol = {**protocol, "protocol": {**protocol["protocol"], "identity_profile": "deferred", "owner_execution_profile": "reference"}}
        compiled = _compile_probe(
            context,
            observer_protocol,
            source,
            {**row, "source_protocol_version": source_protocol_version},
            role=str(role),
            atom_key=atom_key,
            role_seed=int(block_lock["role_seed"]),
            scratch=None,
        )
        result = verify_replay_proof(
            compiled["canonical"],
            context.factory,
            device=device.name,
            tolerance=tolerance,
        )
        device.synchronize()
        rows.append(
            {
                "dataset": context.dataset,
                "block_seed": block_seed,
                "source_index": source_index,
                "source_id": source_id,
                "instrumentation_mode": "uninstrumented",
                **_semantic_identity(
                    compiled,
                    role=str(role),
                    atom_key=atom_key,
                    canonical_verdict=bool(result["passed"]),
                    candidate_verdict=bool(result["passed"]),
                    expected_verdict=bool(compiled["bundle"].sealed.expected_verdict),
                ),
            }
        )
        del compiled, source
        gc.collect()
        device.empty_cache()
    return rows


_IDENTITY_FIELDS = (
    "task_id",
    "role",
    "atom_key",
    "descriptor_sha256",
    "public_envelope_sha256",
    "canonical_proof_sha256",
    "certificate_sha256",
    "canonical_verdict",
    "candidate_verdict",
    "expected_verdict",
    "delivery_canonical_bytes",
    "delivery_canonical_sha256",
)


def _compare_observer_effect(
    instrumented: Sequence[Mapping[str, Any]],
    uninstrumented: Sequence[Mapping[str, Any]],
    *,
    failure_threshold: int,
) -> dict[str, Any]:
    left = {int(row["source_index"]): row for row in instrumented}
    right = {int(row["source_index"]): row for row in uninstrumented}
    mismatches: list[dict[str, Any]] = []
    if set(left) != set(right):
        mismatches.append({"source_index": None, "field": "source_index_set"})
    for source_index in sorted(set(left) & set(right)):
        for field_name in _IDENTITY_FIELDS:
            if left[source_index].get(field_name) != right[source_index].get(field_name):
                mismatches.append(
                    {"source_index": source_index, "field": field_name}
                )
    left_settlement = settlement_identity(
        [
            {
                "role": row["role"],
                "canonical_verdict": row["canonical_verdict"],
                "expected_verdict": row["expected_verdict"],
            }
            for row in instrumented
        ],
        failure_threshold=failure_threshold,
    )
    right_settlement = settlement_identity(
        [
            {
                "role": row["role"],
                "canonical_verdict": row["canonical_verdict"],
                "expected_verdict": row["expected_verdict"],
            }
            for row in uninstrumented
        ],
        failure_threshold=failure_threshold,
    )
    if left_settlement != right_settlement:
        mismatches.append({"source_index": None, "field": "settlement_identity"})
    return {
        "compared_source_count": len(set(left) & set(right)),
        "mismatches": mismatches,
        "instrumented_settlement": left_settlement,
        "uninstrumented_settlement": right_settlement,
        "passed": not mismatches,
    }


