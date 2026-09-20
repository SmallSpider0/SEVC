"""Selected canonical scientific routines; deployment wrappers are omitted."""
from __future__ import annotations


import hashlib

import json

import os

from pathlib import Path

import time

from typing import Any, Iterable, Mapping

import torch

from sevc.core.artifacts import canonical_json_text

from sevc.core.runtime import set_global_seed

from sevc.models import build_model

from sevc.training import WorkerBehavior, produce_worker_update, verify_replay_proof

from sevc.verification.replay_coupled_probes import T2_PROTOCOL_VERSION, assign_block_roles, canonicalize_replay_proof, canonicalize_replay_proof_with_identity, proof_component_hashes, shared_schema_validator, sha256_text as replay_sha256_text, slice_replay_proof

PROTOCOL_VERSION = T2_PROTOCOL_VERSION


WRAPPER_DOMAIN = f"{PROTOCOL_VERSION}|wrapper"


def _factory():
    return build_model("small-mlp", class_count=10)


def _derive_int(domain: str, *parts: object, bits: int = 63) -> int:
    material = "|".join((domain, *(str(part) for part in parts)))
    value = int.from_bytes(hashlib.sha256(material.encode("utf-8")).digest()[:8], "big")
    return value & ((1 << bits) - 1)


def _sync() -> None:
    torch.mps.synchronize()


def _jsonl_text(rows: Iterable[Mapping[str, Any]]) -> str:
    return "".join(
        json.dumps(dict(row), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
        for row in rows
    )


def _append_jsonl(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
    *,
    fsync: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(_jsonl_text(rows))
        handle.flush()
        if fsync:
            os.fsync(handle.fileno())


def _fixture_batches(source_seed: int) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(source_seed))
    return tuple(
        (
            torch.randn((2, 1, 28, 28), generator=generator),
            torch.as_tensor([index % 10, (index + 1) % 10], dtype=torch.long),
        )
        for index in range(4)
    )


def _produce_source(
    block_seed: int,
    source_index: int,
    device: str,
    *,
    protocol_version: str | None = None,
) -> dict[str, Any]:
    active_protocol = protocol_version or PROTOCOL_VERSION
    source_domain = f"{active_protocol}|source"
    commitment_domain = f"{active_protocol}|source-commitment"
    source_id = replay_sha256_text(
        f"{active_protocol}|source-id|{block_seed}|{source_index}"
    )
    source_seed = _derive_int(source_domain, block_seed, source_index, bits=32)
    set_global_seed(source_seed)
    started = time.perf_counter()
    update = produce_worker_update(
        _factory().to(device),
        _factory,
        _fixture_batches(source_seed),
        WorkerBehavior.NORMAL,
        device=device,
        learning_rate=0.01,
        momentum=0.9,
        local_epochs=1,
        max_batches=4,
        capture_replay=True,
    )
    _sync()
    proof = update.replay_proof
    if proof is None:
        raise RuntimeError("formal T2 source did not emit a replay proof")
    hashes = proof_component_hashes(proof)
    commitment_payload = {
        "protocol_version": active_protocol,
        "block_seed": int(block_seed),
        "source_index": int(source_index),
        "source_id": source_id,
        "source_seed": int(source_seed),
        "proof_sha256": hashes["proof_sha256"],
    }
    commitment = replay_sha256_text(
        f"{commitment_domain}|{canonical_json_text(commitment_payload)}"
    )
    return {
        **commitment_payload,
        "source_commitment": commitment,
        "component_hashes": hashes,
        "proof": proof,
        "production_seconds": time.perf_counter() - started,
    }


def _build_block_sources(
    block_seed: int,
    device: str,
    *,
    commitment_path: Path | None,
    protocol_version: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], float]:
    active_protocol = protocol_version or PROTOCOL_VERSION
    source_domain = f"{active_protocol}|source"
    commitment_domain = f"{active_protocol}|source-commitment"
    role_domain = f"{active_protocol}|role-selection"
    wrapper_domain = f"{active_protocol}|wrapper"
    sources: list[dict[str, Any]] = []
    chain = "0" * 64
    source_seconds = 0.0
    for source_index in range(40):
        source = _produce_source(
            block_seed,
            source_index,
            device,
            protocol_version=active_protocol,
        )
        source_seconds += float(source["production_seconds"])
        deterministic = {
            key: source[key]
            for key in (
                "protocol_version",
                "block_seed",
                "source_index",
                "source_id",
                "source_seed",
                "proof_sha256",
                "source_commitment",
            )
        }
        chain = replay_sha256_text(
            f"{chain}|{canonical_json_text(deterministic)}"
        )
        row = {
            **deterministic,
            "checkpoint_sha256": source["component_hashes"]["checkpoint_sha256"],
            "optimizer_sha256": source["component_hashes"]["optimizer_sha256"],
            "commitment_ordinal": source_index,
            "commitment_chain_sha256": chain,
            "committed_at_ns": time.time_ns(),
        }
        source["commitment_row"] = row
        sources.append(source)
        if commitment_path is not None:
            _append_jsonl(commitment_path, (row,), fsync=False)
    if commitment_path is not None:
        with commitment_path.open("a", encoding="utf-8") as handle:
            handle.flush()
            os.fsync(handle.fileno())
    role_release_started = time.perf_counter()
    role_seed = _derive_int(role_domain, block_seed, chain)
    roles = assign_block_roles(
        [str(source["source_id"]) for source in sources],
        post_commit_seed=role_seed,
        protocol_version=active_protocol,
    )
    role_seconds = time.perf_counter() - role_release_started
    receipt = {
        "block_seed": int(block_seed),
        "commitment_count": len(sources),
        "commitment_chain_sha256": chain,
        "role_seed": int(role_seed),
        "role_domain": role_domain,
        "wrapper_domain": wrapper_domain,
        "source_domain": source_domain,
        "source_commitment_domain": commitment_domain,
        "role_release_at_ns": time.time_ns(),
        "roles": {
            source_id: {"role": role, "atom_key": atom}
            for source_id, (role, atom) in sorted(roles.items())
        },
    }
    if any(int(source["commitment_row"]["committed_at_ns"]) >= receipt["role_release_at_ns"] for source in sources):
        raise RuntimeError("source commitment did not precede role release")
    return sources, receipt, role_seconds


def _timed_replay(proof, *, transitions: int | None = None) -> tuple[dict[str, Any], float]:
    candidate = proof if transitions is None else slice_replay_proof(proof, transitions)
    _sync()
    started = time.perf_counter()
    result = verify_replay_proof(candidate, _factory, device="mps", tolerance=1e-5)
    _sync()
    return result, time.perf_counter() - started


def _timed_canonical_replay(
    bundle,
    *,
    transitions: int | None = None,
    protocol_version: str | None = None,
    wrapper_seed_domain: str | None = None,
):
    active_protocol = protocol_version or PROTOCOL_VERSION
    active_wrapper_domain = wrapper_seed_domain or WRAPPER_DOMAIN
    _sync()
    canonical_started = time.perf_counter()
    identity_profile = getattr(bundle, "identity_profile", "reference")
    schema_validator_profile = getattr(bundle, "schema_validator_profile", "fresh")
    if schema_validator_profile == "shared":
        schema_model = getattr(bundle, "schema_validator", None)
        if schema_model is None:
            raise RuntimeError("shared schema-validator profile has no validator")
        if schema_model is not shared_schema_validator("small-mlp", _factory):
            raise RuntimeError("shared schema-validator identity drift")
    else:
        schema_model = _factory()
    if identity_profile in {"fused", "deferred"}:
        canonical, verified_plan, canonical_hashes, _ = (
            canonicalize_replay_proof_with_identity(
                bundle.wrapped_proof,
                schema_model,
                bundle.descriptor,
                protocol_version=active_protocol,
                wrapper_seed_domain=active_wrapper_domain,
            )
        )
        bundle.component_hashes["canonical"] = canonical_hashes
        if identity_profile == "deferred" and "candidate" not in bundle.component_hashes:
            bundle.component_hashes["candidate"] = dict(canonical_hashes)
    else:
        canonical, verified_plan = canonicalize_replay_proof(
            bundle.wrapped_proof,
            schema_model,
            bundle.descriptor,
            protocol_version=active_protocol,
            wrapper_seed_domain=active_wrapper_domain,
        )
    canonical_seconds = time.perf_counter() - canonical_started
    result, replay_seconds = _timed_replay(canonical, transitions=transitions)
    return canonical, verified_plan, result, canonical_seconds, replay_seconds


