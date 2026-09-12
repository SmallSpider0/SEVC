"""Selected canonical scientific routines; deployment wrappers are omitted."""
from __future__ import annotations


from datetime import datetime, timezone

import math

from typing import Any, Mapping, Sequence

import torch

from sevc.attacks import WorkerBehavior

from sevc.core.artifacts import canonical_json_text, sha256_text

from sevc.core.runtime import set_global_seed

from sevc.data import indices_sha256

from sevc.models import build_model, model_state_sha256

from sevc.training import produce_worker_update, verify_replay_proof

from sevc.verification.replay_coupled_probes import proof_component_hashes

from sevc.verification.sampling_replay_stress import PROTOCOL_VERSION, derive_int, exact_without_replacement_coverage, mutate_one_step_proof, perturb_one_step_proof, polluted_segment_ids, replay_decision, sample_segment_ids, segment_identity, summarize_partial_rows, summarize_tolerance_rows

CHANGE_ID = "experiment-tdsc-sampling-replay-stress-v1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _segment_batch(dataset: Any, ordered_pool: Sequence[int], *, dataset_key: str, source_seed: int, segment_index: int, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, tuple[int, ...]]:
    start = segment_index * batch_size
    indices = tuple(int(value) for value in ordered_pool[start : start + batch_size])
    if len(indices) != batch_size:
        raise ValueError("E2 trainer pool is too small for frozen segment registry")
    data_rows = []
    target_rows = []
    for position, index in enumerate(indices):
        set_global_seed(derive_int(PROTOCOL_VERSION, "sample-transform", dataset_key, source_seed, segment_index, position, index) % 2147483647)
        data, target = dataset[index]
        data_rows.append(data)
        target_rows.append(int(target))
    return torch.stack(data_rows), torch.as_tensor(target_rows, dtype=torch.long), indices


def _finite_replay(result: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(result)
    for key in ("max_abs_difference", "max_model_abs_difference", "max_optimizer_abs_difference", "tolerance"):
        if key in payload and not math.isfinite(float(payload[key])):
            raise ValueError(f"E2 replay emitted non-finite {key}")
    return payload


def _process_segment(
    identity: Mapping[str, Any],
    *,
    dataset: Any,
    ordered_pool: Sequence[int],
    effective: Mapping[str, Any],
    dataset_spec: Mapping[str, Any],
    device: str,
    profile: str,
    worker_id: str,
    physical_gpu_id: int | None,
    implementation: Mapping[str, str],
    tolerance_assignments: Mapping[str, Sequence[tuple[int, int]]],
) -> dict[str, Any]:
    dataset_key = str(identity["dataset"])
    source_seed = int(identity["source_seed"])
    segment_index = int(identity["segment_index"])
    segment_id = str(identity["segment_id"])
    batch_size = int(effective["source_bank"]["batch_size"])
    batch_data, batch_target, batch_indices = _segment_batch(
        dataset,
        ordered_pool,
        dataset_key=dataset_key,
        source_seed=source_seed,
        segment_index=segment_index,
        batch_size=batch_size,
    )
    model_key = str(dataset_spec["model"])
    class_count = int(dataset_spec["class_count"])
    factory = lambda: build_model(model_key, class_count=class_count)
    set_global_seed(derive_int(PROTOCOL_VERSION, "model-anchor", dataset_key, source_seed) % 2147483647)
    anchor = factory()
    anchor_sha256 = model_state_sha256(anchor)
    update = produce_worker_update(
        anchor,
        factory,
        ((batch_data, batch_target),),
        WorkerBehavior.NORMAL,
        device=device,
        learning_rate=float(effective["source_bank"]["learning_rate"]),
        momentum=float(effective["source_bank"]["momentum"]),
        local_epochs=1,
        capture_replay=True,
        max_batches=1,
    )
    proof = update.replay_proof
    if proof is None:
        raise RuntimeError("E2 segment did not emit a replay proof")
    tau = float(effective["source_bank"]["replay_tolerance"])
    clean = _finite_replay(verify_replay_proof(proof, factory, device=device, tolerance=tau))
    if int(clean["checkpoint_count"]) != 1 or clean["state_complete"] is not True:
        raise ValueError("E2 clean replay is not state complete")
    segments_per_bank = int(effective["source_bank"]["segments_per_bank"])
    bank_ids = [segment_identity(dataset_key, source_seed, index)["segment_id"] for index in range(segments_per_bank)]
    high_count = max(int(value) for value in effective["partial_segment"]["polluted_segment_counts"])
    high_set = set(polluted_segment_ids(bank_ids, dataset=dataset_key, source_seed=source_seed, polluted_count=high_count))
    attack_seed = derive_int(PROTOCOL_VERSION, "attack-seed", dataset_key, source_seed)
    attack = None
    mutated_proof = None
    mutation_witness = None
    if segment_id in high_set:
        mutated_proof, mutation_witness = mutate_one_step_proof(
            proof,
            segment_id=segment_id,
            post_commit_seed=attack_seed,
            tolerance=tau,
            magnitude_ratio=float(effective["partial_segment"]["mutation_magnitude_over_tolerance"]),
        )
        attack = {
            "mutation_witness": mutation_witness,
            "replay": _finite_replay(verify_replay_proof(mutated_proof, factory, device=device, tolerance=tau)),
        }
    tolerance_rows = []
    for trace_index, trace_seed in tolerance_assignments.get(segment_id, ()):
        if mutated_proof is None or mutation_witness is None:
            mutated_proof, mutation_witness = mutate_one_step_proof(
                proof,
                segment_id=segment_id,
                post_commit_seed=attack_seed,
                tolerance=tau,
                magnitude_ratio=float(effective["partial_segment"]["mutation_magnitude_over_tolerance"]),
            )
        for origin in effective["tolerance_stress"]["origin_classes"]:
            origin_proof = proof if origin == "honest-reference" else mutated_proof
            coordinate = None if origin == "honest-reference" else mutation_witness
            for ratio in effective["tolerance_stress"]["ratios"]:
                candidate, witness = perturb_one_step_proof(
                    origin_proof,
                    segment_id=segment_id,
                    trace_seed=int(trace_seed),
                    tolerance=tau,
                    ratio=float(ratio),
                    coordinate_witness=coordinate,
                )
                replay = _finite_replay(verify_replay_proof(candidate, factory, device=device, tolerance=tau))
                maximum = float(replay["max_abs_difference"])
                passed = bool(replay["passed"])
                tolerance_rows.append(
                    {
                        "change_id": CHANGE_ID,
                        "dataset": dataset_key,
                        "source_seed": source_seed,
                        "segment_id": segment_id,
                        "trace_index": int(trace_index),
                        "trace_seed": int(trace_seed),
                        "ratio": float(ratio),
                        "origin_class": str(origin),
                        "maximum_residual": maximum,
                        "replay_passed": passed,
                        "decision": replay_decision(maximum, tolerance=tau),
                        "false_reject": origin == "honest-reference" and not passed,
                        "false_accept": origin == "registered-invalid-4tau" and passed,
                        "perturbation_witness": witness,
                    }
                )
    return {
        "schema_version": "sevc-tdsc-e2-segment-unit-v1",
        "change_id": CHANGE_ID,
        "status": "COMPLETE",
        "identity": dict(identity),
        "profile": profile,
        "device": device,
        "worker_id": worker_id,
        "physical_gpu_id": physical_gpu_id,
        "batch_indices": list(batch_indices),
        "batch_indices_sha256": indices_sha256(batch_indices),
        "anchor_state_sha256": anchor_sha256,
        "proof_component_sha256": proof_component_hashes(proof),
        "clean_replay": clean,
        "attack_characterization": attack,
        "tolerance_rows": tolerance_rows,
        "implementation_sha256": dict(implementation),
        "completed_at_utc": _utc_now(),
    }


def _witness_exact(witness: Mapping[str, Any]) -> bool:
    try:
        before = float.fromhex(str(witness["before_hex"]))
        after = float.fromhex(str(witness["after_hex"]))
        return (
            int(witness["checkpoint_index_zero_based"]) == 0
            and math.isfinite(before)
            and math.isfinite(after)
            and after - before == float(witness["observed_delta"])
            and abs(float(witness["observed_delta"])) >= 0.75 * abs(float(witness["requested_delta"]))
        )
    except (KeyError, TypeError, ValueError):
        return False


def _dataset_evidence(dataset: str, units: Sequence[Mapping[str, Any]], effective: Mapping[str, Any]) -> dict[str, Any]:
    segments_per_bank = int(effective["source_bank"]["segments_per_bank"])
    seeds = [int(value) for value in effective["source_seeds"]]
    expected_units = segments_per_bank * len(seeds)
    if len(units) != expected_units:
        raise ValueError("E2 merged dataset segment cardinality drift")
    by_id = {str(row["identity"]["segment_id"]): row for row in units}
    if len(by_id) != len(units):
        raise ValueError("E2 duplicate merged segment ID")
    source_registry = []
    for seed in seeds:
        rows = sorted(
            (row for row in units if int(row["identity"]["source_seed"]) == seed),
            key=lambda row: int(row["identity"]["segment_index"]),
        )
        if len(rows) != segments_per_bank or [int(row["identity"]["segment_index"]) for row in rows] != list(range(segments_per_bank)):
            raise ValueError("E2 source-bank segment coverage drift")
        chain = sha256_text("".join(str(row["proof_component_sha256"]["proof_sha256"]) for row in rows))
        source_registry.append({"dataset": dataset, "source_seed": seed, "segment_count": len(rows), "aggregate_commitment_sha256": chain, "clean_replay_pass_count": sum(bool(row["clean_replay"]["passed"]) for row in rows)})
    high_count = max(int(value) for value in effective["partial_segment"]["polluted_segment_counts"])
    high_rows = [row for row in units if row.get("attack_characterization") is not None]
    expected_high = high_count * len(seeds)
    if len(high_rows) != expected_high:
        raise ValueError("E2 high-intensity characterization cardinality drift")
    tau = float(effective["source_bank"]["replay_tolerance"])
    witness_exact_count = sum(_witness_exact(row["attack_characterization"]["mutation_witness"]) for row in high_rows)
    clean_pass_count = sum(bool(row["clean_replay"]["passed"]) for row in high_rows)
    mutated_fail_count = sum(not bool(row["attack_characterization"]["replay"]["passed"]) for row in high_rows)
    residual_ratios = [float(row["attack_characterization"]["replay"]["max_abs_difference"]) / tau for row in high_rows]
    technical_pass = clean_pass_count == expected_high and witness_exact_count == expected_high
    effect_pass = mutated_fail_count == expected_high and min(residual_ratios) >= float(effective["attack_effect_gate"]["minimum_observed_residual_over_tolerance"])
    attack_gate = {
        "dataset": dataset,
        "expected_segments": expected_high,
        "clean_reference_pass_count": clean_pass_count,
        "mutated_reference_fail_count": mutated_fail_count,
        "mutation_witness_exact_count": witness_exact_count,
        "minimum_observed_residual_over_tolerance": min(residual_ratios),
        "technical_passed": technical_pass,
        "effect_passed": effect_pass,
        "passed": technical_pass and effect_pass,
    }
    partial_rows = []
    trace_seeds = [int(value) for value in effective["partial_segment"]["sampling_trace_seeds"]]
    fractions = list(zip(effective["partial_segment"]["polluted_fractions"], effective["partial_segment"]["polluted_segment_counts"]))
    for fraction, polluted_count in fractions:
        for sampled_count in effective["partial_segment"]["sampled_segment_counts"]:
            exact = exact_without_replacement_coverage(segments_per_bank, int(polluted_count), int(sampled_count))
            for trace_index, trace_seed in enumerate(trace_seeds):
                seed = seeds[trace_index % len(seeds)]
                bank_ids = [segment_identity(dataset, seed, index)["segment_id"] for index in range(segments_per_bank)]
                polluted = set(polluted_segment_ids(bank_ids, dataset=dataset, source_seed=seed, polluted_count=int(polluted_count)))
                sampled = sample_segment_ids(bank_ids, dataset=dataset, source_seed=seed, polluted_count=int(polluted_count), sampled_count=int(sampled_count), trace_seed=trace_seed)
                attacked = [value for value in sampled if value in polluted]
                attacked_detections = sum(not bool(by_id[value]["attack_characterization"]["replay"]["passed"]) for value in attacked)
                clean_failures = sum(not bool(by_id[value]["clean_replay"]["passed"]) for value in sampled if value not in polluted)
                partial_rows.append({
                    "change_id": CHANGE_ID,
                    "dataset": dataset,
                    "source_seed": seed,
                    "trace_index": trace_index,
                    "trace_seed": trace_seed,
                    "polluted_fraction": float(fraction),
                    "polluted_count": int(polluted_count),
                    "sampled_count": int(sampled_count),
                    "sampled_segment_ids": list(sampled),
                    "sampled_segment_ids_sha256": sha256_text(canonical_json_text(list(sampled))),
                    "attacked_sample_count": len(attacked),
                    "attacked_sample_detection_count": attacked_detections,
                    "clean_sample_failure_count": clean_failures,
                    "detected": attacked_detections > 0 or clean_failures > 0,
                    "exact_coverage": exact,
                    "replay_execution_count": len(sampled),
                })
    tolerance_rows = [dict(value) for row in units for value in row["tolerance_rows"]]
    partial_summary = summarize_partial_rows(partial_rows, expected_cell_size=len(trace_seeds))
    tolerance_summary = summarize_tolerance_rows(tolerance_rows, expected_cell_size=int(effective["tolerance_stress"]["traces_per_dataset_ratio_class"]))
    zero_honest = next(row for row in tolerance_summary if row["ratio"] == 0.0 and row["origin_class"] == "honest-reference")
    zero_invalid = next(row for row in tolerance_summary if row["ratio"] == 0.0 and row["origin_class"] == "registered-invalid-4tau")
    zero_anchor = {"dataset": dataset, "honest_false_reject_count": int(zero_honest["false_reject"]), "invalid_false_accept_count": int(zero_invalid["false_accept"]), "passed": int(zero_honest["false_reject"]) == 0 and int(zero_invalid["false_accept"]) == 0}
    return {"source_registry": source_registry, "segment_rows": [dict(row) for row in units], "attack_gate": attack_gate, "partial_rows": partial_rows, "partial_summary": partial_summary, "tolerance_rows": tolerance_rows, "tolerance_summary": tolerance_summary, "zero_anchor": zero_anchor}
