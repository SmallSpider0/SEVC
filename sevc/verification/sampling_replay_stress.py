"""Frozen partial-segment sampling and single-host replay-tolerance primitives."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import math
from statistics import median
from typing import Any, Mapping, Sequence

import torch

from sevc.core.artifacts import canonical_json_text, sha256_text
from sevc.evaluation.statistics import wilson_interval
from sevc.training import ReplayProof, verify_replay_proof


PROTOCOL_VERSION = "sevc-tdsc-e2-sampling-replay-stress-v1"
A3_VARIANTS = (
    "fixed-tau-current",
    "scale-aware-envelope",
    "fp64-canonical-appeal",
    "pathwise-commitment-appeal",
    "envelope-fp64-pathwise",
)


def derive_int(*parts: object) -> int:
    material = "|".join(str(value) for value in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def segment_identity(dataset: str, source_seed: int, segment_index: int) -> dict[str, Any]:
    if not dataset or source_seed < 0 or not 0 <= segment_index < 120:
        raise ValueError("invalid E2 segment identity")
    identity = {
        "dataset": str(dataset),
        "source_seed": int(source_seed),
        "segment_index": int(segment_index),
    }
    identity["segment_id"] = sha256_text(
        canonical_json_text({"protocol": PROTOCOL_VERSION, **identity})
    )[:24]
    return identity


def exact_without_replacement_coverage(total: int, polluted: int, sampled: int) -> float:
    if total <= 0 or not 0 <= polluted <= total or not 0 <= sampled <= total:
        raise ValueError("invalid without-replacement cardinality")
    if polluted == 0 or sampled == 0:
        return 0.0
    clean = total - polluted
    if sampled > clean:
        return 1.0
    return float(1.0 - math.comb(clean, sampled) / math.comb(total, sampled))


def pollution_order(
    segment_ids: Sequence[str], *, dataset: str, source_seed: int
) -> tuple[str, ...]:
    values = tuple(str(value) for value in segment_ids)
    if not values or len(set(values)) != len(values):
        raise ValueError("pollution ordering requires unique segments")
    return tuple(
        sorted(
            values,
            key=lambda value: sha256_text(
                f"{PROTOCOL_VERSION}|pollution-order|{dataset}|{source_seed}|{value}"
            ),
        )
    )


def polluted_segment_ids(
    segment_ids: Sequence[str], *, dataset: str, source_seed: int, polluted_count: int
) -> tuple[str, ...]:
    if not 0 < polluted_count <= len(segment_ids):
        raise ValueError("polluted segment count is outside the segment bank")
    return pollution_order(
        segment_ids, dataset=dataset, source_seed=source_seed
    )[:polluted_count]


def sample_segment_ids(
    segment_ids: Sequence[str],
    *,
    dataset: str,
    source_seed: int,
    polluted_count: int,
    sampled_count: int,
    trace_seed: int,
) -> tuple[str, ...]:
    values = tuple(str(value) for value in segment_ids)
    if not values or len(set(values)) != len(values):
        raise ValueError("sampling requires unique committed segments")
    if not 0 < sampled_count <= len(values) or trace_seed < 0:
        raise ValueError("sampling request is outside the segment bank")
    ordered = sorted(
        values,
        key=lambda value: sha256_text(
            f"{PROTOCOL_VERSION}|sample-without-replacement|{dataset}|"
            f"{source_seed}|{polluted_count}|{sampled_count}|{trace_seed}|{value}"
        ),
    )
    return tuple(ordered[:sampled_count])


def tolerance_segment_id(
    segment_ids: Sequence[str],
    *,
    dataset: str,
    source_seed: int,
    trace_seed: int,
) -> str:
    values = tuple(str(value) for value in segment_ids)
    if not values or len(set(values)) != len(values) or trace_seed < 0:
        raise ValueError("tolerance selection requires unique committed segments")
    return min(
        values,
        key=lambda value: sha256_text(
            f"{PROTOCOL_VERSION}|tolerance-segment|{dataset}|"
            f"{source_seed}|{trace_seed}|{value}"
        ),
    )


def _precision_safe_coordinate(
    checkpoint: Mapping[str, torch.Tensor],
    *,
    selector_parts: Sequence[object],
    requested_delta: float,
) -> tuple[str, int, float, float, int]:
    if not math.isfinite(requested_delta) or requested_delta == 0.0:
        raise ValueError("precision-safe selection requires a finite non-zero delta")
    floating = tuple(
        key
        for key in sorted(checkpoint)
        if checkpoint[key].is_floating_point() and checkpoint[key].numel() > 0
    )
    if not floating:
        raise ValueError("replay checkpoint has no mutable floating coordinate")
    for attempt in range(4096):
        selector = derive_int(*selector_parts, attempt)
        key = floating[selector % len(floating)]
        flat = checkpoint[key].detach().reshape(-1)
        coordinate = (selector // len(floating)) % flat.numel()
        before_tensor = flat[coordinate]
        after_tensor = before_tensor + torch.as_tensor(
            requested_delta, dtype=before_tensor.dtype
        )
        before = float(before_tensor.to(torch.float64))
        after = float(after_tensor.to(torch.float64))
        observed = after - before
        if math.isfinite(observed) and abs(observed) >= 0.75 * abs(requested_delta):
            return key, int(coordinate), before, after, attempt + 1
    raise ValueError("no precision-safe replay checkpoint coordinate was found")


def mutate_one_step_proof(
    proof: ReplayProof,
    *,
    segment_id: str,
    post_commit_seed: int,
    tolerance: float,
    magnitude_ratio: float = 4.0,
) -> tuple[ReplayProof, dict[str, Any]]:
    if len(proof.checkpoints) != 1 or len(proof.batches) != 1:
        raise ValueError("E2 registered mutation requires a one-step replay proof")
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("registered replay tolerance must be finite and positive")
    if not math.isfinite(magnitude_ratio) or magnitude_ratio <= 0.0:
        raise ValueError("registered mutation ratio must be finite and positive")
    sign = 1 if derive_int(PROTOCOL_VERSION, "attack-sign", segment_id, post_commit_seed) % 2 else -1
    requested = sign * magnitude_ratio * tolerance
    checkpoint = proof.checkpoints[0]
    key, coordinate, before, after, attempts = _precision_safe_coordinate(
        checkpoint,
        selector_parts=(PROTOCOL_VERSION, "attack-coordinate", segment_id, post_commit_seed),
        requested_delta=requested,
    )
    changed = dict(checkpoint)
    tensor = changed[key].detach().clone()
    tensor.reshape(-1)[coordinate] = torch.as_tensor(after, dtype=tensor.dtype)
    changed[key] = tensor
    mutated = replace(proof, checkpoints=(changed,))
    witness = {
        "schema_version": "sevc-tdsc-e2-checkpoint-mutation-witness-v1",
        "segment_id": segment_id,
        "checkpoint_index_zero_based": 0,
        "tensor_key": key,
        "flat_coordinate": coordinate,
        "requested_delta": requested,
        "observed_delta": after - before,
        "before_hex": before.hex(),
        "after_hex": after.hex(),
        "magnitude_over_tolerance": magnitude_ratio,
        "precision_safe_attempt_count": attempts,
        "post_commit_prf_sha256": sha256_text(
            f"{PROTOCOL_VERSION}|attack|{segment_id}|{post_commit_seed}"
        ),
    }
    return mutated, witness


def perturb_one_step_proof(
    proof: ReplayProof,
    *,
    segment_id: str,
    trace_seed: int,
    tolerance: float,
    ratio: float,
    coordinate_witness: Mapping[str, Any] | None = None,
) -> tuple[ReplayProof, dict[str, Any]]:
    if len(proof.checkpoints) != 1 or len(proof.batches) != 1:
        raise ValueError("E2 tolerance perturbation requires a one-step replay proof")
    if ratio not in {0.0, 0.25, 0.5, 1.0, 2.0, 4.0}:
        raise ValueError("tolerance ratio is outside the frozen E2 matrix")
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("registered replay tolerance must be finite and positive")
    sign = 1 if derive_int(PROTOCOL_VERSION, "tolerance-sign", segment_id, trace_seed) % 2 else -1
    requested = sign * ratio * tolerance
    checkpoint = proof.checkpoints[0]
    if ratio == 0.0:
        return proof, {
            "schema_version": "sevc-tdsc-e2-tolerance-witness-v1",
            "segment_id": segment_id,
            "tensor_key": None,
            "flat_coordinate": None,
            "requested_delta": 0.0,
            "observed_delta": 0.0,
            "before_hex": None,
            "after_hex": None,
            "ratio": 0.0,
            "sign_prf_sha256": sha256_text(
                f"{PROTOCOL_VERSION}|tolerance-sign|{segment_id}|{trace_seed}"
            ),
        }
    if coordinate_witness is None:
        key, coordinate, before, after, attempts = _precision_safe_coordinate(
            checkpoint,
            selector_parts=(PROTOCOL_VERSION, "tolerance-coordinate", segment_id, trace_seed),
            requested_delta=requested,
        )
    else:
        key = str(coordinate_witness["tensor_key"])
        coordinate = int(coordinate_witness["flat_coordinate"])
        source = checkpoint[key].detach().reshape(-1)
        before_tensor = source[coordinate]
        after_tensor = before_tensor + torch.as_tensor(
            requested, dtype=before_tensor.dtype
        )
        before = float(before_tensor.to(torch.float64))
        after = float(after_tensor.to(torch.float64))
        attempts = 1
        if not math.isfinite(after - before):
            raise ValueError("tolerance perturbation produced a non-finite witness")
    changed = dict(checkpoint)
    tensor = changed[key].detach().clone()
    tensor.reshape(-1)[coordinate] = torch.as_tensor(after, dtype=tensor.dtype)
    changed[key] = tensor
    perturbed = replace(proof, checkpoints=(changed,))
    return perturbed, {
        "schema_version": "sevc-tdsc-e2-tolerance-witness-v1",
        "segment_id": segment_id,
        "tensor_key": key,
        "flat_coordinate": coordinate,
        "requested_delta": requested,
        "observed_delta": after - before,
        "before_hex": before.hex(),
        "after_hex": after.hex(),
        "ratio": ratio,
        "precision_safe_attempt_count": attempts,
        "sign_prf_sha256": sha256_text(
            f"{PROTOCOL_VERSION}|tolerance-sign|{segment_id}|{trace_seed}"
        ),
    }


def replay_decision(maximum_residual: float, *, tolerance: float) -> str:
    if not math.isfinite(maximum_residual) or maximum_residual < 0.0:
        return "hard-reject"
    if maximum_residual <= tolerance:
        return "accept"
    if maximum_residual <= 2.0 * tolerance:
        return "appeal-ambiguous"
    return "hard-reject"


def replay_residual_scale(
    *, tolerance: float, state_norm: float, step: int, dtype_epsilon: float = 2.0**-23
) -> float:
    """Return the public dtype/state/step envelope denominator used by A3."""

    if (
        not math.isfinite(tolerance)
        or tolerance <= 0.0
        or not math.isfinite(state_norm)
        or state_norm < 0.0
        or step <= 0
        or not math.isfinite(dtype_epsilon)
        or dtype_epsilon <= 0.0
    ):
        raise ValueError("invalid replay residual scale input")
    return float(
        tolerance
        + dtype_epsilon * max(1.0, state_norm)
        + dtype_epsilon * tolerance * step
    )


def fit_canonical_appeal_envelope(
    honest_normalized_residuals: Sequence[float],
    invalid_normalized_residuals: Sequence[float],
    *,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Fit the frozen conformal honest bound and conservative attack separator."""

    honest = sorted(float(value) for value in honest_normalized_residuals)
    invalid = sorted(float(value) for value in invalid_normalized_residuals)
    if not honest or not invalid or not 0.5 < confidence < 1.0:
        raise ValueError("A3 envelope fitting requires two non-empty calibration classes")
    if any(not math.isfinite(value) or value < 0.0 for value in (*honest, *invalid)):
        raise ValueError("A3 envelope calibration residual is invalid")
    # Split-conformal one-sided upper order statistic.  Clipping to the maximum
    # is the conservative finite-sample rule when ceil((n+1)q) is n+1.
    upper_index = min(len(honest) - 1, math.ceil((len(honest) + 1) * confidence) - 1)
    honest_upper = honest[upper_index]
    attack_lower = invalid[0]
    separated = attack_lower > honest_upper
    boundary = (honest_upper + attack_lower) / 2.0 if separated else None
    return {
        "schema_version": "sevc-tdsc-a3-envelope-v1",
        "confidence": float(confidence),
        "honest_calibration_count": len(honest),
        "invalid_calibration_count": len(invalid),
        "honest_upper": honest_upper,
        "attack_lower": attack_lower,
        "decision_boundary": boundary,
        "separated": separated,
    }


def _cast_tensor(value: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    result = value.detach().cpu().clone()
    return result.to(dtype=dtype) if result.is_floating_point() else result


def _batch_sequence_sha256(
    batches: Sequence[tuple[torch.Tensor, torch.Tensor]],
) -> str:
    digest = hashlib.sha256(b"sevc-replay-data-order-v1")
    for index, (data, target) in enumerate(batches):
        digest.update(str(index).encode("utf-8"))
        for value in (data, target):
            tensor = value.detach().cpu().contiguous()
            digest.update(str(tensor.dtype).encode("utf-8"))
            digest.update(str(tuple(tensor.shape)).encode("utf-8"))
            digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def cast_replay_proof_dtype(
    proof: ReplayProof, *, dtype: torch.dtype
) -> ReplayProof:
    """Cast every floating replay state and input while preserving proof identity."""

    if dtype not in {torch.float32, torch.float64}:
        raise ValueError("A3 canonical replay supports only float32/float64")
    batches = tuple(
        (_cast_tensor(data, dtype), _cast_tensor(target, dtype))
        for data, target in proof.batches
    )
    cast_state = lambda state: {
        key: _cast_tensor(value, dtype) for key, value in state.items()
    }
    optimizer_initial = (
        None
        if proof.optimizer_initial_state is None
        else cast_state(proof.optimizer_initial_state)
    )
    return replace(
        proof,
        initial_state=cast_state(proof.initial_state),
        batches=batches,
        checkpoints=tuple(cast_state(state) for state in proof.checkpoints),
        optimizer_initial_state=optimizer_initial,
        optimizer_checkpoints=tuple(
            cast_state(state) for state in proof.optimizer_checkpoints
        ),
        data_order_sha256=_batch_sequence_sha256(batches),
    )


def canonical_fp64_replay(
    proof: ReplayProof,
    model_factory: Any,
    *,
    tolerance: float,
) -> dict[str, Any]:
    """Execute the A3 canonical CPU FP64 appeal through the shared replay engine."""

    cast = cast_replay_proof_dtype(proof, dtype=torch.float64)

    def fp64_factory() -> torch.nn.Module:
        return model_factory().to(dtype=torch.float64)

    result = dict(
        verify_replay_proof(
            cast,
            fp64_factory,
            device="cpu",
            tolerance=tolerance,
        )
    )
    result["appeal_dtype"] = "float64"
    result["appeal_device"] = "cpu"
    return result


def canonical_appeal_decision(
    variant: str,
    *,
    fast_normalized_residuals: Sequence[float],
    fp64_normalized_residual: float,
    envelope: Mapping[str, Any],
    maximum_fast_residual: float,
    tolerance: float,
    path_attack_witness: bool,
    technical_valid: bool = True,
) -> dict[str, Any]:
    """Return the mutually exclusive A3 accept/reject/unresolved decision."""

    if variant not in A3_VARIANTS:
        raise KeyError(variant)
    values = tuple(float(value) for value in fast_normalized_residuals)
    finite = (
        values
        and all(math.isfinite(value) and value >= 0.0 for value in values)
        and math.isfinite(fp64_normalized_residual)
        and fp64_normalized_residual >= 0.0
        and math.isfinite(maximum_fast_residual)
        and maximum_fast_residual >= 0.0
    )
    if not technical_valid or not finite:
        return {
            "decision": "unresolved",
            "route": "fail-closed-technical",
            "appealed": False,
        }
    honest_upper = float(envelope["honest_upper"])
    boundary_value = envelope.get("decision_boundary")
    boundary = None if boundary_value is None else float(boundary_value)
    maximum = max(values)
    if variant == "fixed-tau-current":
        current = replay_decision(maximum_fast_residual, tolerance=tolerance)
        return {
            "decision": (
                "accept"
                if current == "accept"
                else "reject"
                if current == "hard-reject"
                else "unresolved"
            ),
            "route": f"current-{current}",
            "appealed": current == "appeal-ambiguous",
        }
    if maximum <= honest_upper:
        return {"decision": "accept", "route": "fast-envelope-accept", "appealed": False}
    pathwise = variant in {
        "pathwise-commitment-appeal",
        "envelope-fp64-pathwise",
    }
    if pathwise and path_attack_witness and boundary is not None:
        return {"decision": "reject", "route": "pathwise-attack-reject", "appealed": False}
    appeal_enabled = variant in {
        "fp64-canonical-appeal",
        "envelope-fp64-pathwise",
    }
    if appeal_enabled:
        if fp64_normalized_residual <= honest_upper:
            return {"decision": "accept", "route": "fp64-appeal-accept", "appealed": True}
        if boundary is not None and fp64_normalized_residual >= boundary:
            return {"decision": "reject", "route": "fp64-appeal-reject", "appealed": True}
        return {"decision": "unresolved", "route": "fp64-appeal-unresolved", "appealed": True}
    if boundary is not None and maximum >= boundary:
        return {"decision": "reject", "route": "envelope-hard-reject", "appealed": False}
    return {"decision": "unresolved", "route": "envelope-unresolved", "appealed": False}


def summarize_partial_rows(
    rows: Sequence[Mapping[str, Any]], *, expected_cell_size: int = 30
) -> list[dict[str, Any]]:
    if expected_cell_size <= 0:
        raise ValueError("partial summary cell size must be positive")
    groups: dict[tuple[str, float, int], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = (str(row["dataset"]), float(row["polluted_fraction"]), int(row["sampled_count"]))
        groups.setdefault(key, []).append(row)
    result = []
    for (dataset, fraction, sampled), values in sorted(groups.items()):
        if len(values) != expected_cell_size:
            raise ValueError("partial summary cell has unexpected trace cardinality")
        detected = sum(bool(row["detected"]) for row in values)
        interval = wilson_interval(detected, len(values))
        exact_values = {float(row["exact_coverage"]) for row in values}
        if len(exact_values) != 1:
            raise ValueError("partial summary cell has inconsistent exact coverage")
        exact = next(iter(exact_values))
        by_seed = {}
        for source_seed in sorted({int(row["source_seed"]) for row in values}):
            subset = [row for row in values if int(row["source_seed"]) == source_seed]
            by_seed[str(source_seed)] = {
                "detected": sum(bool(row["detected"]) for row in subset),
                "traces": len(subset),
            }
        rates = [item["detected"] / item["traces"] for item in by_seed.values()]
        result.append(
            {
                "dataset": dataset,
                "polluted_fraction": fraction,
                "sampled_count": sampled,
                "detected": detected,
                "traces": len(values),
                "empirical_detection_rate": interval.point_estimate,
                "wilson95": {"lower": interval.lower, "upper": interval.upper},
                "exact_coverage": exact,
                "detection_minus_coverage": interval.point_estimate - exact,
                "clean_sample_failure_count": sum(int(row["clean_sample_failure_count"]) for row in values),
                "source_seed_counts": by_seed,
                "source_seed_rate_median": median(rates),
                "source_seed_rate_range": [min(rates), max(rates)],
                "replay_execution_count": sum(int(row["replay_execution_count"]) for row in values),
            }
        )
    return result


def summarize_tolerance_rows(
    rows: Sequence[Mapping[str, Any]], *, expected_cell_size: int = 30
) -> list[dict[str, Any]]:
    if expected_cell_size <= 0:
        raise ValueError("tolerance summary cell size must be positive")
    groups: dict[tuple[str, float, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = (str(row["dataset"]), float(row["ratio"]), str(row["origin_class"]))
        groups.setdefault(key, []).append(row)
    result = []
    for (dataset, ratio, origin), values in sorted(groups.items()):
        if len(values) != expected_cell_size:
            raise ValueError("tolerance summary cell has unexpected trace cardinality")
        false_reject = sum(bool(row["false_reject"]) for row in values)
        false_accept = sum(bool(row["false_accept"]) for row in values)
        appeals = sum(row["decision"] == "appeal-ambiguous" for row in values)
        rejects = sum(row["decision"] == "hard-reject" for row in values)
        residuals = [float(row["maximum_residual"]) for row in values]
        result.append(
            {
                "dataset": dataset,
                "ratio": ratio,
                "origin_class": origin,
                "traces": len(values),
                "false_reject": false_reject,
                "false_reject_wilson95": _interval_payload(false_reject, len(values)),
                "false_accept": false_accept,
                "false_accept_wilson95": _interval_payload(false_accept, len(values)),
                "appeal_ambiguous": appeals,
                "appeal_wilson95": _interval_payload(appeals, len(values)),
                "hard_reject": rejects,
                "residual_median": median(residuals),
                "residual_range": [min(residuals), max(residuals)],
            }
        )
    return result


def _interval_payload(successes: int, total: int) -> dict[str, float | int]:
    interval = wilson_interval(successes, total)
    return {
        "numerator": interval.numerator,
        "denominator": interval.denominator,
        "point_estimate": interval.point_estimate,
        "lower": interval.lower,
        "upper": interval.upper,
    }


__all__ = [
    "A3_VARIANTS",
    "PROTOCOL_VERSION",
    "canonical_appeal_decision",
    "canonical_fp64_replay",
    "cast_replay_proof_dtype",
    "derive_int",
    "exact_without_replacement_coverage",
    "fit_canonical_appeal_envelope",
    "mutate_one_step_proof",
    "perturb_one_step_proof",
    "polluted_segment_ids",
    "pollution_order",
    "replay_decision",
    "replay_residual_scale",
    "sample_segment_ids",
    "segment_identity",
    "summarize_partial_rows",
    "summarize_tolerance_rows",
    "tolerance_segment_id",
]
