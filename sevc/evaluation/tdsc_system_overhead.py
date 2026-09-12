"""Shared accounting, byte encoding and gate aggregation for RCMP system overhead.

This module owns no compilation, replay, wrapper, mutation or certificate behaviour.
It consumes rows produced by the registered RCMP seams and turns them into the two
frozen endpoints, their intervals, the validity gates and the terminal route.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


CANONICAL_PAYLOAD_SCHEMA = "sevc-tdsc-system-overhead-canonical-payload-v1"
PROTOCOL_VERSION = "tdsc-system-overhead-v1"

TIMED_PHASES = (
    "owner_compile",
    "matched_replay",
    "production_replay",
)
PAYLOAD_DIRECTIONS = (
    "trainer_commitment_upload",
    "owner_to_verifier_production_delivery",
    "owner_to_verifier_probe_delivery",
    "verifier_to_owner_report_return",
)
RETENTION_ROLES = (
    "checkpoint_chain",
    "committed_segment",
    "owner_reference_material",
    "certificate",
    "report",
)


# --------------------------------------------------------------------------- #
# Canonical byte encoding
# --------------------------------------------------------------------------- #


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _tensor_record(name: str, tensor: Any) -> Iterable[bytes]:
    value = tensor.detach().cpu().contiguous()
    header = canonical_json_bytes(
        {"k": name, "dtype": str(value.dtype), "shape": list(value.shape)}
    )
    raw = memoryview(value.numpy()).cast("B")
    yield len(header).to_bytes(8, "big")
    yield header
    yield len(raw).to_bytes(8, "big")
    yield raw


def _state_records(label: str, state: Mapping[str, Any]) -> Iterable[bytes]:
    yield canonical_json_bytes({"section": label, "keys": len(state)})
    for key in sorted(state):
        yield from _tensor_record(key, state[key])


def canonical_state_stream(state: Mapping[str, Any]) -> Iterable[bytes]:
    """Stream one name-keyed tensor state deterministically."""

    yield CANONICAL_PAYLOAD_SCHEMA.encode("utf-8")
    yield b"\x00"
    yield from _state_records("state", state)


def canonical_proof_stream(proof: Any) -> Iterable[bytes]:
    """Stream one replay proof deterministically without touching its identity hashes."""

    yield CANONICAL_PAYLOAD_SCHEMA.encode("utf-8")
    yield b"\x00"
    yield canonical_json_bytes(
        {
            "schema_version": getattr(proof, "schema_version", None),
            "learning_rate": float(proof.learning_rate),
            "momentum": float(proof.momentum),
            "rng_state_sha256": getattr(proof, "rng_state_sha256", None),
            "data_order_sha256": getattr(proof, "data_order_sha256", None),
            "criterion_key": getattr(proof, "criterion_key", None),
            "checkpoint_count": len(proof.checkpoints),
            "batch_count": len(proof.batches),
        }
    )
    yield from _state_records("initial_state", proof.initial_state)
    for index, checkpoint in enumerate(proof.checkpoints):
        yield from _state_records(f"checkpoint/{index}", checkpoint)
    optimizer_initial = getattr(proof, "optimizer_initial_state", None)
    if optimizer_initial is not None:
        yield from _state_records("optimizer_initial_state", optimizer_initial)
    for index, checkpoint in enumerate(getattr(proof, "optimizer_checkpoints", ())):
        yield from _state_records(f"optimizer_checkpoint/{index}", checkpoint)
    for index, (inputs, targets) in enumerate(proof.batches):
        yield from _tensor_record(f"batch/{index}/inputs", inputs)
        yield from _tensor_record(f"batch/{index}/targets", targets)


def canonical_payload_stream(payload: Any) -> Iterable[bytes]:
    """Return the one canonical byte encoding used for every accounted transfer."""

    if isinstance(payload, (bytes, bytearray)):
        yield bytes(payload)
        return
    if hasattr(payload, "initial_state") and hasattr(payload, "checkpoints"):
        yield from canonical_proof_stream(payload)
        return
    if isinstance(payload, Mapping) and payload and all(
        hasattr(value, "detach") for value in payload.values()
    ):
        yield from canonical_state_stream(payload)
        return
    yield CANONICAL_PAYLOAD_SCHEMA.encode("utf-8")
    yield b"\x00"
    yield canonical_json_bytes(payload)


def canonical_payload_bytes(payload: Any) -> bytes:
    """Materialize the canonical encoding; prefer the streaming measurement helpers."""

    return b"".join(canonical_payload_stream(payload))


def measure_canonical_payload(payload: Any) -> tuple[int, str]:
    """Return canonical length and digest without materializing the whole payload."""

    digest = hashlib.sha256()
    length = 0
    for chunk in canonical_payload_stream(payload):
        digest.update(chunk)
        length += len(chunk)
    return length, digest.hexdigest()


def write_canonical_payload(payload: Any, path: "Any") -> tuple[int, str, int]:
    """Serialize to disk so that canonical length can be checked against `st_size`."""

    digest = hashlib.sha256()
    length = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        for chunk in canonical_payload_stream(payload):
            handle.write(chunk)
            digest.update(chunk)
            length += len(chunk)
        handle.flush()
    return length, digest.hexdigest(), int(path.stat().st_size)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


# --------------------------------------------------------------------------- #
# Sequential, non-overlapping phase ledger
# --------------------------------------------------------------------------- #


@dataclass
class PhaseLedger:
    """Record disjoint sequential phases so that overlap is structurally impossible."""

    job_start_ns: int
    entries: list[dict[str, Any]] = field(default_factory=list)
    _cursor_ns: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self._cursor_ns = int(self.job_start_ns)

    def record(self, name: str, start_ns: int, end_ns: int) -> None:
        if end_ns < start_ns:
            raise ValueError(f"phase {name!r} ends before it starts")
        if start_ns < self._cursor_ns:
            raise ValueError(f"phase {name!r} overlaps the preceding phase")
        self.entries.append(
            {"phase": name, "start_ns": int(start_ns), "end_ns": int(end_ns)}
        )
        self._cursor_ns = int(end_ns)

    def measure(self, name: str):
        return _PhaseTimer(self, name)

    def totals(self, job_end_ns: int) -> dict[str, Any]:
        if job_end_ns < self._cursor_ns:
            raise ValueError("job wall-clock ends before the last recorded phase")
        named: dict[str, float] = {}
        for entry in self.entries:
            seconds = (entry["end_ns"] - entry["start_ns"]) / 1e9
            named[entry["phase"]] = named.get(entry["phase"], 0.0) + seconds
        wall = (int(job_end_ns) - int(self.job_start_ns)) / 1e9
        named_total = float(sum(named.values()))
        return {
            "named_phase_seconds": named,
            "named_phase_total_seconds": named_total,
            "residual_seconds": wall - named_total,
            "job_wall_seconds": wall,
            "phase_count": len(self.entries),
        }


class _PhaseTimer:
    def __init__(self, ledger: PhaseLedger, name: str) -> None:
        self._ledger = ledger
        self._name = name
        self._start_ns = 0
        self.seconds = 0.0

    def __enter__(self) -> "_PhaseTimer":
        self._start_ns = time.perf_counter_ns()
        return self

    def __exit__(self, *_exc: Any) -> None:
        end_ns = time.perf_counter_ns()
        self._ledger.record(self._name, self._start_ns, end_ns)
        self.seconds = (end_ns - self._start_ns) / 1e9


def closure_tolerance_seconds(job_wall_seconds: float) -> float:
    return max(0.001, 0.01 * float(job_wall_seconds))


def evaluate_closure(totals: Mapping[str, Any]) -> dict[str, Any]:
    """Accounting closes when no phase overlaps and the residual is non-negative."""

    wall = float(totals["job_wall_seconds"])
    residual = float(totals["residual_seconds"])
    named_total = float(totals["named_phase_total_seconds"])
    tolerance = closure_tolerance_seconds(wall)
    reconstructed = named_total + residual
    return {
        "job_wall_seconds": wall,
        "named_phase_total_seconds": named_total,
        "residual_seconds": residual,
        "residual_fraction": residual / wall if wall > 0.0 else math.inf,
        "closure_tolerance_seconds": tolerance,
        "closure_absolute_error_seconds": abs(reconstructed - wall),
        "passed": bool(
            abs(reconstructed - wall) <= tolerance and residual >= -tolerance
        ),
    }


# --------------------------------------------------------------------------- #
# Repetition aggregation and endpoint estimation
# --------------------------------------------------------------------------- #


def aggregate_repetitions(
    rows: Sequence[Mapping[str, Any]],
    *,
    key_fields: Sequence[str],
    value_field: str,
    expected_repetitions: Sequence[int],
) -> dict[tuple, float]:
    """Average the retained timed repetitions inside each measured unit."""

    expected = tuple(int(value) for value in expected_repetitions)
    if not expected or len(set(expected)) != len(expected):
        raise ValueError("expected repetitions must be unique and nonempty")
    grouped: dict[tuple, dict[int, float]] = {}
    for row in rows:
        key = tuple(row[field] for field in key_fields)
        repetition = int(row["repetition"])
        seconds = float(row[value_field])
        if not math.isfinite(seconds) or seconds < 0.0:
            raise ValueError(f"nonfinite or negative timing row: {key}|{repetition}")
        bucket = grouped.setdefault(key, {})
        if repetition in bucket:
            raise ValueError(f"duplicate repetition row: {key}|{repetition}")
        bucket[repetition] = seconds
    result: dict[tuple, float] = {}
    for key, bucket in grouped.items():
        if tuple(sorted(bucket)) != tuple(sorted(expected)):
            raise ValueError(f"incomplete repetition set for unit: {key}")
        result[key] = float(sum(bucket.values()) / len(bucket))
    return result


def ratio_of_sums(
    numerators: Sequence[float], denominators: Sequence[float]
) -> float:
    numerator = float(np.sum(np.asarray(numerators, dtype=np.float64)))
    denominator = float(np.sum(np.asarray(denominators, dtype=np.float64)))
    if denominator <= 0.0:
        raise ValueError("ratio-of-sums denominator must be positive")
    return numerator / denominator


def _paired_block_bootstrap(
    numerators: np.ndarray,
    denominators: np.ndarray,
    *,
    seed: int,
    resamples: int,
) -> np.ndarray:
    if numerators.shape != denominators.shape or numerators.ndim != 1:
        raise ValueError("paired bootstrap requires aligned one-dimensional blocks")
    if numerators.size == 0:
        raise ValueError("paired bootstrap requires at least one block")
    if resamples <= 0:
        raise ValueError("paired bootstrap requires a positive resample count")
    rng = np.random.default_rng(int(seed))
    draws = np.empty(int(resamples), dtype=np.float64)
    cursor = 0
    while cursor < int(resamples):
        count = min(500, int(resamples) - cursor)
        sampled = rng.integers(0, numerators.size, size=(count, numerators.size))
        draws[cursor : cursor + count] = (
            numerators[sampled].sum(axis=1) / denominators[sampled].sum(axis=1)
        )
        cursor += count
    return draws


def endpoint_estimate(
    block_numerators: Sequence[float],
    block_denominators: Sequence[float],
    *,
    seed: int,
    resamples: int,
    cell_quantile: float,
    descriptive_alpha: float = 0.05,
) -> dict[str, Any]:
    """Estimate one endpoint with its simultaneous and descriptive bounds."""

    numerators = np.asarray(block_numerators, dtype=np.float64)
    denominators = np.asarray(block_denominators, dtype=np.float64)
    if np.any(~np.isfinite(numerators)) or np.any(~np.isfinite(denominators)):
        raise ValueError("endpoint rows contain nonfinite seconds")
    if np.any(numerators < 0.0) or np.any(denominators <= 0.0):
        raise ValueError("endpoint rows contain invalid seconds")
    if not 0.0 < cell_quantile < 1.0:
        raise ValueError("cell quantile must lie strictly inside (0, 1)")
    draws = _paired_block_bootstrap(
        numerators, denominators, seed=seed, resamples=resamples
    )
    point = float(numerators.sum() / denominators.sum())
    return {
        "block_count": int(numerators.size),
        "numerator_seconds_total": float(numerators.sum()),
        "denominator_seconds_total": float(denominators.sum()),
        "point_estimator": "ratio_of_sums_across_blocks",
        "point": point,
        "bootstrap_seed": int(seed),
        "bootstrap_resamples": int(resamples),
        "cell_quantile": float(cell_quantile),
        "simultaneous_one_sided_upper": float(
            np.quantile(draws, cell_quantile, method="higher")
        ),
        "descriptive_two_sided_lower": float(
            np.quantile(draws, descriptive_alpha / 2.0, method="lower")
        ),
        "descriptive_two_sided_upper": float(
            np.quantile(draws, 1.0 - descriptive_alpha / 2.0, method="higher")
        ),
    }


def bonferroni_cell_quantile(*, familywise_alpha: float, cell_count: int) -> float:
    if not 0.0 < familywise_alpha < 1.0 or cell_count <= 0:
        raise ValueError("invalid familywise correction parameters")
    return 1.0 - familywise_alpha / float(cell_count)


# --------------------------------------------------------------------------- #
# Settlement identity used only as an observer-effect invariant
# --------------------------------------------------------------------------- #


def settlement_identity(
    rows: Sequence[Mapping[str, Any]], *, failure_threshold: int
) -> dict[str, Any]:
    """Derive one block-level settlement route from measured probe verdicts."""

    probes = [row for row in rows if str(row["role"]) in {"control", "challenge"}]
    if not probes:
        raise ValueError("settlement identity requires at least one probe row")
    mismatches = sum(
        int(bool(row["canonical_verdict"]) != bool(row["expected_verdict"]))
        for row in probes
    )
    production = [row for row in rows if str(row["role"]) == "production"]
    execution_supported = all(bool(row["canonical_verdict"]) for row in production)
    if mismatches >= int(failure_threshold):
        route = "reject"
    elif execution_supported:
        route = "settle"
    else:
        route = "defer"
    return {
        "probe_count": len(probes),
        "production_count": len(production),
        "probe_mismatch_count": mismatches,
        "failure_threshold": int(failure_threshold),
        "service_supported": mismatches < int(failure_threshold),
        "execution_supported": bool(execution_supported),
        "trainer_decision": "submit_committed_segment",
        "settlement_route": route,
    }


# --------------------------------------------------------------------------- #
# Byte, retention and ledger accounting
# --------------------------------------------------------------------------- #


def accumulate_bytes(
    rows: Sequence[Mapping[str, Any]],
    *,
    field_name: str,
    categories: Sequence[str],
) -> tuple[dict[str, int], int]:
    """Total canonical bytes by category and count the physical cross-checks."""

    totals = {category: 0 for category in categories}
    cross_checked = 0
    for row in rows:
        category = str(row[field_name])
        if category not in totals:
            raise ValueError(f"unregistered byte category: {category}")
        value = int(row["canonical_bytes"])
        if value < 0:
            raise ValueError("canonical byte counts must be non-negative")
        physical = row.get("physical_st_size")
        if physical is not None:
            if int(physical) != value:
                raise ValueError(
                    f"canonical bytes disagree with serialized st_size: {category}"
                )
            cross_checked += 1
        totals[category] += value
    return totals, cross_checked


def byte_accounting_summary(
    payload_rows: Sequence[Mapping[str, Any]],
    retention_rows: Sequence[Mapping[str, Any]],
    ledger_rows: Sequence[Mapping[str, Any]],
    *,
    temporary_peak_bytes: int,
) -> dict[str, Any]:
    payload, payload_checked = accumulate_bytes(
        payload_rows, field_name="direction", categories=PAYLOAD_DIRECTIONS
    )
    retention, retention_checked = accumulate_bytes(
        retention_rows, field_name="retention_role", categories=RETENTION_ROLES
    )
    ledger_bytes = sum(int(row["canonical_bytes"]) for row in ledger_rows)
    return {
        "canonical_serializer": "sevc.evaluation.tdsc_system_overhead.canonical_payload_bytes",
        "serializer_schema_version": CANONICAL_PAYLOAD_SCHEMA,
        "application_payload_bytes_by_direction": payload,
        "application_payload_bytes_total": int(sum(payload.values())),
        "logical_retained_bytes_by_role": retention,
        "logical_retained_bytes_total": int(sum(retention.values())),
        "temporary_peak_bytes": int(temporary_peak_bytes),
        "prototype_ledger_append_count": len(ledger_rows),
        "prototype_ledger_canonical_bytes": int(ledger_bytes),
        "payload_rows": len(payload_rows),
        "retention_rows": len(retention_rows),
        "physically_cross_checked_payload_rows": int(payload_checked),
        "physically_cross_checked_retention_rows": int(retention_checked),
        "observed_network_bytes": None,
        "retransmission_count": None,
        "filesystem_allocated_bytes": None,
        "blockchain_consensus_or_gas": None,
        "interpretation": (
            "Canonical prototype bytes and records. Latency, throughput, retries, "
            "replication, consensus and gas remain unmeasured."
        ),
    }


# --------------------------------------------------------------------------- #
# Validity gates, endpoint gates and terminal routing
# --------------------------------------------------------------------------- #


def evaluate_cardinality(
    counts: Mapping[str, int], expected: Mapping[str, int]
) -> dict[str, Any]:
    mismatches = {
        key: {"observed": int(counts.get(key, -1)), "expected": int(value)}
        for key, value in expected.items()
        if int(counts.get(key, -1)) != int(value)
    }
    return {
        "expected": {key: int(value) for key, value in expected.items()},
        "observed": {key: int(counts.get(key, -1)) for key in expected},
        "mismatches": mismatches,
        "passed": not mismatches,
    }


def evaluate_dataset(
    *,
    dataset: str,
    block_rows: Sequence[Mapping[str, Any]],
    cardinality: Mapping[str, Any],
    identity: Mapping[str, Any],
    timing: Mapping[str, Any],
    closure: Mapping[str, Any],
    observer_effect: Mapping[str, Any],
    bytes_summary: Mapping[str, Any],
    seed: int,
    resamples: int,
    familywise_alpha: float,
    gate_cell_count: int,
    g1_upper_below: float,
    g2_upper_below: float,
    additional_thresholds: Mapping[str, float] | None = None,
    inherited_endpoint_invariance: bool | None = None,
) -> dict[str, Any]:
    """Aggregate one dataset into its two endpoints and its validity verdict."""

    ordered = sorted(block_rows, key=lambda row: int(row["block_seed"]))
    seeds = [int(row["block_seed"]) for row in ordered]
    if len(set(seeds)) != len(seeds):
        raise ValueError("dataset aggregation requires unique block summaries")
    quantile = bonferroni_cell_quantile(
        familywise_alpha=familywise_alpha, cell_count=gate_cell_count
    )
    g1 = endpoint_estimate(
        [float(row["g1_numerator_seconds"]) for row in ordered],
        [float(row["g1_denominator_seconds"]) for row in ordered],
        seed=seed,
        resamples=resamples,
        cell_quantile=quantile,
    )
    g2 = endpoint_estimate(
        [float(row["g2_numerator_seconds"]) for row in ordered],
        [float(row["g2_denominator_seconds"]) for row in ordered],
        seed=seed,
        resamples=resamples,
        cell_quantile=quantile,
    )
    g1["threshold_upper_below"] = float(g1_upper_below)
    g1["passed"] = bool(g1["simultaneous_one_sided_upper"] < float(g1_upper_below))
    g2["threshold_upper_below"] = float(g2_upper_below)
    g2["passed"] = bool(g2["simultaneous_one_sided_upper"] < float(g2_upper_below))
    additional = {}
    for name, threshold in (additional_thresholds or {}).items():
        estimate = endpoint_estimate(
            [float(row[f"{name.lower()}_numerator_seconds"]) for row in ordered],
            [float(row[f"{name.lower()}_denominator_seconds"]) for row in ordered],
            seed=seed, resamples=resamples, cell_quantile=quantile,
        )
        estimate["threshold_upper_below"] = float(threshold)
        estimate["passed"] = estimate["simultaneous_one_sided_upper"] < threshold
        additional[name] = estimate
    validity = {
        "source_and_identity": bool(identity["passed"]),
        "cuda_timing": bool(timing["passed"]),
        "accounting_closure": bool(closure["passed"]),
        "observer_effect": bool(observer_effect["passed"]),
        "cardinality": bool(cardinality["passed"]),
    }
    if inherited_endpoint_invariance is not None:
        validity["inherited_endpoint_invariance"] = bool(inherited_endpoint_invariance)
    evidence_valid = all(validity.values())
    endpoints_passed = bool(g1["passed"] and g2["passed"] and all(e["passed"] for e in additional.values()))
    if not evidence_valid:
        verdict = "VOID_RERUN"
    elif endpoints_passed:
        verdict = "PASS"
    else:
        verdict = "ACCEPTED_NEGATIVE"
    return {
        "schema_version": "sevc-tdsc-system-overhead-dataset-gate-v1",
        "protocol_version": PROTOCOL_VERSION,
        "dataset": dataset,
        "block_count": len(ordered),
        "block_seed_first": seeds[0],
        "block_seed_last": seeds[-1],
        "statistics": {
            "cluster_unit": "registered block",
            "point_estimator": "ratio_of_sums_across_blocks",
            "bootstrap_seed": int(seed),
            "bootstrap_resamples": int(resamples),
            "familywise_alpha": float(familywise_alpha),
            "gate_cell_count": int(gate_cell_count),
            "multiplicity": "Bonferroni",
            "gate_cell_quantile": quantile,
        },
        "G1": g1,
        "G2": g2,
        **additional,
        "validity_gates": validity,
        "cardinality": dict(cardinality),
        "identity": dict(identity),
        "timing": dict(timing),
        "closure": dict(closure),
        "observer_effect": dict(observer_effect),
        "bytes": dict(bytes_summary),
        "evidence_valid": evidence_valid,
        "endpoints_passed": endpoints_passed,
        "verdict": verdict,
    }


def route_terminal(
    dataset_gates: Sequence[Mapping[str, Any]],
    *,
    stop_loss_dataset: str,
    required_datasets: Sequence[str],
    independent_audit_passed: bool,
    stop_loss_enabled: bool = True,
) -> dict[str, Any]:
    """Route the campaign without letting a valid negative become a technical failure."""

    by_dataset = {str(gate["dataset"]): gate for gate in dataset_gates}
    completed = [name for name in required_datasets if name in by_dataset]
    if not completed:
        raise ValueError("terminal routing requires at least one completed dataset")
    if any(str(by_dataset[name]["verdict"]) == "VOID_RERUN" for name in completed):
        return {
            "verdict": "VOID_RERUN",
            "terminal": "TECHNICAL_VALIDITY_FAILURE",
            "paper_routing": "NO_PAPER_CHANGE",
            "completed_datasets": completed,
            "independent_audit_passed": bool(independent_audit_passed),
        }
    if not independent_audit_passed:
        return {
            "verdict": "VOID_RERUN",
            "terminal": "INDEPENDENT_AUDIT_FAILURE",
            "paper_routing": "NO_PAPER_CHANGE",
            "completed_datasets": completed,
            "independent_audit_passed": False,
        }
    gating_dataset = str(stop_loss_dataset)
    substituted = False
    if gating_dataset not in required_datasets:
        gating_dataset = str(required_datasets[0])
        substituted = True
    stop_loss = by_dataset.get(gating_dataset)
    if stop_loss is None:
        raise ValueError("terminal routing requires the gating dataset")
    if stop_loss_enabled and not bool(stop_loss["endpoints_passed"]):
        return {
            "verdict": "ACCEPTED_NEGATIVE",
            "terminal": "CIFAR10_STOP_LOSS" if not substituted else "STOP_LOSS",
            "paper_routing": "PAPER_CHANGE_REQUIRED",
            "completed_datasets": completed,
            "generalization_forbidden_beyond": gating_dataset,
            "gating_dataset": gating_dataset,
            "gating_dataset_substituted": substituted,
            "independent_audit_passed": True,
        }
    missing = [name for name in required_datasets if name not in by_dataset]
    if missing:
        return {
            "verdict": "VOID_RERUN",
            "terminal": "MANDATORY_CONFIRMATION_INCOMPLETE",
            "paper_routing": "NO_PAPER_CHANGE",
            "completed_datasets": completed,
            "missing_datasets": missing,
            "independent_audit_passed": True,
        }
    if all(bool(by_dataset[name]["endpoints_passed"]) for name in required_datasets):
        return {
            "verdict": "PASS",
            "terminal": "THREE_DATASET_SYMMETRIC_OVERHEAD_MEASURED" if all("G4" in by_dataset[n] for n in required_datasets) else "THREE_DATASET_OVERHEAD_MEASURED",
            "paper_routing": "PAPER_CHANGE_REQUIRED",
            "completed_datasets": completed,
            "independent_audit_passed": True,
        }
    boundary_confirmed = all(
        not by_dataset[name].get("G1", {}).get("passed", True)
        and all(by_dataset[name].get(endpoint, {}).get("passed", False) for endpoint in ("G2", "G3", "G4"))
        for name in required_datasets
    )
    return {
        "verdict": "ACCEPTED_NEGATIVE",
        "terminal": "G1_BOUNDARY_CONFIRMED" if boundary_confirmed else "THREE_DATASET_MIXED_OR_FAIL",
        "paper_routing": "PAPER_CHANGE_REQUIRED",
        "completed_datasets": completed,
        "failed_datasets": [
            name
            for name in required_datasets
            if not bool(by_dataset[name]["endpoints_passed"])
        ],
        "independent_audit_passed": True,
    }


def verify_received_commitment(payload: Any, sealed_digest: str) -> str:
    """Recompute the delivered canonical digest and reject a mismatched commitment."""
    _, received = measure_canonical_payload(payload)
    if received != sealed_digest:
        raise ValueError("received payload commitment mismatch")
    return received


def verify_inherited_endpoints(reference: Mapping[str, Any]) -> dict[str, Any]:
    """Replay sealed v2 estimates with their original multiplicity, without timing."""
    results = {}
    for dataset, entry in reference["datasets"].items():
        gate = entry["gate"]
        stats = gate["statistics"]
        ordered = sorted(entry["block_summaries"], key=lambda row: int(row["block_seed"]))
        checks = {}
        for endpoint in ("G1", "G2"):
            key = endpoint.lower()
            observed = endpoint_estimate(
                [row[f"{key}_numerator_seconds"] for row in ordered],
                [row[f"{key}_denominator_seconds"] for row in ordered],
                seed=stats["bootstrap_seed"], resamples=stats["bootstrap_resamples"],
                cell_quantile=stats["gate_cell_quantile"],
            )
            checks[endpoint] = all(value == gate[endpoint][field] for field, value in observed.items())
        results[dataset] = checks
    return {"passed": set(results) == {"cifar10", "mnist", "cifar100"}
        and all(all(checks.values()) for checks in results.values()),
        "datasets": results, "reference_change_id": reference["change_id"],
        "v3_bounds_use_wider_12_cell_quantile": True}


def registered_prediction_comparison(gates, predictions, reference) -> dict[str, Any]:
    rows = []
    for gate in gates:
        dataset = gate["dataset"]
        checks = {}
        for endpoint in ("G3", "G4"):
            low, high = predictions[f"{endpoint}_point_within"]
            point = gate[endpoint]["point"]
            checks[endpoint] = {"point": point, "range": [low, high], "held": low <= point <= high}
        for endpoint in ("G1", "G2"):
            previous = reference["datasets"][dataset]["gate"][endpoint]["point"]
            point = gate[endpoint]["point"]
            relative = abs(point - previous) / abs(previous)
            tolerance = predictions[f"{endpoint}_replication_relative_tolerance"]
            checks[endpoint] = {"point": point, "v2_point": previous,
                "relative_difference": relative, "relative_tolerance": tolerance,
                "held": relative <= tolerance}
        rows.append({"dataset": dataset, "predictions": checks})
    return {"datasets": rows, "shared_prefix_blocks": 5, "formal_blocks_per_dataset": 40,
        "samples_disjoint": False, "prediction_failure_authorizes_rerun": False,
        "multiplicity_note": "v2 used 6 cells; v3 uses 12 cells. Point definitions are unchanged."}
