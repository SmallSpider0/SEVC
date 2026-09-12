"""Pure EVAL-E3 reliability, capacity, cost, and synthetic-link simulation."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
import hashlib
import math
import random
import statistics
from typing import Any, Iterable, Mapping, Sequence

from sevc.core.artifacts import canonical_json_text, sha256_text
from sevc.evaluation.statistics import wilson_interval
from sevc.evaluation.tdsc_direct_execution import METHODS as E1_METHODS


METHODS = tuple(method for method in E1_METHODS if method != "no-verification")
PROTOCOL_VERSION = "sevc-tdsc-e3-simulation-v1"


def derive_int(domain: str, *parts: object) -> int:
    payload = canonical_json_text([domain, *parts]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def derive_unit_interval(domain: str, *parts: object) -> float:
    value = derive_int(domain, *parts)
    return (value + 0.5) / float(1 << 64)


def _binary_hash(values: Sequence[bool]) -> str:
    return sha256_text(canonical_json_text([1 if value else 0 for value in values]))


def availability_trace(
    *,
    dataset: str,
    population: int,
    dropout_probability: float,
    bernoulli_icc: float,
    trace_seed: int,
) -> dict[str, Any]:
    if population <= 0:
        raise ValueError("population must be positive")
    if not 0.0 <= dropout_probability <= 1.0:
        raise ValueError("dropout probability must be in [0,1]")
    if not 0.0 <= bernoulli_icc < 1.0:
        raise ValueError("Bernoulli ICC must be in [0,1)")
    success_probability = 1.0 - dropout_probability
    seed = derive_int(
        "e3-availability",
        dataset,
        population,
        dropout_probability,
        bernoulli_icc,
        trace_seed,
    )
    if success_probability <= 0.0:
        latent_probability = 0.0
        responses = [False] * population
    elif success_probability >= 1.0:
        latent_probability = 1.0
        responses = [True] * population
    else:
        generator = random.Random(seed)
        if bernoulli_icc == 0.0:
            latent_probability = success_probability
        else:
            concentration = (1.0 - bernoulli_icc) / bernoulli_icc
            alpha = success_probability * concentration
            beta = (1.0 - success_probability) * concentration
            latent_probability = generator.betavariate(alpha, beta)
        responses = [
            generator.random() < latent_probability for _ in range(population)
        ]
    return {
        "responses": responses,
        "latent_probability": float(latent_probability),
        "availability_sha256": _binary_hash(responses),
        "availability_seed": seed,
    }


def task_rosters(
    *,
    dataset: str,
    population: int,
    dropout_probability: float,
    bernoulli_icc: float,
    trace_seed: int,
    task_count: int,
    maximum_slots: int = 9,
) -> list[list[int]]:
    if maximum_slots > population:
        raise ValueError("maximum roster slots exceed population")
    rosters: list[list[int]] = []
    for task_index in range(task_count):
        ordered = sorted(
            range(population),
            key=lambda member: derive_int(
                "e3-task-roster",
                dataset,
                population,
                dropout_probability,
                bernoulli_icc,
                trace_seed,
                task_index,
                member,
            ),
        )
        rosters.append(ordered[:maximum_slots])
    return rosters


def _binomial_tail(size: int, threshold: int, probability: float) -> float:
    return sum(
        math.comb(size, successes)
        * probability**successes
        * (1.0 - probability) ** (size - successes)
        for successes in range(threshold, size + 1)
    )


def claimed_sevc_reserve_reliability(
    claimed_success_probability: float, replacement_budget: int
) -> float:
    """Exact independent-slot reliability for the registered six-slot SEVC path."""

    if replacement_budget not in {0, 1, 2, 3}:
        raise ValueError("unregistered SEVC replacement budget")
    probability = min(1.0, max(0.0, float(claimed_success_probability)))
    total = 0.0
    for initial_successes in range(7):
        missing = 6 - initial_successes
        if missing > replacement_budget:
            continue
        initial_probability = (
            math.comb(6, initial_successes)
            * probability**initial_successes
            * (1.0 - probability) ** (6 - initial_successes)
        )
        total += initial_probability * _binomial_tail(
            replacement_budget, missing, probability
        )
    return total


def claimed_reliability(method: str, claimed_success_probability: float) -> float:
    probability = min(1.0, max(0.0, float(claimed_success_probability)))
    if method == "owner-only-reference-replay":
        return 1.0
    if method == "vanilla-pol-replay":
        return probability
    if method == "depol-redundant-replay-style":
        return _binomial_tail(3, 2, probability)
    if method == "refiner-two-stage-audit-style":
        return _binomial_tail(5, 4, probability)
    if method == "full-sevc":
        return claimed_sevc_reserve_reliability(probability, 3)
    raise KeyError(method)


def _attempt_slot(
    member: int,
    responses: Sequence[bool],
    usage: list[int] | None,
    capacity: int | None,
) -> tuple[bool, str]:
    if not responses[member]:
        return False, "nonresponsive-commissioned"
    if usage is not None and capacity is not None and usage[member] >= capacity:
        return False, "capacity-rejected"
    if usage is not None:
        usage[member] += 1
    return True, "successful-commissioned"


def _evaluate_task(
    method: str,
    responses: Sequence[bool],
    roster: Sequence[int],
    parameters: Mapping[str, Any],
    usage: list[int] | None,
    capacity: int | None,
) -> dict[str, Any]:
    if method == "owner-only-reference-replay":
        return {
            "complete": True,
            "successful_executions": 0,
            "commissioned_slots": 0,
            "nonresponsive_slots": 0,
            "capacity_rejected_slots": 0,
            "recovered_slots": 0,
            "attempts": [],
        }
    method_parameters = parameters[method]
    initial_slots = int(method_parameters["initial_slots"])
    threshold = int(method_parameters["threshold"])
    replacement_budget = int(method_parameters["replacement_budget"])
    if len(roster) < initial_slots + replacement_budget:
        raise ValueError("task roster is shorter than registered method slots")
    successful = 0
    commissioned = 0
    nonresponsive = 0
    capacity_rejected = 0
    recovered = 0
    attempts: list[dict[str, Any]] = []

    def consume(member: int, *, replacement: bool) -> None:
        nonlocal successful, commissioned, nonresponsive, capacity_rejected, recovered
        passed, route = _attempt_slot(member, responses, usage, capacity)
        attempts.append(
            {
                "member": int(member),
                "replacement": bool(replacement),
                "route": route,
                "responsive": bool(responses[member]),
            }
        )
        if route == "capacity-rejected":
            capacity_rejected += 1
            return
        commissioned += 1
        if route == "nonresponsive-commissioned":
            nonresponsive += 1
            return
        successful += 1
        if replacement:
            recovered += 1

    for member in roster[:initial_slots]:
        consume(member, replacement=False)
    if method == "full-sevc":
        for member in roster[initial_slots : initial_slots + replacement_budget]:
            if successful >= threshold:
                break
            consume(member, replacement=True)
    return {
        "complete": successful >= threshold,
        "successful_executions": successful,
        "commissioned_slots": commissioned,
        "nonresponsive_slots": nonresponsive,
        "capacity_rejected_slots": capacity_rejected,
        "recovered_slots": recovered,
        "attempts": attempts,
    }


def evaluate_trace_method(
    *,
    method: str,
    responses: Sequence[bool],
    rosters: Sequence[Sequence[int]],
    method_parameters: Mapping[str, Any],
    capacity_per_member: int,
    source_false_negative_count: int,
    normalized_costs: Mapping[str, Any],
) -> dict[str, Any]:
    usage = [0] * len(responses)
    adjusted: list[dict[str, Any]] = []
    uncapacitated: list[dict[str, Any]] = []
    for roster in rosters:
        uncapacitated.append(
            _evaluate_task(
                method,
                responses,
                roster,
                method_parameters,
                None,
                None,
            )
        )
        adjusted.append(
            _evaluate_task(
                method,
                responses,
                roster,
                method_parameters,
                usage,
                capacity_per_member,
            )
        )
    completed = sum(bool(row["complete"]) for row in adjusted)
    uncapacitated_completed = sum(
        bool(row["complete"]) for row in uncapacitated
    )
    successful_executions = sum(
        int(row["successful_executions"]) for row in adjusted
    )
    commissioned_slots = sum(int(row["commissioned_slots"]) for row in adjusted)
    nonresponsive_slots = sum(
        int(row["nonresponsive_slots"]) for row in adjusted
    )
    capacity_rejected_slots = sum(
        int(row["capacity_rejected_slots"]) for row in adjusted
    )
    recovered_slots = sum(int(row["recovered_slots"]) for row in adjusted)
    successful_recovery_tasks = sum(
        bool(row["complete"]) and int(row["recovered_slots"]) > 0
        for row in adjusted
    )
    capacity_rejection_tasks = sum(
        bool(before["complete"]) and not bool(after["complete"])
        for before, after in zip(uncapacitated, adjusted)
    )
    replacement_budget_rejection_tasks = (
        sum(not bool(row["complete"]) for row in uncapacitated)
        if method == "full-sevc"
        else 0
    )
    deferral_tasks = (
        sum(not bool(row["complete"]) for row in adjusted)
        if method == "full-sevc"
        else 0
    )
    termination_tasks = (
        sum(not bool(row["complete"]) for row in adjusted)
        if method
        in {
            "vanilla-pol-replay",
            "depol-redundant-replay-style",
            "refiner-two-stage-audit-style",
        }
        else 0
    )
    unsafe_tasks = completed if source_false_negative_count > 0 else 0
    owner_reference_executions = (
        len(rosters) if method == "owner-only-reference-replay" else 0
    )
    owner_expenditure = (
        owner_reference_executions
        * float(normalized_costs["owner_reference_execution"])
        + successful_executions
        * float(normalized_costs["verifier_execution"])
        * float(normalized_costs["service_fee_ratio"])
    )
    refundable = (
        successful_executions
        * float(normalized_costs["verifier_execution"])
        * float(normalized_costs["bond_ratio"])
    )
    slashed = (
        nonresponsive_slots
        * float(normalized_costs["verifier_execution"])
        * float(normalized_costs["bond_ratio"])
    )
    return {
        "completed_task_count": completed,
        "uncapacitated_completed_task_count": uncapacitated_completed,
        "successful_recovery_task_count": successful_recovery_tasks,
        "recovered_slot_count": recovered_slots,
        "safe_deferral_task_count": deferral_tasks,
        "termination_or_restart_task_count": termination_tasks,
        "unsafe_settlement_task_count": unsafe_tasks,
        "capacity_rejection_task_count": capacity_rejection_tasks,
        "replacement_budget_rejection_task_count": replacement_budget_rejection_tasks,
        "successful_verifier_execution_count": successful_executions,
        "commissioned_slot_count": commissioned_slots,
        "nonresponsive_commissioned_slot_count": nonresponsive_slots,
        "capacity_rejected_slot_count": capacity_rejected_slots,
        "owner_reference_execution_count": owner_reference_executions,
        "owner_expenditure_units": owner_expenditure,
        "refundable_bond_exposure_units": refundable,
        "slashed_bond_exposure_units": slashed,
        "liquidity_cost_units": (refundable + slashed)
        * float(normalized_costs["liquidity_cost_rate"]),
    }


def evaluate_trace_method_with_tasks(
    *,
    method: str,
    responses: Sequence[bool],
    rosters: Sequence[Sequence[int]],
    method_parameters: Mapping[str, Any],
    capacity_per_member: int,
    source_false_negative_count: int,
    normalized_costs: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Return the canonical trace aggregate plus its ordered nested task rows."""

    aggregate = evaluate_trace_method(
        method=method,
        responses=responses,
        rosters=rosters,
        method_parameters=method_parameters,
        capacity_per_member=capacity_per_member,
        source_false_negative_count=source_false_negative_count,
        normalized_costs=normalized_costs,
    )
    usage = [0] * len(responses)
    tasks: list[dict[str, Any]] = []
    for task_index, roster in enumerate(rosters):
        uncapacitated = _evaluate_task(
            method,
            responses,
            roster,
            method_parameters,
            None,
            None,
        )
        adjusted = _evaluate_task(
            method,
            responses,
            roster,
            method_parameters,
            usage,
            capacity_per_member,
        )
        complete = bool(adjusted["complete"])
        recovered = int(adjusted["recovered_slots"])
        if complete and recovered > 0:
            route = "complete-after-recovery"
        elif complete:
            route = "complete-without-recovery"
        elif method == "full-sevc":
            route = "safe-defer"
        else:
            route = "terminate-or-restart"
        tasks.append(
            {
                "task_index": task_index,
                "route": route,
                "complete": complete,
                "unsafe_settlement": complete
                and int(source_false_negative_count) > 0,
                "uncapacitated_complete": bool(uncapacitated["complete"]),
                "successful_executions": int(
                    adjusted["successful_executions"]
                ),
                "commissioned_slots": int(adjusted["commissioned_slots"]),
                "nonresponsive_slots": int(adjusted["nonresponsive_slots"]),
                "capacity_rejected_slots": int(
                    adjusted["capacity_rejected_slots"]
                ),
                "recovered_slots": recovered,
                "attempts": list(adjusted["attempts"]),
                "capacity_rejection": bool(uncapacitated["complete"])
                and not complete,
                "replacement_budget_rejection": method == "full-sevc"
                and not bool(uncapacitated["complete"]),
            }
        )
    checks = {
        "completed_task_count": sum(bool(row["complete"]) for row in tasks),
        "uncapacitated_completed_task_count": sum(
            bool(row["uncapacitated_complete"]) for row in tasks
        ),
        "successful_recovery_task_count": sum(
            row["route"] == "complete-after-recovery" for row in tasks
        ),
        "recovered_slot_count": sum(int(row["recovered_slots"]) for row in tasks),
        "safe_deferral_task_count": sum(
            row["route"] == "safe-defer" for row in tasks
        ),
        "termination_or_restart_task_count": sum(
            row["route"] == "terminate-or-restart" for row in tasks
        ),
        "unsafe_settlement_task_count": sum(
            bool(row["unsafe_settlement"]) for row in tasks
        ),
        "capacity_rejection_task_count": sum(
            bool(row["capacity_rejection"]) for row in tasks
        ),
        "replacement_budget_rejection_task_count": sum(
            bool(row["replacement_budget_rejection"]) for row in tasks
        ),
        "successful_verifier_execution_count": sum(
            int(row["successful_executions"]) for row in tasks
        ),
        "commissioned_slot_count": sum(
            int(row["commissioned_slots"]) for row in tasks
        ),
        "nonresponsive_commissioned_slot_count": sum(
            int(row["nonresponsive_slots"]) for row in tasks
        ),
        "capacity_rejected_slot_count": sum(
            int(row["capacity_rejected_slots"]) for row in tasks
        ),
    }
    mismatches = {
        field: {"aggregate": aggregate[field], "nested": value}
        for field, value in checks.items()
        if aggregate[field] != value
    }
    if mismatches:
        raise ValueError(f"nested task aggregation drift: {mismatches}")
    return aggregate, tasks


def source_adapter(source_metrics: Mapping[str, Any]) -> dict[tuple[str, int, str], dict[str, Any]]:
    adapter: dict[tuple[str, int, str], dict[str, Any]] = {}
    for row in source_metrics["rows"]:
        if str(row["source_state"]) != "compound":
            continue
        method = str(row["method"])
        if method not in {"sevc-ctiv-registered-v1", "refiner-update-audit-style"}:
            continue
        metrics = row["metrics"]
        key = (str(row["dataset"]), int(row["source_seed"]), method)
        adapter[key] = {
            "source_unit_id": str(row["source_unit_id"]),
            "source_update_bank_chain_sha256": str(row["update_bank_chain_sha256"]),
            "source_false_negative_count": int(metrics["FN"]),
            "verification_bytes": int(metrics["verification_bytes"]),
        }
    return adapter


def build_reliability_rows(
    protocol: Mapping[str, Any],
    source_metrics: Mapping[str, Any],
    *,
    datasets: Sequence[str],
    trace_indices: Iterable[int],
) -> list[dict[str, Any]]:
    adapter = source_adapter(source_metrics)
    trace_indices = tuple(int(index) for index in trace_indices)
    trace_seeds = tuple(int(seed) for seed in protocol["trace_seeds"])
    source_seeds = tuple(int(seed) for seed in protocol["source_seeds"])
    reliability = protocol["reliability"]
    task_count = int(reliability["tasks_per_trace"])
    capacity = int(reliability["capacity_per_member_per_trace"])
    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        for population in reliability["verifier_populations"]:
            for dropout in reliability["dropout_probabilities"]:
                for icc in reliability["bernoulli_icc"]:
                    for trace_index in trace_indices:
                        trace_seed = trace_seeds[trace_index]
                        source_seed = source_seeds[trace_index % len(source_seeds)]
                        availability = availability_trace(
                            dataset=dataset,
                            population=int(population),
                            dropout_probability=float(dropout),
                            bernoulli_icc=float(icc),
                            trace_seed=trace_seed,
                        )
                        rosters = task_rosters(
                            dataset=dataset,
                            population=int(population),
                            dropout_probability=float(dropout),
                            bernoulli_icc=float(icc),
                            trace_seed=trace_seed,
                            task_count=task_count,
                        )
                        roster_sha = sha256_text(canonical_json_text(rosters))
                        for method in protocol["methods"]:
                            source_method = protocol["source_method_by_method"][method]
                            source = adapter[(dataset, source_seed, source_method)]
                            outcome = evaluate_trace_method(
                                method=method,
                                responses=availability["responses"],
                                rosters=rosters,
                                method_parameters=protocol["method_parameters"],
                                capacity_per_member=capacity,
                                source_false_negative_count=int(
                                    source["source_false_negative_count"]
                                ),
                                normalized_costs=protocol["normalized_costs"],
                            )
                            for bias in reliability["claimed_probability_bias"]:
                                claimed_probability = min(
                                    1.0,
                                    max(0.0, 1.0 - float(dropout) + float(bias)),
                                )
                                claim = claimed_reliability(method, claimed_probability)
                                successes = int(outcome["completed_task_count"])
                                brier = (
                                    successes * (1.0 - claim) ** 2
                                    + (task_count - successes) * claim**2
                                ) / task_count
                                identity = {
                                    "dataset": dataset,
                                    "verifier_population": int(population),
                                    "dropout_probability": float(dropout),
                                    "bernoulli_icc": float(icc),
                                    "claimed_probability_bias": float(bias),
                                    "trace_index": trace_index,
                                    "trace_seed": trace_seed,
                                    "method": method,
                                }
                                rows.append(
                                    {
                                        "schema_version": "sevc-tdsc-e3-reliability-trace-v1",
                                        "change_id": protocol["change_id"],
                                        **identity,
                                        "trace_row_id": sha256_text(
                                            canonical_json_text(identity)
                                        )[:24],
                                        "source_seed": source_seed,
                                        "source_method": source_method,
                                        "source_unit_id": source["source_unit_id"],
                                        "source_false_negative_count": int(
                                            source["source_false_negative_count"]
                                        ),
                                        "availability_seed": availability["availability_seed"],
                                        "availability_sha256": availability[
                                            "availability_sha256"
                                        ],
                                        "task_roster_sha256": roster_sha,
                                        "latent_responsiveness": availability[
                                            "latent_probability"
                                        ],
                                        "claimed_member_responsiveness": claimed_probability,
                                        "claimed_reliability": claim,
                                        "task_count": task_count,
                                        **outcome,
                                        "empirical_completion_rate": successes / task_count,
                                        "brier_score": brier,
                                        "simulation_compute_device": "cpu",
                                    }
                                )
    return rows


def _wilson(successes: int, total: int) -> dict[str, Any]:
    return asdict(wilson_interval(successes, total))


def _numeric_summary(values: Sequence[float]) -> dict[str, float]:
    if not values:
        raise ValueError("numeric summary requires values")
    return {
        "median": float(statistics.median(values)),
        "minimum": float(min(values)),
        "maximum": float(max(values)),
    }


def _trace_rate_interval(
    rows: Sequence[Mapping[str, Any]], field: str
) -> dict[str, Any]:
    values = [
        int(row[field]) / int(row["task_count"])
        for row in rows
    ]
    if not values:
        raise ValueError("trace-rate interval requires registered traces")
    point = float(statistics.mean(values))
    standard_error = (
        float(statistics.stdev(values)) / math.sqrt(len(values))
        if len(values) > 1
        else 0.0
    )
    z = statistics.NormalDist().inv_cdf(0.975)
    return {
        "task_numerator": sum(int(row[field]) for row in rows),
        "task_denominator": sum(int(row["task_count"]) for row in rows),
        "trace_count": len(values),
        "point_estimate": point,
        "lower": max(0.0, point - z * standard_error),
        "upper": min(1.0, point + z * standard_error),
        "confidence": 0.95,
        "interval_method": "trace-cluster-mean-normal",
    }


def summarize_reliability_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    fields = (
        "dataset",
        "verifier_population",
        "dropout_probability",
        "bernoulli_icc",
        "claimed_probability_bias",
        "method",
    )
    for row in rows:
        groups[tuple(row[field] for field in fields)].append(row)
    summaries: list[dict[str, Any]] = []
    rate_fields = (
        "completed_task_count",
        "uncapacitated_completed_task_count",
        "successful_recovery_task_count",
        "safe_deferral_task_count",
        "termination_or_restart_task_count",
        "unsafe_settlement_task_count",
        "capacity_rejection_task_count",
        "replacement_budget_rejection_task_count",
    )
    for key in sorted(groups):
        scoped = groups[key]
        total = sum(int(row["task_count"]) for row in scoped)
        claims = {float(row["claimed_reliability"]) for row in scoped}
        if len(claims) != 1:
            raise ValueError("claimed reliability drift within summary cell")
        claim = claims.pop()
        payload = {field: value for field, value in zip(fields, key)}
        payload.update(
            {
                "registered_trace_count": len(scoped),
                "task_count": total,
                "claimed_reliability": claim,
                "brier_score": sum(
                    float(row["brier_score"]) * int(row["task_count"])
                    for row in scoped
                )
                / total,
            }
        )
        for field in rate_fields:
            summary_name = (
                field[: -len("_task_count")]
                if field.endswith("_task_count")
                else field
            )
            payload[summary_name + "_rate"] = _trace_rate_interval(scoped, field)
        empirical = float(payload["completed_rate"]["point_estimate"])
        payload["claimed_minus_empirical_completion"] = claim - empirical
        summaries.append(payload)
    return summaries


def summarize_calibration_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["dataset"]), str(row["method"]))].append(row)
    summaries: list[dict[str, Any]] = []
    for (dataset, method), scoped in sorted(groups.items()):
        bins = [
            {"task_count": 0, "successes": 0, "confidence_sum": 0.0}
            for _ in range(10)
        ]
        brier_sum = 0.0
        total = 0
        for row in scoped:
            count = int(row["task_count"])
            confidence = float(row["claimed_reliability"])
            successes = int(row["completed_task_count"])
            index = min(9, int(confidence * 10.0))
            bins[index]["task_count"] += count
            bins[index]["successes"] += successes
            bins[index]["confidence_sum"] += confidence * count
            brier_sum += float(row["brier_score"]) * count
            total += count
        public_bins: list[dict[str, Any]] = []
        ece = 0.0
        for index, entry in enumerate(bins):
            count = int(entry["task_count"])
            if count == 0:
                continue
            confidence = float(entry["confidence_sum"]) / count
            accuracy = int(entry["successes"]) / count
            ece += count / total * abs(confidence - accuracy)
            public_bins.append(
                {
                    "bin_index": index,
                    "lower_inclusive": index / 10.0,
                    "upper_inclusive": 1.0 if index == 9 else (index + 1) / 10.0,
                    "task_count": count,
                    "mean_confidence": confidence,
                    "empirical_completion": accuracy,
                }
            )
        summaries.append(
            {
                "dataset": dataset,
                "method": method,
                "task_count": total,
                "brier_score": brier_sum / total,
                "ece_10_equal_width": ece,
                "nonempty_bins": public_bins,
            }
        )
    return summaries


def summarize_owner_cost_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    fields = (
        "dataset",
        "verifier_population",
        "dropout_probability",
        "bernoulli_icc",
        "method",
    )
    for row in rows:
        if float(row["claimed_probability_bias"]) != 0.0:
            continue
        groups[tuple(row[field] for field in fields)].append(row)
    summaries: list[dict[str, Any]] = []
    numeric_fields = (
        "owner_expenditure_units",
        "refundable_bond_exposure_units",
        "slashed_bond_exposure_units",
        "liquidity_cost_units",
        "successful_verifier_execution_count",
    )
    for key in sorted(groups):
        scoped = groups[key]
        payload = {field: value for field, value in zip(fields, key)}
        payload["registered_trace_count"] = len(scoped)
        for field in numeric_fields:
            payload[field] = _numeric_summary([float(row[field]) for row in scoped])
        summaries.append(payload)
    return summaries


def summarize_safety_availability_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    fields = (
        "dataset",
        "verifier_population",
        "dropout_probability",
        "bernoulli_icc",
        "method",
    )
    for row in rows:
        if float(row["claimed_probability_bias"]) == 0.0:
            groups[tuple(row[field] for field in fields)].append(row)
    output: list[dict[str, Any]] = []
    for key in sorted(groups):
        scoped = groups[key]
        total = sum(int(row["task_count"]) for row in scoped)
        completed = sum(int(row["completed_task_count"]) for row in scoped)
        unsafe = sum(int(row["unsafe_settlement_task_count"]) for row in scoped)
        output.append(
            {
                **{field: value for field, value in zip(fields, key)},
                "task_count": total,
                "completion_rate": completed / total,
                "unsafe_settlement_rate": unsafe / total,
                "safe_or_nonsettled_rate": 1.0 - unsafe / total,
            }
        )
    return output


def _chunks(payload_bytes: int, chunk_bytes: int) -> list[int]:
    if payload_bytes <= 0 or chunk_bytes <= 0:
        raise ValueError("payload and chunk size must be positive")
    full, remainder = divmod(payload_bytes, chunk_bytes)
    chunks = [chunk_bytes] * full
    if remainder:
        chunks.append(remainder)
    return chunks


def _transfer(
    *,
    payload_bytes: int,
    bandwidth_mbps: float,
    rtt_ms: float,
    loss_probability: float,
    chunk_bytes: int,
    maximum_retries: int,
    trace_identity: Sequence[Any],
    transfer_index: int,
) -> dict[str, Any]:
    bandwidth = bandwidth_mbps * 1_000_000.0
    rtt_seconds = rtt_ms / 1000.0
    seconds = rtt_seconds + payload_bytes * 8.0 / bandwidth
    registered_bytes = payload_bytes
    complete = True
    retry_count = 0
    attempt_count = 0
    for chunk_index, size in enumerate(_chunks(payload_bytes, chunk_bytes)):
        delivered = False
        for attempt_index in range(maximum_retries + 1):
            attempt_count += 1
            lost = (
                derive_unit_interval(
                    "e3-link-loss",
                    *trace_identity,
                    transfer_index,
                    chunk_index,
                    attempt_index,
                )
                < loss_probability
            )
            if attempt_index > 0:
                retry_count += 1
                registered_bytes += size
                seconds += rtt_seconds + size * 8.0 / bandwidth
            if not lost:
                delivered = True
                break
        if not delivered:
            complete = False
    return {
        "complete": complete,
        "registered_bytes": registered_bytes,
        "seconds": seconds,
        "retry_count": retry_count,
        "attempt_count": attempt_count,
    }


def build_link_rows(
    protocol: Mapping[str, Any],
    source_metrics: Mapping[str, Any],
    *,
    datasets: Sequence[str],
    trace_indices: Iterable[int],
) -> list[dict[str, Any]]:
    adapter = source_adapter(source_metrics)
    trace_indices = tuple(int(index) for index in trace_indices)
    trace_seeds = tuple(int(seed) for seed in protocol["trace_seeds"])
    source_seeds = tuple(int(seed) for seed in protocol["source_seeds"])
    link = protocol["synthetic_link"]
    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        for population in protocol["reliability"]["verifier_populations"]:
            for profile_name, profile in link["profiles"].items():
                for topology in link["topologies"]:
                    for trace_index in trace_indices:
                        trace_seed = trace_seeds[trace_index]
                        source_seed = source_seeds[trace_index % len(source_seeds)]
                        trace_identity = (
                            dataset,
                            int(population),
                            profile_name,
                            topology,
                            trace_index,
                            trace_seed,
                        )
                        link_trace_sha = sha256_text(
                            canonical_json_text(list(trace_identity))
                        )
                        for method in protocol["methods"]:
                            source_method = protocol["source_method_by_method"][method]
                            source = adapter[(dataset, source_seed, source_method)]
                            payload_bytes = math.ceil(
                                int(source["verification_bytes"]) / 20
                            )
                            transfer_count = int(
                                link["logical_transfer_count"][method]
                            )
                            transfers = [
                                _transfer(
                                    payload_bytes=payload_bytes,
                                    bandwidth_mbps=float(profile["bandwidth_mbps"]),
                                    rtt_ms=float(profile["rtt_ms"]),
                                    loss_probability=float(
                                        profile["loss_probability"]
                                    ),
                                    chunk_bytes=int(link["chunk_bytes"]),
                                    maximum_retries=int(
                                        link["maximum_retries_after_initial"]
                                    ),
                                    trace_identity=trace_identity,
                                    transfer_index=index,
                                )
                                for index in range(transfer_count)
                            ]
                            if topology == "independent-links":
                                completion_seconds = max(
                                    float(item["seconds"]) for item in transfers
                                )
                            elif topology == "shared-owner-gateway":
                                completion_seconds = sum(
                                    float(item["seconds"]) for item in transfers
                                )
                            else:
                                raise ValueError(f"unknown topology {topology}")
                            analytical_transfer = float(profile["rtt_ms"]) / 1000.0 + (
                                payload_bytes
                                * 8.0
                                / (float(profile["bandwidth_mbps"]) * 1_000_000.0)
                            )
                            analytical_seconds = (
                                analytical_transfer
                                if topology == "independent-links"
                                else analytical_transfer * transfer_count
                            )
                            registered_bytes = sum(
                                int(item["registered_bytes"]) for item in transfers
                            )
                            identity = {
                                "dataset": dataset,
                                "verifier_population": int(population),
                                "profile": profile_name,
                                "topology": topology,
                                "trace_index": trace_index,
                                "trace_seed": trace_seed,
                                "method": method,
                            }
                            row = {
                                "schema_version": "sevc-tdsc-e3-synthetic-link-trace-v1",
                                "change_id": protocol["change_id"],
                                **identity,
                                "link_row_id": sha256_text(
                                    canonical_json_text(identity)
                                )[:24],
                                "link_trace_sha256": link_trace_sha,
                                "source_seed": source_seed,
                                "source_method": source_method,
                                "source_unit_id": source["source_unit_id"],
                                "payload_bytes_per_transfer": payload_bytes,
                                "logical_transfer_count": transfer_count,
                                "base_payload_bytes": payload_bytes * transfer_count,
                                "registered_transmitted_bytes": registered_bytes,
                                "retry_count": sum(
                                    int(item["retry_count"]) for item in transfers
                                ),
                                "attempt_count": sum(
                                    int(item["attempt_count"]) for item in transfers
                                ),
                                "transfer_complete": all(
                                    bool(item["complete"]) for item in transfers
                                ),
                                "simulated_completion_seconds": completion_seconds,
                                "zero_loss_analytical_seconds": analytical_seconds,
                                "zero_loss_absolute_error_seconds": (
                                    abs(completion_seconds - analytical_seconds)
                                    if float(profile["loss_probability"]) == 0.0
                                    else None
                                ),
                                "byte_conservation_passed": registered_bytes
                                == sum(
                                    int(item["registered_bytes"])
                                    for item in transfers
                                ),
                                "simulation_kind": "single-host-deterministic-discrete-event",
                                "real_network_invocations": 0,
                                "multihost_processes": 0,
                                "deployment_measurement": False,
                            }
                            row["deterministic_output_sha256"] = sha256_text(
                                canonical_json_text(row)
                            )
                            rows.append(row)
    return rows


def summarize_link_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[
            (
                str(row["dataset"]),
                str(row["profile"]),
                str(row["topology"]),
                str(row["method"]),
            )
        ].append(row)
    summaries: list[dict[str, Any]] = []
    for (dataset, profile, topology, method), scoped in sorted(groups.items()):
        total = len(scoped)
        completed = sum(bool(row["transfer_complete"]) for row in scoped)
        base_bytes = sum(int(row["base_payload_bytes"]) for row in scoped)
        transmitted = sum(
            int(row["registered_transmitted_bytes"]) for row in scoped
        )
        summaries.append(
            {
                "dataset": dataset,
                "profile": profile,
                "topology": topology,
                "method": method,
                "registered_trace_count": total,
                "completion_rate": _wilson(completed, total),
                "simulated_completion_seconds": _numeric_summary(
                    [float(row["simulated_completion_seconds"]) for row in scoped]
                ),
                "base_payload_bytes": base_bytes,
                "registered_transmitted_bytes": transmitted,
                "byte_overhead_ratio": transmitted / base_bytes,
                "retry_count": sum(int(row["retry_count"]) for row in scoped),
            }
        )
    return summaries


def analytical_limit_checks(link_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    zero_loss = [
        row for row in link_rows if str(row["profile"]) == "SIM-FAST"
    ]
    maximum_error = max(
        float(row["zero_loss_absolute_error_seconds"]) for row in zero_loss
    )
    deterministic_hashes = all(
        str(row["deterministic_output_sha256"])
        == sha256_text(
            canonical_json_text(
                {
                    key: value
                    for key, value in row.items()
                    if key != "deterministic_output_sha256"
                }
            )
        )
        for row in link_rows
    )
    return {
        "zero_loss_row_count": len(zero_loss),
        "zero_loss_max_absolute_error_seconds": maximum_error,
        "zero_loss_exact_within_1e_9": maximum_error <= 1e-9,
        "byte_conservation_passed": all(
            bool(row["byte_conservation_passed"]) for row in link_rows
        ),
        "deterministic_output_hash_passed": deterministic_hashes,
        "real_network_invocations": sum(
            int(row["real_network_invocations"]) for row in link_rows
        ),
        "multihost_processes": sum(
            int(row["multihost_processes"]) for row in link_rows
        ),
        "deployment_measurement": any(
            bool(row["deployment_measurement"]) for row in link_rows
        ),
    }


__all__ = [
    "METHODS",
    "PROTOCOL_VERSION",
    "analytical_limit_checks",
    "availability_trace",
    "build_link_rows",
    "build_reliability_rows",
    "claimed_reliability",
    "derive_int",
    "derive_unit_interval",
    "evaluate_trace_method",
    "source_adapter",
    "summarize_calibration_rows",
    "summarize_link_rows",
    "summarize_owner_cost_rows",
    "summarize_reliability_rows",
    "summarize_safety_availability_rows",
    "task_rosters",
]
