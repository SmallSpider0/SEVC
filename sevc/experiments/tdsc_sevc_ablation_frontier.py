"""Selected canonical scientific routines; deployment wrappers are omitted."""
from __future__ import annotations


from collections import defaultdict

import copy

from dataclasses import asdict

import hashlib

import statistics

from typing import Any, Mapping, Sequence

import numpy as np

from sevc.core.artifacts import canonical_json_text, sha256_text

from sevc.evaluation.statistics import wilson_interval

from sevc.evaluation.tdsc_direct_execution import DirectExecutionContext, SEVC_RESERVE_VARIANTS

from sevc.evaluation.tdsc_failure_scale_simulation import availability_trace, claimed_sevc_reserve_reliability, evaluate_trace_method_with_tasks, source_adapter, task_rosters

CHANGE_ID = "experiment-tdsc-sevc-ablation-frontier-v1"


VARIANT_ORDER = (
    "sevc-reserve-b0",
    "sevc-reserve-b1",
    "sevc-reserve-b2",
    "full-sevc",
)


def _direct_row(
    trace: Mapping[str, Any],
    source_unit: Mapping[str, Any],
    protocol: Mapping[str, Any],
    variant: str,
) -> dict[str, Any]:
    budget = int(protocol["variants"][variant]["replacement_budget"])
    direct_protocol = {
        "method_parameters": {
            "sevc_committee_count": int(protocol["fixed_method_contract"]["committee_count"]),
            "sevc_committee_size": int(protocol["fixed_method_contract"]["committee_size"]),
            "sevc_replacement_budget": budget,
        }
    }
    context = DirectExecutionContext(
        trace=trace,
        source_unit=source_unit,
        protocol=direct_protocol,
    )
    decision = SEVC_RESERVE_VARIANTS.get(variant)(context)
    source_rows = {
        str(row["method"]): row for row in source_unit["trajectory_rows"]
    }
    source_metrics = source_rows[decision.source_method]["metrics"]
    costs = protocol["normalized_costs"]
    unsafe = bool(
        trace["trainer_bad"]
        and decision.native_outcome == "accept"
        and int(source_metrics["FN"]) > 0
    )
    complete = decision.native_outcome not in {"defer", "terminate", "restart"}
    if complete and decision.recovered_slots > 0:
        route = "complete-after-recovery"
    elif complete:
        route = "complete-without-recovery"
    elif decision.native_outcome == "defer":
        route = "safe-defer"
    else:
        route = "terminate-or-restart"
    owner_expenditure = (
        decision.owner_reference_executions * float(costs["owner_reference_execution"])
        + (decision.replay_executions + decision.audit_executions)
        * float(costs["verifier_execution"])
        * float(costs["service_fee_ratio"])
    )
    service_cost = (
        decision.replay_executions + decision.audit_executions
    ) * float(costs["verifier_execution"])
    bond = (
        decision.commissioned_verifier_slots
        * float(costs["verifier_execution"])
        * float(costs["bond_ratio"])
    )
    state = (
        "compound"
        if str(trace["system_state"]) == "compound-trainer-verifier-failure"
        else str(trace["system_state"])
    )
    identity = {
        "dataset": str(trace["dataset"]),
        "state": state,
        "trace_id": str(trace["trace_id"]),
        "variant": variant,
    }
    return {
        "schema_version": "sevc-tdsc-e4-direct-variant-outcome-v1",
        "change_id": CHANGE_ID,
        **identity,
        "outcome_id": sha256_text(canonical_json_text(identity))[:24],
        "predecessor_state": str(trace["system_state"]),
        "trace_index": int(trace["trace_index"]),
        "trace_seed": int(trace["trace_seed"]),
        "source_unit_id": str(trace["source_unit_id"]),
        "source_seed": int(trace["source_seed"]),
        "replacement_budget": budget,
        "native_outcome": decision.native_outcome,
        "route": route,
        "complete": complete,
        "unsafe_settlement": unsafe,
        "successful_recovery": decision.successful_recovery,
        "recovered_slots": int(decision.recovered_slots),
        "reserve_activation_count": int(decision.recovered_slots),
        "safe_deferral": decision.native_outcome == "defer",
        "termination_or_restart": decision.native_outcome in {"terminate", "restart"},
        "replay_executions": int(decision.replay_executions),
        "audit_executions": int(decision.audit_executions),
        "owner_reference_executions": int(decision.owner_reference_executions),
        "owner_expenditure_units": owner_expenditure,
        "verifier_service_cost_units": service_cost,
        "combined_expenditure_units": owner_expenditure + service_cost,
        "bond_exposure_units": bond,
        "source_method": decision.source_method,
        "source_update_bank_chain_sha256": str(
            source_rows[decision.source_method]["update_bank_chain_sha256"]
        ),
    }


def _failure_rows(
    *,
    dataset: str,
    trace_indices: Sequence[int],
    protocol: Mapping[str, Any],
    e3_protocol: Mapping[str, Any],
    source_metrics: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    adapter = source_adapter(source_metrics)
    e4b = protocol["e4_b"]
    trace_seeds = tuple(int(value) for value in e3_protocol["trace_seeds"])
    source_seeds = tuple(int(value) for value in e3_protocol["source_seeds"])
    method_parameters = copy.deepcopy(dict(e3_protocol["method_parameters"]))
    normalized_costs = protocol["normalized_costs"]
    trace_rows: list[dict[str, Any]] = []
    task_rows: list[dict[str, Any]] = []
    for population in e4b["verifier_populations"]:
        for dropout in e4b["dropout_probabilities"]:
            for icc in e4b["bernoulli_icc"]:
                for trace_index in trace_indices:
                    trace_seed = trace_seeds[int(trace_index)]
                    source_seed = source_seeds[int(trace_index) % len(source_seeds)]
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
                        task_count=int(e4b["tasks_per_trace"]),
                    )
                    roster_sha = sha256_text(canonical_json_text(rosters))
                    source = adapter[(dataset, source_seed, "sevc-ctiv-registered-v1")]
                    for variant in VARIANT_ORDER:
                        budget = int(protocol["variants"][variant]["replacement_budget"])
                        variant_parameters = copy.deepcopy(method_parameters)
                        variant_parameters["full-sevc"]["replacement_budget"] = budget
                        aggregate, nested = evaluate_trace_method_with_tasks(
                            method="full-sevc",
                            responses=availability["responses"],
                            rosters=rosters,
                            method_parameters=variant_parameters,
                            capacity_per_member=int(e4b["capacity_per_member_per_trace"]),
                            source_false_negative_count=int(
                                source["source_false_negative_count"]
                            ),
                            normalized_costs=normalized_costs,
                        )
                        claimed = claimed_sevc_reserve_reliability(
                            1.0 - float(dropout), budget
                        )
                        completed = int(aggregate["completed_task_count"])
                        task_count = int(e4b["tasks_per_trace"])
                        brier = (
                            completed * (1.0 - claimed) ** 2
                            + (task_count - completed) * claimed**2
                        ) / task_count
                        identity = {
                            "dataset": dataset,
                            "verifier_population": int(population),
                            "dropout_probability": float(dropout),
                            "bernoulli_icc": float(icc),
                            "trace_index": int(trace_index),
                            "trace_seed": trace_seed,
                            "variant": variant,
                        }
                        service = (
                            int(aggregate["successful_verifier_execution_count"])
                            * float(normalized_costs["verifier_execution"])
                        )
                        trace_row = {
                            "schema_version": "sevc-tdsc-e4-failure-trace-variant-v1",
                            "change_id": CHANGE_ID,
                            **identity,
                            "trace_variant_row_id": sha256_text(
                                canonical_json_text(identity)
                            )[:24],
                            "replacement_budget": budget,
                            "claimed_probability_bias": 0.0,
                            "claimed_member_responsiveness": 1.0 - float(dropout),
                            "claimed_reliability": claimed,
                            "source_seed": source_seed,
                            "source_method": "sevc-ctiv-registered-v1",
                            "source_unit_id": source["source_unit_id"],
                            "source_false_negative_count": int(
                                source["source_false_negative_count"]
                            ),
                            "availability_seed": availability["availability_seed"],
                            "availability_sha256": availability["availability_sha256"],
                            "task_roster_sha256": roster_sha,
                            "latent_responsiveness": availability["latent_probability"],
                            "task_count": task_count,
                            **aggregate,
                            "empirical_completion_rate": completed / task_count,
                            "brier_score": brier,
                            "service_expenditure_units": service,
                            "combined_expenditure_units": float(
                                aggregate["owner_expenditure_units"]
                            )
                            + service,
                            "reserve_activation_task_count": sum(
                                int(row["recovered_slots"]) > 0 for row in nested
                            ),
                            "simulation_compute_device": "cpu",
                        }
                        trace_rows.append(trace_row)
                        for task in nested:
                            task_identity = {
                                **identity,
                                "task_index": int(task["task_index"]),
                            }
                            task_rows.append(
                                {
                                    "schema_version": "sevc-tdsc-e4-nested-task-observation-v1",
                                    "change_id": CHANGE_ID,
                                    **task_identity,
                                    "trace_variant_row_id": trace_row[
                                        "trace_variant_row_id"
                                    ],
                                    "task_observation_id": sha256_text(
                                        canonical_json_text(task_identity)
                                    )[:24],
                                    "replacement_budget": budget,
                                    "source_unit_id": source["source_unit_id"],
                                    "availability_sha256": availability[
                                        "availability_sha256"
                                    ],
                                    "task_roster_sha256": roster_sha,
                                    "roster_member_ids": list(
                                        rosters[int(task["task_index"])]
                                    ),
                                    **task,
                                }
                            )
    return trace_rows, task_rows


def _wilson(successes: int, total: int) -> dict[str, Any]:
    return {"x": successes, "n": total, **asdict(wilson_interval(successes, total))}


def _numeric(values: Sequence[float]) -> dict[str, float]:
    return {
        "mean": float(statistics.mean(values)),
        "median": float(statistics.median(values)),
        "minimum": float(min(values)),
        "maximum": float(max(values)),
    }


def _summarize_direct(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["dataset"], row["state"], row["variant"])].append(row)
    output = []
    binary = (
        "complete",
        "unsafe_settlement",
        "successful_recovery",
        "safe_deferral",
        "termination_or_restart",
    )
    numeric = (
        "recovered_slots",
        "replay_executions",
        "owner_expenditure_units",
        "verifier_service_cost_units",
        "combined_expenditure_units",
        "bond_exposure_units",
    )
    for key in sorted(groups):
        scoped = groups[key]
        payload = {
            "dataset": key[0],
            "state": key[1],
            "variant": key[2],
            "replacement_budget": int(scoped[0]["replacement_budget"]),
            "registered_trace_count": len(scoped),
            "route_counts": dict(
                sorted(
                    {
                        route: sum(str(row["route"]) == route for row in scoped)
                        for route in {
                            "complete-without-recovery",
                            "complete-after-recovery",
                            "safe-defer",
                            "terminate-or-restart",
                        }
                    }.items()
                )
            ),
        }
        for field in binary:
            payload[field] = _wilson(sum(bool(row[field]) for row in scoped), len(scoped))
        for field in numeric:
            payload[field] = _numeric([float(row[field]) for row in scoped])
        output.append(payload)
    return output


def _summarize_failure(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    fields = (
        "dataset",
        "verifier_population",
        "dropout_probability",
        "bernoulli_icc",
        "variant",
    )
    for row in rows:
        groups[tuple(row[field] for field in fields)].append(row)
    output = []
    task_counts = (
        "completed_task_count",
        "successful_recovery_task_count",
        "safe_deferral_task_count",
        "unsafe_settlement_task_count",
        "capacity_rejection_task_count",
    )
    numeric = (
        "successful_verifier_execution_count",
        "commissioned_slot_count",
        "capacity_rejected_slot_count",
        "recovered_slot_count",
        "owner_expenditure_units",
        "service_expenditure_units",
        "combined_expenditure_units",
        "refundable_bond_exposure_units",
        "slashed_bond_exposure_units",
        "liquidity_cost_units",
    )
    for key in sorted(groups):
        scoped = groups[key]
        task_n = sum(int(row["task_count"]) for row in scoped)
        payload = {field: value for field, value in zip(fields, key)}
        payload.update(
            {
                "replacement_budget": int(scoped[0]["replacement_budget"]),
                "registered_trace_count": len(scoped),
                "nested_task_count": task_n,
                "trace_any_completion": _wilson(
                    sum(int(row["completed_task_count"]) > 0 for row in scoped), len(scoped)
                ),
                "trace_all_tasks_complete": _wilson(
                    sum(
                        int(row["completed_task_count"]) == int(row["task_count"])
                        for row in scoped
                    ),
                    len(scoped),
                ),
                "trace_any_recovery": _wilson(
                    sum(int(row["successful_recovery_task_count"]) > 0 for row in scoped),
                    len(scoped),
                ),
                "trace_any_deferral": _wilson(
                    sum(int(row["safe_deferral_task_count"]) > 0 for row in scoped),
                    len(scoped),
                ),
            }
        )
        for field in task_counts:
            x = sum(int(row[field]) for row in scoped)
            payload[field] = {
                "task_x": x,
                "task_n": task_n,
                "descriptive_task_rate": x / task_n,
                "trace_rate_mean": statistics.mean(
                    int(row[field]) / int(row["task_count"]) for row in scoped
                ),
            }
        for field in numeric:
            payload[field] = _numeric([float(row[field]) for row in scoped])
        output.append(payload)
    return output


def _bootstrap_seed(base: int, identity: Sequence[Any]) -> int:
    payload = canonical_json_text([base, *identity]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _bootstrap_effects(
    direct: Sequence[Mapping[str, Any]], failure: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]
) -> list[dict[str, Any]]:
    statistics_contract = protocol["statistics"]
    resamples = int(statistics_contract["paired_bootstrap_resamples"])
    base_seed = int(statistics_contract["paired_bootstrap_seed"])
    contrasts = (
        ("b3-b0", "full-sevc", "sevc-reserve-b0"),
        ("b1-b0", "sevc-reserve-b1", "sevc-reserve-b0"),
        ("b2-b1", "sevc-reserve-b2", "sevc-reserve-b1"),
        ("b3-b2", "full-sevc", "sevc-reserve-b2"),
    )
    output: list[dict[str, Any]] = []

    direct_groups: dict[tuple[Any, ...], dict[str, list[Mapping[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in direct:
        direct_groups[(row["dataset"], row["state"])][str(row["variant"])].append(row)
    for cell, variants in sorted(direct_groups.items()):
        ordered = {
            variant: sorted(rows, key=lambda row: int(row["trace_index"]))
            for variant, rows in variants.items()
        }
        n = len(next(iter(ordered.values())))
        rng = np.random.default_rng(_bootstrap_seed(base_seed, ["e4-a", *cell]))
        indices = rng.integers(0, n, size=(resamples, n), endpoint=False)
        endpoints = {
            "completion_rate": lambda row: float(bool(row["complete"])),
            "recovery_rate": lambda row: float(bool(row["successful_recovery"])),
            "combined_expenditure_units": lambda row: float(row["combined_expenditure_units"]),
        }
        for contrast, high, low in contrasts:
            for endpoint, extract in endpoints.items():
                differences = np.asarray(
                    [
                        extract(high_row) - extract(low_row)
                        for high_row, low_row in zip(ordered[high], ordered[low])
                    ],
                    dtype=float,
                )
                samples = differences[indices].mean(axis=1)
                output.append(
                    {
                        "matrix": "E4-A",
                        "dataset": cell[0],
                        "state": cell[1],
                        "contrast": contrast,
                        "endpoint": endpoint,
                        "paired_trace_count": n,
                        "resamples": resamples,
                        "seed": base_seed,
                        "point_estimate": float(differences.mean()),
                        "bootstrap_mean": float(samples.mean()),
                        "lower95": float(np.quantile(samples, 0.025)),
                        "upper95": float(np.quantile(samples, 0.975)),
                    }
                )

    failure_groups: dict[tuple[Any, ...], dict[str, list[Mapping[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in failure:
        cell = (
            row["dataset"],
            row["verifier_population"],
            row["dropout_probability"],
            row["bernoulli_icc"],
        )
        failure_groups[cell][str(row["variant"])].append(row)
    for cell, variants in sorted(failure_groups.items()):
        ordered = {
            variant: sorted(rows, key=lambda row: int(row["trace_index"]))
            for variant, rows in variants.items()
        }
        n = len(next(iter(ordered.values())))
        rng = np.random.default_rng(_bootstrap_seed(base_seed, ["e4-b", *cell]))
        indices = rng.integers(0, n, size=(resamples, n), endpoint=False)
        endpoints = {
            "completion_rate": lambda row: int(row["completed_task_count"])
            / int(row["task_count"]),
            "recovery_rate": lambda row: int(row["successful_recovery_task_count"])
            / int(row["task_count"]),
            "combined_expenditure_units": lambda row: float(row["combined_expenditure_units"]),
        }
        for contrast, high, low in contrasts:
            for endpoint, extract in endpoints.items():
                differences = np.asarray(
                    [
                        extract(high_row) - extract(low_row)
                        for high_row, low_row in zip(ordered[high], ordered[low])
                    ],
                    dtype=float,
                )
                samples = differences[indices].mean(axis=1)
                output.append(
                    {
                        "matrix": "E4-B",
                        "dataset": cell[0],
                        "verifier_population": cell[1],
                        "dropout_probability": cell[2],
                        "bernoulli_icc": cell[3],
                        "contrast": contrast,
                        "endpoint": endpoint,
                        "paired_trace_count": n,
                        "resamples": resamples,
                        "seed": base_seed,
                        "point_estimate": float(differences.mean()),
                        "bootstrap_mean": float(samples.mean()),
                        "lower95": float(np.quantile(samples, 0.025)),
                        "upper95": float(np.quantile(samples, 0.975)),
                    }
                )
    return output


