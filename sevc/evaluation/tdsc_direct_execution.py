"""Pure EVAL-E1 trace expansion and B0--B5 system comparison."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import statistics
from typing import Any, Callable, Mapping, Sequence

from sevc.core.artifacts import canonical_json_text, sha256_text
from sevc.core.registry import Registry
from sevc.evaluation.statistics import wilson_interval


METHODS = (
    "no-verification",
    "owner-only-reference-replay",
    "vanilla-pol-replay",
    "depol-redundant-replay-style",
    "refiner-two-stage-audit-style",
    "full-sevc",
)


@dataclass(frozen=True)
class DirectExecutionContext:
    trace: Mapping[str, Any]
    source_unit: Mapping[str, Any]
    protocol: Mapping[str, Any]


@dataclass(frozen=True)
class MethodDecision:
    native_outcome: str
    source_method: str
    replay_executions: int = 0
    audit_executions: int = 0
    owner_reference_executions: int = 0
    commissioned_verifier_slots: int = 0
    successful_recovery: bool = False
    recovered_slots: int = 0
    slow_path_invoked: bool = False


SYSTEM_METHODS: Registry[Callable[[DirectExecutionContext], MethodDecision]] = Registry(
    "direct execution method"
)
SEVC_RESERVE_VARIANTS: Registry[
    Callable[[DirectExecutionContext], MethodDecision]
] = Registry("SEVC reserve-budget variant")
SEVC_RESERVE_BUDGETS = {
    "sevc-reserve-b0": 0,
    "sevc-reserve-b1": 1,
    "sevc-reserve-b2": 2,
    "full-sevc": 3,
}


def _method_row(source_unit: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    rows = {
        str(row["method"]): row for row in source_unit["trajectory_rows"]
    }
    if set(rows) != {
        "none",
        "sevc-ctiv-registered-v1",
        "refiner-update-audit-style",
    }:
        raise ValueError("E1 source unit method identity drift")
    return rows[key]


def _method_signal_outcome(
    context: DirectExecutionContext, source_method: str
) -> str:
    metrics = _method_row(context.source_unit, source_method)["metrics"]
    trainer_bad = bool(context.trace["trainer_bad"])
    if trainer_bad:
        all_bad_detected = int(metrics["TP"]) > 0 and int(metrics["FN"]) == 0
        return "reject" if all_bad_detected else "accept"
    return "reject" if int(metrics["FP"]) > 0 else "accept"


def _responsive(role: str) -> bool:
    return role == "responsive"


@SYSTEM_METHODS.register("no-verification")
def _no_verification(context: DirectExecutionContext) -> MethodDecision:
    del context
    return MethodDecision(native_outcome="accept", source_method="none")


@SYSTEM_METHODS.register("owner-only-reference-replay")
def _owner_only(context: DirectExecutionContext) -> MethodDecision:
    return MethodDecision(
        native_outcome=_method_signal_outcome(
            context, "sevc-ctiv-registered-v1"
        ),
        source_method="sevc-ctiv-registered-v1",
        owner_reference_executions=1,
    )


@SYSTEM_METHODS.register("vanilla-pol-replay")
def _vanilla_pol(context: DirectExecutionContext) -> MethodDecision:
    first_role = str(context.trace["roster_roles"][0])
    if not _responsive(first_role):
        return MethodDecision(
            native_outcome="terminate",
            source_method="sevc-ctiv-registered-v1",
            commissioned_verifier_slots=1,
        )
    return MethodDecision(
        native_outcome=_method_signal_outcome(
            context, "sevc-ctiv-registered-v1"
        ),
        source_method="sevc-ctiv-registered-v1",
        replay_executions=1,
        commissioned_verifier_slots=1,
    )


@SYSTEM_METHODS.register("depol-redundant-replay-style")
def _depol_style(context: DirectExecutionContext) -> MethodDecision:
    group_size = int(context.protocol["method_parameters"]["depol_group_size"])
    majority = int(context.protocol["method_parameters"]["depol_majority"])
    roles = tuple(str(value) for value in context.trace["roster_roles"][:group_size])
    responsive = sum(_responsive(role) for role in roles)
    if responsive < majority:
        return MethodDecision(
            native_outcome="terminate",
            source_method="sevc-ctiv-registered-v1",
            replay_executions=responsive,
            commissioned_verifier_slots=group_size,
        )
    return MethodDecision(
        native_outcome=_method_signal_outcome(
            context, "sevc-ctiv-registered-v1"
        ),
        source_method="sevc-ctiv-registered-v1",
        replay_executions=responsive,
        commissioned_verifier_slots=group_size,
        slow_path_invoked=responsive < group_size,
    )


@SYSTEM_METHODS.register("refiner-two-stage-audit-style")
def _refiner_style(context: DirectExecutionContext) -> MethodDecision:
    committee_size = int(
        context.protocol["method_parameters"]["refiner_committee_size"]
    )
    supermajority = int(
        context.protocol["method_parameters"]["refiner_supermajority"]
    )
    roles = tuple(
        str(value) for value in context.trace["roster_roles"][:committee_size]
    )
    responsive = sum(_responsive(role) for role in roles)
    if responsive < supermajority:
        return MethodDecision(
            native_outcome="terminate",
            source_method="refiner-update-audit-style",
            audit_executions=responsive,
            commissioned_verifier_slots=committee_size,
        )
    return MethodDecision(
        native_outcome=_method_signal_outcome(
            context, "refiner-update-audit-style"
        ),
        source_method="refiner-update-audit-style",
        audit_executions=responsive,
        commissioned_verifier_slots=committee_size,
    )


def evaluate_sevc_reserve_variant(
    context: DirectExecutionContext, replacement_budget: int
) -> MethodDecision:
    """Evaluate the one SEVC reserve path with an explicit registered budget."""

    if replacement_budget not in set(SEVC_RESERVE_BUDGETS.values()):
        raise ValueError("unregistered SEVC replacement budget")
    parameters = context.protocol["method_parameters"]
    initial_slots = int(parameters["sevc_committee_count"]) * int(
        parameters["sevc_committee_size"]
    )
    roles = tuple(str(value) for value in context.trace["roster_roles"])
    initial_responsive = sum(_responsive(role) for role in roles[:initial_slots])
    missing = initial_slots - initial_responsive
    reserve_responsive = sum(_responsive(role) for role in roles[initial_slots:])
    recovered = min(missing, replacement_budget, reserve_responsive)
    completed = initial_responsive + recovered
    if completed < initial_slots:
        return MethodDecision(
            native_outcome="defer",
            source_method="sevc-ctiv-registered-v1",
            replay_executions=completed,
            commissioned_verifier_slots=initial_slots + recovered,
            successful_recovery=False,
            recovered_slots=recovered,
        )
    return MethodDecision(
        native_outcome=_method_signal_outcome(
            context, "sevc-ctiv-registered-v1"
        ),
        source_method="sevc-ctiv-registered-v1",
        replay_executions=completed,
        commissioned_verifier_slots=initial_slots + recovered,
        successful_recovery=recovered > 0,
        recovered_slots=recovered,
    )


@SEVC_RESERVE_VARIANTS.register("sevc-reserve-b0")
def _sevc_reserve_b0(context: DirectExecutionContext) -> MethodDecision:
    return evaluate_sevc_reserve_variant(context, 0)


@SEVC_RESERVE_VARIANTS.register("sevc-reserve-b1")
def _sevc_reserve_b1(context: DirectExecutionContext) -> MethodDecision:
    return evaluate_sevc_reserve_variant(context, 1)


@SEVC_RESERVE_VARIANTS.register("sevc-reserve-b2")
def _sevc_reserve_b2(context: DirectExecutionContext) -> MethodDecision:
    return evaluate_sevc_reserve_variant(context, 2)


@SEVC_RESERVE_VARIANTS.register("full-sevc")
def _sevc_reserve_b3(context: DirectExecutionContext) -> MethodDecision:
    return evaluate_sevc_reserve_variant(context, 3)


@SYSTEM_METHODS.register("full-sevc")
def _full_sevc(context: DirectExecutionContext) -> MethodDecision:
    replacement_budget = int(
        context.protocol["method_parameters"]["sevc_replacement_budget"]
    )
    return evaluate_sevc_reserve_variant(context, replacement_budget)


def _role_for_member(member_id: int) -> str:
    if 0 <= member_id <= 16:
        return "responsive"
    if 17 <= member_id <= 29:
        return "selective-lazy"
    if 30 <= member_id <= 32:
        return "dropout"
    raise ValueError(f"invalid roster member {member_id}")


def _roster_order(trace_seed: int, size: int) -> tuple[int, ...]:
    return tuple(
        sorted(
            range(size),
            key=lambda member_id: hashlib.sha256(
                f"{int(trace_seed)}|{member_id}".encode("utf-8")
            ).hexdigest(),
        )
    )


def build_trace_registry(
    source_units: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
    *,
    datasets: Sequence[str],
    trace_limit: int | None = None,
) -> list[dict[str, Any]]:
    seeds = tuple(int(value) for value in protocol["formal_paired_seeds"])
    trace_seeds = tuple(int(value) for value in protocol["trace_seeds"])
    if trace_limit is not None:
        if trace_limit <= 0 or trace_limit > len(trace_seeds):
            raise ValueError("invalid E1 trace limit")
        trace_seeds = trace_seeds[:trace_limit]
    units = {
        (
            str(unit["identity"]["dataset"]),
            str(unit["identity"]["state"]),
            int(unit["identity"]["seed"]),
        ): unit
        for unit in source_units
    }
    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        for system_state in protocol["system_states"]:
            mapping = protocol["state_source_mapping"][system_state]
            source_state = str(mapping["source_state"])
            available = [
                units[(dataset, source_state, seed)]
                for seed in seeds
                if (dataset, source_state, seed) in units
            ]
            if not available:
                raise ValueError(
                    f"no E1 source units for {dataset}/{source_state}"
                )
            for trace_index, trace_seed in enumerate(trace_seeds):
                source_unit = available[trace_index % len(available)]
                roster_order = _roster_order(
                    trace_seed, int(protocol["roster"]["size"])
                )
                if bool(mapping["verifier_failures"]):
                    roster_roles = tuple(_role_for_member(value) for value in roster_order)
                else:
                    roster_roles = tuple("responsive" for _ in roster_order)
                identity = {
                    "change_id": protocol["change_id"],
                    "dataset": dataset,
                    "system_state": system_state,
                    "trace_index": trace_index,
                    "trace_seed": trace_seed,
                    "source_unit_id": source_unit["identity"]["source_unit_id"],
                }
                rows.append(
                    {
                        **identity,
                        "trace_id": sha256_text(canonical_json_text(identity))[:24],
                        "source_state": source_state,
                        "source_seed": int(source_unit["identity"]["seed"]),
                        "trainer_bad": bool(mapping["trainer_bad"]),
                        "verifier_failures": bool(mapping["verifier_failures"]),
                        "roster_order": list(roster_order),
                        "roster_roles": list(roster_roles),
                        "roster_order_sha256": sha256_text(
                            canonical_json_text(list(roster_order))
                        ),
                    }
                )
    if len({row["trace_id"] for row in rows}) != len(rows):
        raise ValueError("duplicate E1 trace identity")
    return rows


def _event_chain(
    row: Mapping[str, Any], decision: MethodDecision
) -> tuple[list[dict[str, Any]], str]:
    payloads = (
        {"trace_id": row["trace_id"], "method": row["method"]},
        {
            "source_method": decision.source_method,
            "replay_executions": decision.replay_executions,
            "audit_executions": decision.audit_executions,
        },
        {
            "native_outcome": decision.native_outcome,
            "slow_path_invoked": decision.slow_path_invoked,
        },
        {
            "successful_recovery": decision.successful_recovery,
            "recovered_slots": decision.recovered_slots,
        },
    )
    events: list[dict[str, Any]] = []
    parent: str | None = None
    for stage, payload in zip(
        ("assignment", "verification", "settlement", "recovery"), payloads
    ):
        payload_sha = sha256_text(canonical_json_text(payload))
        event_id = sha256_text(
            canonical_json_text(
                {
                    "trace_id": row["trace_id"],
                    "method": row["method"],
                    "stage": stage,
                    "parent_event_id": parent,
                    "payload_sha256": payload_sha,
                }
            )
        )[:32]
        events.append(
            {
                "dataset": row["dataset"],
                "system_state": row["system_state"],
                "trace_id": row["trace_id"],
                "method": row["method"],
                "stage": stage,
                "event_id": event_id,
                "parent_event_id": parent,
                "payload_sha256": payload_sha,
            }
        )
        parent = event_id
    return events, sha256_text(
        canonical_json_text([event["event_id"] for event in events])
    )


def build_direct_outcomes(
    source_units: Sequence[Mapping[str, Any]],
    trace_registry: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
    fidelity: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    units = {
        str(unit["identity"]["source_unit_id"]): unit for unit in source_units
    }
    visibility = {
        str(row["key"]): list(row["information"])
        for row in fidelity["methods"].values()
    }
    costs = protocol["normalized_costs"]
    rows: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    for trace in trace_registry:
        source_unit = units[str(trace["source_unit_id"])]
        context = DirectExecutionContext(
            trace=trace, source_unit=source_unit, protocol=protocol
        )
        for method in METHODS:
            decision = SYSTEM_METHODS.get(method)(context)
            source_metrics = _method_row(
                source_unit, decision.source_method
            )["metrics"]
            unsafe = bool(
                trace["trainer_bad"]
                and decision.native_outcome == "accept"
                and int(source_metrics["FN"]) > 0
            )
            row = {
                "schema_version": "sevc-tdsc-e1-direct-outcome-v1",
                "change_id": protocol["change_id"],
                "dataset": trace["dataset"],
                "system_state": trace["system_state"],
                "trace_id": trace["trace_id"],
                "trace_seed": int(trace["trace_seed"]),
                "source_unit_id": trace["source_unit_id"],
                "source_seed": int(trace["source_seed"]),
                "method": method,
                "source_method": decision.source_method,
                "native_outcome": decision.native_outcome,
                "unsafe_settlement": unsafe,
                "successful_recovery": decision.successful_recovery,
                "recovered_slots": decision.recovered_slots,
                "safe_deferral": decision.native_outcome == "defer",
                "termination_or_restart": decision.native_outcome
                in {"terminate", "restart"},
                "slow_path_invoked": decision.slow_path_invoked,
                "replay_executions": decision.replay_executions,
                "audit_executions": decision.audit_executions,
                "owner_reference_executions": decision.owner_reference_executions,
                "owner_expenditure_units": (
                    decision.owner_reference_executions
                    * float(costs["owner_reference_execution"])
                    + (decision.replay_executions + decision.audit_executions)
                    * float(costs["verifier_execution"])
                    * float(costs["service_fee_ratio"])
                ),
                "verifier_service_cost_units": (
                    decision.replay_executions + decision.audit_executions
                )
                * float(costs["verifier_execution"]),
                "bond_exposure_units": decision.commissioned_verifier_slots
                * float(costs["verifier_execution"])
                * float(costs["bond_ratio"]),
                "visible_information": visibility[method],
                "source_trainer_TPR": float(source_metrics["trainer_TPR"]),
                "source_trainer_FPR": float(source_metrics["trainer_FPR"]),
                "source_clean_accuracy": float(source_metrics["clean_test_accuracy"]),
                "source_triggered_ASR": float(source_metrics["triggered_ASR"]),
                "source_update_bank_chain_sha256": _method_row(
                    source_unit, decision.source_method
                )["update_bank_chain_sha256"],
            }
            chain, chain_sha = _event_chain(row, decision)
            row["event_chain_sha256"] = chain_sha
            rows.append(row)
            events.extend(chain)
    logical = {
        (row["trace_id"], row["method"]) for row in rows
    }
    if len(logical) != len(rows):
        raise ValueError("duplicate E1 outcome identity")
    return rows, events


def _binary_summary(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, Any]:
    successes = sum(bool(row[field]) for row in rows)
    return asdict(wilson_interval(successes, len(rows)))


def _numeric_summary(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, Any]:
    values = [float(row[field]) for row in rows]
    return {
        "denominator": len(values),
        "median": float(statistics.median(values)),
        "minimum": min(values),
        "maximum": max(values),
    }


def evaluate_dataset_safety(
    outcomes: Sequence[Mapping[str, Any]], dataset: str, expected_traces: int
) -> dict[str, Any]:
    scoped = [row for row in outcomes if str(row["dataset"]) == dataset]
    reasons: list[str] = []
    for state in (
        "trainer-segment-deviation",
        "compound-trainer-verifier-failure",
    ):
        b0 = [
            row
            for row in scoped
            if row["system_state"] == state and row["method"] == "no-verification"
        ]
        b5 = [
            row
            for row in scoped
            if row["system_state"] == state and row["method"] == "full-sevc"
        ]
        if len(b0) != expected_traces or len(b5) != expected_traces:
            reasons.append(f"cardinality:{state}:{len(b0)}:{len(b5)}")
            continue
        b0_unsafe = sum(bool(row["unsafe_settlement"]) for row in b0)
        b5_unsafe = sum(bool(row["unsafe_settlement"]) for row in b5)
        safe_routes = sum(
            row["native_outcome"] in {"reject", "defer"}
            and not bool(row["unsafe_settlement"])
            for row in b5
        )
        if b5_unsafe != 0:
            reasons.append(f"full-sevc-unsafe:{state}:{b5_unsafe}")
        if b0_unsafe <= b5_unsafe:
            reasons.append(f"no-safety-contrast:{state}:{b0_unsafe}:{b5_unsafe}")
        if safe_routes != expected_traces:
            reasons.append(f"full-sevc-safe-route:{state}:{safe_routes}")
    return {
        "dataset": dataset,
        "valid": not any(reason.startswith("cardinality:") for reason in reasons),
        "passed": not reasons,
        "reasons": reasons,
    }


def build_comparison_summary(
    outcomes: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
    *,
    datasets: Sequence[str],
    expected_traces: int,
    expected_source_banks: int,
    source_units: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    cells: list[dict[str, Any]] = []
    for dataset in datasets:
        for state in protocol["system_states"]:
            for method in METHODS:
                scoped = [
                    row
                    for row in outcomes
                    if row["dataset"] == dataset
                    and row["system_state"] == state
                    and row["method"] == method
                ]
                if len(scoped) != expected_traces:
                    raise ValueError(
                        f"E1 summary cardinality drift: {dataset}/{state}/{method}"
                    )
                cells.append(
                    {
                        "dataset": dataset,
                        "system_state": state,
                        "method": method,
                        "unsafe_settlement": _binary_summary(
                            scoped, "unsafe_settlement"
                        ),
                        "successful_recovery": _binary_summary(
                            scoped, "successful_recovery"
                        ),
                        "safe_deferral": _binary_summary(scoped, "safe_deferral"),
                        "termination_or_restart": _binary_summary(
                            scoped, "termination_or_restart"
                        ),
                        "replay_executions": _numeric_summary(
                            scoped, "replay_executions"
                        ),
                        "audit_executions": _numeric_summary(
                            scoped, "audit_executions"
                        ),
                        "owner_reference_executions": _numeric_summary(
                            scoped, "owner_reference_executions"
                        ),
                        "owner_expenditure_units": _numeric_summary(
                            scoped, "owner_expenditure_units"
                        ),
                        "verifier_service_cost_units": _numeric_summary(
                            scoped, "verifier_service_cost_units"
                        ),
                        "bond_exposure_units": _numeric_summary(
                            scoped, "bond_exposure_units"
                        ),
                    }
                )
    dataset_gates = {
        dataset: evaluate_dataset_safety(outcomes, dataset, expected_traces)
        for dataset in datasets
    }
    expected_trace_count = len(datasets) * len(protocol["system_states"]) * expected_traces
    integrity = {
        "source_bank_count": len(source_units),
        "expected_source_bank_count": expected_source_banks,
        "expanded_trace_count": len({str(row["trace_id"]) for row in outcomes}),
        "expected_expanded_trace_count": expected_trace_count,
        "outcome_count": len(outcomes),
        "expected_outcome_count": expected_trace_count * len(METHODS),
        "summary_cell_count": len(cells),
        "expected_summary_cell_count": len(datasets)
        * len(protocol["system_states"])
        * len(METHODS),
        "six_methods_per_trace": all(
            {
                str(row["method"])
                for row in outcomes
                if row["trace_id"] == trace_id
            }
            == set(METHODS)
            for trace_id in {str(row["trace_id"]) for row in outcomes}
        ),
    }
    integrity["passed"] = (
        integrity["source_bank_count"] == integrity["expected_source_bank_count"]
        and integrity["expanded_trace_count"]
        == integrity["expected_expanded_trace_count"]
        and integrity["outcome_count"] == integrity["expected_outcome_count"]
        and integrity["summary_cell_count"]
        == integrity["expected_summary_cell_count"]
        and integrity["six_methods_per_trace"]
    )
    return {
        "schema_version": "sevc-tdsc-e1-comparison-summary-v1",
        "change_id": protocol["change_id"],
        "integrity": integrity,
        "dataset_gates": dataset_gates,
        "cells": cells,
        "passed": bool(
            integrity["passed"]
            and all(value["passed"] for value in dataset_gates.values())
        ),
        "pooled_override_forbidden": True,
    }


def extract_source_metrics(
    source_units: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    rows = []
    for unit in source_units:
        for trajectory in unit["trajectory_rows"]:
            rows.append(
                {
                    "dataset": unit["identity"]["dataset"],
                    "source_state": unit["identity"]["state"],
                    "source_seed": int(unit["identity"]["seed"]),
                    "source_unit_id": unit["identity"]["source_unit_id"],
                    "method": trajectory["method"],
                    "update_bank_chain_sha256": trajectory[
                        "update_bank_chain_sha256"
                    ],
                    "metrics": trajectory["metrics"],
                    "method_details": trajectory["method_details"],
                }
            )
    return {
        "schema_version": "sevc-tdsc-e1-source-metrics-v1",
        "source_bank_count": len(source_units),
        "source_method_row_count": len(rows),
        "rows": rows,
    }


__all__ = [
    "METHODS",
    "SYSTEM_METHODS",
    "SEVC_RESERVE_BUDGETS",
    "SEVC_RESERVE_VARIANTS",
    "evaluate_sevc_reserve_variant",
    "DirectExecutionContext",
    "MethodDecision",
    "build_comparison_summary",
    "build_direct_outcomes",
    "build_trace_registry",
    "evaluate_dataset_safety",
    "extract_source_metrics",
]
