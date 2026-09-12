"""Frozen COMP-E3 RCMP ablation and sensitivity evaluation.

The evaluator consumes immutable T3/T4/CIFAR-100 source ledgers.  It never
trains a model or creates a trainer trajectory; all variants are settlement
counterfactuals over the same block-level action and cost observations.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from sevc.evaluation.rcmp_gates import (
    DEVIATIONS,
    attack_bound_rows,
    evaluate_action_gate,
    evaluate_utility_gate,
)


VARIANTS = (
    "production-only",
    "public-probe",
    "independent-hidden-probe",
    "full-RCMP",
)

CHALLENGE_ATOM_ORDER = (
    "cp3-negative",
    "cp3-positive",
    "cp4-negative",
    "cp4-positive",
)
_CHALLENGE_ATOM_RANK = {
    atom: index for index, atom in enumerate(CHALLENGE_ATOM_ORDER)
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def verify_manifest(path: Path) -> list[str]:
    failures: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        target = path.parent / relative
        observed = _sha256(target) if target.is_file() else None
        if observed != expected:
            failures.append(relative)
    return failures


def load_source(dataset: str, identity: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(str(identity["root"]))
    stage = str(identity["stage"])
    stage_root = root / "evidence" / stage
    raw = stage_root / "raw"
    checks = {
        "root_manifest_file_sha256": _sha256(root / "manifest.sha256"),
        "result_index_sha256": _sha256(root / "result_index.json"),
        "gate_sha256": _sha256(stage_root / "gate_result.json"),
        "raw_manifest_file_sha256": _sha256(raw / "manifest.sha256"),
        "source_ledger_sha256": _sha256(raw / "source_ledger.jsonl"),
    }
    mismatches = [key for key, value in checks.items() if identity.get(key) != value]
    root_manifest_failures = verify_manifest(root / "manifest.sha256")
    raw_manifest_failures = verify_manifest(raw / "manifest.sha256")
    source_rows = _jsonl(raw / "source_ledger.jsonl")
    block_rows = _jsonl(raw / "block_ledger.jsonl")
    action_rows = _jsonl(raw / "action_ledger.jsonl")
    if len(source_rows) != int(identity["sources"]):
        mismatches.append("source_cardinality")
    if len(block_rows) != int(identity["blocks"]):
        mismatches.append("block_cardinality")
    if len(action_rows) != int(identity["sources"]):
        mismatches.append("action_cardinality")
    if any(
        row.get("dataset") is not None and str(row.get("dataset")) != dataset
        for row in source_rows
    ):
        mismatches.append("dataset_identity")
    return {
        "dataset": dataset,
        "root": root,
        "stage": stage,
        "stage_root": stage_root,
        "raw": raw,
        "gate": _json(stage_root / "gate_result.json"),
        "source_rows": source_rows,
        "block_rows": block_rows,
        "action_rows": action_rows,
        "hashes": checks,
        "mismatches": sorted(set(mismatches)),
        "root_manifest_failures": root_manifest_failures,
        "raw_manifest_failures": raw_manifest_failures,
        "passed": not mismatches and not root_manifest_failures and not raw_manifest_failures,
    }


def _variant_blocks(
    block_rows: Sequence[Mapping[str, Any]],
    variant: str,
    *,
    action_rows: Sequence[Mapping[str, Any]] | None = None,
    probe_pair_count: int | None = None,
    failure_threshold: int | None = None,
) -> list[dict[str, Any]]:
    probe_requested = probe_pair_count is not None or failure_threshold is not None
    if probe_requested and (probe_pair_count is None or failure_threshold is None):
        raise ValueError("probe pair count and failure threshold must be supplied together")
    mismatch_counts: dict[tuple[int, str], int] | None = None
    if probe_requested:
        if action_rows is None:
            raise ValueError("probe sensitivity requires task-level action rows")
        mismatch_counts = _probe_mismatch_counts(
            action_rows, probe_pair_count=int(probe_pair_count)
        )
    result: list[dict[str, Any]] = []
    for source in block_rows:
        row = dict(source)
        if "block_seed" not in row:
            raise ValueError("block row is missing block_seed")
        block_seed = int(row["block_seed"])
        for action in DEVIATIONS:
            detected = bool(row[f"{action}_detected"])
            if variant in {"production-only", "public-probe"}:
                detected = False
            elif mismatch_counts is not None:
                key = (block_seed, action)
                if key not in mismatch_counts:
                    raise ValueError(
                        f"action ledger is missing block/action group {key}"
                    )
                detected = mismatch_counts[key] >= int(failure_threshold)
            row[f"{action}_detected"] = detected
        result.append(row)
    return result


def _probe_mismatch_counts(
    action_rows: Sequence[Mapping[str, Any]],
    *,
    probe_pair_count: int,
) -> dict[tuple[int, str], int]:
    """Reconstruct frozen probe-prefix mismatches from task-level action rows.

    Each block/action group must contain exactly four controls and four challenges.
    Controls are ordered by source index; challenges use the preregistered atom order
    and then source index.  The function deliberately has no block-level fallback.
    """

    if probe_pair_count not in {2, 3, 4}:
        raise ValueError(f"unsupported probe pair count: {probe_pair_count}")
    required = {
        "action",
        "atom_key",
        "block_seed",
        "mismatch",
        "repetition",
        "role",
        "source_index",
    }
    grouped: dict[tuple[int, str], dict[str, list[Mapping[str, Any]]]] = {}
    for index, row in enumerate(action_rows):
        missing = sorted(required.difference(row))
        if missing:
            raise ValueError(f"action row {index} is missing fields: {missing}")
        action = str(row["action"])
        if action not in DEVIATIONS:
            continue
        role = str(row["role"])
        if role not in {"control", "challenge"}:
            raise ValueError(f"invalid role in action row {index}: {role}")
        if type(row["mismatch"]) is not bool:
            raise ValueError(f"non-boolean mismatch in action row {index}")
        if int(row["repetition"]) != 1:
            raise ValueError(f"unexpected repetition in action row {index}")
        atom = row["atom_key"]
        if role == "control" and atom is not None:
            raise ValueError(f"control row {index} has an atom key")
        if role == "challenge" and atom not in _CHALLENGE_ATOM_RANK:
            raise ValueError(f"challenge row {index} has invalid atom key: {atom}")
        key = (int(row["block_seed"]), action)
        grouped.setdefault(key, {"control": [], "challenge": []})[role].append(row)

    counts: dict[tuple[int, str], int] = {}
    for key, roles in grouped.items():
        controls = roles["control"]
        challenges = roles["challenge"]
        if len(controls) != 4 or len(challenges) != 4:
            raise ValueError(
                f"action ledger group {key} must have 4 controls and 4 challenges"
            )
        control_indices = [int(row["source_index"]) for row in controls]
        challenge_keys = [
            (str(row["atom_key"]), int(row["source_index"])) for row in challenges
        ]
        if len(set(control_indices)) != 4 or len(set(challenge_keys)) != 4:
            raise ValueError(f"duplicate task identity in action ledger group {key}")
        if {atom for atom, _ in challenge_keys} != set(CHALLENGE_ATOM_ORDER):
            raise ValueError(f"incomplete challenge atom coverage in group {key}")
        selected_controls = sorted(
            controls, key=lambda row: int(row["source_index"])
        )[:probe_pair_count]
        selected_challenges = sorted(
            challenges,
            key=lambda row: (
                _CHALLENGE_ATOM_RANK[str(row["atom_key"])],
                int(row["source_index"]),
            ),
        )[:probe_pair_count]
        counts[key] = sum(
            bool(row["mismatch"])
            for row in selected_controls + selected_challenges
        )
    return counts


def _variant_parameters(variant: str, full_gate: Mapping[str, Any]) -> dict[str, float]:
    recognition = full_gate["recognizability"]
    if variant == "full-RCMP":
        return {
            "lambda_F": float(recognition["lambda_F_simultaneous_upper95"]),
            "auc": float(recognition["grouped_auc_upper95"]),
            "coverage": 1.0,
            "eta_cov": 0.0,
        }
    if variant == "public-probe":
        return {"lambda_F": 1.0, "auc": 1.0, "coverage": 1.0, "eta_cov": 0.0}
    if variant == "independent-hidden-probe":
        return {
            "lambda_F": float(recognition["lambda_F_simultaneous_upper95"]),
            "auc": float(recognition["grouped_auc_upper95"]),
            "coverage": 0.0,
            "eta_cov": 1.0,
        }
    return {"lambda_F": 1.0, "auc": 1.0, "coverage": 0.0, "eta_cov": 1.0}


def _summary(
    dataset: str,
    variant: str,
    block_rows: Sequence[Mapping[str, Any]],
    full_gate: Mapping[str, Any],
    *,
    service_fee_ratio: float = 1.25,
    bond_ratio: float = 0.5,
    liquidity_cost_rate: float = 0.01,
    lambda_override: float | None = None,
    point_family: str = "ablation",
    point_id: str = "registered",
) -> dict[str, Any]:
    action = evaluate_action_gate(
        block_rows, seed=20260812, resamples=10000, alpha=0.05
    )
    utility = evaluate_utility_gate(
        block_rows,
        service_fee_ratio=service_fee_ratio,
        bond_ratio=bond_ratio,
        liquidity_cost_rate=liquidity_cost_rate,
        seed=20260812,
        resamples=10000,
        alpha=0.05,
    )
    params = _variant_parameters(variant, full_gate)
    lambda_value = params["lambda_F"] if lambda_override is None else float(lambda_override)
    probe_error = float(action["H_false_settlement_upper95"])
    bound = attack_bound_rows(
        probe_error,
        eta_cov=float(params["eta_cov"]),
        lambda_grid=(lambda_value,),
    )[0]["B_attack"]
    overhead = full_gate["overhead"]
    if variant == "production-only":
        online, all_in = 0.0, 0.0
    else:
        online = float(overhead["pooled_online_overhead_ratio"])
        all_in = float(overhead["conservative_all_in_ratio"])
    lower = utility["utility_margin_simultaneous_lower95"]
    feasible = bool(
        utility["honest_participation_margin_lower95"] >= 0.0
        and min(float(lower[action]) for action in DEVIATIONS) > 0.0
    )
    return {
        "dataset": dataset,
        "variant": variant,
        "point_family": point_family,
        "point_id": point_id,
        "block_count": len(block_rows),
        "H_false_settlement_rate": action["H_false_settlement_rate"],
        "H_false_settlement_upper95": action["H_false_settlement_upper95"],
        "L_detection_rate": action["detection_rate"]["L"],
        "C+_detection_rate": action["detection_rate"]["C+"],
        "C-_detection_rate": action["detection_rate"]["C-"],
        "D_detection_rate": action["detection_rate"]["D"],
        "L_detection_lower95": action["detection_simultaneous_lower95"]["L"],
        "C+_detection_lower95": action["detection_simultaneous_lower95"]["C+"],
        "C-_detection_lower95": action["detection_simultaneous_lower95"]["C-"],
        "D_detection_lower95": action["detection_simultaneous_lower95"]["D"],
        "lambda_F_upper95": lambda_value,
        "grouped_auc_upper95": params["auc"],
        "honest_participation_lower95": utility["honest_participation_margin_lower95"],
        "L_utility_margin_lower95": lower["L"],
        "C+_utility_margin_lower95": lower["C+"],
        "C-_utility_margin_lower95": lower["C-"],
        "D_utility_margin_lower95": lower["D"],
        "attack_coverage": params["coverage"],
        "eta_cov": params["eta_cov"],
        "registered_attack_bound": bound,
        "online_overhead_ratio": online,
        "all_in_overhead_ratio": all_in,
        "feasible_payment": feasible,
    }


def evaluate_dataset(dataset: str, source: Mapping[str, Any]) -> dict[str, Any]:
    gate = source["gate"]
    base = list(source["block_rows"])
    ablations = [
        _summary(dataset, variant, _variant_blocks(base, variant), gate)
        for variant in VARIANTS
    ]
    sensitivity: list[dict[str, Any]] = []
    for variant in VARIANTS:
        variant_base = _variant_blocks(base, variant)
        for fee in (1.0, 1.25, 1.5):
            for bond in (0.25, 0.5, 0.75):
                for liquidity in (0.0, 0.01, 0.02):
                    row = _summary(
                        dataset, variant, variant_base, gate,
                        service_fee_ratio=fee,
                        bond_ratio=bond,
                        liquidity_cost_rate=liquidity,
                        point_family="payment",
                        point_id=f"fee={fee}|bond={bond}|liquidity={liquidity}",
                    )
                    row.update({"service_fee_ratio": fee, "bond_ratio": bond, "liquidity_cost_rate": liquidity})
                    sensitivity.append(row)
        for pairs in (2, 3, 4):
            for threshold in range(1, pairs + 1):
                rows = _variant_blocks(
                    base, variant,
                    action_rows=source["action_rows"],
                    probe_pair_count=pairs,
                    failure_threshold=threshold,
                )
                row = _summary(
                    dataset, variant, rows, gate,
                    point_family="probe",
                    point_id=f"pairs={pairs}|threshold={threshold}",
                )
                row.update({"prefix_pair_count": pairs, "failure_threshold": threshold})
                sensitivity.append(row)
        for leakage in (0.0, 0.05, 0.1, 0.15, 0.2, 0.3):
            row = _summary(
                dataset, variant, variant_base, gate,
                lambda_override=leakage,
                point_family="leakage",
                point_id=f"lambda_F={leakage}",
            )
            row["lambda_F_grid"] = leakage
            sensitivity.append(row)
    full = next(row for row in ablations if row["variant"] == "full-RCMP")
    public = next(row for row in ablations if row["variant"] == "public-probe")
    independent = next(row for row in ablations if row["variant"] == "independent-hidden-probe")
    production = next(row for row in ablations if row["variant"] == "production-only")
    detection_keys = tuple(f"{name}_detection_rate" for name in DEVIATIONS)
    component = {
        "public_probe_secrecy_supported": bool(
            full["lambda_F_upper95"] < public["lambda_F_upper95"]
            and full["grouped_auc_upper95"] < public["grouped_auc_upper95"]
        ),
        "hidden_probe_coupling_supported": bool(
            full["registered_attack_bound"] < independent["registered_attack_bound"]
        ),
        "probe_detection_supported": bool(
            min(full[key] for key in detection_keys)
            > min(production[key] for key in detection_keys)
        ),
        "protected_H_noninferior": bool(
            full["H_false_settlement_upper95"]
            <= max(public["H_false_settlement_upper95"], independent["H_false_settlement_upper95"], production["H_false_settlement_upper95"])
        ),
    }
    component["passed"] = all(component.values())
    full_gate = {
        "H": full["H_false_settlement_upper95"] <= 0.1,
        "detection": min(full[key] for key in tuple(f"{name}_detection_lower95" for name in DEVIATIONS)) >= 0.8,
        "lambda_F": full["lambda_F_upper95"] <= 0.1,
        "auc": full["grouped_auc_upper95"] <= 0.65,
        "utility": min(full[f"{name}_utility_margin_lower95"] for name in DEVIATIONS) > 0.0 and full["honest_participation_lower95"] >= 0.0,
        "attack": full["registered_attack_bound"] <= 0.2,
        "overhead": full["online_overhead_ratio"] <= 0.25,
    }
    full_gate["passed"] = all(full_gate.values())
    return {
        "dataset": dataset,
        "source_identity_passed": bool(source["passed"]),
        "ablation_rows": ablations,
        "sensitivity_rows": sensitivity,
        "component_gate": component,
        "full_rcmp_gate": full_gate,
        "trainer_training_units": 0,
        "fresh_paired_replay_blocks": 0,
    }


__all__ = [
    "CHALLENGE_ATOM_ORDER",
    "VARIANTS",
    "evaluate_dataset",
    "load_source",
    "verify_manifest",
]
