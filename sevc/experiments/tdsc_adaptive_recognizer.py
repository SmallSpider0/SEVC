"""Selected canonical scientific routines; deployment wrappers are omitted."""
from __future__ import annotations


import math

from typing import Any, Mapping, Sequence

import numpy as np

from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

from sevc.core.artifacts import canonical_json_text, sha256_text

from sevc.verification.adaptive_recognizer import V2_HASH_FIELDS, deterministic_history_indices, feature_matrix_v2, fit_predict_probe_probability_v2, v2_feature_identity

from sevc.verification.verifier_behaviors import BEHAVIORS

V2_CHANGE_ID = "experiment-tdsc-adaptive-recognizer-v2"


V3_CHANGE_ID = "experiment-tdsc-adaptive-recognizer-v3"


def _matched_random_flags(
    *,
    dataset: str,
    block_seed: int,
    observation_budget: int,
    task_ids: Sequence[str],
    replay_count: int,
    control_seed: int,
) -> tuple[bool, ...]:
    if not 0 <= replay_count <= len(task_ids):
        raise ValueError("adaptive v2 matched-control replay count is invalid")
    ordered = sorted(
        range(len(task_ids)),
        key=lambda index: (
            sha256_text(
                f"{dataset}|{control_seed}|{block_seed}|{observation_budget}|"
                f"{task_ids[index]}"
            ),
            task_ids[index],
        ),
    )
    selected = set(ordered[:replay_count])
    return tuple(index in selected for index in range(len(task_ids)))


def _predict_dataset_v2(
    dataset: str,
    rows: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    discriminator = protocol["discriminator"]
    selectors, feature_names = v2_feature_identity(
        rows,
        expected_selector_count=int(discriminator["features"]["selector_count"]),
        expected_selector_names_sha256=str(
            discriminator["features"]["selector_names_sha256"]
        ),
        expected_feature_count=int(discriminator["feature_count"]),
        expected_feature_names_sha256=str(discriminator["feature_names_sha256"]),
        expected_feature_schema_sha256=str(discriminator["feature_schema_sha256"]),
        expected_source_feature_schema_sha256=str(
            discriminator["features"]["source_feature_schema_sha256"]
        ),
        expected_parameter_sha256=str(
            discriminator["features"]["selector_parameter_identity_sha256"]
        ),
        expected_hash_fields=discriminator["features"]["hash_fields"],
    )
    matrix = feature_matrix_v2(rows, selectors)
    labels = np.asarray([int(row["is_probe"]) for row in rows], dtype=int)
    all_blocks = sorted({int(row["block_seed"]) for row in rows})
    selected_blocks = [all_blocks[index] for index in protocol["block_indices"]]
    row_indices_by_block = {
        block: tuple(
            index
            for index, row in enumerate(rows)
            if int(row["block_seed"]) == block
        )
        for block in selected_blocks
    }
    base_seed = int(protocol["statistics"]["bootstrap_seed_by_dataset"][dataset])
    control_seed = int(protocol["statistics"]["control_seed_by_dataset"][dataset])
    threshold = float(discriminator["probability_threshold"])
    output: list[dict[str, Any]] = []
    ensemble_fit_count = 0
    component_fit_count = 0
    for block in selected_blocks:
        test_indices = row_indices_by_block[block]
        history = deterministic_history_indices(
            rows,
            heldout_block_seed=block,
            dataset=dataset,
            order_seed=base_seed,
        )
        if set(test_indices) & set(history):
            raise RuntimeError("adaptive v2 history overlaps held-out block")
        task_ids = tuple(str(rows[index]["task_id"]) for index in test_indices)
        for budget in protocol["observation_budget_ladder"]:
            random_state = base_seed + int(budget) + (int(block) % 1_000_003)
            probabilities, receipt = fit_predict_probe_probability_v2(
                matrix=matrix,
                labels=labels,
                history_indices=history,
                test_indices=test_indices,
                observation_budget=int(budget),
                random_state=random_state,
                component_models=discriminator["component_models"],
            )
            ensemble_fit_count += int(receipt.fitted)
            component_fit_count += int(receipt.component_fit_count)
            predicted = tuple(float(value) >= threshold for value in probabilities)
            random_flags = _matched_random_flags(
                dataset=dataset,
                block_seed=block,
                observation_budget=int(budget),
                task_ids=task_ids,
                replay_count=sum(predicted),
                control_seed=control_seed,
            )
            if sum(random_flags) != sum(predicted):
                raise RuntimeError("adaptive v2 matched-control effort drift")
            for source_index, probability, predicted_probe, random_replay in zip(
                test_indices, probabilities, predicted, random_flags
            ):
                source = rows[source_index]
                role = str(source["role"])
                adaptive_action = "honest-replay" if predicted_probe else "always-pass"
                random_action = "honest-replay" if random_replay else "always-pass"
                row = {
                    "schema_version": "sevc-tdsc-adaptive-prediction-v2",
                    "change_id": V2_CHANGE_ID,
                    "dataset": dataset,
                    "model": protocol["datasets"][dataset]["model"],
                    "block_seed": block,
                    "task_id": source["task_id"],
                    "source_index": int(source["source_index"]),
                    "public_envelope_sha256": source["public_envelope_sha256"],
                    "observation_budget": int(budget),
                    "history_row_count": receipt.history_row_count,
                    "history_probe_count": receipt.history_probe_count,
                    "history_identity_sha256": receipt.history_identity_sha256,
                    "history_heldout_overlap_count": 0,
                    "discriminator_model": receipt.model,
                    "probe_probability": float(probability),
                    "predicted_probe": bool(predicted_probe),
                    "candidate_action": adaptive_action,
                    "matched_random_honest_replay": bool(random_replay),
                    "matched_random_action": random_action,
                    "effort_match": bool(predicted_probe) == bool(random_replay),
                    "evaluation_role": role,
                    "evaluation_is_probe": bool(source["is_probe"]),
                    "current_label_read_before_prediction": False,
                    "adaptive_detected_event": bool(
                        role == "challenge" and not predicted_probe
                    ),
                    "random_detected_event": bool(
                        role == "challenge" and not random_replay
                    ),
                    "adaptive_unsupported_release": bool(
                        role == "production" and not predicted_probe
                    ),
                    "random_unsupported_release": bool(
                        role == "production" and not random_replay
                    ),
                    "classifier_correct": bool(predicted_probe)
                    == bool(source["is_probe"]),
                    "feature_count": len(feature_names),
                    "feature_names_sha256": discriminator["feature_names_sha256"],
                    "feature_schema_sha256": discriminator["feature_schema_sha256"],
                    "public_identity_projection_fields": list(V2_HASH_FIELDS),
                    "existing_ledger_analysis_only": True,
                }
                row["row_sha256"] = sha256_text(canonical_json_text(row))
                output.append(row)
    return output, {
        "selected_blocks": selected_blocks,
        "source_row_count": len(rows),
        "feature_count": len(feature_names),
        "feature_names_sha256": discriminator["feature_names_sha256"],
        "ensemble_fit_count": ensemble_fit_count,
        "component_model_fit_count": component_fit_count,
    }


def _bootstrap_summary(
    values: Sequence[float],
    *,
    seed: int,
    resamples: int,
    alpha: float,
) -> dict[str, float]:
    vector = np.asarray(values, dtype=float)
    if vector.ndim != 1 or not len(vector) or np.any(~np.isfinite(vector)):
        raise ValueError("adaptive v2 bootstrap requires a finite vector")
    point = float(vector.mean())
    if np.all(vector == vector[0]):
        return {"point": point, "lower": point, "upper": point}
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, len(vector), size=(int(resamples), len(vector)))
    estimates = vector[indices].mean(axis=1)
    return {
        "point": point,
        "lower": float(np.quantile(estimates, alpha / 2.0, method="higher")),
        "upper": float(np.quantile(estimates, 1.0 - alpha / 2.0, method="lower")),
    }


def _one_sided_block_lower(
    values: Sequence[float],
    *,
    seed: int,
    resamples: int,
    alpha: float,
) -> float:
    vector = np.asarray(values, dtype=float)
    if vector.ndim != 1 or not len(vector) or np.any(~np.isfinite(vector)):
        raise ValueError("adaptive v2 one-sided bootstrap requires a finite vector")
    if np.all(vector == vector[0]):
        return float(vector[0])
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, len(vector), size=(int(resamples), len(vector)))
    estimates = vector[indices].mean(axis=1)
    return float(np.quantile(estimates, alpha, method="higher"))


def _block_statistics_v2(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, list[float]]:
    result = {
        "balanced_accuracy": [],
        "adaptive_detection": [],
        "random_detection": [],
        "adaptive_probe_replay": [],
        "random_probe_replay": [],
        "adaptive_unsupported_release": [],
        "random_unsupported_release": [],
        "adaptive_effort": [],
        "random_effort": [],
    }
    for block in sorted({int(row["block_seed"]) for row in rows}):
        selected = [row for row in rows if int(row["block_seed"]) == block]
        labels = np.asarray([int(row["evaluation_is_probe"]) for row in selected])
        predicted = np.asarray([int(row["predicted_probe"]) for row in selected])
        probes = [row for row in selected if row["evaluation_is_probe"]]
        production = [row for row in selected if row["evaluation_role"] == "production"]
        challenges = [row for row in selected if row["evaluation_role"] == "challenge"]
        if len(selected) != 40 or len(probes) != 8 or len(production) != 32 or len(challenges) != 4:
            raise ValueError("adaptive v2 held-out block shape drift")
        result["balanced_accuracy"].append(
            float(balanced_accuracy_score(labels, predicted))
        )
        result["adaptive_detection"].append(
            float(any(row["adaptive_detected_event"] for row in challenges))
        )
        result["random_detection"].append(
            float(any(row["random_detected_event"] for row in challenges))
        )
        result["adaptive_probe_replay"].append(
            float(np.mean([row["predicted_probe"] for row in probes]))
        )
        result["random_probe_replay"].append(
            float(np.mean([row["matched_random_honest_replay"] for row in probes]))
        )
        result["adaptive_unsupported_release"].append(
            float(np.mean([row["adaptive_unsupported_release"] for row in production]))
        )
        result["random_unsupported_release"].append(
            float(np.mean([row["random_unsupported_release"] for row in production]))
        )
        result["adaptive_effort"].append(
            float(np.mean([row["predicted_probe"] for row in selected]))
        )
        result["random_effort"].append(
            float(np.mean([row["matched_random_honest_replay"] for row in selected]))
        )
    return result


def _rung_metrics_v2(
    dataset: str,
    rows: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    seed = int(protocol["statistics"]["bootstrap_seed_by_dataset"][dataset])
    resamples = int(protocol["statistics"]["resamples"])
    for rung_index, budget in enumerate(protocol["observation_budget_ladder"]):
        selected = [
            row for row in rows if int(row["observation_budget"]) == int(budget)
        ]
        labels = np.asarray([int(row["evaluation_is_probe"]) for row in selected])
        predicted = np.asarray([int(row["predicted_probe"]) for row in selected])
        scores = np.asarray([float(row["probe_probability"]) for row in selected])
        blocks = _block_statistics_v2(selected)
        rung_seed = seed + rung_index * 100
        adaptive_detection = _bootstrap_summary(
            blocks["adaptive_detection"], seed=rung_seed + 1,
            resamples=resamples, alpha=0.05,
        )
        random_detection = _bootstrap_summary(
            blocks["random_detection"], seed=rung_seed + 2,
            resamples=resamples, alpha=0.05,
        )
        balanced = _bootstrap_summary(
            blocks["balanced_accuracy"], seed=rung_seed + 3,
            resamples=resamples, alpha=0.05,
        )
        random_minus_adaptive_detection = [
            random - adaptive
            for random, adaptive in zip(
                blocks["random_detection"], blocks["adaptive_detection"]
            )
        ]
        adaptive_minus_random_probe_replay = [
            adaptive - random
            for adaptive, random in zip(
                blocks["adaptive_probe_replay"], blocks["random_probe_replay"]
            )
        ]
        adaptive_minus_random_unsupported = [
            adaptive - random
            for adaptive, random in zip(
                blocks["adaptive_unsupported_release"],
                blocks["random_unsupported_release"],
            )
        ]
        results.append(
            {
                "observation_budget": int(budget),
                "heldout_block_count": len(blocks["balanced_accuracy"]),
                "prediction_count": len(selected),
                "classifier_accuracy": float(accuracy_score(labels, predicted)),
                "classifier_balanced_accuracy": float(
                    balanced_accuracy_score(labels, predicted)
                ),
                "classifier_balanced_accuracy_two_sided95": balanced,
                "classifier_roc_auc": float(roc_auc_score(labels, scores)),
                "adaptive_honest_replay_fraction": float(predicted.mean()),
                "matched_random_honest_replay_fraction": float(
                    np.mean([row["matched_random_honest_replay"] for row in selected])
                ),
                "effort_count_difference": int(
                    predicted.sum()
                    - sum(row["matched_random_honest_replay"] for row in selected)
                ),
                "adaptive_detection": adaptive_detection,
                "matched_random_detection": random_detection,
                "random_minus_adaptive_detection_two_sided95": _bootstrap_summary(
                    random_minus_adaptive_detection,
                    seed=rung_seed + 4,
                    resamples=resamples,
                    alpha=0.05,
                ),
                "adaptive_probe_replay_rate": float(
                    np.mean(blocks["adaptive_probe_replay"])
                ),
                "matched_random_probe_replay_rate": float(
                    np.mean(blocks["random_probe_replay"])
                ),
                "adaptive_minus_random_probe_replay_two_sided95": _bootstrap_summary(
                    adaptive_minus_random_probe_replay,
                    seed=rung_seed + 5,
                    resamples=resamples,
                    alpha=0.05,
                ),
                "adaptive_unsupported_release_rate": float(
                    np.mean(blocks["adaptive_unsupported_release"])
                ),
                "matched_random_unsupported_release_rate": float(
                    np.mean(blocks["random_unsupported_release"])
                ),
                "adaptive_minus_random_unsupported_release_two_sided95": _bootstrap_summary(
                    adaptive_minus_random_unsupported,
                    seed=rung_seed + 6,
                    resamples=resamples,
                    alpha=0.05,
                ),
            }
        )
    return results


def _validity_permutation_pvalue(
    rows: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    resamples: int,
) -> float:
    blocks = sorted({int(row["block_seed"]) for row in rows})
    observed = []
    rng = np.random.default_rng(int(seed))
    null_sum = np.zeros(int(resamples), dtype=float)
    for block in blocks:
        selected = [row for row in rows if int(row["block_seed"]) == block]
        labels = np.asarray([int(row["evaluation_is_probe"]) for row in selected])
        predicted = np.asarray([int(row["predicted_probe"]) for row in selected])
        positives = int(labels.sum())
        negatives = len(labels) - positives
        predicted_positives = int(predicted.sum())
        if positives <= 0 or negatives <= 0:
            raise ValueError("adaptive v2 permutation block lacks both classes")
        observed.append(float(balanced_accuracy_score(labels, predicted)))
        true_positives = rng.hypergeometric(
            ngood=predicted_positives,
            nbad=len(labels) - predicted_positives,
            nsample=positives,
            size=int(resamples),
        )
        true_negatives = negatives - (predicted_positives - true_positives)
        null_sum += 0.5 * (
            true_positives / positives + true_negatives / negatives
        )
    observed_mean = float(np.mean(observed))
    null_mean = null_sum / len(blocks)
    return float((1 + np.count_nonzero(null_mean >= observed_mean)) / (resamples + 1))


def _all_finite(value: Any) -> bool:
    if isinstance(value, Mapping):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_finite(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def _dataset_gate_v2(
    dataset: str,
    prediction_rows: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any],
    protocol: Mapping[str, Any],
    interface: Mapping[str, Any],
) -> dict[str, Any]:
    curve = _rung_metrics_v2(dataset, prediction_rows, protocol)
    expected_blocks = len(protocol["block_indices"])
    expected_predictions = expected_blocks * 40 * len(
        protocol["observation_budget_ladder"]
    )
    groups: dict[tuple[int, int], list[Mapping[str, Any]]] = {}
    for row in prediction_rows:
        key = (int(row["block_seed"]), int(row["observation_budget"]))
        groups.setdefault(key, []).append(row)
    effort_matched = all(
        sum(row["predicted_probe"] for row in rows)
        == sum(row["matched_random_honest_replay"] for row in rows)
        for rows in groups.values()
    )
    integrity_checks = {
        "information_boundary": interface.get("passed") is True
        and all(
            row["history_heldout_overlap_count"] == 0
            and row["current_label_read_before_prediction"] is False
            for row in prediction_rows
        ),
        "identity_join": metadata["identity_join_count"]
        == metadata["source_row_count"],
        "feature_identity": metadata["feature_count"]
        == int(protocol["discriminator"]["feature_count"])
        and metadata["feature_names_sha256"]
        == protocol["discriminator"]["feature_names_sha256"],
        "like_for_like": protocol["adaptive_behavior"]["cheaper_action"]
        == "always-pass"
        and "always-pass" in BEHAVIORS.keys(),
        "effort_matched_control": effort_matched,
        "curve_completeness": len(prediction_rows) == expected_predictions
        and len(curve) == len(protocol["observation_budget_ladder"])
        and all(
            row["heldout_block_count"] == expected_blocks
            and row["prediction_count"] == expected_blocks * 40
            for row in curve
        ),
        "finite_metrics": _all_finite(curve),
    }
    integrity_passed = all(integrity_checks.values())
    confirmatory: dict[str, Any]
    if protocol["scientifically_eligible"]:
        final_budget = int(protocol["statistics"]["final_rung"])
        final_rows = [
            row
            for row in prediction_rows
            if int(row["observation_budget"]) == final_budget
        ]
        if len(final_rows) != expected_blocks * 40:
            raise ValueError("adaptive v2 final-rung cardinality drift")
        block_values = _block_statistics_v2(final_rows)["balanced_accuracy"]
        alpha = float(protocol["statistics"]["one_sided_dataset_alpha"])
        point = float(np.mean(block_values))
        lower = _one_sided_block_lower(
            block_values,
            seed=int(protocol["statistics"]["bootstrap_seed_by_dataset"][dataset])
            + 7000,
            resamples=int(protocol["statistics"]["resamples"]),
            alpha=alpha,
        )
        pvalue = _validity_permutation_pvalue(
            final_rows,
            seed=int(protocol["statistics"]["permutation_seed_by_dataset"][dataset]),
            resamples=int(protocol["permutation_resamples"]),
        )
        confirmatory = {
            "eligible": True,
            "observation_budget": final_budget,
            "balanced_accuracy": point,
            "one_sided_block_bootstrap_lower": lower,
            "one_sided_dataset_alpha": alpha,
            "permutation_pvalue": pvalue,
            "permutation_resamples": int(protocol["permutation_resamples"]),
            "point_above_chance": point > 0.5,
            "lower_above_chance": lower > 0.5,
            "permutation_significant": pvalue <= alpha,
        }
        confirmatory["passed"] = bool(
            confirmatory["point_above_chance"]
            and confirmatory["lower_above_chance"]
            and confirmatory["permutation_significant"]
        )
    else:
        confirmatory = {
            "eligible": False,
            "observation_budget": max(protocol["observation_budget_ladder"]),
            "passed": False,
            "reason": "tiny profile is not confirmatory",
        }
    return {
        "schema_version": "sevc-tdsc-adaptive-dataset-gate-v2",
        "change_id": V2_CHANGE_ID,
        "dataset": dataset,
        "source_row_count": metadata["source_row_count"],
        "identity_row_count": metadata["identity_row_count"],
        "identity_join_count": metadata["identity_join_count"],
        "heldout_block_count": expected_blocks,
        "feature_count": metadata["feature_count"],
        "feature_names_sha256": metadata["feature_names_sha256"],
        "ensemble_fit_count": metadata["ensemble_fit_count"],
        "component_model_fit_count": metadata["component_model_fit_count"],
        "prediction_row_count": len(prediction_rows),
        "curve": curve,
        "confirmatory_validity": confirmatory,
        "integrity_checks": integrity_checks,
        "integrity_passed": integrity_passed,
        "passed": bool(
            integrity_passed
            and (
                confirmatory["passed"]
                if protocol["scientifically_eligible"]
                else True
            )
        ),
    }


def _max_detection_drop_v3(
    dataset: str,
    prediction_rows: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    budgets = [
        int(value) for value in protocol["observation_budget_ladder"]
    ]
    baseline_budget = budgets[0]
    blocks = sorted({int(row["block_seed"]) for row in prediction_rows})
    by_budget: dict[int, list[float]] = {}
    for budget in budgets:
        selected = [
            row
            for row in prediction_rows
            if int(row["observation_budget"]) == budget
        ]
        by_budget[budget] = _block_statistics_v2(selected)["adaptive_detection"]
    baseline = np.asarray(by_budget[baseline_budget], dtype=float)
    drop_matrix = np.stack(
        [baseline - np.asarray(by_budget[budget], dtype=float) for budget in budgets[1:]],
        axis=1,
    )
    point = float(np.max(drop_matrix.mean(axis=0)))
    resamples = int(protocol["statistics"]["resamples"])
    alpha = float(protocol["statistics"]["per_dataset_alpha"])
    rng = np.random.default_rng(
        int(protocol["statistics"]["bootstrap_seed_by_dataset"][dataset]) + 9100
    )
    maxima: list[float] = []
    remaining = resamples
    while remaining:
        batch = min(1000, remaining)
        indices = rng.integers(0, len(blocks), size=(batch, len(blocks)))
        estimates = drop_matrix[indices].mean(axis=1)
        maxima.extend(float(value) for value in np.max(estimates, axis=1))
        remaining -= batch
    upper = float(
        np.quantile(np.asarray(maxima), 1.0 - alpha, method="lower")
    )
    margin = float(protocol["statistics"]["max_detection_drop_margin"])
    return {
        "baseline_budget": baseline_budget,
        "compared_budgets": budgets[1:],
        "max_drop_point": point,
        "max_drop_upper": upper,
        "per_dataset_alpha": alpha,
        "practical_margin": margin,
        "no_material_degradation": upper <= margin,
    }


def _recognizability_v3(
    dataset: str,
    prediction_rows: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    final_budget = int(protocol["statistics"]["final_rung"])
    final_rows = [
        row
        for row in prediction_rows
        if int(row["observation_budget"]) == final_budget
    ]
    expected = len(protocol["block_indices"]) * 40
    if len(final_rows) != expected:
        raise ValueError("adaptive v3 final-rung cardinality drift")
    values = _block_statistics_v2(final_rows)["balanced_accuracy"]
    alpha = float(protocol["statistics"]["per_dataset_alpha"])
    interval = _bootstrap_summary(
        values,
        seed=int(protocol["statistics"]["bootstrap_seed_by_dataset"][dataset])
        + 7000,
        resamples=int(protocol["statistics"]["resamples"]),
        alpha=alpha,
    )
    pvalue = _validity_permutation_pvalue(
        final_rows,
        seed=int(protocol["statistics"]["permutation_seed_by_dataset"][dataset]),
        resamples=int(protocol["permutation_resamples"]),
    )
    lower_margin, upper_margin = (
        float(value)
        for value in protocol["statistics"]["practical_equivalence_interval"]
    )
    equivalent = bool(
        interval["lower"] > lower_margin and interval["upper"] < upper_margin
    )
    signal = bool(interval["lower"] > 0.5 and pvalue <= alpha)
    if equivalent and signal:
        classification = "STATISTICALLY_LEARNABLE_BUT_PRACTICALLY_EQUIVALENT"
    elif equivalent:
        classification = "PRACTICALLY_EQUIVALENT_TO_CHANCE"
    elif signal:
        classification = "STATISTICALLY_LEARNABLE_SIGNAL"
    else:
        classification = "INDETERMINATE_WITHIN_REGISTERED_BUDGET"
    return {
        "eligible": bool(protocol["scientifically_eligible"]),
        "observation_budget": final_budget,
        "balanced_accuracy": interval["point"],
        "simultaneous_two_sided_interval": interval,
        "simultaneous_confidence": 1.0 - alpha,
        "per_dataset_alpha": alpha,
        "permutation_pvalue": pvalue,
        "permutation_resamples": int(protocol["permutation_resamples"]),
        "practical_equivalence_interval": [lower_margin, upper_margin],
        "practically_equivalent_to_chance": equivalent,
        "statistically_learnable_signal": signal,
        "classification": (
            classification
            if protocol["scientifically_eligible"]
            else "PILOT_ONLY"
        ),
        "nonsignificant_permutation_is_equivalence_evidence": False,
    }


def _dataset_gate_v3(
    dataset: str,
    prediction_rows: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any],
    protocol: Mapping[str, Any],
    interface: Mapping[str, Any],
) -> dict[str, Any]:
    inherited = _dataset_gate_v2(
        dataset, prediction_rows, metadata, protocol, interface
    )
    recognizability = _recognizability_v3(dataset, prediction_rows, protocol)
    non_degradation = _max_detection_drop_v3(dataset, prediction_rows, protocol)
    integrity = dict(inherited["integrity_checks"])
    integrity["scientific_direction_not_used_as_gate"] = True
    return {
        "schema_version": "sevc-tdsc-adaptive-dataset-gate-v3",
        "change_id": V3_CHANGE_ID,
        "dataset": dataset,
        "source_row_count": metadata["source_row_count"],
        "identity_row_count": metadata["identity_row_count"],
        "identity_join_count": metadata["identity_join_count"],
        "heldout_block_count": len(protocol["block_indices"]),
        "feature_count": metadata["feature_count"],
        "feature_names_sha256": metadata["feature_names_sha256"],
        "ensemble_fit_count": metadata["ensemble_fit_count"],
        "component_model_fit_count": metadata["component_model_fit_count"],
        "prediction_row_count": len(prediction_rows),
        "curve": inherited["curve"],
        "recognizability": recognizability,
        "detection_non_degradation": non_degradation,
        "integrity_checks": integrity,
        "integrity_passed": all(integrity.values()),
        "passed": all(integrity.values()),
    }


