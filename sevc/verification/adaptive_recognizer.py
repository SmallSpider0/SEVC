"""Frozen cross-fitted probe recognizer for the TDSC adaptive experiment."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class AdaptiveFitReceipt:
    observation_budget: int
    history_row_count: int
    history_probe_count: int
    history_identity_sha256: str
    model: str
    fitted: bool
    component_fit_count: int = 1


V2_HASH_FIELDS = (
    "task_id",
    "public_envelope_sha256",
    "source_commitment",
    "permutation_id",
    "descriptor_sha256",
    "candidate_proof_sha256",
)


def selector_feature_names(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_count: int,
    expected_names_sha256: str,
    expected_schema_sha256: str,
    expected_parameter_sha256: str,
) -> tuple[str, ...]:
    if not rows:
        raise ValueError("adaptive recognizer requires source rows")
    names = tuple(sorted(str(key) for key in rows[0]["predictions"]))
    encoded = json.dumps(names, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if len(names) != expected_count or hashlib.sha256(encoded.encode("utf-8")).hexdigest() != expected_names_sha256:
        raise ValueError("adaptive selector feature-name identity drift")
    expected_set = set(names)
    for row in rows:
        if set(row["predictions"]) != expected_set:
            raise ValueError("adaptive selector feature set drift")
        if row.get("feature_schema_sha256") != expected_schema_sha256:
            raise ValueError("adaptive feature schema drift")
        if row.get("selector_parameter_identity_sha256") != expected_parameter_sha256:
            raise ValueError("adaptive selector parameter drift")
        if any(not math.isfinite(float(row["predictions"][name]["score"])) for name in names):
            raise ValueError("adaptive source contains nonfinite selector score")
    return names


def feature_matrix(
    rows: Sequence[Mapping[str, Any]], names: Sequence[str]
) -> np.ndarray:
    matrix = np.asarray(
        [[float(row["predictions"][name]["score"]) for name in names] for row in rows],
        dtype=float,
    )
    if matrix.ndim != 2 or np.any(~np.isfinite(matrix)):
        raise ValueError("adaptive feature matrix must be finite and two-dimensional")
    return matrix


def v2_feature_identity(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_selector_count: int,
    expected_selector_names_sha256: str,
    expected_feature_count: int,
    expected_feature_names_sha256: str,
    expected_feature_schema_sha256: str,
    expected_source_feature_schema_sha256: str,
    expected_parameter_sha256: str,
    expected_hash_fields: Sequence[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Validate and freeze the recorded v2 public feature projection."""

    selectors = selector_feature_names(
        rows,
        expected_count=expected_selector_count,
        expected_names_sha256=expected_selector_names_sha256,
        expected_schema_sha256=expected_source_feature_schema_sha256,
        expected_parameter_sha256=expected_parameter_sha256,
    )
    hash_fields = tuple(str(value) for value in expected_hash_fields)
    if hash_fields != V2_HASH_FIELDS:
        raise ValueError("adaptive v2 public hash-field identity drift")
    feature_names = tuple(
        [f"score:{name}" for name in selectors]
        + [f"decision:{name}" for name in selectors]
        + [f"log1p_runtime_seconds:{name}" for name in selectors]
        + [
            f"hash_byte:{field}:{index:02d}"
            for field in hash_fields
            for index in range(32)
        ]
    )
    encoded = json.dumps(
        feature_names, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    schema = {
        "selector_order": selectors,
        "selector_components": (
            "score",
            "decision",
            "log1p_runtime_seconds",
        ),
        "hash_fields": hash_fields,
        "hash_encoding": "32 unsigned bytes divided by 255",
        "feature_order": feature_names,
    }
    schema_encoded = json.dumps(
        schema, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    if (
        len(feature_names) != expected_feature_count
        or hashlib.sha256(encoded.encode("utf-8")).hexdigest()
        != expected_feature_names_sha256
        or hashlib.sha256(schema_encoded.encode("utf-8")).hexdigest()
        != expected_feature_schema_sha256
    ):
        raise ValueError("adaptive v2 feature identity drift")
    for row in rows:
        identity = row.get("public_identity")
        if not isinstance(identity, Mapping) or set(identity) != set(hash_fields):
            raise ValueError("adaptive v2 public identity projection drift")
        if identity["task_id"] != row.get("task_id"):
            raise ValueError("adaptive v2 task identity disagreement")
        if identity["public_envelope_sha256"] != row.get("public_envelope_sha256"):
            raise ValueError("adaptive v2 public envelope identity disagreement")
        for field in hash_fields:
            value = str(identity[field])
            try:
                decoded = bytes.fromhex(value)
            except ValueError as exc:
                raise ValueError(f"adaptive v2 non-hex public identity: {field}") from exc
            if len(decoded) != 32:
                raise ValueError(f"adaptive v2 public identity is not SHA-256: {field}")
        for name in selectors:
            prediction = row["predictions"][name]
            runtime = float(prediction["runtime_seconds"])
            if not isinstance(prediction.get("decision"), bool):
                raise ValueError("adaptive v2 selector decision must be boolean")
            if not math.isfinite(runtime) or runtime < 0.0:
                raise ValueError("adaptive v2 selector runtime must be finite and nonnegative")
    return selectors, feature_names


def feature_matrix_v2(
    rows: Sequence[Mapping[str, Any]], selectors: Sequence[str]
) -> np.ndarray:
    """Build the exact 615-column v2 public feature matrix."""

    matrix_rows: list[list[float]] = []
    for row in rows:
        predictions = row["predictions"]
        identity = row["public_identity"]
        values = [float(predictions[name]["score"]) for name in selectors]
        values.extend(float(bool(predictions[name]["decision"])) for name in selectors)
        values.extend(
            math.log1p(float(predictions[name]["runtime_seconds"]))
            for name in selectors
        )
        for field in V2_HASH_FIELDS:
            values.extend(value / 255.0 for value in bytes.fromhex(str(identity[field])))
        matrix_rows.append(values)
    matrix = np.asarray(matrix_rows, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != 615 or np.any(~np.isfinite(matrix)):
        raise ValueError("adaptive v2 feature matrix shape or finiteness failure")
    return matrix


def deterministic_history_indices(
    rows: Sequence[Mapping[str, Any]],
    *,
    heldout_block_seed: int,
    dataset: str,
    order_seed: int,
) -> tuple[int, ...]:
    candidates = [
        index for index, row in enumerate(rows)
        if int(row["block_seed"]) != int(heldout_block_seed)
    ]
    return tuple(
        sorted(
            candidates,
            key=lambda index: (
                hashlib.sha256(
                    f"{dataset}|{order_seed}|{rows[index]['task_id']}".encode("utf-8")
                ).hexdigest(),
                str(rows[index]["task_id"]),
            ),
        )
    )


def fit_predict_probe_probability(
    *,
    matrix: np.ndarray,
    labels: np.ndarray,
    history_indices: Sequence[int],
    test_indices: Sequence[int],
    observation_budget: int,
    random_state: int,
    c_value: float,
    max_iter: int,
) -> tuple[np.ndarray, AdaptiveFitReceipt]:
    if observation_budget < 0 or observation_budget > len(history_indices):
        raise ValueError("adaptive observation budget is outside the history pool")
    selected = tuple(history_indices[:observation_budget])
    history_ids = ",".join(str(index) for index in selected)
    identity = hashlib.sha256(history_ids.encode("utf-8")).hexdigest()
    if observation_budget == 0:
        return (
            np.zeros(len(test_indices), dtype=float),
            AdaptiveFitReceipt(
                0, 0, 0, identity, "constant-production-prior", False, 0
            ),
        )
    train_y = labels[np.asarray(selected, dtype=int)]
    if set(np.unique(train_y).tolist()) != {0, 1}:
        raise ValueError("adaptive history prefix must contain both disclosed labels")
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            solver="liblinear",
            C=float(c_value),
            class_weight="balanced",
            max_iter=int(max_iter),
            random_state=int(random_state),
        ),
    )
    model.fit(matrix[np.asarray(selected, dtype=int)], train_y)
    probabilities = model.predict_proba(matrix[np.asarray(test_indices, dtype=int)])[:, 1]
    if np.any(~np.isfinite(probabilities)):
        raise ValueError("adaptive discriminator produced nonfinite probability")
    return probabilities, AdaptiveFitReceipt(
        observation_budget=int(observation_budget),
        history_row_count=len(selected),
        history_probe_count=int(train_y.sum()),
        history_identity_sha256=identity,
        model="standard-scaler+l2-logistic-liblinear",
        fitted=True,
        component_fit_count=1,
    )


def fit_predict_probe_probability_v2(
    *,
    matrix: np.ndarray,
    labels: np.ndarray,
    history_indices: Sequence[int],
    test_indices: Sequence[int],
    observation_budget: int,
    random_state: int,
    component_models: Mapping[str, Mapping[str, Any]],
) -> tuple[np.ndarray, AdaptiveFitReceipt]:
    """Fit the frozen three-branch successor ensemble and average probabilities."""

    if observation_budget < 0 or observation_budget > len(history_indices):
        raise ValueError("adaptive v2 observation budget is outside the history pool")
    selected = tuple(history_indices[:observation_budget])
    identity = hashlib.sha256(
        ",".join(str(index) for index in selected).encode("utf-8")
    ).hexdigest()
    if observation_budget == 0:
        return (
            np.zeros(len(test_indices), dtype=float),
            AdaptiveFitReceipt(
                0, 0, 0, identity, "constant-production-prior", False, 0
            ),
        )
    train_indices = np.asarray(selected, dtype=int)
    heldout_indices = np.asarray(test_indices, dtype=int)
    train_y = labels[train_indices]
    if set(np.unique(train_y).tolist()) != {0, 1}:
        raise ValueError("adaptive v2 history prefix must contain both disclosed labels")
    logistic_config = component_models["logistic"]
    logistic = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            solver=str(logistic_config["solver"]),
            penalty=str(logistic_config["penalty"]),
            C=float(logistic_config["C"]),
            class_weight=str(logistic_config["class_weight"]),
            max_iter=int(logistic_config["max_iter"]),
            random_state=int(random_state),
        ),
    )
    histogram_config = component_models["hist_gradient_boosting"]
    histogram = HistGradientBoostingClassifier(
        learning_rate=float(histogram_config["learning_rate"]),
        max_iter=int(histogram_config["max_iter"]),
        max_leaf_nodes=int(histogram_config["max_leaf_nodes"]),
        min_samples_leaf=int(histogram_config["min_samples_leaf"]),
        l2_regularization=float(histogram_config["l2_regularization"]),
        max_bins=int(histogram_config["max_bins"]),
        class_weight=str(histogram_config["class_weight"]),
        early_stopping=bool(histogram_config["early_stopping"]),
        random_state=int(random_state) + 1,
    )
    forest_config = component_models["extra_trees"]
    forest = ExtraTreesClassifier(
        n_estimators=int(forest_config["n_estimators"]),
        criterion=str(forest_config["criterion"]),
        max_depth=forest_config["max_depth"],
        min_samples_split=int(forest_config["min_samples_split"]),
        min_samples_leaf=int(forest_config["min_samples_leaf"]),
        max_features=str(forest_config["max_features"]),
        bootstrap=bool(forest_config["bootstrap"]),
        class_weight=str(forest_config["class_weight"]),
        n_jobs=int(forest_config["n_jobs"]),
        random_state=int(random_state) + 2,
    )
    probabilities = []
    for model in (logistic, histogram, forest):
        model.fit(matrix[train_indices], train_y)
        probabilities.append(model.predict_proba(matrix[heldout_indices])[:, 1])
    averaged = np.mean(np.stack(probabilities, axis=0), axis=0)
    if np.any(~np.isfinite(averaged)) or np.any((averaged < 0.0) | (averaged > 1.0)):
        raise ValueError("adaptive v2 ensemble produced invalid probability")
    return averaged, AdaptiveFitReceipt(
        observation_budget=int(observation_budget),
        history_row_count=len(selected),
        history_probe_count=int(train_y.sum()),
        history_identity_sha256=identity,
        model="equal-soft-vote[scaled-l2-logistic,hist-gradient-boosting,extra-trees]",
        fitted=True,
        component_fit_count=3,
    )


__all__ = [
    "AdaptiveFitReceipt",
    "deterministic_history_indices",
    "feature_matrix",
    "feature_matrix_v2",
    "fit_predict_probe_probability",
    "fit_predict_probe_probability_v2",
    "selector_feature_names",
    "v2_feature_identity",
    "V2_HASH_FIELDS",
]
