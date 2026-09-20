"""Frozen block-cluster statistics and gate semantics for RCMP-C-TIV."""

from __future__ import annotations

import hashlib
import json
import math
import time
from typing import Any, Mapping, Sequence

import numpy as np


ACTIONS = ("H", "L", "C+", "C-", "D")
DEVIATIONS = ("L", "C+", "C-", "D")


from sevc.core.statistical_bounds import clopper_pearson_bound


def _finite(values: Sequence[float]) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1 or not len(result) or not np.isfinite(result).all():
        raise ValueError("RCMP statistic requires a non-empty finite vector")
    return result


def one_sided_cluster_bound(
    values: Sequence[float],
    *,
    side: str,
    seed: int,
    resamples: int,
    alpha: float,
) -> float:
    vector = _finite(values)
    if side not in {"lower", "upper"}:
        raise ValueError("one-sided bound must be lower or upper")
    if not 0 < alpha < 1 or resamples <= 0:
        raise ValueError("invalid bootstrap parameters")
    if np.all(vector == vector[0]):
        return float(vector[0])
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, len(vector), size=(int(resamples), len(vector)))
    means = vector[indices].mean(axis=1)
    quantile = alpha if side == "lower" else 1.0 - alpha
    return float(np.quantile(means, quantile, method="higher"))


def simultaneous_lower_bounds(
    values_by_key: Mapping[str, Sequence[float]],
    *,
    seed: int,
    resamples: int,
    alpha: float,
) -> dict[str, float]:
    keys = tuple(sorted(values_by_key))
    matrix = np.stack([_finite(values_by_key[key]) for key in keys], axis=1)
    if len({len(values_by_key[key]) for key in keys}) != 1:
        raise ValueError("simultaneous vectors must have the same block count")
    estimates = matrix.mean(axis=0)
    if np.all(matrix == matrix[0:1, :]):
        return {key: float(estimates[index]) for index, key in enumerate(keys)}
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, len(matrix), size=(int(resamples), len(matrix)))
    bootstrap = matrix[indices].mean(axis=1)
    max_shortfall = np.max(estimates[None, :] - bootstrap, axis=1)
    correction = float(np.quantile(max_shortfall, 1.0 - alpha, method="higher"))
    return {
        key: float(estimates[index] - correction) for index, key in enumerate(keys)
    }


def _hash_feature(value: str) -> float:
    return int(value[:16], 16) / float((1 << 64) - 1)


def _byte_features(encoded: bytes) -> dict[str, float]:
    if not encoded:
        raise ValueError("selector envelope must not be empty")
    counts = np.bincount(np.frombuffer(encoded, dtype=np.uint8), minlength=256)
    probabilities = counts[counts > 0].astype(np.float64) / len(encoded)
    entropy = float(-np.sum(probabilities * np.log2(probabilities)))
    nibble = np.zeros(16, dtype=np.float64)
    raw = np.frombuffer(encoded, dtype=np.uint8)
    np.add.at(nibble, raw >> 4, 1)
    np.add.at(nibble, raw & 15, 1)
    nibble /= 2.0 * len(encoded)
    result = {
        "serialized_length": float(len(encoded)),
        "serialized_byte_entropy": entropy,
    }
    result.update({f"serialized_nibble_{index:02d}": float(value) for index, value in enumerate(nibble)})
    return result


def selector_features_from_envelope(
    encoded: bytes,
) -> tuple[dict[str, float], dict[str, str], dict[str, Any]]:
    """Parse one immutable public envelope and extract the frozen features."""

    payload = json.loads(encoded.decode("utf-8"))
    required = {
        "schema_version",
        "protocol_version",
        "task_id",
        "source_commitment",
        "wrapper_id",
        "wrapper_descriptor",
        "permutation_id",
        "report_nonce_sha256",
        "checkpoint_count",
        "batch_count",
        "tensor_count",
        "tensor_payload_bytes",
        "model_code",
        "wrapped_proof_identity",
        "public_state_features",
    }
    if set(payload) != required:
        raise ValueError("public selector envelope schema mismatch")
    descriptor = payload["wrapper_descriptor"]
    proof_identity = payload["wrapped_proof_identity"]
    hash_fields = {
        "task_id": str(payload["task_id"]),
        "source_commitment": str(payload["source_commitment"]),
        "descriptor_commitment": str(descriptor["descriptor_commitment"]),
        "wrapped_proof_sha256": str(proof_identity["proof_sha256"]),
    }
    if any(len(value) != 64 for value in hash_fields.values()):
        raise ValueError("selector hash identity must be SHA-256")
    features = {
        "checkpoint_count": float(payload["checkpoint_count"]),
        "batch_count": float(payload["batch_count"]),
        "tensor_count": float(payload["tensor_count"]),
        "tensor_payload_bytes": float(payload["tensor_payload_bytes"]),
        "model_code": float(payload["model_code"]),
    }
    features.update(_byte_features(encoded))
    for name, value in hash_fields.items():
        features[f"hash64:{name}"] = _hash_feature(value)
    public_state = payload["public_state_features"]
    if public_state.get("schema_version") != "sevc-rcmp-public-state-features-v2":
        raise ValueError("public state feature schema mismatch")
    summaries = public_state["summaries"]
    summary_metrics = (
        "tensor_count",
        "element_count",
        "payload_bytes",
        "finite_fraction",
        "zero_fraction",
        "mean",
        "std",
        "l1",
        "l2",
        "linf",
    )
    for group_name in sorted(summaries):
        if set(summaries[group_name]) != set(summary_metrics):
            raise ValueError("public state summary schema mismatch")
        for metric in summary_metrics:
            features[f"summary:{group_name}:{metric}"] = float(
                summaries[group_name][metric]
            )
    sketch = public_state["task_selected_coordinate_sketch"]
    if len(sketch) != 64 or [int(row["slot"]) for row in sketch] != list(range(64)):
        raise ValueError("public coordinate sketch schema mismatch")
    for row in sketch:
        features[f"sketch:{int(row['slot']):02d}"] = float(row["value"])
    vector = np.asarray(list(features.values()), dtype=np.float64)
    if not np.isfinite(vector).all():
        raise ValueError("non-finite selector feature")
    return features, hash_fields, payload


def selector_features_from_public_view(
    view: Any,
    *,
    required_feature_names: Sequence[str] | None = None,
) -> tuple[dict[str, float], dict[str, str], dict[str, Any]]:
    """Extract the frozen v3 feature schema from a compact public proof view.

    Envelope/hash selectors stay envelope-only.  Proof summaries are computed
    lazily only when a selector depends on them.  Legacy nibble features are
    reconstructed from the same deterministic full public view so the only
    intentional feature changes are real compact length and entropy.
    """

    from sevc.verification.replay_coupled_probes import (
        T2_COMPACT_TASK_SCHEMA,
        T2_TASK_SCHEMA,
        canonical_json,
    )

    encoded = bytes(view.compact_envelope_bytes)
    payload = json.loads(encoded.decode("utf-8"))
    required = {
        "schema_version",
        "protocol_version",
        "task_id",
        "source_commitment",
        "wrapper_id",
        "wrapper_descriptor",
        "permutation_id",
        "report_nonce_sha256",
        "checkpoint_count",
        "batch_count",
        "tensor_count",
        "tensor_payload_bytes",
        "model_code",
        "wrapped_proof_identity",
        "public_proof_content_address",
    }
    if set(payload) != required or payload["schema_version"] != T2_COMPACT_TASK_SCHEMA:
        raise ValueError("compact public selector envelope schema mismatch")
    if payload["public_proof_content_address"] != payload["wrapped_proof_identity"]["proof_sha256"]:
        raise ValueError("compact public proof content address mismatch")
    descriptor = payload["wrapper_descriptor"]
    proof_identity = payload["wrapped_proof_identity"]
    hash_fields = {
        "task_id": str(payload["task_id"]),
        "source_commitment": str(payload["source_commitment"]),
        "descriptor_commitment": str(descriptor["descriptor_commitment"]),
        "wrapped_proof_sha256": str(proof_identity["proof_sha256"]),
    }
    if any(len(value) != 64 for value in hash_fields.values()):
        raise ValueError("selector hash identity must be SHA-256")
    features = {
        "checkpoint_count": float(payload["checkpoint_count"]),
        "batch_count": float(payload["batch_count"]),
        "tensor_count": float(payload["tensor_count"]),
        "tensor_payload_bytes": float(payload["tensor_payload_bytes"]),
        "model_code": float(payload["model_code"]),
    }
    compact_byte_features = _byte_features(encoded)
    features.update(compact_byte_features)
    for name, value in hash_fields.items():
        features[f"hash64:{name}"] = _hash_feature(value)
    required_names = set(required_feature_names or ())
    needs_proof = required_feature_names is None or any(
        name.startswith(("summary:", "sketch:", "serialized_nibble_"))
        for name in required_names
    )
    if needs_proof:
        public_state = view.public_state_features()
        legacy_payload = dict(payload)
        legacy_payload["schema_version"] = T2_TASK_SCHEMA
        legacy_payload.pop("public_proof_content_address")
        legacy_payload["public_state_features"] = public_state
        legacy_bytes = canonical_json(legacy_payload).encode("utf-8")
        legacy_features, legacy_hashes, _ = selector_features_from_envelope(
            legacy_bytes
        )
        if legacy_hashes != hash_fields:
            raise ValueError("compact/full selector hash identity mismatch")
        for name, value in legacy_features.items():
            if name not in {"serialized_length", "serialized_byte_entropy"}:
                features[name] = value
    if required_feature_names is not None:
        missing = sorted(set(required_feature_names) - set(features))
        if missing:
            raise ValueError(f"compact selector feature dependency missing: {missing}")
    vector = np.asarray(list(features.values()), dtype=np.float64)
    if not np.isfinite(vector).all():
        raise ValueError("non-finite compact selector feature")
    return features, hash_fields, payload


def _threshold_parameter(values: np.ndarray, labels: np.ndarray, direction: str) -> float:
    candidates = np.unique(np.quantile(values, np.linspace(0.05, 0.95, 19)))
    if not len(candidates):
        return 0.0
    best_threshold = float(candidates[0])
    best_advantage = -1.0
    for threshold in candidates:
        prediction = values <= threshold if direction == "le" else values >= threshold
        advantage = _advantage(prediction.astype(np.float64), labels)
        if advantage > best_advantage:
            best_advantage = advantage
            best_threshold = float(threshold)
    return best_threshold


def _fit_selector_family_from_extracted(
    extracted: Sequence[tuple[Mapping[str, float], Mapping[str, str], Any]],
    labels: Sequence[int],
    *,
    seed: int,
) -> dict[str, Any]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeClassifier

    if len(extracted) != len(labels) or not extracted:
        raise ValueError("selector fit identity mismatch")
    feature_names = tuple(extracted[0][0])
    if any(tuple(row[0]) != feature_names for row in extracted):
        raise ValueError("selector feature ordering drift")
    matrix = np.asarray(
        [[row[0][name] for name in feature_names] for row in extracted],
        dtype=np.float64,
    )
    targets = np.asarray(labels, dtype=np.int64)
    if set(np.unique(targets)) != {0, 1}:
        raise ValueError("selector fit requires production and probe labels")
    selectors: list[dict[str, Any]] = []
    scalar_names = (
        "checkpoint_count",
        "batch_count",
        "tensor_count",
        "tensor_payload_bytes",
        "model_code",
        "serialized_length",
        "serialized_byte_entropy",
    )
    for name in scalar_names:
        feature_index = feature_names.index(name)
        for direction in ("le", "ge"):
            selectors.append(
                {
                    "selector_key": f"scalar-threshold:{name}:{direction}",
                    "family": "metadata" if name != "serialized_length" else "serialized-length",
                    "kind": "scalar-threshold",
                    "feature_name": name,
                    "direction": direction,
                    "threshold": _threshold_parameter(
                        matrix[:, feature_index], targets, direction
                    ),
                }
            )
    for hash_name in (
        "task_id",
        "source_commitment",
        "descriptor_commitment",
        "wrapped_proof_sha256",
    ):
        for modulus in (2, 4, 8, 16):
            for bucket in range(modulus):
                selectors.append(
                    {
                        "selector_key": f"hash-bucket:{hash_name}:mod={modulus}:bucket={bucket}",
                        "family": "hash-bucket",
                        "kind": "hash-bucket",
                        "hash_name": hash_name,
                        "modulus": modulus,
                        "bucket": bucket,
                    }
                )
    for c_value in (0.01, 0.1, 1.0, 10.0):
        scaler = StandardScaler().fit(matrix)
        transformed = scaler.transform(matrix)
        model = LogisticRegression(
            C=c_value,
            penalty="l2",
            solver="liblinear",
            random_state=int(seed),
            max_iter=1000,
        ).fit(transformed, targets)
        selectors.append(
            {
                "selector_key": f"l2-logistic-C={c_value}",
                "family": "l2-logistic",
                "kind": "l2-logistic",
                "C": c_value,
                "mean": scaler.mean_.astype(float).tolist(),
                "scale": scaler.scale_.astype(float).tolist(),
                "coef": model.coef_[0].astype(float).tolist(),
                "intercept": float(model.intercept_[0]),
            }
        )
    for depth in (1, 2, 3):
        model = DecisionTreeClassifier(
            max_depth=depth,
            random_state=int(seed),
        ).fit(matrix, targets)
        tree = model.tree_
        values = tree.value
        selectors.append(
            {
                "selector_key": f"decision-tree-depth={depth}",
                "family": "decision-tree",
                "kind": "decision-tree",
                "depth": depth,
                "children_left": tree.children_left.astype(int).tolist(),
                "children_right": tree.children_right.astype(int).tolist(),
                "feature": tree.feature.astype(int).tolist(),
                "threshold": tree.threshold.astype(float).tolist(),
                "class_counts": [
                    np.asarray(value, dtype=np.float64).reshape(-1).astype(float).tolist()
                    for value in values
                ],
            }
        )
    feature_schema = {
        "schema_version": "sevc-rcmp-selector-features-v2",
        "feature_names": list(feature_names),
    }
    feature_schema_sha256 = hashlib.sha256(
        json.dumps(feature_schema, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": "sevc-rcmp-selector-parameters-v2",
        "fit_seed": int(seed),
        "feature_schema": feature_schema,
        "feature_schema_sha256": feature_schema_sha256,
        "selectors": selectors,
    }


def fit_selector_family(
    envelopes: Sequence[bytes],
    labels: Sequence[int],
    *,
    seed: int,
) -> dict[str, Any]:
    """Fit the preregistered finite family on selector-fit blocks only."""

    return _fit_selector_family_from_extracted(
        [selector_features_from_envelope(value) for value in envelopes],
        labels,
        seed=seed,
    )


def fit_selector_family_from_features(
    feature_rows: Sequence[Mapping[str, float]],
    hash_rows: Sequence[Mapping[str, str]],
    labels: Sequence[int],
    *,
    seed: int,
) -> dict[str, Any]:
    """Fit the same 141-selector family from an audited feature ledger."""

    if len(feature_rows) != len(hash_rows):
        raise ValueError("selector feature/hash identity mismatch")
    return _fit_selector_family_from_extracted(
        [(dict(features), dict(hashes), None) for features, hashes in zip(feature_rows, hash_rows)],
        labels,
        seed=seed,
    )


def evaluate_frozen_selector(
    envelope: Any,
    selector: Mapping[str, Any],
    feature_names: Sequence[str],
) -> tuple[bool, float]:
    """Measure one selector from bytes through parse, features and decision."""

    kind = selector["kind"]
    required_names: Sequence[str]
    if kind == "scalar-threshold":
        required_names = (str(selector["feature_name"]),)
    elif kind == "hash-bucket":
        required_names = ()
    elif kind == "decision-tree":
        required_names = tuple(
            feature_names[int(index)]
            for index in sorted({int(value) for value in selector["feature"] if int(value) >= 0})
        )
    else:
        required_names = tuple(feature_names)
    if hasattr(envelope, "public_state_features"):
        features, hash_fields, _ = selector_features_from_public_view(
            envelope,
            required_feature_names=required_names,
        )
    else:
        features, hash_fields, _ = selector_features_from_envelope(envelope)
    return evaluate_frozen_selector_from_features(
        features, hash_fields, selector, feature_names
    )


def evaluate_frozen_selector_from_features(
    features: Mapping[str, float],
    hash_fields: Mapping[str, str],
    selector: Mapping[str, Any],
    feature_names: Sequence[str],
) -> tuple[bool, float]:
    """Evaluate one frozen selector from an already audited public feature row."""

    kind = selector["kind"]
    vector = (
        np.asarray([features[name] for name in feature_names], dtype=np.float64)
        if kind in {"l2-logistic", "decision-tree"}
        else None
    )
    if kind == "scalar-threshold":
        value = float(features[str(selector["feature_name"])])
        threshold = float(selector["threshold"])
        decision = value <= threshold if selector["direction"] == "le" else value >= threshold
        score = threshold - value if selector["direction"] == "le" else value - threshold
        return bool(decision), float(score)
    if kind == "hash-bucket":
        value = int(hash_fields[str(selector["hash_name"])], 16)
        decision = value % int(selector["modulus"]) == int(selector["bucket"])
        return bool(decision), float(decision)
    if kind == "l2-logistic":
        mean = np.asarray(selector["mean"], dtype=np.float64)
        scale = np.asarray(selector["scale"], dtype=np.float64)
        coefficient = np.asarray(selector["coef"], dtype=np.float64)
        assert vector is not None
        logit = float(np.dot((vector - mean) / scale, coefficient) + float(selector["intercept"]))
        probability = 1.0 / (1.0 + math.exp(-max(-700.0, min(700.0, logit))))
        return probability >= 0.5, probability
    if kind == "decision-tree":
        node = 0
        left = selector["children_left"]
        right = selector["children_right"]
        feature = selector["feature"]
        threshold = selector["threshold"]
        assert vector is not None
        while int(left[node]) != int(right[node]):
            node = int(left[node]) if vector[int(feature[node])] <= float(threshold[node]) else int(right[node])
        counts = np.asarray(selector["class_counts"][node], dtype=np.float64)
        probability = float(counts[1] / counts.sum()) if counts.sum() and len(counts) > 1 else 0.0
        return probability >= 0.5, probability
    raise ValueError(f"unknown frozen selector kind: {kind}")


def evaluate_grouped_crossfit_selector_oracle(
    feature_rows: Sequence[Mapping[str, float]],
    hash_rows: Sequence[Mapping[str, str]],
    labels: Sequence[int],
    block_ids: Sequence[int],
    *,
    seed: int,
    resamples: int,
    alpha: float,
    folds: int = 5,
) -> dict[str, Any]:
    """Run the fixed 141-algorithm free-feature grouped cross-fit oracle."""

    from sklearn.model_selection import GroupKFold

    if not (
        len(feature_rows)
        == len(hash_rows)
        == len(labels)
        == len(block_ids)
    ) or not feature_rows:
        raise ValueError("counterfactual selector oracle identity mismatch")
    targets = np.asarray(labels, dtype=np.int64)
    groups = np.asarray(block_ids, dtype=np.int64)
    if set(np.unique(targets)) != {0, 1}:
        raise ValueError("counterfactual selector oracle requires both labels")
    if len(np.unique(groups)) != 160 or int(folds) != 5:
        raise ValueError("counterfactual selector oracle requires frozen 160 blocks / 5 folds")
    splitter = GroupKFold(n_splits=int(folds))
    prediction_rows: list[dict[str, Any] | None] = [None] * len(targets)
    expected_keys: tuple[str, ...] | None = None
    feature_schema_sha256: str | None = None
    fold_receipts = []
    for fold_index, (train, test) in enumerate(
        splitter.split(np.zeros(len(targets)), targets, groups), start=1
    ):
        parameters = fit_selector_family_from_features(
            [feature_rows[index] for index in train],
            [hash_rows[index] for index in train],
            targets[train].tolist(),
            seed=int(seed) + fold_index,
        )
        keys = tuple(str(row["selector_key"]) for row in parameters["selectors"])
        if expected_keys is None:
            expected_keys = keys
            feature_schema_sha256 = str(parameters["feature_schema_sha256"])
        elif keys != expected_keys or parameters["feature_schema_sha256"] != feature_schema_sha256:
            raise ValueError("counterfactual selector algorithm/schema drift across folds")
        feature_names = tuple(parameters["feature_schema"]["feature_names"])
        for index in test:
            predictions = {}
            for selector in parameters["selectors"]:
                decision, score = evaluate_frozen_selector_from_features(
                    feature_rows[index],
                    hash_rows[index],
                    selector,
                    feature_names,
                )
                predictions[str(selector["selector_key"])] = {
                    "decision": bool(decision),
                    "score": float(score),
                    "runtime_seconds": 0.0,
                }
            prediction_rows[int(index)] = {
                "block_seed": int(groups[index]),
                "is_probe": int(targets[index]),
                "predictions": predictions,
            }
        fold_receipts.append(
            {
                "fold": fold_index,
                "train_block_count": len(np.unique(groups[train])),
                "test_block_count": len(np.unique(groups[test])),
                "train_row_count": len(train),
                "test_row_count": len(test),
                "selector_count": len(keys),
            }
        )
    if expected_keys is None or len(expected_keys) != 141 or any(
        row is None for row in prediction_rows
    ):
        raise ValueError("counterfactual selector oracle coverage failure")
    result = evaluate_fixed_selector_recognizability(
        [row for row in prediction_rows if row is not None],
        registered_selector_keys=expected_keys,
        honest_fit_median_replay_seconds=1.0,
        seed=int(seed),
        resamples=int(resamples),
        alpha=float(alpha),
    )
    result.update(
        {
            "counterfactual_group_folds": int(folds),
            "free_feature_oracle": True,
            "source_or_replay_generation_count": 0,
            "selector_keys": list(expected_keys),
            "feature_schema_sha256": feature_schema_sha256,
            "fold_receipts": fold_receipts,
        }
    )
    return result


def benchmark_selector_family(
    envelopes: Sequence[Any],
    parameters: Mapping[str, Any],
) -> list[dict[str, Any]]:
    feature_names = parameters["feature_schema"]["feature_names"]
    inventory = []
    for selector in parameters["selectors"]:
        runtimes = []
        for envelope in envelopes:
            started = time.perf_counter_ns()
            evaluate_frozen_selector(envelope, selector, feature_names)
            runtimes.append((time.perf_counter_ns() - started) / 1_000_000_000.0)
        inventory.append(
            {
                "selector_key": selector["selector_key"],
                "family": selector["family"],
                "runtime_p95_seconds": float(
                    np.quantile(runtimes, 0.95, method="higher")
                ),
                "runtime_sample_count": len(runtimes),
            }
        )
    return inventory


def _block_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("block AUC requires both production and probe tasks")
    return float(roc_auc_score(labels, scores))


def evaluate_fixed_selector_recognizability(
    prediction_rows: Sequence[Mapping[str, Any]],
    *,
    registered_selector_keys: Sequence[str],
    honest_fit_median_replay_seconds: float,
    seed: int,
    resamples: int,
    alpha: float,
) -> dict[str, Any]:
    """Evaluate fixed selectors with paired whole-block centered-max bounds."""

    selectors = tuple(registered_selector_keys)
    if not selectors or resamples <= 0:
        raise ValueError("fixed selector evaluation requires selectors and resamples")
    blocks = sorted({int(row["block_seed"]) for row in prediction_rows})
    rows_by_block = {
        block: [row for row in prediction_rows if int(row["block_seed"]) == block]
        for block in blocks
    }
    contributions = np.zeros((len(blocks), len(selectors)), dtype=np.float64)
    block_aucs = np.zeros_like(contributions)
    runtimes: dict[str, list[float]] = {key: [] for key in selectors}
    for block_index, block in enumerate(blocks):
        rows = rows_by_block[block]
        labels = np.asarray([int(row["is_probe"]) for row in rows], dtype=np.int64)
        if len(rows) != 40 or int(labels.sum()) != 8:
            raise ValueError("recognizability block must contain 8 probes and 32 production tasks")
        for selector_index, key in enumerate(selectors):
            decisions = np.asarray(
                [float(bool(row["predictions"][key]["decision"])) for row in rows],
                dtype=np.float64,
            )
            scores = np.asarray(
                [float(row["predictions"][key]["score"]) for row in rows],
                dtype=np.float64,
            )
            runtimes[key].extend(
                float(row["predictions"][key]["runtime_seconds"]) for row in rows
            )
            contributions[block_index, selector_index] = (
                decisions[labels == 1].mean() - decisions[labels == 0].mean()
            )
            block_aucs[block_index, selector_index] = _block_auc(labels, scores)
    point_mean = contributions.mean(axis=0)
    lambda_point = np.abs(point_mean)
    auc_mean = block_aucs.mean(axis=0)
    auc_point = 0.5 + np.abs(auc_mean - 0.5)
    rng = np.random.default_rng(int(seed))
    lambda_max_statistics = np.empty(int(resamples), dtype=np.float64)
    auc_max_statistics = np.empty(int(resamples), dtype=np.float64)
    cursor = 0
    chunk_size = 250
    while cursor < int(resamples):
        count = min(chunk_size, int(resamples) - cursor)
        sampled = rng.integers(0, len(blocks), size=(count, len(blocks)))
        sampled_lambda_mean = contributions[sampled].mean(axis=1)
        sampled_auc_mean = block_aucs[sampled].mean(axis=1)
        lambda_max_statistics[cursor : cursor + count] = np.max(
            np.abs(sampled_lambda_mean - point_mean[None, :]), axis=1
        )
        auc_max_statistics[cursor : cursor + count] = np.max(
            np.abs(sampled_auc_mean - auc_mean[None, :]), axis=1
        )
        cursor += count
    lambda_correction = float(
        np.quantile(lambda_max_statistics, 1.0 - alpha, method="higher")
    )
    auc_correction = float(
        np.quantile(auc_max_statistics, 1.0 - alpha, method="higher")
    )
    inventory = []
    for index, key in enumerate(selectors):
        runtime_p95 = float(np.quantile(runtimes[key], 0.95, method="higher"))
        inventory.append(
            {
                "selector_key": key,
                "lambda_hat": float(lambda_point[index]),
                "lambda_simultaneous_upper95": min(
                    1.0, float(lambda_point[index] + lambda_correction)
                ),
                "grouped_auc": float(auc_point[index]),
                "grouped_auc_simultaneous_upper95": min(
                    1.0, float(auc_point[index] + auc_correction)
                ),
                "formal_runtime_p95_seconds": runtime_p95,
                "formal_runtime_ratio": runtime_p95
                / max(float(honest_fit_median_replay_seconds), 1e-12),
            }
        )
    return {
        "block_count": len(blocks),
        "registered_selector_count": len(selectors),
        "bootstrap_seed": int(seed),
        "bootstrap_resamples": int(resamples),
        "alpha": float(alpha),
        "lambda_centered_max_correction": lambda_correction,
        "lambda_F_simultaneous_upper95": max(
            row["lambda_simultaneous_upper95"] for row in inventory
        ),
        "auc_centered_max_correction": auc_correction,
        "grouped_auc_upper95": max(
            row["grouped_auc_simultaneous_upper95"] for row in inventory
        ),
        "maximum_registered_runtime_ratio": max(
            row["formal_runtime_ratio"] for row in inventory
        ),
        "auc_is_total_variation": False,
        "oracle_role_bit_constructible": False,
        "inventory": inventory,
    }


def _public_feature_matrix(
    public_rows: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, list[str]]:
    scalar_names = [
        "checkpoint_count",
        "batch_count",
        "tensor_count",
        "tensor_payload_bytes",
        "model_code",
    ]
    hash_names = [
        "task_id",
        "source_commitment",
        "permutation_id",
        "report_nonce_sha256",
    ]
    names = scalar_names + [f"{name}_hash64" for name in hash_names] + ["serialized_length"]
    rows = []
    for row in public_rows:
        values = [float(row[name]) for name in scalar_names]
        values.extend(_hash_feature(str(row[name])) for name in hash_names)
        serialized = len(
            __import__("json").dumps(
                dict(row), sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        )
        values.append(float(serialized))
        rows.append(values)
    matrix = np.asarray(rows, dtype=np.float64)
    if not np.isfinite(matrix).all():
        raise ValueError("non-finite public selector feature")
    return matrix, names


def _advantage(prediction: np.ndarray, labels: np.ndarray) -> float:
    probe = prediction[labels == 1]
    production = prediction[labels == 0]
    if not len(probe) or not len(production):
        raise ValueError("recognizability evaluation lacks both roles")
    return abs(float(probe.mean() - production.mean()))


def _block_bootstrap_selector_bounds(
    predictions: Mapping[str, np.ndarray],
    scores: Mapping[str, np.ndarray],
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    seed: int,
    resamples: int,
    alpha: float,
) -> tuple[float, float]:
    from sklearn.metrics import roc_auc_score

    selectors = tuple(sorted(predictions))
    unique_groups = np.unique(groups)
    if not selectors or not len(unique_groups):
        raise ValueError("selector bootstrap requires selectors and blocks")
    point_advantages = {
        key: _advantage(predictions[key], labels) for key in selectors
    }
    point_aucs = {
        key: float(roc_auc_score(labels, scores[key])) for key in selectors
    }
    if resamples <= 0:
        raise ValueError("selector bootstrap requires positive resamples")
    rng = np.random.default_rng(int(seed))
    max_advantages = []
    max_aucs = []
    group_indices = {group: np.flatnonzero(groups == group) for group in unique_groups}
    for _ in range(int(resamples)):
        sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        selected = np.concatenate([group_indices[group] for group in sampled])
        sampled_labels = labels[selected]
        if len(np.unique(sampled_labels)) != 2:
            continue
        max_advantages.append(
            max(_advantage(predictions[key][selected], sampled_labels) for key in selectors)
        )
        max_aucs.append(
            max(float(roc_auc_score(sampled_labels, scores[key][selected])) for key in selectors)
        )
    if not max_advantages:
        raise ValueError("selector bootstrap produced no valid sample")
    advantage_upper = max(
        max(point_advantages.values()),
        float(np.quantile(max_advantages, 1.0 - alpha, method="higher")),
    )
    auc_upper = max(
        max(point_aucs.values()),
        float(np.quantile(max_aucs, 1.0 - alpha, method="higher")),
    )
    return advantage_upper, auc_upper


def evaluate_recognizability(
    public_rows: Sequence[Mapping[str, Any]],
    labels: Sequence[int],
    block_ids: Sequence[str],
    *,
    honest_median_replay_seconds: float,
    seed: int,
    resamples: int,
    alpha: float,
    runtime_budget_ratio: float,
) -> dict[str, Any]:
    """Evaluate the preregistered cheap selector family on grouped folds."""

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeClassifier

    if len(public_rows) != len(labels) or len(labels) != len(block_ids):
        raise ValueError("recognizability row identity mismatch")
    matrix, feature_names = _public_feature_matrix(public_rows)
    targets = np.asarray(labels, dtype=np.int64)
    groups = np.asarray(block_ids)
    folds = min(5, len(np.unique(groups)))
    splitter = GroupKFold(n_splits=folds)
    predictions: dict[str, np.ndarray] = {}
    scores: dict[str, np.ndarray] = {}
    runtimes: dict[str, float] = {}

    def evaluate_model(key: str, factory: Any) -> None:
        predicted = np.zeros(len(targets), dtype=np.float64)
        scored = np.zeros(len(targets), dtype=np.float64)
        runtime_total = 0.0
        runtime_count = 0
        for train, test in splitter.split(matrix, targets, groups):
            model = factory()
            model.fit(matrix[train], targets[train])
            started = time.perf_counter_ns()
            probabilities = model.predict_proba(matrix[test])[:, 1]
            runtime_total += (time.perf_counter_ns() - started) / 1_000_000_000.0
            runtime_count += len(test)
            scored[test] = probabilities
            predicted[test] = probabilities >= 0.5
        predictions[key] = predicted
        scores[key] = scored
        runtimes[key] = runtime_total / max(1, runtime_count)

    for c_value in (0.01, 0.1, 1.0, 10.0):
        evaluate_model(
            f"l2-logistic-C={c_value}",
            lambda c_value=c_value: make_pipeline(
                StandardScaler(),
                LogisticRegression(C=c_value, penalty="l2", solver="liblinear", random_state=seed),
            ),
        )
    for depth in (1, 2, 3):
        evaluate_model(
            f"decision-tree-depth={depth}",
            lambda depth=depth: DecisionTreeClassifier(max_depth=depth, random_state=seed),
        )

    for feature_index, feature_name in enumerate(feature_names):
        values = matrix[:, feature_index]
        thresholds = np.unique(np.quantile(values, np.linspace(0.05, 0.95, 19)))
        best_key = f"scalar-threshold:{feature_name}"
        predicted = np.zeros(len(targets), dtype=np.float64)
        scored = np.zeros(len(targets), dtype=np.float64)
        runtime_total = 0.0
        runtime_count = 0
        for train, test in splitter.split(matrix, targets, groups):
            best_threshold = float(thresholds[0]) if len(thresholds) else 0.0
            best_direction = 1
            best = -1.0
            for threshold in thresholds:
                for direction in (-1, 1):
                    train_prediction = (
                        direction * (values[train] - threshold) >= 0
                    ).astype(np.float64)
                    advantage = _advantage(train_prediction, targets[train])
                    if advantage > best:
                        best = advantage
                        best_threshold = float(threshold)
                        best_direction = direction
            started = time.perf_counter_ns()
            test_score = best_direction * (values[test] - best_threshold)
            runtime_total += (time.perf_counter_ns() - started) / 1_000_000_000.0
            runtime_count += len(test)
            scored[test] = test_score
            predicted[test] = test_score >= 0
        predictions[best_key] = predicted
        scores[best_key] = scored
        runtimes[best_key] = runtime_total / max(1, runtime_count)

    for hash_name in ("task_id", "source_commitment", "permutation_id", "report_nonce_sha256"):
        values = np.asarray([int(str(row[hash_name]), 16) for row in public_rows], dtype=object)
        for modulus in (2, 4, 8, 16):
            for bucket in range(modulus):
                key = f"hash-bucket:{hash_name}:mod={modulus}:bucket={bucket}"
                started = time.perf_counter_ns()
                prediction = np.asarray(
                    [float(int(value) % modulus == bucket) for value in values],
                    dtype=np.float64,
                )
                elapsed = (time.perf_counter_ns() - started) / 1_000_000_000.0
                predictions[key] = prediction
                scores[key] = prediction.copy()
                runtimes[key] = elapsed / max(1, len(prediction))

    budget_seconds = float(runtime_budget_ratio) * float(honest_median_replay_seconds)
    registered = {
        key: value for key, value in predictions.items() if runtimes[key] <= budget_seconds
    }
    registered_scores = {key: scores[key] for key in registered}
    if not registered:
        raise ValueError("no selector satisfies the preregistered runtime budget")
    lambda_upper, auc_upper = _block_bootstrap_selector_bounds(
        registered,
        registered_scores,
        targets,
        groups,
        seed=seed,
        resamples=resamples,
        alpha=alpha,
    )
    inventory = []
    for key in sorted(predictions):
        inventory.append(
            {
                "selector_key": key,
                "registered": key in registered,
                "runtime_seconds_per_task": runtimes[key],
                "runtime_ratio": runtimes[key] / max(honest_median_replay_seconds, 1e-12),
                "lambda_hat": _advantage(predictions[key], targets),
                "grouped_auc": float(roc_auc_score(targets, scores[key])),
            }
        )
    return {
        "selector_count": len(inventory),
        "registered_selector_count": len(registered),
        "runtime_budget_seconds": budget_seconds,
        "maximum_registered_runtime_ratio": max(
            runtimes[key] / max(honest_median_replay_seconds, 1e-12)
            for key in registered
        ),
        "lambda_F_simultaneous_upper95": float(lambda_upper),
        "grouped_auc_upper95": float(auc_upper),
        "oracle_role_bit_constructible": False,
        "auc_is_total_variation": False,
        "inventory": inventory,
    }


def evaluate_action_gate(
    block_rows: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    resamples: int,
    alpha: float,
) -> dict[str, Any]:
    """Apply the frozen block-level exact T2 action bounds.

    ``seed`` and ``resamples`` remain accepted for the older public API, but
    exact binomial bounds intentionally do not consume either value.
    """

    del seed, resamples
    if not block_rows:
        raise ValueError("action gate requires block rows")
    false_h = [int(bool(row["H_false_settlement"])) for row in block_rows]
    detection = {
        action: [int(bool(row[f"{action}_detected"])) for row in block_rows]
        for action in DEVIATIONS
    }
    trials = len(block_rows)
    deviation_alpha = float(alpha) / len(DEVIATIONS)
    lower = {
        action: clopper_pearson_bound(
            sum(values),
            trials,
            side="lower",
            alpha=deviation_alpha,
        )
        for action, values in detection.items()
    }
    return {
        "block_count": trials,
        "H_false_settlement_rate": float(np.mean(false_h)),
        "H_false_settlement_count": sum(false_h),
        "H_false_settlement_upper95": clopper_pearson_bound(
            sum(false_h),
            trials,
            side="upper",
            alpha=alpha,
        ),
        "detection_rate": {
            action: float(np.mean(values)) for action, values in detection.items()
        },
        "detection_count": {
            action: int(sum(values)) for action, values in detection.items()
        },
        "detection_simultaneous_lower95": lower,
        "honest_method": "one-sided Clopper-Pearson alpha=0.05",
        "deviation_method": "one-sided Clopper-Pearson Bonferroni alpha/4",
        "deviation_alpha_each": deviation_alpha,
    }


def evaluate_utility_gate(
    block_rows: Sequence[Mapping[str, Any]],
    *,
    service_fee_ratio: float,
    bond_ratio: float,
    liquidity_cost_rate: float,
    seed: int,
    resamples: int,
    alpha: float,
) -> dict[str, Any]:
    margins: dict[str, list[float]] = {action: [] for action in DEVIATIONS}
    participation = []
    for row in block_rows:
        honest_cost = float(row["H_cost_seconds"])
        if not math.isfinite(honest_cost) or honest_cost <= 0:
            raise ValueError("honest replay cost must be finite and positive")
        service_fee = service_fee_ratio * honest_cost
        bond = bond_ratio * honest_cost
        participation.append(service_fee - honest_cost - liquidity_cost_rate * bond)
        for action in DEVIATIONS:
            action_cost = float(row[f"{action}_cost_seconds"])
            pass_h = 1.0 - float(bool(row["H_false_settlement"]))
            pass_a = 1.0 - float(bool(row[f"{action}_detected"]))
            slash_h = float(bool(row["H_false_settlement"]))
            slash_a = float(bool(row[f"{action}_detected"]))
            margin = (
                (pass_h - pass_a) * service_fee
                + (slash_a - slash_h) * bond
                - (honest_cost - action_cost)
            )
            margins[action].append(float(margin))
    lower = simultaneous_lower_bounds(
        margins, seed=seed, resamples=resamples, alpha=alpha
    )
    return {
        "honest_participation_margin_mean": float(np.mean(participation)),
        "honest_participation_margin_lower95": one_sided_cluster_bound(
            participation,
            side="lower",
            seed=seed,
            resamples=resamples,
            alpha=alpha,
        ),
        "utility_margin_mean": {
            action: float(np.mean(values)) for action, values in margins.items()
        },
        "utility_margin_simultaneous_lower95": lower,
    }


def attack_bound_rows(
    probe_error_upper: float, *, eta_cov: float, lambda_grid: Sequence[float]
) -> list[dict[str, float]]:
    return [
        {
            "lambda": float(value),
            "probe_error_upper": float(probe_error_upper),
            "eta_cov": float(eta_cov),
            "B_attack": min(1.0, float(probe_error_upper + value + eta_cov)),
        }
        for value in lambda_grid
    ]


def conditional_t3_screen_verdict(
    *,
    violations: int,
    action: Mapping[str, Any],
    normalized_utility: Mapping[str, Any],
    repetition_utility_point_estimates: Mapping[str, Mapping[str, float]],
    attack_coupling_count: int,
    atom_counts: Mapping[str, int],
    witness_disagreements: int,
    environment_clean: bool,
    auditor_passed: bool,
    environment_blocking: bool = True,
) -> tuple[str, list[str]]:
    """Route the nonformal six-block conditional T3 utility screen.

    This deliberately does not call or modify :func:`formal_verdict`: exact
    six-of-six detection is only a directional front gate, never a formal
    Clopper--Pearson T3 claim.
    """

    reasons: list[str] = []
    if int(violations) != 0:
        reasons.append("integrity_certificate_or_checker_violation")
    if int(action["block_count"]) != 6:
        reasons.append("screen_block_count_not_six")
    if int(action["H_false_settlement_count"]) != 0:
        reasons.append("honest_false_settlement_nonzero")
    if any(int(action["detection_count"][key]) != 6 for key in DEVIATIONS):
        reasons.append("deviation_detection_not_six_of_six")
    if int(attack_coupling_count) != 24 or any(
        int(atom_counts.get(key, 0)) != 6
        for key in ("cp3-positive", "cp3-negative", "cp4-positive", "cp4-negative")
    ):
        reasons.append("attack_coupling_or_atom_coverage_failed")
    if int(witness_disagreements) != 0:
        reasons.append("witness_full_replay_disagreement")
    if float(normalized_utility["honest_participation_margin_lower95"]) < 0.10:
        reasons.append("normalized_honest_participation_below_0.10")
    if any(
        float(normalized_utility["utility_margin_simultaneous_lower95"][key])
        <= 0.05
        for key in DEVIATIONS
    ):
        reasons.append("normalized_deviation_utility_not_strictly_above_0.05")
    for repetition, values in repetition_utility_point_estimates.items():
        required = ("honest_participation", *DEVIATIONS)
        if any(float(values.get(key, math.nan)) <= 0.0 for key in required):
            reasons.append(f"repetition_{repetition}_core_utility_not_positive")
    if environment_blocking and not environment_clean:
        reasons.append("environment_not_clean_throughout")
    if not auditor_passed:
        reasons.append("independent_auditor_failed")
    return (
        ("CONDITIONAL_T3_40_BLOCK_WORTH_RUNNING", [])
        if not reasons
        else ("CONTINUE_MECHANISM_REVIEW", reasons)
    )


def t2_verdict(
    *,
    violations: int,
    action: Mapping[str, Any],
    recognizability: Mapping[str, Any],
    coverage: float,
    eta_cov: float,
    overhead_ratio: float,
    checker_disagreements: int,
    overhead_paired_upper95: float | None = None,
    witness_disagreements: int = 0,
) -> tuple[str, list[str]]:
    reasons = []
    if violations or checker_disagreements or witness_disagreements:
        reasons.append("certificate_or_checker_violation")
    if any(
        float(action["detection_simultaneous_lower95"][key]) < 0.8
        for key in DEVIATIONS
    ):
        reasons.append("action_separation_below_0.80")
    if float(action["H_false_settlement_upper95"]) > 0.05:
        reasons.append("epsilon_H_above_0.05")
    recognition_failure = (
        float(recognizability["maximum_registered_runtime_ratio"]) > 0.05
        or float(recognizability["lambda_F_simultaneous_upper95"]) > 0.10
        or float(recognizability["grouped_auc_upper95"]) > 0.65
        or float(coverage) != 1.0
        or float(eta_cov) != 0.0
    )
    if recognition_failure:
        reasons.append("recognizability_or_coverage_gate_failed")
    if float(overhead_ratio) > 0.25:
        reasons.append("probe_overhead_above_0.25")
    if (
        overhead_paired_upper95 is not None
        and float(overhead_paired_upper95) > 0.25
    ):
        reasons.append("probe_overhead_paired_upper95_above_0.25")
    if not reasons:
        return "PASS", []
    if recognition_failure:
        return "RECOGNIZABILITY_OR_COVERAGE_FAILED", reasons
    return "PROBE_CERTIFICATE_INVALID", reasons


def evaluate_probe_overhead(
    block_rows: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    resamples: int,
    alpha: float,
) -> dict[str, Any]:
    """Aggregate frozen whole-block online and all-in T2 overhead rows."""

    if not block_rows or len({int(row["block_seed"]) for row in block_rows}) != len(
        block_rows
    ):
        raise ValueError("probe overhead requires unique nonempty block rows")
    if resamples <= 0 or not 0.0 < alpha < 1.0:
        raise ValueError("invalid probe overhead bootstrap parameters")
    ordered = sorted(block_rows, key=lambda row: int(row["block_seed"]))
    online = np.asarray(
        [float(row["online_numerator_seconds"]) for row in ordered],
        dtype=np.float64,
    )
    all_in = np.asarray(
        [float(row["all_in_numerator_seconds"]) for row in ordered],
        dtype=np.float64,
    )
    denominator = np.asarray(
        [float(row["production_H_denominator_seconds"]) for row in ordered],
        dtype=np.float64,
    )
    if (
        np.any(online < 0.0)
        or np.any(all_in < online)
        or np.any(denominator <= 0.0)
    ):
        raise ValueError("probe overhead timing rows are invalid")
    online_ratio = float(online.sum()) / float(denominator.sum())
    all_in_ratio = float(all_in.sum()) / float(denominator.sum())
    rng = np.random.default_rng(int(seed))
    bootstrap = np.empty(int(resamples), dtype=np.float64)
    cursor = 0
    while cursor < int(resamples):
        count = min(500, int(resamples) - cursor)
        sampled = rng.integers(0, len(ordered), size=(count, len(ordered)))
        bootstrap[cursor : cursor + count] = (
            online[sampled].sum(axis=1) / denominator[sampled].sum(axis=1)
        )
        cursor += count
    return {
        "cluster_unit": "formal block",
        "block_count": len(ordered),
        "bootstrap_seed": int(seed),
        "bootstrap_resamples": int(resamples),
        "alpha": float(alpha),
        "online_numerator_seconds_total": float(online.sum()),
        "all_in_numerator_seconds_total": float(all_in.sum()),
        "production_H_denominator_seconds_total": float(denominator.sum()),
        "pooled_online_overhead_ratio": online_ratio,
        "paired_whole_block_one_sided_upper95": float(
            np.quantile(bootstrap, 1.0 - alpha, method="higher")
        ),
        "conservative_all_in_ratio": all_in_ratio,
    }


def evaluate_probe_compiler_authorization(
    block_rows: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    resamples: int,
    alpha: float,
) -> dict[str, Any]:
    """Evaluate the frozen twenty-block engineering authorization margins.

    Repeated optimized measurements must already be aggregated inside each
    row.  The block, rather than a timing repetition or task, is the sole
    bootstrap cluster.
    """

    if len(block_rows) != 20 or len({int(row["block_seed"]) for row in block_rows}) != 20:
        raise ValueError("compiler authorization requires twenty unique blocks")
    if resamples <= 0 or not 0.0 < alpha < 1.0:
        raise ValueError("invalid compiler authorization bootstrap parameters")
    ordered = sorted(block_rows, key=lambda row: int(row["block_seed"]))
    baseline_compile = np.asarray(
        [float(row["baseline_compile_group_seconds"]) for row in ordered],
        dtype=np.float64,
    )
    optimized_compile = np.asarray(
        [float(row["optimized_compile_group_seconds"]) for row in ordered],
        dtype=np.float64,
    )
    deployable = np.asarray(
        [float(row["optimized_deployable_numerator_seconds"]) for row in ordered],
        dtype=np.float64,
    )
    all_in = np.asarray(
        [float(row["optimized_all_in_numerator_seconds"]) for row in ordered],
        dtype=np.float64,
    )
    denominator = np.asarray(
        [float(row["production_H_denominator_seconds"]) for row in ordered],
        dtype=np.float64,
    )
    if (
        np.any(baseline_compile <= 0.0)
        or np.any(optimized_compile < 0.0)
        or np.any(deployable < 0.0)
        or np.any(all_in < deployable)
        or np.any(denominator <= 0.0)
    ):
        raise ValueError("compiler authorization timing rows are invalid")
    compile_reduction = 1.0 - float(optimized_compile.sum()) / float(
        baseline_compile.sum()
    )
    pooled_ratio = float(deployable.sum()) / float(denominator.sum())
    all_in_ratio = float(all_in.sum()) / float(denominator.sum())
    rng = np.random.default_rng(int(seed))
    bootstrap = np.empty(int(resamples), dtype=np.float64)
    cursor = 0
    while cursor < int(resamples):
        count = min(500, int(resamples) - cursor)
        sampled = rng.integers(0, len(ordered), size=(count, len(ordered)))
        sampled_num = deployable[sampled].sum(axis=1)
        sampled_den = denominator[sampled].sum(axis=1)
        bootstrap[cursor : cursor + count] = sampled_num / sampled_den
        cursor += count
    upper95 = float(np.quantile(bootstrap, 1.0 - alpha, method="higher"))
    return {
        "cluster_unit": "development block",
        "block_count": len(ordered),
        "measurement_repetitions_aggregated_within_block": 3,
        "bootstrap_seed": int(seed),
        "bootstrap_resamples": int(resamples),
        "alpha": float(alpha),
        "compile_group_reduction": compile_reduction,
        "pooled_online_deployable_overhead_ratio": pooled_ratio,
        "paired_whole_block_one_sided_upper95": upper95,
        "conservative_all_in_ratio": all_in_ratio,
        "baseline_compile_group_seconds_total": float(baseline_compile.sum()),
        "optimized_compile_group_seconds_total": float(optimized_compile.sum()),
        "optimized_deployable_numerator_seconds_total": float(deployable.sum()),
        "optimized_all_in_numerator_seconds_total": float(all_in.sum()),
        "production_H_denominator_seconds_total": float(denominator.sum()),
    }


def evaluate_overhead_screen(
    block_rows: Sequence[Mapping[str, Any]],
    repeat_rows: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    resamples: int,
    alpha: float,
    expected_repetitions: Sequence[int] = (1, 2, 3),
) -> dict[str, Any]:
    """Aggregate a staged canonical-overhead screen with blocks as clusters.

    Each block row must already average the three fixed timing repetitions.
    Repetition-specific pooled ratios are recomputed from the unaggregated rows;
    timing repetitions never become bootstrap clusters.
    """

    if not block_rows:
        raise ValueError("overhead screen requires at least one block")
    ordered = sorted(block_rows, key=lambda row: int(row["block_seed"]))
    block_seeds = [int(row["block_seed"]) for row in ordered]
    if len(set(block_seeds)) != len(block_seeds):
        raise ValueError("overhead screen requires unique block aggregates")
    if resamples <= 0 or not 0.0 < alpha < 1.0:
        raise ValueError("invalid overhead screen bootstrap parameters")
    repetitions = tuple(int(value) for value in expected_repetitions)
    if not repetitions or len(set(repetitions)) != len(repetitions):
        raise ValueError("overhead screen repetitions must be unique and nonempty")
    expected_repeat_keys = {
        (block_seed, repetition)
        for block_seed in block_seeds
        for repetition in repetitions
    }
    observed_repeat_keys = {
        (int(row["block_seed"]), int(row["repetition"]))
        for row in repeat_rows
    }
    if observed_repeat_keys != expected_repeat_keys or len(repeat_rows) != len(
        expected_repeat_keys
    ):
        raise ValueError("overhead screen requires exactly three repeats per block")

    online = np.asarray(
        [float(row["online_numerator_seconds_mean"]) for row in ordered],
        dtype=np.float64,
    )
    all_in = np.asarray(
        [float(row["all_in_numerator_seconds_mean"]) for row in ordered],
        dtype=np.float64,
    )
    denominator = np.asarray(
        [float(row["production_H_denominator_seconds_mean"]) for row in ordered],
        dtype=np.float64,
    )
    if (
        np.any(online < 0.0)
        or np.any(all_in < online)
        or np.any(denominator <= 0.0)
    ):
        raise ValueError("overhead screen timing rows are invalid")

    repetition_pooled: dict[str, float] = {}
    for repetition in repetitions:
        selected = [
            row for row in repeat_rows if int(row["repetition"]) == repetition
        ]
        selected.sort(key=lambda row: int(row["block_seed"]))
        numerator_total = sum(
            float(row["online_numerator_seconds"]) for row in selected
        )
        denominator_total = sum(
            float(row["production_H_denominator_seconds"]) for row in selected
        )
        repetition_pooled[str(repetition)] = numerator_total / denominator_total

    rng = np.random.default_rng(int(seed))
    bootstrap = np.empty(int(resamples), dtype=np.float64)
    cursor = 0
    while cursor < int(resamples):
        count = min(500, int(resamples) - cursor)
        sampled = rng.integers(0, len(ordered), size=(count, len(ordered)))
        bootstrap[cursor : cursor + count] = (
            online[sampled].sum(axis=1) / denominator[sampled].sum(axis=1)
        )
        cursor += count

    return {
        "cluster_unit": "block",
        "block_count": len(ordered),
        "measurement_repetitions_aggregated_within_block": len(repetitions),
        "bootstrap_seed": int(seed),
        "bootstrap_resamples": int(resamples),
        "alpha": float(alpha),
        "online_numerator_seconds_total": float(online.sum()),
        "all_in_numerator_seconds_total": float(all_in.sum()),
        "production_H_denominator_seconds_total": float(denominator.sum()),
        "pooled_online_overhead_ratio": float(online.sum())
        / float(denominator.sum()),
        "repetition_pooled_online_overhead_ratios": repetition_pooled,
        "paired_whole_block_one_sided_upper95": float(
            np.quantile(bootstrap, 1.0 - alpha, method="higher")
        ),
        "conservative_all_in_ratio": float(all_in.sum())
        / float(denominator.sum()),
    }


def formal_verdict(
    *,
    action: Mapping[str, Any],
    recognizability: Mapping[str, Any],
    utility: Mapping[str, Any],
    attack_bound_design: float,
    overhead_ratio: float,
    violations: int,
    t4_extra_violations: int = 0,
) -> tuple[str, list[str]]:
    reasons = []
    if float(action["H_false_settlement_upper95"]) > 0.10:
        reasons.append("honest_false_settlement_above_0.10")
    if any(
        float(action["detection_simultaneous_lower95"][key]) < 0.8
        for key in DEVIATIONS
    ):
        reasons.append("deviation_detection_below_0.80")
    if (
        float(recognizability["lambda_F_simultaneous_upper95"]) > 0.10
        or float(recognizability["grouped_auc_upper95"]) > 0.65
    ):
        reasons.append("recognizability_gate_failed")
    if float(utility["honest_participation_margin_lower95"]) < 0:
        reasons.append("honest_participation_negative")
    if any(
        float(utility["utility_margin_simultaneous_lower95"][key]) <= 0
        for key in DEVIATIONS
    ):
        reasons.append("utility_margin_not_strictly_positive")
    if float(attack_bound_design) > 0.20:
        reasons.append("registered_attack_bound_above_0.20")
    if float(overhead_ratio) > 0.25:
        reasons.append("probe_overhead_above_0.25")
    if violations or t4_extra_violations:
        reasons.append("identity_protocol_coverage_or_checker_violation")
    return ("PASS", []) if not reasons else ("FAIL", reasons)


__all__ = [
    "ACTIONS",
    "DEVIATIONS",
    "attack_bound_rows",
    "benchmark_selector_family",
    "clopper_pearson_bound",
    "conditional_t3_screen_verdict",
    "evaluate_action_gate",
    "evaluate_fixed_selector_recognizability",
    "evaluate_frozen_selector",
    "evaluate_recognizability",
    "evaluate_probe_compiler_authorization",
    "evaluate_probe_overhead",
    "evaluate_utility_gate",
    "fit_selector_family",
    "formal_verdict",
    "one_sided_cluster_bound",
    "selector_features_from_envelope",
    "simultaneous_lower_bounds",
    "t2_verdict",
]
