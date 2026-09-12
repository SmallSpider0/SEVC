"""Shared aggregation/decision interface for COMP-E2 robustness methods."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import math
import time
from typing import Callable, Mapping, Sequence

import torch
import torch.nn as nn

from sevc.training import ReplayProof, verify_replay_proof
from sevc.verification.methods import _parameter_distance_matrix


@dataclass(frozen=True)
class SubmittedUpdate:
    trainer_id: int
    model: nn.Module
    samples_seen: int
    replay_proof: ReplayProof | None


@dataclass(frozen=True)
class RobustAggregationResult:
    method_key: str
    aggregate: nn.Module
    predicted_malicious: tuple[bool, ...]
    scores: tuple[float, ...]
    selected_trainers: tuple[int, ...]
    verification_wall_seconds: float
    verification_bytes: int
    details: Mapping[str, object]


def _floating_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach()
        for key, value in model.state_dict().items()
        if value.is_floating_point() or value.is_complex()
    }


def _weighted_delta_aggregate(
    anchor: nn.Module,
    submissions: Sequence[SubmittedUpdate],
    weights: Sequence[float],
) -> nn.Module:
    if not submissions or len(submissions) != len(weights):
        raise ValueError("submissions and weights must have equal non-zero length")
    normalized = tuple(float(value) for value in weights)
    total_weight = sum(normalized)
    if not math.isfinite(total_weight) or total_weight <= 0:
        raise ValueError("aggregation weights must have positive finite sum")
    result = copy.deepcopy(anchor)
    anchor_state = anchor.state_dict()
    states = [submission.model.state_dict() for submission in submissions]
    result_state = result.state_dict()
    with torch.no_grad():
        for key, value in result_state.items():
            source = anchor_state[key].detach().to(value.device)
            if value.is_floating_point() or value.is_complex():
                value.copy_(source)
                for weight, state in zip(normalized, states):
                    delta = state[key].detach().to(
                        device=value.device, dtype=value.dtype
                    ) - source.to(dtype=value.dtype)
                    value.add_(delta, alpha=weight / total_weight)
            else:
                value.copy_(source)
    result.load_state_dict(result_state)
    return result


def _proof_nbytes(proof: ReplayProof | None) -> int:
    if proof is None:
        return 0
    total = 0
    mappings = [proof.initial_state, *(proof.checkpoints or ())]
    if proof.optimizer_initial_state is not None:
        mappings.append(proof.optimizer_initial_state)
    mappings.extend(proof.optimizer_checkpoints)
    for mapping in mappings:
        total += sum(
            int(value.numel() * value.element_size()) for value in mapping.values()
        )
    for data, target in proof.batches:
        total += int(data.numel() * data.element_size())
        total += int(target.numel() * target.element_size())
    return total


def _delta_dot_norms(
    anchor: nn.Module, left: nn.Module, right: nn.Module
) -> tuple[float, float, float]:
    anchor_state = _floating_state(anchor)
    left_state = _floating_state(left)
    right_state = _floating_state(right)
    dot = left_norm = right_norm = 0.0
    for key, anchor_value in anchor_state.items():
        left_delta = (
            left_state[key].detach().to(dtype=torch.float64, device="cpu")
            - anchor_value.detach().to(dtype=torch.float64, device="cpu")
        )
        right_delta = (
            right_state[key].detach().to(dtype=torch.float64, device="cpu")
            - anchor_value.detach().to(dtype=torch.float64, device="cpu")
        )
        dot += float(torch.sum(left_delta * right_delta))
        left_norm += float(torch.sum(left_delta * left_delta))
        right_norm += float(torch.sum(right_delta * right_delta))
    return dot, math.sqrt(left_norm), math.sqrt(right_norm)


def _batched_root_geometry(
    anchor: nn.Module,
    submissions: Sequence[SubmittedUpdate],
    root_model: nn.Module,
    *,
    compute_device: str,
    chunk_elements: int,
) -> tuple[list[float], list[float], float]:
    """Compute all FLTrust dot products in bounded device batches."""

    if chunk_elements <= 0:
        raise ValueError("distance_chunk_elements must be positive")
    device = torch.device(compute_device)
    count = len(submissions)
    dots = torch.zeros(count, dtype=torch.float64, device=device)
    update_norm_sq = torch.zeros(count, dtype=torch.float64, device=device)
    root_norm_sq = torch.zeros((), dtype=torch.float64, device=device)
    anchor_state = anchor.state_dict()
    root_state = root_model.state_dict()
    submission_states = [row.model.state_dict() for row in submissions]
    with torch.no_grad():
        for key, anchor_value in anchor_state.items():
            if not anchor_value.is_floating_point():
                continue
            anchor_flat = anchor_value.detach().cpu().reshape(-1)
            root_flat = root_state[key].detach().cpu().reshape(-1)
            values = [state[key].detach().cpu().reshape(-1) for state in submission_states]
            for start in range(0, anchor_flat.numel(), chunk_elements):
                stop = min(anchor_flat.numel(), start + chunk_elements)
                base = anchor_flat[start:stop].to(device=device, dtype=torch.float64)
                root_delta = root_flat[start:stop].to(
                    device=device, dtype=torch.float64
                ) - base
                block = torch.stack(
                    [
                        value[start:stop].to(device=device, dtype=torch.float64)
                        - base
                        for value in values
                    ]
                )
                dots.add_(block @ root_delta)
                update_norm_sq.add_((block * block).sum(dim=1))
                root_norm_sq.add_(torch.sum(root_delta * root_delta))
    return (
        dots.detach().cpu().tolist(),
        torch.sqrt(torch.clamp(update_norm_sq, min=0.0)).detach().cpu().tolist(),
        math.sqrt(max(0.0, float(root_norm_sq.detach().cpu()))),
    )


def aggregate_submissions(
    method_key: str,
    *,
    anchor: nn.Module,
    submissions: Sequence[SubmittedUpdate],
    contribution_weights: Sequence[float],
    root_update: SubmittedUpdate | None = None,
    replay_model_factory: Callable[[], nn.Module] | None = None,
    replay_device: str = "cpu",
    replay_tolerance: float = 1e-5,
    distance_chunk_elements: int = 65536,
    vector_compute_device: str | None = None,
    trusted_validation_evaluator: Callable[[nn.Module], tuple[float, float]] | None = None,
) -> RobustAggregationResult:
    """Apply one frozen defense without mutating or regenerating submissions."""

    if method_key == "reproduction-oracle":
        raise ValueError("reproduction-oracle is forbidden for COMP-E2")
    if len(submissions) != len(contribution_weights) or not submissions:
        raise ValueError("submission/weight cardinality mismatch")
    started = time.perf_counter()
    count = len(submissions)

    if method_key == "none":
        selected = tuple(range(count))
        predicted = tuple(False for _ in submissions)
        scores = tuple(1.0 for _ in submissions)
        aggregate = _weighted_delta_aggregate(
            anchor, submissions, contribution_weights
        )
        details: dict[str, object] = {"aggregation": "contribution-weighted"}
        verification_bytes = 0
    elif method_key == "multi-krum":
        f_value = math.floor(0.4 * count)
        closest_count = count - f_value - 2
        select_count = closest_count
        if closest_count <= 0:
            raise ValueError("Multi-Krum dimensions are invalid")
        matrix = _parameter_distance_matrix(
            [item.model for item in submissions],
            chunk_elements=distance_chunk_elements,
            compute_device=vector_compute_device or "cpu",
        )
        krum_scores = []
        for index in range(count):
            distances = sorted(
                float(matrix[index, other])
                for other in range(count)
                if other != index
            )
            krum_scores.append(sum(distances[:closest_count]))
        selected = tuple(sorted(range(count), key=lambda idx: (krum_scores[idx], idx))[:select_count])
        selected_set = set(selected)
        predicted = tuple(index not in selected_set for index in range(count))
        scores = tuple(-value for value in krum_scores)
        aggregate = _weighted_delta_aggregate(
            anchor,
            [submissions[index] for index in selected],
            [1.0 for _ in selected],
        )
        details = {
            "f": f_value,
            "closest_count": closest_count,
            "select_count": select_count,
        }
        verification_bytes = sum(
            sum(
                int(value.numel() * value.element_size())
                for value in submission.model.state_dict().values()
            )
            for submission in submissions
        )
    elif method_key == "fltrust":
        if root_update is None:
            raise ValueError("FLTrust requires one trusted-root update")
        trusts: list[float] = []
        rescale_weights: list[float] = []
        if vector_compute_device:
            dot_values, update_norms, root_norm = _batched_root_geometry(
                anchor,
                submissions,
                root_update.model,
                compute_device=vector_compute_device,
                chunk_elements=distance_chunk_elements,
            )
        else:
            root_dot, root_norm, _ = _delta_dot_norms(
                anchor, root_update.model, root_update.model
            )
            del root_dot
            dot_values = []
            update_norms = []
            for submission in submissions:
                dot, update_norm, observed_root_norm = _delta_dot_norms(
                    anchor, submission.model, root_update.model
                )
                if not math.isclose(
                    root_norm, observed_root_norm, rel_tol=1e-12, abs_tol=1e-12
                ):
                    raise AssertionError("FLTrust root norm drift")
                dot_values.append(dot)
                update_norms.append(update_norm)
        for dot, update_norm in zip(dot_values, update_norms):
            cosine = 0.0 if update_norm == 0 or root_norm == 0 else dot / (update_norm * root_norm)
            trust = max(0.0, cosine)
            trusts.append(trust)
            rescale_weights.append(0.0 if update_norm == 0 else trust * root_norm / update_norm)
        if sum(rescale_weights) <= 0:
            aggregate = copy.deepcopy(anchor)
            selected = ()
        else:
            aggregate = _weighted_delta_aggregate(anchor, submissions, rescale_weights)
            selected = tuple(index for index, trust in enumerate(trusts) if trust > 0)
        predicted = tuple(trust <= 0 for trust in trusts)
        scores = tuple(trusts)
        details = {
            "trusted_root_assumption": True,
            "root_sample_count": root_update.samples_seen,
            "root_update_norm": root_norm,
            "vector_compute_device": vector_compute_device or "cpu",
        }
        verification_bytes = sum(
            sum(
                int(value.numel() * value.element_size())
                for value in submission.model.state_dict().values()
            )
            for submission in submissions
        )
    elif method_key == "sevc-ctiv-registered-v1":
        if replay_model_factory is None:
            raise ValueError("registered SEVC requires replay_model_factory")
        passed: list[bool] = []
        replay_rows: list[Mapping[str, object] | None] = []
        for submission in submissions:
            if submission.replay_proof is None:
                passed.append(False)
                replay_rows.append(None)
                continue
            result = verify_replay_proof(
                submission.replay_proof,
                replay_model_factory,
                device=replay_device,
                tolerance=replay_tolerance,
            )
            passed.append(bool(result["passed"]))
            replay_rows.append(result)
        predicted = tuple(not value for value in passed)
        selected = tuple(index for index, value in enumerate(passed) if value)
        if selected:
            aggregate = _weighted_delta_aggregate(
                anchor,
                [submissions[index] for index in selected],
                [contribution_weights[index] for index in selected],
            )
        else:
            aggregate = copy.deepcopy(anchor)
        scores = tuple(1.0 if value else 0.0 for value in passed)
        details = {
            "registered_implementation": "sevc-ctiv-registered-v1",
            "ground_truth_behavior_access": False,
            "replay_results": replay_rows,
            "safe_anchor_fallback": not bool(selected),
        }
        verification_bytes = sum(_proof_nbytes(item.replay_proof) for item in submissions)
    elif method_key == "refiner-update-audit-style":
        if trusted_validation_evaluator is None:
            raise ValueError(
                "Refiner-style update audit requires trusted_validation_evaluator"
            )
        validation_losses: list[float | None] = []
        validation_accuracies: list[float | None] = []
        validation_metric_valid: list[bool] = []
        validation_invalid_reasons: list[list[str]] = []
        for submission in submissions:
            accuracy, loss = trusted_validation_evaluator(submission.model)
            accuracy_value = float(accuracy)
            loss_value = float(loss)
            reasons = []
            if not math.isfinite(accuracy_value):
                reasons.append("nonfinite_accuracy")
            if not math.isfinite(loss_value):
                reasons.append("nonfinite_loss")
            valid = not reasons
            validation_metric_valid.append(valid)
            validation_invalid_reasons.append(reasons)
            validation_accuracies.append(accuracy_value if valid else None)
            validation_losses.append(loss_value if valid else None)
        ordered_losses = sorted(
            loss
            for loss, valid in zip(validation_losses, validation_metric_valid)
            if valid and loss is not None
        )
        if ordered_losses:
            middle = len(ordered_losses) // 2
            median_loss = (
                ordered_losses[middle]
                if len(ordered_losses) % 2
                else 0.5 * (ordered_losses[middle - 1] + ordered_losses[middle])
            )
            threshold: float | None = 2.0 * median_loss - min(ordered_losses)
        else:
            threshold = None
        predicted = tuple(
            not valid or threshold is None or loss is None or loss > threshold
            for loss, valid in zip(validation_losses, validation_metric_valid)
        )
        selected = tuple(index for index, value in enumerate(predicted) if not value)
        if selected:
            aggregate = _weighted_delta_aggregate(
                anchor,
                [submissions[index] for index in selected],
                [contribution_weights[index] for index in selected],
            )
        else:
            aggregate = copy.deepcopy(anchor)
        invalid_score = -1.0e300
        scores = tuple(
            -loss if valid and loss is not None else invalid_score
            for loss, valid in zip(validation_losses, validation_metric_valid)
        )
        details = {
            "registered_implementation": "refiner-update-audit-style",
            "trusted_validation_data": True,
            "loss_threshold_rule": (
                "nonfinite->reject; "
                "2*median(finite-current-round-loss)-minimum(finite-current-round-loss)"
            ),
            "history_rounds": 1,
            "cross_round_copy_history": False,
            "ground_truth_behavior_access": False,
            "validation_losses": validation_losses,
            "validation_accuracies": validation_accuracies,
            "validation_metric_valid": validation_metric_valid,
            "validation_invalid_reasons": validation_invalid_reasons,
            "validation_nonfinite_count": sum(not value for value in validation_metric_valid),
            "invalid_score_sentinel": invalid_score,
            "loss_threshold": threshold,
            "safe_anchor_fallback": not bool(selected),
        }
        verification_bytes = sum(
            sum(
                int(value.numel() * value.element_size())
                for value in submission.model.state_dict().values()
            )
            for submission in submissions
        )
    else:
        raise KeyError(f"unsupported COMP-E2 method: {method_key}")

    return RobustAggregationResult(
        method_key=method_key,
        aggregate=aggregate,
        predicted_malicious=predicted,
        scores=scores,
        selected_trainers=selected,
        verification_wall_seconds=time.perf_counter() - started,
        verification_bytes=verification_bytes,
        details=details,
    )


__all__ = [
    "RobustAggregationResult",
    "SubmittedUpdate",
    "aggregate_submissions",
]
