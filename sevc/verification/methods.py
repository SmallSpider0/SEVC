"""Historical baselines and C-TIV variants behind one verification interface."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import math
import random
import time
from typing import Callable, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from sevc.core.registry import Registry
from sevc.training import ReplayProof, verify_replay_proof


Evaluator = Callable[[nn.Module], Tuple[float, float]]


@dataclass(frozen=True)
class VerificationContext:
    global_model: nn.Module
    local_models: tuple[nn.Module, ...]
    evaluator: Evaluator
    ground_truth_malicious: tuple[bool, ...]
    seed: int
    shapley_samples: int = 3
    replay_proofs: tuple[ReplayProof | None, ...] = ()
    replay_model_factory: Callable[[], nn.Module] | None = None
    replay_device: str = "cpu"
    replay_tolerance: float = 1e-6
    distance_chunk_elements: int = 65536
    compute_device: str | None = None


@dataclass(frozen=True)
class StrategyOutput:
    scores: tuple[float, ...] | None = None
    predictions: tuple[bool, ...] | None = None
    details: dict[str, object] | None = None


@dataclass(frozen=True)
class VerificationResult:
    method_key: str
    elapsed_seconds: float
    scores: tuple[float, ...] | None
    predicted_malicious: tuple[bool, ...]
    accuracy: float
    f1: float
    details: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


METHODS: Registry[Callable[[VerificationContext], StrategyOutput]] = Registry(
    "verification method"
)


def _load_state_average(
    target: nn.Module,
    states: Sequence[dict[str, torch.Tensor]],
    indices: Sequence[int],
    *,
    floating_sums: dict[str, torch.Tensor] | None = None,
) -> nn.Module:
    if not indices:
        raise ValueError("state average requires at least one model")
    target_state = target.state_dict()
    first = int(indices[0])
    for key, value in target_state.items():
        if value.is_floating_point() or value.is_complex():
            if floating_sums is None:
                total = torch.zeros_like(value)
                for index in indices:
                    total.add_(states[int(index)][key].detach().to(value.device))
            else:
                total = floating_sums[key]
            target_state[key] = total / len(indices)
        else:
            target_state[key] = states[first][key].detach().to(value.device)
    target.load_state_dict(target_state)
    return target


def _parameter_distance_matrix(
    models: Sequence[nn.Module],
    *,
    chunk_elements: int,
    compute_device: str | None = None,
) -> torch.Tensor:
    if chunk_elements <= 0:
        raise ValueError("distance_chunk_elements must be positive")
    count = len(models)
    if count == 0:
        raise ValueError("distance matrix requires models")
    parameters = [tuple(model.parameters()) for model in models]
    if any(len(items) != len(parameters[0]) for items in parameters):
        raise ValueError("all models must have the same parameter structure")
    device = torch.device(compute_device) if compute_device else parameters[0][0].device
    squared = torch.zeros((count, count), dtype=torch.float32, device=device)
    with torch.no_grad():
        for position in range(len(parameters[0])):
            flattened = [items[position].detach().reshape(-1) for items in parameters]
            if any(value.numel() != flattened[0].numel() for value in flattened):
                raise ValueError("model parameter shapes do not match")
            for start in range(0, flattened[0].numel(), chunk_elements):
                stop = min(flattened[0].numel(), start + chunk_elements)
                block = torch.stack(
                    [
                        value[start:stop].to(device=device, dtype=torch.float32)
                        for value in flattened
                    ]
                )
                norms = (block * block).sum(dim=1)
                squared.add_(
                    torch.clamp(
                        norms[:, None] + norms[None, :] - 2.0 * (block @ block.T),
                        min=0.0,
                    )
                )
    squared.fill_diagonal_(0.0)
    return torch.sqrt(torch.clamp(squared, min=0.0))


@METHODS.register("none")
def _none(context: VerificationContext) -> StrategyOutput:
    return StrategyOutput(predictions=tuple(False for _ in context.local_models))


@METHODS.register("test-acc")
def _test_accuracy(context: VerificationContext) -> StrategyOutput:
    global_accuracy, _ = context.evaluator(context.global_model)
    return StrategyOutput(
        scores=tuple(
            context.evaluator(model)[0] - global_accuracy for model in context.local_models
        )
    )


@METHODS.register("sampled-shapley")
def _sampled_shapley(context: VerificationContext) -> StrategyOutput:
    count = len(context.local_models)
    if context.shapley_samples <= 0:
        raise ValueError("shapley_samples must be positive")
    rng = random.Random(context.seed)
    contributions = np.zeros(count, dtype=float)
    states = [model.state_dict() for model in context.local_models]
    for _ in range(context.shapley_samples):
        permutation = rng.sample(range(count), count)
        previous = 0.0
        candidate = copy.deepcopy(context.local_models[permutation[0]])
        if context.compute_device is not None:
            candidate = candidate.to(context.compute_device)
        running_sums = {
            key: torch.zeros_like(value)
            for key, value in candidate.state_dict().items()
            if value.is_floating_point() or value.is_complex()
        }
        prefix: list[int] = []
        for position, worker_index in enumerate(permutation):
            prefix.append(worker_index)
            for key, total in running_sums.items():
                total.add_(states[worker_index][key].detach().to(total.device))
            current = context.evaluator(
                _load_state_average(
                    candidate,
                    states,
                    prefix,
                    floating_sums=running_sums,
                )
            )[0]
            contributions[worker_index] += current - previous
            previous = current
    return StrategyOutput(scores=tuple(contributions / context.shapley_samples))


@METHODS.register("influence")
def _influence(context: VerificationContext) -> StrategyOutput:
    count = len(context.local_models)
    states = [model.state_dict() for model in context.local_models]
    candidate = copy.deepcopy(context.local_models[0])
    if context.compute_device is not None:
        candidate = candidate.to(context.compute_device)
    total_sums: dict[str, torch.Tensor] = {}
    for key, value in candidate.state_dict().items():
        if not (value.is_floating_point() or value.is_complex()):
            continue
        total = torch.zeros_like(value)
        for state in states:
            total.add_(state[key].detach().to(total.device))
        total_sums[key] = total
    full_loss = context.evaluator(
        _load_state_average(
            candidate,
            states,
            tuple(range(count)),
            floating_sums=total_sums,
        )
    )[1]
    scores: list[float] = []
    for index in range(count):
        remaining = tuple(
            candidate_index
            for candidate_index in range(count)
            if candidate_index != index
        )
        if not remaining:
            scores.append(0.0)
        else:
            leave_one_out = {
                key: total - states[index][key].detach().to(total.device)
                for key, total in total_sums.items()
            }
            loss_without = context.evaluator(
                _load_state_average(
                    candidate,
                    states,
                    remaining,
                    floating_sums=leave_one_out,
                )
            )[1]
            scores.append(loss_without - full_loss)
    return StrategyOutput(scores=tuple(scores))


@METHODS.register("multi-krum")
def _multi_krum(context: VerificationContext) -> StrategyOutput:
    count = len(context.local_models)
    assumed_byzantine = math.ceil(count / 2) - 1
    closest_count = max(0, count - assumed_byzantine - 2)
    matrix = _parameter_distance_matrix(
        context.local_models,
        chunk_elements=context.distance_chunk_elements,
        compute_device=context.compute_device,
    ).detach().cpu()
    scores: list[float] = []
    for index in range(count):
        distances = sorted(
            float(matrix[index, other_index])
            for other_index in range(count)
            if other_index != index
        )
        scores.append(-sum(distances[:closest_count]))
    return StrategyOutput(scores=tuple(scores))


@METHODS.register("update-significance")
def _update_significance(context: VerificationContext) -> StrategyOutput:
    states = [dict(model.named_parameters()) for model in context.local_models]
    scores: list[float] = []
    for state in states:
        squared = 0.0
        for name, parameter in state.items():
            average = torch.stack(
                [candidate[name].detach().to(parameter.device) for candidate in states]
            ).mean(dim=0)
            squared += float(torch.norm(parameter.detach() - average).cpu()) ** 2
        scores.append(math.sqrt(squared))
    return StrategyOutput(scores=tuple(scores))


@METHODS.register("reproduction-oracle")
def _reproduction_oracle(context: VerificationContext) -> StrategyOutput:
    return StrategyOutput(
        predictions=context.ground_truth_malicious,
        details={"historical_semantics": "oracle-returned-ground-truth"},
    )


@METHODS.register("trajectory-replay")
def _trajectory_replay(context: VerificationContext) -> StrategyOutput:
    if context.replay_model_factory is None:
        raise ValueError("trajectory replay requires replay_model_factory")
    if len(context.replay_proofs) != len(context.local_models):
        raise ValueError("trajectory replay proof count must match local models")
    predictions: list[bool] = []
    proof_results: list[dict[str, object] | None] = []
    for proof in context.replay_proofs:
        if proof is None:
            predictions.append(True)
            proof_results.append(None)
            continue
        result = verify_replay_proof(
            proof,
            context.replay_model_factory,
            device=context.replay_device,
            tolerance=context.replay_tolerance,
        )
        predictions.append(not bool(result["passed"]))
        proof_results.append(result)
    return StrategyOutput(
        predictions=tuple(predictions), details={"proof_results": proof_results}
    )


_ALIASES = {
    "test_acc": "test-acc",
    "shapley": "sampled-shapley",
    "multi_krum": "multi-krum",
    "update_significance": "update-significance",
    "reproduction": "reproduction-oracle",
}


def _binary_metrics(
    truth: Sequence[bool], prediction: Sequence[bool]
) -> tuple[float, float]:
    if len(truth) != len(prediction) or not truth:
        raise ValueError("truth and prediction must have equal non-zero length")
    correct = sum(bool(left) == bool(right) for left, right in zip(truth, prediction))
    true_positive = sum(bool(left) and bool(right) for left, right in zip(truth, prediction))
    false_positive = sum(not bool(left) and bool(right) for left, right in zip(truth, prediction))
    false_negative = sum(bool(left) and not bool(right) for left, right in zip(truth, prediction))
    denominator = 2 * true_positive + false_positive + false_negative
    f1 = 0.0 if denominator == 0 else 2 * true_positive / denominator
    return correct / len(truth), f1


def run_verification(method_key: str, context: VerificationContext) -> VerificationResult:
    canonical_key = _ALIASES.get(method_key, method_key)
    if context.replay_device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif context.replay_device == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()
    started = time.perf_counter()
    output = METHODS.get(canonical_key)(context)
    if context.replay_device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif context.replay_device == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()
    if output.predictions is not None:
        predictions = output.predictions
    elif output.scores is not None:
        scores_array = np.asarray(output.scores, dtype=float)
        threshold = float(np.mean(scores_array) - np.std(scores_array))
        predictions = tuple(bool(value <= threshold) for value in scores_array)
    else:
        raise ValueError(f"verification method {canonical_key} returned no decision")
    accuracy, f1 = _binary_metrics(context.ground_truth_malicious, predictions)
    return VerificationResult(
        method_key=canonical_key,
        elapsed_seconds=time.perf_counter() - started,
        scores=output.scores,
        predicted_malicious=predictions,
        accuracy=accuracy,
        f1=f1,
        details={
            **(output.details or {}),
            "clock": "time.perf_counter",
            "device_synchronized": context.replay_device in {"cuda", "mps"},
        },
    )
