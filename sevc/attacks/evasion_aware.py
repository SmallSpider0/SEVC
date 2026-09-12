"""Frozen COMP-E2 evasion-aware poisoning and model-replacement attack."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import math
from typing import Any, Callable, Iterable, Mapping, Sequence

import torch
import torch.nn as nn
import torch.optim as optim

_NORMALIZATION = {
    "mnist": ((0.1307,), (0.3081,)),
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    "cifar100": ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
}


def apply_checkerboard_trigger(
    data: torch.Tensor, *, dataset_key: str
) -> torch.Tensor:
    """Apply the frozen bottom-right 4x4 trigger before normalization."""

    key = str(dataset_key).lower()
    if key not in _NORMALIZATION:
        raise ValueError(f"unsupported checkerboard dataset: {dataset_key}")
    mean_values, std_values = _NORMALIZATION[key]
    if data.ndim != 4 or int(data.shape[1]) != len(mean_values):
        raise ValueError(f"checkerboard trigger shape mismatch for {key}")
    height, width = int(data.shape[-2]), int(data.shape[-1])
    if height < 4 or width < 4:
        raise ValueError("checkerboard trigger requires at least 4x4 inputs")
    result = data.clone()
    mean = torch.tensor(
        mean_values, dtype=result.dtype, device=result.device
    ).view(1, len(mean_values), 1, 1)
    std = torch.tensor(
        std_values, dtype=result.dtype, device=result.device
    ).view(1, len(std_values), 1, 1)
    raw = result * std + mean
    for row in range(4):
        for column in range(4):
            raw[:, :, height - 4 + row, width - 4 + column] = float(
                (row + column) % 2
            )
    return (raw - mean) / std


def apply_cifar10_checkerboard_trigger(data: torch.Tensor) -> torch.Tensor:
    """Apply a 4x4 bottom-right checkerboard before normalization."""

    if data.ndim != 4 or data.shape[1:] != (3, 32, 32):
        raise ValueError("COMP-E2 trigger requires normalized CIFAR-10 NCHW batches")
    return apply_checkerboard_trigger(data, dataset_key="cifar10")


def poison_selection(
    change_id: str,
    formal_seed: int,
    trainer_id: int,
    indices: Sequence[int],
    labels: Sequence[int],
    *,
    target_class: int = 0,
    fraction: float = 0.5,
) -> frozenset[int]:
    if not 0 < fraction <= 1:
        raise ValueError("poison fraction must be in (0, 1]")
    eligible = [int(index) for index in indices if int(labels[int(index)]) != target_class]
    ordered = sorted(
        eligible,
        key=lambda sample_index: hashlib.sha256(
            f"{change_id}|{formal_seed}|{trainer_id}|{sample_index}|poison".encode("utf-8")
        ).digest(),
    )
    return frozenset(ordered[: math.floor(len(ordered) * fraction)])


def _cpu_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _momentum_state(model: nn.Module, optimizer: optim.Optimizer) -> dict[str, torch.Tensor]:
    rows: dict[str, torch.Tensor] = {}
    for name, parameter in model.named_parameters():
        value = optimizer.state.get(parameter, {}).get("momentum_buffer")
        if value is not None:
            rows[name] = value.detach().cpu().clone()
    return rows


def train_evasion_aware_update(
    anchor: nn.Module,
    model_factory: Callable[[], nn.Module],
    batches: Iterable[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    poisoned_indices: frozenset[int],
    *,
    device: str,
    learning_rate: float,
    momentum: float,
    local_epochs: int = 1,
    target_class: int = 0,
    proximal_penalty: float = 1e-4,
    dataset_key: str = "cifar10",
) -> Any:
    """Train the preregistered clean+backdoor+proximal local objective."""

    # Import lazily to avoid the attacks <-> training package initialization
    # cycle while reusing the single canonical singleton-BatchNorm forward.
    from sevc.training.engine import ReplayProof, WorkerUpdate, _training_logits

    if local_epochs <= 0 or learning_rate <= 0 or momentum < 0 or proximal_penalty < 0:
        raise ValueError("invalid evasion-aware training hyperparameters")
    model = model_factory().to(device)
    model.load_state_dict(anchor.state_dict())
    anchor_parameters = {
        name: value.detach().to(device).clone()
        for name, value in anchor.named_parameters()
    }
    materialized = tuple(batches)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    loss_tensors: list[torch.Tensor] = []
    samples_seen = 0
    proof_initial: dict[str, torch.Tensor] = {}
    proof_optimizer_initial: dict[str, torch.Tensor] = {}
    proof_batch: tuple[torch.Tensor, torch.Tensor] | None = None
    proof_checkpoint: dict[str, torch.Tensor] | None = None
    proof_optimizer_checkpoint: dict[str, torch.Tensor] | None = None
    total_steps = len(materialized) * local_epochs
    step = 0
    model.train()
    for _ in range(local_epochs):
        for clean, target, sample_indices in materialized:
            if step == total_steps - 1:
                proof_initial = _cpu_state(model)
                proof_optimizer_initial = _momentum_state(model, optimizer)
                proof_batch = (clean.detach().cpu().clone(), target.detach().cpu().clone())
            asynchronous = str(device).startswith("cuda") and clean.is_pinned()
            clean_device = clean.to(device, non_blocking=asynchronous)
            target_device = target.to(device, non_blocking=asynchronous)
            sample_values = [int(value) for value in sample_indices.tolist()]
            poison_positions = [
                position
                for position, sample_index in enumerate(sample_values)
                if sample_index in poisoned_indices
            ]
            optimizer.zero_grad()
            clean_loss = criterion(_training_logits(model, clean_device), target_device)
            backdoor_loss = torch.zeros((), device=device, dtype=clean_loss.dtype)
            if poison_positions:
                positions = torch.tensor(poison_positions, dtype=torch.long, device=device)
                triggered = apply_checkerboard_trigger(
                    clean_device.index_select(0, positions),
                    dataset_key=dataset_key,
                )
                targets = torch.full(
                    (len(poison_positions),), target_class, dtype=torch.long, device=device
                )
                backdoor_loss = criterion(
                    _training_logits(model, triggered), targets
                )
            proximal = torch.zeros((), device=device, dtype=clean_loss.dtype)
            for name, parameter in model.named_parameters():
                proximal = proximal + torch.sum(
                    (parameter - anchor_parameters[name].to(parameter.dtype)) ** 2
                )
            loss = clean_loss + backdoor_loss + proximal_penalty * proximal
            loss.backward()
            optimizer.step()
            loss_tensors.append(loss.detach())
            samples_seen += int(target.shape[0])
            if step == total_steps - 1:
                proof_checkpoint = _cpu_state(model)
                proof_optimizer_checkpoint = _momentum_state(model, optimizer)
            step += 1
    if proof_batch is None or proof_checkpoint is None or proof_optimizer_checkpoint is None:
        raise ValueError("evasion-aware trainer received no batches")
    proof = ReplayProof(
        initial_state=proof_initial,
        batches=(proof_batch,),
        checkpoints=(proof_checkpoint,),
        learning_rate=learning_rate,
        momentum=momentum,
        schema_version="sevc-state-complete-replay-proof-v2",
        optimizer_initial_state=proof_optimizer_initial,
        optimizer_checkpoints=(proof_optimizer_checkpoint,),
        criterion_key="clean-ce+backdoor-ce+1e-4-proximal",
    )
    from sevc.attacks import WorkerBehavior

    loss_values = (
        torch.stack(loss_tensors).detach().cpu().tolist() if loss_tensors else []
    )
    return WorkerUpdate(
        model=model,
        behavior=WorkerBehavior.ADVERSARIAL,
        samples_seen=samples_seen,
        mean_loss=sum(loss_values) / len(loss_values),
        replay_proof=proof,
    )


def _median_honest_delta(
    anchor: nn.Module, honest_models: Sequence[nn.Module], *, chunk_elements: int = 65536
) -> dict[str, torch.Tensor]:
    if not honest_models:
        raise ValueError("evasion center requires honest updates")
    anchor_state = anchor.state_dict()
    states = [model.state_dict() for model in honest_models]
    result: dict[str, torch.Tensor] = {}
    for key, anchor_value in anchor_state.items():
        if not anchor_value.is_floating_point():
            continue
        flat_anchor = anchor_value.detach().cpu().reshape(-1).to(torch.float32)
        output = torch.empty_like(flat_anchor)
        for start in range(0, flat_anchor.numel(), chunk_elements):
            stop = min(flat_anchor.numel(), start + chunk_elements)
            block = torch.stack(
                [
                    state[key].detach().cpu().reshape(-1)[start:stop].to(torch.float32)
                    - flat_anchor[start:stop]
                    for state in states
                ]
            )
            output[start:stop] = torch.median(block, dim=0).values
        result[key] = output.reshape(anchor_value.shape)
    return result


def _delta_distance(
    anchor: nn.Module, model: nn.Module, center: Mapping[str, torch.Tensor]
) -> float:
    squared = 0.0
    for key, anchor_value in anchor.state_dict().items():
        if not anchor_value.is_floating_point():
            continue
        delta = model.state_dict()[key].detach().cpu().to(torch.float32) - anchor_value.detach().cpu().to(torch.float32)
        squared += float(torch.sum((delta - center[key]) ** 2))
    return math.sqrt(squared)


@dataclass(frozen=True)
class EvasionProjectionContext:
    """Round-local honest reference shared by every malicious submission."""

    center: Mapping[str, torch.Tensor]
    median_distance: float
    center_norm_sq: float


def prepare_evasion_projection(
    anchor: nn.Module, honest_models: Sequence[nn.Module]
) -> EvasionProjectionContext:
    """Materialize the frozen honest center once for one submitted-update bank."""

    center = _median_honest_delta(anchor, honest_models)
    honest_distances = sorted(
        _delta_distance(anchor, model, center) for model in honest_models
    )
    median_distance = float(
        torch.tensor(honest_distances, dtype=torch.float64).median()
    )
    center_norm_sq = sum(
        float(torch.sum(value.to(torch.float64) ** 2)) for value in center.values()
    )
    return EvasionProjectionContext(
        center=center,
        median_distance=median_distance,
        center_norm_sq=center_norm_sq,
    )


def project_evasion_aware_update(
    anchor: nn.Module,
    raw_model: nn.Module,
    honest_models: Sequence[nn.Module] | None = None,
    *,
    replacement_scale: float = 2.5,
    projection_context: EvasionProjectionContext | None = None,
) -> tuple[nn.Module, float, float, float]:
    """Apply raw replacement and the algebraically equivalent 1/64 search.

    The old implementation deep-copied the complete model for every one of 65
    grid points.  Because every candidate is ``c + s(r-c)`` in update space,
    distance and cosine are determined by three round-local dot products.  We
    select the same discrete grid point and materialize only that model.
    """

    if replacement_scale <= 0:
        raise ValueError("replacement scale must be positive")
    if projection_context is None:
        if honest_models is None:
            raise ValueError("evasion projection requires honest models or a prepared context")
        projection_context = prepare_evasion_projection(anchor, honest_models)
    center = projection_context.center
    median_distance = projection_context.median_distance
    anchor_state = anchor.state_dict()
    raw_state = raw_model.state_dict()

    raw_direction_norm_sq = 0.0
    center_raw_direction_dot = 0.0
    for key, anchor_value in anchor_state.items():
        if not anchor_value.is_floating_point():
            continue
        raw_delta = (
            raw_state[key].detach().cpu().to(torch.float64)
            - anchor_value.detach().cpu().to(torch.float64)
        ) * replacement_scale
        center_value = center[key].to(torch.float64)
        direction = raw_delta - center_value
        raw_direction_norm_sq += float(torch.sum(direction * direction))
        center_raw_direction_dot += float(torch.sum(center_value * direction))

    def candidate_for(scale: float) -> nn.Module:
        candidate = copy.deepcopy(anchor)
        state = candidate.state_dict()
        with torch.no_grad():
            for key, value in state.items():
                base = anchor_state[key].detach().to(value.device)
                if value.is_floating_point():
                    raw_delta = (
                        raw_state[key].detach().to(value.device, value.dtype)
                        - base.to(value.dtype)
                    ) * replacement_scale
                    center_value = center[key].to(value.device, value.dtype)
                    value.copy_(base.to(value.dtype) + center_value + scale * (raw_delta - center_value))
                else:
                    value.copy_(base)
        candidate.load_state_dict(state)
        return candidate

    center_norm_sq = projection_context.center_norm_sq
    chosen_scale = 0.0
    chosen_cosine = 1.0 if center_norm_sq > 0 else 0.0
    chosen_distance = 0.0
    for ordinal in range(65):
        scale = ordinal / 64.0
        dot = center_norm_sq + scale * center_raw_direction_dot
        norm_sq = (
            center_norm_sq
            + 2.0 * scale * center_raw_direction_dot
            + scale * scale * raw_direction_norm_sq
        )
        cosine = 0.0 if norm_sq == 0 or center_norm_sq == 0 else dot / math.sqrt(norm_sq * center_norm_sq)
        distance = scale * math.sqrt(max(0.0, raw_direction_norm_sq))
        if distance <= median_distance and cosine >= 0.5:
            chosen_scale = scale
            chosen_cosine = cosine
            chosen_distance = distance
    chosen_model = candidate_for(chosen_scale)
    return chosen_model, chosen_scale, chosen_distance, chosen_cosine


def apply_strong_model_replacement(
    anchor: nn.Module,
    raw_model: nn.Module,
    *,
    replacement_scale: float = 2.5,
) -> nn.Module:
    """Apply the preregistered strong model-replacement delta without evasion clipping."""

    if replacement_scale <= 0:
        raise ValueError("replacement scale must be positive")
    candidate = copy.deepcopy(anchor)
    anchor_state = anchor.state_dict()
    raw_state = raw_model.state_dict()
    state = candidate.state_dict()
    with torch.no_grad():
        for key, value in state.items():
            base = anchor_state[key].detach().to(value.device)
            if value.is_floating_point():
                raw = raw_state[key].detach().to(value.device, value.dtype)
                replacement = (
                    base.to(value.dtype)
                    + replacement_scale * (raw - base.to(value.dtype))
                )
                if not torch.isfinite(replacement).all():
                    raise ValueError("strong model replacement produced non-finite state")
                value.copy_(replacement)
            else:
                value.copy_(base)
    candidate.load_state_dict(state)
    return candidate


__all__ = [
    "EvasionProjectionContext",
    "apply_checkerboard_trigger",
    "apply_cifar10_checkerboard_trigger",
    "apply_strong_model_replacement",
    "poison_selection",
    "prepare_evasion_projection",
    "project_evasion_aware_update",
    "train_evasion_aware_update",
]
