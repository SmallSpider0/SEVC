"""Single collaborative-training implementation used by all experiment variants."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import math
from typing import Any, Callable, Iterable, Sequence

import torch
import torch.nn as nn
import torch.optim as optim

from sevc.attacks import WorkerBehavior, prepare_worker_model


ModelFactory = Callable[[], nn.Module]


@dataclass(frozen=True)
class ReplayProof:
    initial_state: dict[str, torch.Tensor]
    batches: tuple[tuple[torch.Tensor, torch.Tensor], ...]
    checkpoints: tuple[dict[str, torch.Tensor], ...]
    learning_rate: float
    momentum: float
    schema_version: str = "sevc-replay-proof-v1"
    optimizer_initial_state: dict[str, torch.Tensor] | None = None
    optimizer_checkpoints: tuple[dict[str, torch.Tensor], ...] = ()
    rng_state_sha256: str | None = None
    data_order_sha256: str | None = None
    criterion_key: str | None = None


@dataclass(frozen=True)
class WorkerUpdate:
    model: nn.Module
    behavior: WorkerBehavior
    samples_seen: int
    mean_loss: float | None
    replay_proof: ReplayProof | None


def _cpu_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def _optimizer_momentum_state(
    model: nn.Module, optimizer: optim.Optimizer
) -> dict[str, torch.Tensor]:
    """Return name-keyed momentum without exposing unstable optimizer IDs."""

    result: dict[str, torch.Tensor] = {}
    for name, parameter in model.named_parameters():
        buffer = optimizer.state.get(parameter, {}).get("momentum_buffer")
        if buffer is not None:
            result[name] = buffer.detach().cpu().clone()
    return result


def _load_optimizer_momentum_state(
    model: nn.Module,
    optimizer: optim.Optimizer,
    state: dict[str, torch.Tensor],
) -> None:
    parameters = dict(model.named_parameters())
    unknown = sorted(set(state) - set(parameters))
    if unknown:
        raise ValueError(f"optimizer state references unknown parameters: {unknown}")
    for name, value in state.items():
        parameter = parameters[name]
        if value.shape != parameter.shape:
            raise ValueError(f"optimizer momentum shape mismatch for {name}")
        optimizer.state[parameter]["momentum_buffer"] = value.detach().to(
            device=parameter.device, dtype=parameter.dtype
        ).clone()


def _tensor_sequence_sha256(
    batches: Sequence[tuple[torch.Tensor, torch.Tensor]],
) -> str:
    digest = hashlib.sha256(b"sevc-replay-data-order-v1")
    for index, (data, target) in enumerate(batches):
        digest.update(str(index).encode("utf-8"))
        for value in (data, target):
            tensor = value.detach().cpu().contiguous()
            digest.update(str(tensor.dtype).encode("utf-8"))
            digest.update(str(tuple(tensor.shape)).encode("utf-8"))
            digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _logits(output: Any) -> torch.Tensor:
    logits = output.logits if hasattr(output, "logits") else output
    if not isinstance(logits, torch.Tensor):
        raise TypeError("model output must be a tensor or expose .logits")
    return logits


def _training_logits(
    model: nn.Module, data: torch.Tensor
) -> torch.Tensor:
    """Run a training forward while preserving singleton logical batches.

    PyTorch BatchNorm cannot compute training statistics when a layer receives
    exactly one value per channel (for example ``[1, C, 1, 1]`` at the end of
    ResNet).  ``drop_last=False`` is part of the frozen data contract, so the
    sample must neither be discarded nor duplicated.  For only the affected
    BatchNorm layer and only that forward, use its existing running statistics;
    affine parameters and the rest of the model remain in training mode.
    """

    if data.ndim == 0 or int(data.shape[0]) != 1:
        return _logits(model(data))

    toggled: set[nn.Module] = set()

    def before(module: nn.Module, inputs: tuple[Any, ...]) -> None:
        if not module.training or not inputs:
            return
        value = inputs[0]
        if not isinstance(value, torch.Tensor) or value.ndim < 2:
            return
        channel_count = int(value.shape[1])
        if channel_count <= 0:
            return
        values_per_channel = int(value.numel()) // channel_count
        if values_per_channel == 1:
            module.training = False
            toggled.add(module)

    def after(module: nn.Module, _inputs: tuple[Any, ...], _output: Any) -> None:
        if module in toggled:
            module.training = True
            toggled.remove(module)

    handles = []
    batch_norm_types = (
        nn.BatchNorm1d,
        nn.BatchNorm2d,
        nn.BatchNorm3d,
        nn.SyncBatchNorm,
    )
    for module in model.modules():
        if isinstance(module, batch_norm_types):
            handles.append(module.register_forward_pre_hook(before))
            handles.append(module.register_forward_hook(after))
    try:
        return _logits(model(data))
    finally:
        for module in toggled:
            module.training = True
        for handle in handles:
            handle.remove()


def produce_worker_update(
    global_model: nn.Module,
    model_factory: ModelFactory,
    batches: Iterable[tuple[torch.Tensor, torch.Tensor]],
    behavior: WorkerBehavior,
    *,
    device: str,
    learning_rate: float,
    momentum: float,
    local_epochs: int = 1,
    max_batches: int | None = None,
    capture_replay: bool = False,
    replay_tail_batches: int | None = None,
    logical_microbatch_size: int | None = None,
    initial_momentum_state: dict[str, torch.Tensor] | None = None,
    milestone_callback=None,
    milestone_steps: tuple[int, ...] | None = None,
    initial_rng_state: dict | None = None,
) -> WorkerUpdate:
    if local_epochs <= 0 or learning_rate <= 0 or momentum < 0:
        raise ValueError("invalid local-training hyperparameters")
    prepared = prepare_worker_model(
        behavior,
        global_model,
        model_factory,
        device,
    )
    if not prepared.should_train:
        return WorkerUpdate(
            model=prepared.model,
            behavior=behavior,
            samples_seen=0,
            mean_loss=None,
            replay_proof=None,
        )
    model = prepared.model

    materialized_batches = list(batches)
    if max_batches is not None:
        materialized_batches = materialized_batches[:max_batches]
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    if initial_momentum_state is not None:
        _load_optimizer_momentum_state(model, optimizer, initial_momentum_state)
    if initial_rng_state is not None:
        torch.set_rng_state(initial_rng_state['cpu_rng'])
        if str(device).startswith('cuda'):
            torch.cuda.set_rng_state_all(initial_rng_state['cuda_rng'])
    if replay_tail_batches is not None and (
        not capture_replay or replay_tail_batches <= 0
    ):
        raise ValueError("replay_tail_batches requires positive capture_replay")
    if logical_microbatch_size is not None and logical_microbatch_size <= 0:
        raise ValueError("logical_microbatch_size must be positive when provided")
    total_steps = len(materialized_batches) * local_epochs
    capture_start = (
        0
        if replay_tail_batches is None
        else max(0, total_steps - replay_tail_batches)
    )
    initial_state = _cpu_state(model) if capture_replay and capture_start == 0 else {}
    optimizer_initial_state = (
        _optimizer_momentum_state(model, optimizer)
        if capture_replay and capture_start == 0
        else None
    )
    rng_state_sha256 = (
        hashlib.sha256(torch.get_rng_state().cpu().numpy().tobytes()).hexdigest()
        if capture_replay
        else None
    )
    replay_batches: list[tuple[torch.Tensor, torch.Tensor]] = []
    checkpoints: list[dict[str, torch.Tensor]] = []
    optimizer_checkpoints: list[dict[str, torch.Tensor]] = []
    loss_tensors: list[torch.Tensor] = []
    samples_seen = 0
    model.train()
    step = 0
    for _ in range(local_epochs):
        for data, target in materialized_batches:
            if capture_replay and step == capture_start and not initial_state:
                initial_state = _cpu_state(model)
                optimizer_initial_state = _optimizer_momentum_state(model, optimizer)
            asynchronous = str(device).startswith("cuda") and data.is_pinned()
            data_on_device = data.to(device, non_blocking=asynchronous)
            target_on_device = target.to(device, non_blocking=asynchronous)
            optimizer.zero_grad()
            if (
                logical_microbatch_size is None
                or int(data_on_device.shape[0]) <= logical_microbatch_size
            ):
                loss = criterion(
                    _training_logits(model, data_on_device), target_on_device
                )
                loss.backward()
            else:
                # Preserve one optimizer step and the mean-loss semantics of the
                # original logical batch while bounding activation residency.
                # This path is enabled only for the FP32 ViT runtime repair;
                # the logical batch, sample order, LR and momentum are unchanged.
                logical_count = int(target_on_device.shape[0])
                chunk_loss_sums: list[torch.Tensor] = []
                for start in range(0, logical_count, logical_microbatch_size):
                    stop = min(start + logical_microbatch_size, logical_count)
                    chunk_loss_sum = nn.functional.cross_entropy(
                        _training_logits(model, data_on_device[start:stop]),
                        target_on_device[start:stop],
                        reduction="sum",
                    )
                    (chunk_loss_sum / logical_count).backward()
                    chunk_loss_sums.append(chunk_loss_sum.detach())
                loss = torch.stack(chunk_loss_sums).sum() / logical_count
            optimizer.step()
            # Keep the scalar on the execution device until the local-training
            # loop is complete.  Copying every scalar to the host here forces
            # one CUDA/MPS synchronization per batch and serializes otherwise
            # independent input preparation and device work.  The deferred
            # conversion below preserves the original Python-float summation
            # order and therefore the stored mean-loss semantics.
            loss_tensors.append(loss.detach())
            samples_seen += int(target.shape[0])
            if capture_replay and step >= capture_start:
                replay_batches.append((data.detach().cpu().clone(), target.detach().cpu().clone()))
                checkpoints.append(_cpu_state(model))
                optimizer_checkpoints.append(
                    _optimizer_momentum_state(model, optimizer)
                )
            step += 1
            if milestone_callback is not None and (milestone_steps is None or step in milestone_steps):
                milestone_callback(step, model, _optimizer_momentum_state(model, optimizer))
    proof = (
        ReplayProof(
            initial_state=initial_state,
            batches=tuple(replay_batches),
            checkpoints=tuple(checkpoints),
            learning_rate=learning_rate,
            momentum=momentum,
            schema_version="sevc-state-complete-replay-proof-v2",
            optimizer_initial_state=optimizer_initial_state,
            optimizer_checkpoints=tuple(optimizer_checkpoints),
            rng_state_sha256=rng_state_sha256,
            data_order_sha256=_tensor_sequence_sha256(tuple(replay_batches)),
            criterion_key="torch.nn.CrossEntropyLoss",
        )
        if capture_replay
        else None
    )
    losses = (
        torch.stack(loss_tensors).detach().cpu().tolist()
        if loss_tensors
        else []
    )
    return WorkerUpdate(
        model=model,
        behavior=behavior,
        samples_seen=samples_seen,
        mean_loss=float(sum(losses) / len(losses)) if losses else None,
        replay_proof=proof,
    )


def _state_max_abs_difference(
    left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]
) -> float:
    if left.keys() != right.keys():
        return math.inf
    maximum = 0.0
    for key in left:
        left_value = left[key]
        right_value = right[key]
        if left_value.shape != right_value.shape:
            return math.inf
        if left_value.is_floating_point() or left_value.is_complex():
            difference = float(
                torch.max(torch.abs(left_value.to(torch.float64) - right_value.to(torch.float64)))
            )
        else:
            difference = 0.0 if torch.equal(left_value, right_value) else math.inf
        maximum = max(maximum, difference)
    return maximum


def _resident_state_max_abs_difference(left, right) -> float:
    """Same float64 maximum residual, with one scalar synchronization per state."""
    if left.keys() != right.keys():
        return math.inf
    maxima = []
    for key, value in left.items():
        expected = right[key]
        if value.shape != expected.shape:
            return math.inf
        if value.is_floating_point() or value.is_complex():
            maxima.append(torch.amax(torch.abs(value.detach().to(torch.float64) -
                                               expected.to(device=value.device, dtype=torch.float64, non_blocking=True))))
        else:
            matches = torch.all(value.detach() == expected.to(value.device, non_blocking=True))
            maxima.append(torch.where(matches, torch.tensor(0., device=value.device, dtype=torch.float64),
                                       torch.tensor(math.inf, device=value.device, dtype=torch.float64)))
    return float(torch.stack(maxima).amax().item()) if maxima else 0.


def verify_replay_proof(
    proof: ReplayProof,
    model_factory: ModelFactory,
    *,
    device: str,
    tolerance: float,
    logical_microbatch_size: int | None = None,
    comparison_device: str = "cpu",
    state_transform: Callable | None = None,
) -> dict[str, object]:
    if comparison_device not in {"cpu", "replay"}:
        raise ValueError("unknown replay residual comparison device")
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    if logical_microbatch_size is not None and logical_microbatch_size <= 0:
        raise ValueError("logical_microbatch_size must be positive when provided")
    model = model_factory().to(device)
    initial_state = state_transform(proof.initial_state, False) if state_transform else proof.initial_state
    if comparison_device == "replay" and str(device).startswith("cuda"):
        # Queue all required H2D state copies on this lane before loading; avoid
        # a host synchronization for every individual parameter/buffer.
        initial_state = {key:value.to(device,non_blocking=True) for key,value in initial_state.items()}
    model.load_state_dict(initial_state)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), lr=proof.learning_rate, momentum=proof.momentum
    )
    state_complete = proof.schema_version == "sevc-state-complete-replay-proof-v2"
    if state_complete:
        if (
            proof.optimizer_initial_state is None
            or len(proof.optimizer_checkpoints) != len(proof.checkpoints)
            or proof.rng_state_sha256 is None
            or proof.data_order_sha256 is None
            or proof.criterion_key != "torch.nn.CrossEntropyLoss"
            or proof.data_order_sha256 != _tensor_sequence_sha256(proof.batches)
        ):
            return {
                "passed": False,
                "checkpoint_count": 0,
                "max_abs_difference": math.inf,
                "max_optimizer_abs_difference": math.inf,
                "state_complete": False,
                "tolerance": tolerance,
            }
        _load_optimizer_momentum_state(
            model, optimizer, state_transform(proof.optimizer_initial_state, True)
            if state_transform else proof.optimizer_initial_state
        )
    differences: list[float] = []
    optimizer_differences: list[float] = []
    model.train()
    for (data, target), expected_state in zip(proof.batches, proof.checkpoints):
        if state_transform is not None:
            expected_state = state_transform(expected_state, False)
        optimizer.zero_grad()
        asynchronous = str(device).startswith("cuda") and data.is_pinned()
        data_on_device = data.to(device, non_blocking=asynchronous)
        target_on_device = target.to(device, non_blocking=asynchronous)
        if (
            logical_microbatch_size is None
            or int(data_on_device.shape[0]) <= logical_microbatch_size
        ):
            loss = criterion(
                _training_logits(model, data_on_device), target_on_device
            )
            loss.backward()
        else:
            logical_count = int(target_on_device.shape[0])
            for start in range(0, logical_count, logical_microbatch_size):
                stop = min(start + logical_microbatch_size, logical_count)
                chunk_loss_sum = nn.functional.cross_entropy(
                    _training_logits(model, data_on_device[start:stop]),
                    target_on_device[start:stop],
                    reduction="sum",
                )
                (chunk_loss_sum / logical_count).backward()
        optimizer.step()
        differences.append(
            _resident_state_max_abs_difference(model.state_dict(), expected_state)
            if comparison_device == "replay"
            else _state_max_abs_difference(_cpu_state(model), expected_state)
        )
        if state_complete:
            expected_optimizer = proof.optimizer_checkpoints[len(differences) - 1]
            if state_transform is not None:
                expected_optimizer = state_transform(expected_optimizer, True)
            optimizer_differences.append(
                _resident_state_max_abs_difference(
                    {name: optimizer.state[parameter]["momentum_buffer"]
                     for name, parameter in model.named_parameters()
                     if "momentum_buffer" in optimizer.state.get(parameter, {})},
                    expected_optimizer,
                ) if comparison_device == "replay" else _state_max_abs_difference(
                    _optimizer_momentum_state(model, optimizer),
                    expected_optimizer,
                )
            )
    maximum_model = max(differences, default=0.0)
    maximum_optimizer = max(optimizer_differences, default=0.0)
    maximum = max(maximum_model, maximum_optimizer)
    return {
        "passed": math.isfinite(maximum) and maximum <= tolerance,
        "checkpoint_count": len(differences),
        "max_abs_difference": maximum,
        "max_model_abs_difference": maximum_model,
        "max_optimizer_abs_difference": maximum_optimizer,
        "state_complete": state_complete,
        "tolerance": tolerance,
    }


def average_models(
    models: Sequence[nn.Module], excluded: Sequence[bool] | None = None
) -> nn.Module:
    if not models:
        raise ValueError("cannot aggregate an empty model list")
    exclusion = tuple(excluded) if excluded is not None else tuple(False for _ in models)
    if len(exclusion) != len(models):
        raise ValueError("exclusion mask length must match models")
    accepted = [model for model, skip in zip(models, exclusion) if not skip]
    if not accepted:
        raise ValueError("cannot aggregate when every model is excluded")
    result = copy.deepcopy(accepted[0])
    result_state = result.state_dict()
    states = [model.state_dict() for model in accepted]
    with torch.no_grad():
        for key, value in result_state.items():
            if value.is_floating_point() or value.is_complex():
                value.zero_()
                for state in states:
                    value.add_(
                        state[key].detach().to(
                            device=value.device, dtype=value.dtype
                        )
                    )
                value.div_(len(states))
            else:
                value.copy_(states[0][key].detach().to(value.device))
    result.load_state_dict(result_state)
    return result


def evaluate_model(
    model: nn.Module,
    batches: Iterable[tuple[torch.Tensor, torch.Tensor]],
    *,
    device: str,
    max_fused_batches: int = 1,
    max_fused_input_bytes: int | None = None,
) -> tuple[float, float]:
    if max_fused_batches <= 0:
        raise ValueError("max_fused_batches must be positive")
    if max_fused_input_bytes is not None and max_fused_input_bytes <= 0:
        raise ValueError("max_fused_input_bytes must be positive when provided")
    criterion = nn.CrossEntropyLoss(reduction="sum")
    model.eval()
    sample_count = 0
    loss_tensors: list[torch.Tensor] = []
    correct_tensors: list[torch.Tensor] = []

    def consume(
        group: Sequence[tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        nonlocal sample_count
        if len(group) == 1:
            data, target = group[0]
            logits_chunks = (_logits(model(data.to(device))),)
            target_chunks = (target.to(device),)
        else:
            sizes = tuple(int(target.shape[0]) for _, target in group)
            data_on_device = torch.cat(
                tuple(data for data, _ in group), dim=0
            ).to(device)
            target_on_device = torch.cat(
                tuple(target for _, target in group), dim=0
            ).to(device)
            logits_chunks = _logits(model(data_on_device)).split(sizes, dim=0)
            target_chunks = target_on_device.split(sizes, dim=0)
        # Fusion changes only kernel launch shape.  Loss and correct-count
        # reductions retain the original logical-batch boundaries and order.
        for logits, target_on_device in zip(logits_chunks, target_chunks):
            loss_tensors.append(criterion(logits, target_on_device).detach())
            correct_tensors.append(
                (logits.argmax(dim=1) == target_on_device).sum().detach()
            )
            sample_count += int(target_on_device.shape[0])

    group: list[tuple[torch.Tensor, torch.Tensor]] = []
    group_input_bytes = 0
    with torch.no_grad():
        for data, target in batches:
            input_bytes = sum(
                value.numel() * value.element_size() for value in (data, target)
            )
            would_exceed_bytes = bool(
                group
                and max_fused_input_bytes is not None
                and group_input_bytes + input_bytes > max_fused_input_bytes
            )
            if group and (
                len(group) >= max_fused_batches or would_exceed_bytes
            ):
                consume(group)
                group = []
                group_input_bytes = 0
            group.append((data, target))
            group_input_bytes += input_bytes
        if group:
            consume(group)
    if sample_count == 0:
        raise ValueError("evaluation requires at least one sample")
    loss_values = torch.stack(loss_tensors).detach().cpu().tolist()
    correct_values = torch.stack(correct_tensors).detach().cpu().tolist()
    loss_total = float(sum(loss_values))
    correct = int(sum(correct_values))
    return 100.0 * correct / sample_count, loss_total / sample_count
