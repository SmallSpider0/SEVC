"""State-complete coordinate permutations for registered RCMP models."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Callable, Mapping

import torch
import torch.nn as nn

from sevc.core.operation_probe import operation_scope


@dataclass(frozen=True)
class TensorTransform:
    """One tensor's channel permutation on zero or more dimensions."""

    dimension_maps: tuple[tuple[int, str], ...]


@dataclass(frozen=True)
class ReplayStatePermutation:
    model_key: str
    seed: int
    channel_maps: dict[str, tuple[int, ...]]
    tensor_transforms: dict[str, TensorTransform]

    @property
    def permutation_id(self) -> str:
        digest = hashlib.sha256()
        digest.update(self.model_key.encode("utf-8"))
        digest.update(str(self.seed).encode("utf-8"))
        for name, values in sorted(self.channel_maps.items()):
            digest.update(name.encode("utf-8"))
            digest.update(",".join(str(value) for value in values).encode("utf-8"))
        return digest.hexdigest()

    def to_dict(self) -> dict[str, object]:
        return {
            "model_key": self.model_key,
            "seed": self.seed,
            "permutation_id": self.permutation_id,
            "channel_maps": {
                key: list(value) for key, value in sorted(self.channel_maps.items())
            },
            "tensor_transforms": {
                key: [
                    {"dimension": dimension, "channel_map": channel_map}
                    for dimension, channel_map in value.dimension_maps
                ]
                for key, value in sorted(self.tensor_transforms.items())
            },
        }


def _permutation(size: int, *, seed: int, domain: str) -> tuple[int, ...]:
    generator = torch.Generator(device="cpu")
    material = hashlib.sha256(f"{seed}|{domain}".encode("utf-8")).digest()
    generator.manual_seed(int.from_bytes(material[:8], "big") & ((1 << 63) - 1))
    values = torch.randperm(size, generator=generator).tolist()
    if size > 1 and values == list(range(size)):
        values[0], values[1] = values[1], values[0]
    return tuple(int(value) for value in values)


def _identity() -> TensorTransform:
    return TensorTransform(())


def _mlp_transforms(model: nn.Module) -> tuple[dict[str, tuple[int, ...]], dict[str, TensorTransform]]:
    state = model.state_dict()
    expected = {
        "network.1.weight",
        "network.1.bias",
        "network.3.weight",
        "network.3.bias",
        "network.5.weight",
        "network.5.bias",
    }
    if set(state) != expected:
        raise ValueError("small-mlp state schema drift")
    maps = {"hidden0": (), "hidden1": ()}
    transforms = {
        "network.1.weight": TensorTransform(((0, "hidden0"),)),
        "network.1.bias": TensorTransform(((0, "hidden0"),)),
        "network.3.weight": TensorTransform(((0, "hidden1"), (1, "hidden0"))),
        "network.3.bias": TensorTransform(((0, "hidden1"),)),
        "network.5.weight": TensorTransform(((1, "hidden1"),)),
        "network.5.bias": _identity(),
    }
    maps["hidden0"] = tuple(range(int(state["network.1.bias"].shape[0])))
    maps["hidden1"] = tuple(range(int(state["network.3.bias"].shape[0])))
    return maps, transforms


def _resnet_transforms(model: nn.Module) -> tuple[dict[str, tuple[int, ...]], dict[str, TensorTransform]]:
    state = model.state_dict()
    # The stem and layer1 share one external channel map because every layer1
    # BasicBlock has an identity skip connection.
    maps = {
        "stage0": tuple(range(64)),
        "stage2": tuple(range(128)),
        "stage3": tuple(range(256)),
        "stage4": tuple(range(512)),
    }
    transforms: dict[str, TensorTransform] = {}

    def conv(key: str, output_map: str, input_map: str | None = None) -> None:
        dims = [(0, output_map)]
        if input_map is not None:
            dims.append((1, input_map))
        transforms[f"{key}.weight"] = TensorTransform(tuple(dims))
        if f"{key}.bias" in state:
            transforms[f"{key}.bias"] = TensorTransform(((0, output_map),))

    def batch_norm(key: str, channel_map: str) -> None:
        for suffix in ("weight", "bias", "running_mean", "running_var"):
            transforms[f"{key}.{suffix}"] = TensorTransform(((0, channel_map),))
        transforms[f"{key}.num_batches_tracked"] = _identity()

    conv("conv1", "stage0")
    batch_norm("bn1", "stage0")
    input_map = "stage0"
    for layer_index, output_map in enumerate(("stage0", "stage2", "stage3", "stage4"), start=1):
        for block_index in range(2):
            prefix = f"layer{layer_index}.{block_index}"
            block_input = input_map if block_index == 0 else output_map
            conv(f"{prefix}.conv1", output_map, block_input)
            batch_norm(f"{prefix}.bn1", output_map)
            conv(f"{prefix}.conv2", output_map, output_map)
            batch_norm(f"{prefix}.bn2", output_map)
            if f"{prefix}.downsample.0.weight" in state:
                conv(f"{prefix}.downsample.0", output_map, block_input)
                batch_norm(f"{prefix}.downsample.1", output_map)
        input_map = output_map
    transforms["fc.weight"] = TensorTransform(((1, "stage4"),))
    transforms["fc.bias"] = _identity()
    missing = sorted(set(state) - set(transforms))
    extra = sorted(set(transforms) - set(state))
    if missing or extra:
        raise ValueError(f"resnet18 state schema drift; missing={missing}, extra={extra}")
    return maps, transforms


def build_replay_state_permutation(
    model: nn.Module, model_key: str, *, seed: int
) -> ReplayStatePermutation:
    key = model_key.lower()
    if key == "small-mlp":
        identity_maps, transforms = _mlp_transforms(model)
    elif key == "resnet18":
        identity_maps, transforms = _resnet_transforms(model)
    else:
        raise ValueError(f"unsupported replay permutation model: {model_key}")
    channel_maps = {
        name: _permutation(len(values), seed=seed, domain=f"{key}|{name}")
        for name, values in identity_maps.items()
    }
    return ReplayStatePermutation(
        model_key=key,
        seed=int(seed),
        channel_maps=channel_maps,
        tensor_transforms=transforms,
    )


def inverse_replay_state_permutation(
    plan: ReplayStatePermutation,
) -> ReplayStatePermutation:
    inverse = {}
    for name, values in plan.channel_maps.items():
        result = [0] * len(values)
        for new_index, old_index in enumerate(values):
            result[old_index] = new_index
        inverse[name] = tuple(result)
    return ReplayStatePermutation(
        model_key=plan.model_key,
        seed=plan.seed,
        channel_maps=inverse,
        tensor_transforms=plan.tensor_transforms,
    )


def replay_state_permutation_from_descriptor(
    model: nn.Module,
    descriptor: Mapping[str, Any],
) -> ReplayStatePermutation:
    """Validate and reconstruct a permutation from a public descriptor.

    The descriptor is deliberately self-contained so a verifier does not need
    any sealed task role or truth to recover the canonical coordinate system.
    """

    expected_fields = {
        "model_key",
        "seed",
        "permutation_id",
        "channel_maps",
        "tensor_transforms",
    }
    if set(descriptor) != expected_fields:
        raise ValueError("replay permutation descriptor schema mismatch")
    model_key = str(descriptor["model_key"]).lower()
    seed = int(descriptor["seed"])
    raw_maps = descriptor["channel_maps"]
    raw_transforms = descriptor["tensor_transforms"]
    if not isinstance(raw_maps, Mapping) or not isinstance(raw_transforms, Mapping):
        raise ValueError("replay permutation descriptor maps must be objects")
    channel_maps: dict[str, tuple[int, ...]] = {}
    for name, raw_values in raw_maps.items():
        if not isinstance(raw_values, (list, tuple)):
            raise ValueError(f"channel map is not a sequence: {name}")
        values = tuple(int(value) for value in raw_values)
        if sorted(values) != list(range(len(values))):
            raise ValueError(f"channel map is not a bijection: {name}")
        channel_maps[str(name)] = values
    transforms: dict[str, TensorTransform] = {}
    for key, raw_items in raw_transforms.items():
        if not isinstance(raw_items, (list, tuple)):
            raise ValueError(f"tensor transform is not a sequence: {key}")
        items: list[tuple[int, str]] = []
        for raw_item in raw_items:
            if not isinstance(raw_item, Mapping) or set(raw_item) != {
                "dimension",
                "channel_map",
            }:
                raise ValueError(f"tensor transform item schema mismatch: {key}")
            items.append(
                (int(raw_item["dimension"]), str(raw_item["channel_map"]))
            )
        if len({dimension for dimension, _ in items}) != len(items):
            raise ValueError(f"tensor dimension mapped more than once: {key}")
        if any(map_name not in channel_maps for _, map_name in items):
            raise ValueError(f"tensor transform references an unknown channel map: {key}")
        transforms[str(key)] = TensorTransform(tuple(items))
    plan = ReplayStatePermutation(
        model_key=model_key,
        seed=seed,
        channel_maps=channel_maps,
        tensor_transforms=transforms,
    )
    if str(descriptor["permutation_id"]) != plan.permutation_id:
        raise ValueError("replay permutation identifier mismatch")
    expected = build_replay_state_permutation(model, model_key, seed=seed)
    if plan.channel_maps != expected.channel_maps:
        raise ValueError("replay permutation channel maps do not match the public seed")
    if plan.tensor_transforms != expected.tensor_transforms:
        raise ValueError("replay permutation tensor schema mismatch")
    return plan


def transform_replay_state(
    state: Mapping[str, torch.Tensor],
    plan: ReplayStatePermutation,
    *,
    require_complete: bool = True,
    transform_device: str | None = None,
    output_device: str | None = None,
    tensor_observer: Callable[[str, torch.Tensor, torch.Tensor], None] | None = None,
) -> dict[str, torch.Tensor]:
    if require_complete and set(state) != set(plan.tensor_transforms):
        missing = sorted(set(state) - set(plan.tensor_transforms))
        absent = sorted(set(plan.tensor_transforms) - set(state))
        raise ValueError(f"state transform coverage failure; unsupported={missing}, absent={absent}")
    # Queue a whole state on one stream; synchronize CPU destinations once,
    # before observers consume bytes. No stream, index, or answer cache is shared.
    target = torch.device(transform_device) if transform_device is not None else None
    destination = torch.device(output_device) if output_device is not None else None
    if target is not None and target.type not in {"cpu", "cuda"}:
        raise ValueError("state transform supports only CPU or CUDA")
    maps = {}
    result: dict[str, torch.Tensor] = {}
    cuda_to_cpu = False
    for key in sorted(state):
        value = state[key]
        transform = plan.tensor_transforms.get(key)
        if transform is None:
            if require_complete:
                raise ValueError(f"unsupported state key: {key}")
            result[key] = value.detach().clone()
            continue
        # index_select already allocates its result.  Avoid cloning the full
        # tensor immediately before that allocation; tensors with an identity
        # transform still receive an owned copy.
        if transform.dimension_maps:
            tensor = value.detach()
        else:
            with operation_scope("identity_clone"):
                tensor = value.detach().clone()
        if target is not None:
            with operation_scope("h2d_enqueue"):
                tensor = tensor.to(target, non_blocking=target.type == "cuda")
        used_dimensions: set[int] = set()
        for dimension, map_name in transform.dimension_maps:
            if dimension in used_dimensions:
                raise ValueError(f"tensor dimension mapped more than once: {key}:{dimension}")
            used_dimensions.add(dimension)
            map_key = (map_name, str(tensor.device))
            if map_key not in maps:
                with operation_scope("index_creation"):
                    maps[map_key] = torch.tensor(plan.channel_maps[map_name], dtype=torch.long, device=tensor.device)
            indices = maps[map_key]
            if tensor.ndim <= dimension or tensor.shape[dimension] != len(indices):
                raise ValueError(f"state shape does not match channel map: {key}:{dimension}")
            with operation_scope("permutation_enqueue"):
                tensor = tensor.index_select(dimension, indices.to(tensor.device))
        end_device = destination if destination is not None else value.device
        cuda_to_cpu |= tensor.is_cuda and end_device.type == "cpu"
        with operation_scope("d2h_enqueue"):
            # Long-lived CPU deliveries must not allocate a pinned tensor per state.
            # GPU destinations still enqueue asynchronously on their owning lane.
            result[key] = tensor.to(end_device, non_blocking=end_device.type != "cpu")
    if cuda_to_cpu:
        with operation_scope("d2h_completion_wait"):
            torch.cuda.current_stream(target).synchronize()
    if tensor_observer is not None:
        with operation_scope("observer_hash"):
            for key in sorted(state):
                tensor_observer(key, state[key], result[key])
    return result


def transform_optimizer_momentum(
    state: Mapping[str, torch.Tensor],
    plan: ReplayStatePermutation,
    *,
    transform_device: str | None = None,
    output_device: str | None = None,
    tensor_observer: Callable[[str, torch.Tensor, torch.Tensor], None] | None = None,
) -> dict[str, torch.Tensor]:
    parameter_keys = {
        key for key in plan.tensor_transforms if key.endswith((".weight", ".bias"))
    }
    unknown = sorted(set(state) - parameter_keys)
    if unknown:
        raise ValueError(f"unsupported optimizer momentum keys: {unknown}")
    return transform_replay_state(
        state,
        plan,
        require_complete=False,
        transform_device=transform_device,
        output_device=output_device,
        tensor_observer=tensor_observer,
    )


def replay_state_inventory(
    model: nn.Module, plan: ReplayStatePermutation
) -> dict[str, object]:
    state = model.state_dict()
    unsupported = sorted(set(state) - set(plan.tensor_transforms))
    absent = sorted(set(plan.tensor_transforms) - set(state))
    mapped = []
    multiply_mapped = []
    for key in sorted(set(state) & set(plan.tensor_transforms)):
        dimensions = [item[0] for item in plan.tensor_transforms[key].dimension_maps]
        if len(dimensions) != len(set(dimensions)):
            multiply_mapped.append(key)
        mapped.append(
            {
                "key": key,
                "dtype": str(state[key].dtype),
                "shape": list(state[key].shape),
                "dimension_maps": [
                    {"dimension": dimension, "channel_map": channel_map}
                    for dimension, channel_map in plan.tensor_transforms[key].dimension_maps
                ],
            }
        )
    parameter_keys = {name for name, _ in model.named_parameters()}
    buffer_keys = {name for name, _ in model.named_buffers()}
    return {
        "model_key": plan.model_key,
        "permutation_id": plan.permutation_id,
        "state_key_count": len(state),
        "parameter_key_count": len(parameter_keys),
        "buffer_key_count": len(buffer_keys),
        "unmapped_parameter_keys": sorted(parameter_keys & set(unsupported)),
        "unmapped_buffer_keys": sorted(buffer_keys & set(unsupported)),
        "absent_transform_keys": absent,
        "multiply_mapped_keys": multiply_mapped,
        "unsupported_layer_count": len(unsupported) + len(absent),
        "mapped": mapped,
    }


__all__ = [
    "ReplayStatePermutation",
    "TensorTransform",
    "build_replay_state_permutation",
    "inverse_replay_state_permutation",
    "replay_state_permutation_from_descriptor",
    "replay_state_inventory",
    "transform_optimizer_momentum",
    "transform_replay_state",
]
