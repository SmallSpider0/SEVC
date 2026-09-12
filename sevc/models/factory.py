"""Model variants migrated from the historical prototype behind one factory."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch.nn as nn


class _MLP(nn.Module):
    def __init__(self, hidden_layers: int, class_count: int, input_features: int) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Flatten(), nn.Linear(input_features, 512), nn.ReLU()]
        for _ in range(hidden_layers - 1):
            layers.extend([nn.Linear(512, 512), nn.ReLU()])
        layers.append(nn.Linear(512, class_count))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.network(inputs)


class SmallMLP(_MLP):
    def __init__(self, class_count: int = 10, input_features: int = 28 * 28) -> None:
        super().__init__(2, class_count, input_features)


class MediumMLP(_MLP):
    def __init__(self, class_count: int = 10, input_features: int = 28 * 28) -> None:
        super().__init__(3, class_count, input_features)


class LargeMLP(_MLP):
    def __init__(self, class_count: int = 10, input_features: int = 28 * 28) -> None:
        super().__init__(4, class_count, input_features)


def build_model(
    model_key: str,
    *,
    class_count: int = 10,
    input_features: int = 28 * 28,
    pretrained_path: Path | None = None,
    pretrained_revision: str | None = None,
    local_files_only: bool = True,
    ignore_mismatched_sizes: bool = False,
    load_pretrained: bool = True,
):
    key = model_key.lower()
    if key == "small-mlp":
        return SmallMLP(class_count, input_features)
    if key == "medium-mlp":
        return MediumMLP(class_count, input_features)
    if key == "large-mlp":
        return LargeMLP(class_count, input_features)
    if key in {"resnet18", "resnet18-native-32"}:
        from torchvision import models

        return models.resnet18(num_classes=class_count)
    if key == "torchvision-vit":
        from torchvision import models

        return models.vit_b_16(weights=None, num_classes=class_count)
    if key == "hf-vit":
        try:
            from transformers import ViTConfig, ViTForImageClassification
        except ImportError as exc:
            raise RuntimeError(
                "model key 'hf-vit' requires the optional transformers dependency"
            ) from exc
        source = (
            str(pretrained_path)
            if pretrained_path
            else "google/vit-base-patch16-224-in21k"
        )
        if load_pretrained:
            return ViTForImageClassification.from_pretrained(
                source,
                revision=None if pretrained_path else pretrained_revision,
                num_labels=class_count,
                ignore_mismatched_sizes=ignore_mismatched_sizes,
                local_files_only=local_files_only,
            )
        config = ViTConfig.from_pretrained(
            source,
            revision=None if pretrained_path else pretrained_revision,
            local_files_only=local_files_only,
        )
        config.num_labels = int(class_count)
        config.id2label = {index: f"LABEL_{index}" for index in range(class_count)}
        config.label2id = {value: key for key, value in config.id2label.items()}
        return ViTForImageClassification(config)
    raise ValueError(f"unsupported model key: {model_key}")


def resolve_hf_snapshot(
    repository: str,
    *,
    revision: str,
    cache_root: Path,
    allow_download: bool,
) -> Path:
    """Resolve exactly the frozen files needed by the historical ViT workload."""

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("Hugging Face snapshot resolution requires transformers") from exc
    cache_root.mkdir(parents=True, exist_ok=True)
    arguments = {
        "repo_id": repository,
        "revision": revision,
        "cache_dir": str(cache_root),
        "allow_patterns": (
            "config.json",
            "preprocessor_config.json",
            "model.safetensors",
            "pytorch_model.bin",
        ),
    }
    try:
        path = snapshot_download(**arguments, local_files_only=True)
    except Exception:
        if not allow_download:
            raise
        path = snapshot_download(**arguments, local_files_only=False)
    return Path(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hf_snapshot_lock(
    snapshot_path: Path,
    *,
    repository: str,
    requested_revision: str,
) -> dict[str, Any]:
    files = []
    for path in sorted(candidate for candidate in snapshot_path.rglob("*") if candidate.is_file()):
        files.append(
            {
                "path": str(path.relative_to(snapshot_path)),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    if not files or not any(
        item["path"] in {"model.safetensors", "pytorch_model.bin"} for item in files
    ):
        raise ValueError("Hugging Face snapshot has no model weights")
    aggregate = hashlib.sha256()
    for item in files:
        aggregate.update(
            f"{item['sha256']}  {item['bytes']}  {item['path']}\n".encode("utf-8")
        )
    return {
        "schema_version": "sevc-hf-model-snapshot-lock-v1",
        "repository": repository,
        "requested_revision": requested_revision,
        "resolved_revision": snapshot_path.name,
        "snapshot_path": str(snapshot_path),
        "file_count": len(files),
        "total_bytes": sum(int(item["bytes"]) for item in files),
        "aggregate_sha256": aggregate.hexdigest(),
        "files": files,
        "passed": snapshot_path.name == requested_revision,
    }


def model_state_nbytes(model: nn.Module) -> int:
    return int(
        sum(value.numel() * value.element_size() for value in model.state_dict().values())
    )


def model_state_sha256(model: nn.Module) -> str:
    """Hash tensor names, dtypes, shapes, and canonical contiguous CPU bytes."""

    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        metadata = json.dumps(
            {
                "name": name,
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        digest.update(metadata.encode("utf-8"))
        digest.update(b"\0")
        digest.update(tensor.numpy().tobytes(order="C"))
        digest.update(b"\n")
    return digest.hexdigest()


def model_parameter_count(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))
