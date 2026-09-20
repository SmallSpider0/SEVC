"""Selected canonical scientific routines; deployment wrappers are omitted."""
from __future__ import annotations


from dataclasses import asdict, is_dataclass

from datetime import datetime, timezone

import hashlib

import json

import os

from pathlib import Path

import platform

import tempfile

from typing import Any

import numpy as np

from .runtime import device_capabilities

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_text(payload: Any) -> str:
    """Return the one compact JSON representation used for content identities."""

    return json.dumps(
        _jsonable(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        _jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True
    ) + "\n"
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(encoded)
        temporary_path = Path(handle.name)
    os.replace(temporary_path, path)


def ensure_experiment_output_root(
    output_root: Path,
    repo_root: Path,
    change_id: str,
    *,
    required_parent: Path | None = None,
) -> Path:
    if not output_root.is_absolute():
        raise ValueError("experiment output root must be absolute")
    resolved_output = output_root.resolve(strict=False)
    resolved_repo = repo_root.resolve(strict=False)
    if resolved_output == resolved_repo or resolved_repo in resolved_output.parents:
        raise ValueError("experiment output root must be outside the repository")
    required_parent = (
        required_parent
        if required_parent is not None
        else resolved_output.parent
    ).resolve(strict=False)
    if resolved_output != required_parent and required_parent not in resolved_output.parents:
        raise ValueError(
            f"output root must be below {required_parent}, got {resolved_output}"
        )
    resolved_output.mkdir(parents=True, exist_ok=True)
    return resolved_output


def capture_environment() -> dict[str, Any]:
    dependency_versions: dict[str, str | None] = {}
    for name in (
        "numpy",
        "scipy",
        "sklearn",
        "torch",
        "torchvision",
        "matplotlib",
        "transformers",
        "huggingface_hub",
    ):
        try:
            module = __import__(name)
        except ImportError:
            dependency_versions[name] = None
        else:
            dependency_versions[name] = str(getattr(module, "__version__", "unknown"))
    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "device_capabilities": device_capabilities(),
        "dependencies": dependency_versions,
    }


