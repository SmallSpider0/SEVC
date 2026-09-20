"""Deterministic seed and device handling shared by every experiment."""

from __future__ import annotations

import ctypes
import gc
import os
import random
from typing import Any

import numpy as np


def set_global_seed(seed: int, *, deterministic: bool = True) -> None:
    if seed < 0:
        raise ValueError("seed must be non-negative")
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        strict = (torch.are_deterministic_algorithms_enabled()
                  and not torch.is_deterministic_algorithms_warn_only_enabled())
        torch.use_deterministic_algorithms(True, warn_only=not strict)


def configure_torch_thread_caps_from_environment() -> dict[str, int]:
    """Apply explicit operational CPU thread caps before experiment work starts."""

    import torch

    intra_op = int(os.environ.get("SEVC_TORCH_NUM_THREADS", os.environ.get("OMP_NUM_THREADS", "1")))
    inter_op = int(os.environ.get("SEVC_TORCH_INTEROP_THREADS", "1"))
    if intra_op <= 0 or inter_op <= 0:
        raise ValueError("torch thread caps must be positive")
    torch.set_num_threads(intra_op)
    try:
        torch.set_num_interop_threads(inter_op)
    except RuntimeError:
        if torch.get_num_interop_threads() != inter_op:
            raise
    return {
        "torch_num_threads": int(torch.get_num_threads()),
        "torch_num_interop_threads": int(torch.get_num_interop_threads()),
    }


def release_process_memory(*, cuda: bool) -> None:
    """Release unreachable round-local objects and return free arenas on Linux."""

    gc.collect()
    if cuda:
        import torch

        torch.cuda.empty_cache()
    if os.name == "posix":
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except (AttributeError, OSError):
            pass


def device_capabilities() -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {
            "torch_available": False,
            "mps_built": False,
            "mps_available": False,
            "cuda_available": False,
        }
    return {
        "torch_available": True,
        "torch_version": torch.__version__,
        "mps_built": bool(torch.backends.mps.is_built()),
        "mps_available": bool(torch.backends.mps.is_available()),
        "cuda_available": bool(torch.cuda.is_available()),
    }


def resolve_device(preference: str = "mps") -> str:
    """Resolve an explicit device without silently requesting unavailable CUDA."""

    normalized = preference.lower()
    if normalized not in {"mps", "cuda", "cpu", "auto"}:
        raise ValueError(f"unsupported device preference: {preference}")
    capabilities = device_capabilities()
    if normalized == "cpu":
        return "cpu"
    if normalized in {"mps", "auto"} and capabilities["mps_available"]:
        return "mps"
    if normalized in {"cuda", "auto", "mps"} and capabilities["cuda_available"]:
        return "cuda"
    return "cpu"
