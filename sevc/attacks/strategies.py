"""Historical worker behaviors behind explicit registered strategy keys."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable

import torch.nn as nn

from sevc.core.registry import Registry


class WorkerBehavior(str, Enum):
    NORMAL = "normal"
    FREERIDER = "freerider"
    ADVERSARIAL = "adversarial"


@dataclass(frozen=True)
class PreparedWorkerModel:
    model: nn.Module
    should_train: bool


Strategy = Callable[[nn.Module, Callable[[], nn.Module], str], PreparedWorkerModel]
STRATEGIES: Registry[Strategy] = Registry("worker behavior")


@STRATEGIES.register("normal")
def _normal(
    global_model: nn.Module,
    model_factory: Callable[[], nn.Module],
    device: str,
) -> PreparedWorkerModel:
    model = model_factory().to(device)
    model.load_state_dict(global_model.state_dict())
    return PreparedWorkerModel(model, True)


@STRATEGIES.register("freerider")
def _freerider(
    global_model: nn.Module,
    model_factory: Callable[[], nn.Module],
    device: str,
) -> PreparedWorkerModel:
    model = model_factory().to(device)
    model.load_state_dict(global_model.state_dict())
    return PreparedWorkerModel(model, False)


@STRATEGIES.register("adversarial")
def _adversarial(
    global_model: nn.Module,
    model_factory: Callable[[], nn.Module],
    device: str,
) -> PreparedWorkerModel:
    del global_model
    return PreparedWorkerModel(model_factory().to(device), False)


def prepare_worker_model(
    behavior: WorkerBehavior | str,
    global_model: nn.Module,
    model_factory: Callable[[], nn.Module],
    device: str,
) -> PreparedWorkerModel:
    key = behavior.value if isinstance(behavior, WorkerBehavior) else str(behavior)
    return STRATEGIES.get(key)(global_model, model_factory, device)
