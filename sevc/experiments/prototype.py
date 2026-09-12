"""Canonical end-to-end SEVC prototype assembled from replaceable modules."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Iterable, Sequence

import torch
import torch.nn as nn

from sevc.training import WorkerBehavior, average_models, evaluate_model, produce_worker_update
from sevc.verification import VerificationContext, VerificationResult, run_verification


@dataclass(frozen=True)
class LocalTrainingConfig:
    learning_rate: float
    momentum: float
    local_epochs: int = 1
    max_batches_per_worker: int | None = None
    replay_tolerance: float = 1e-6


@dataclass(frozen=True)
class PrototypeRoundResult:
    device: str
    worker_behaviors: tuple[str, ...]
    samples_seen: tuple[int, ...]
    mean_losses: tuple[float | None, ...]
    before_accuracy: float
    before_loss: float
    after_accuracy: float
    after_loss: float
    primary_verification_method: str
    verification_results: dict[str, VerificationResult]

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["verification_results"] = {
            key: value.to_dict() for key, value in self.verification_results.items()
        }
        return payload


class SEVCPrototype:
    """One stateful composition root; experiments vary registered module keys only."""

    def __init__(
        self,
        *,
        global_model: nn.Module,
        model_factory: Callable[[], nn.Module],
        evaluation_batches: Iterable[tuple[torch.Tensor, torch.Tensor]],
        device: str,
        seed: int,
        training: LocalTrainingConfig,
    ) -> None:
        self.global_model = global_model.to(device)
        self.model_factory = model_factory
        self.evaluation_batches = evaluation_batches
        self.device = device
        self.seed = seed
        self.training = training

    def run_round(
        self,
        worker_batches: Sequence[Iterable[tuple[torch.Tensor, torch.Tensor]]],
        behaviors: Sequence[WorkerBehavior],
        *,
        verification_method: str,
        audit_methods: Sequence[str] = (),
    ) -> PrototypeRoundResult:
        if len(worker_batches) != len(behaviors) or not worker_batches:
            raise ValueError("worker batches and behaviors must have equal non-zero length")
        before_accuracy, before_loss = evaluate_model(
            self.global_model, self.evaluation_batches, device=self.device
        )
        all_methods = tuple(dict.fromkeys((verification_method, *audit_methods)))
        capture_replay = any(method == "trajectory-replay" for method in all_methods)
        updates = tuple(
            produce_worker_update(
                self.global_model,
                self.model_factory,
                batches,
                behavior,
                device=self.device,
                learning_rate=self.training.learning_rate,
                momentum=self.training.momentum,
                local_epochs=self.training.local_epochs,
                max_batches=self.training.max_batches_per_worker,
                capture_replay=capture_replay and behavior is WorkerBehavior.NORMAL,
            )
            for batches, behavior in zip(worker_batches, behaviors)
        )
        evaluator = lambda model: evaluate_model(
            model, self.evaluation_batches, device=self.device
        )
        context = VerificationContext(
            global_model=self.global_model,
            local_models=tuple(update.model for update in updates),
            evaluator=evaluator,
            ground_truth_malicious=tuple(
                behavior is not WorkerBehavior.NORMAL for behavior in behaviors
            ),
            seed=self.seed,
            replay_proofs=tuple(update.replay_proof for update in updates),
            replay_model_factory=self.model_factory,
            replay_device=self.device,
            replay_tolerance=self.training.replay_tolerance,
        )
        verification_results = {
            method: run_verification(method, context) for method in all_methods
        }
        primary = verification_results[verification_method]
        self.global_model = average_models(
            tuple(update.model for update in updates), primary.predicted_malicious
        )
        after_accuracy, after_loss = evaluator(self.global_model)
        return PrototypeRoundResult(
            device=self.device,
            worker_behaviors=tuple(behavior.value for behavior in behaviors),
            samples_seen=tuple(update.samples_seen for update in updates),
            mean_losses=tuple(update.mean_loss for update in updates),
            before_accuracy=before_accuracy,
            before_loss=before_loss,
            after_accuracy=after_accuracy,
            after_loss=after_loss,
            primary_verification_method=verification_method,
            verification_results=verification_results,
        )
