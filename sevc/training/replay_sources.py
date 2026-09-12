"""Dataset-backed short replay sources shared by experiment assemblers."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence
import threading

import torch

from sevc.core.runtime import set_global_seed
from sevc.data import load_dataset_bundle
from sevc.models import build_model
from sevc.training import WorkerBehavior, produce_worker_update


class ReplayDatasetContext:
    def __init__(self, dataset: str, spec: Mapping[str, Any], data_root: Path, device: Any):
        self.dataset = dataset
        self.model_key = str(spec["model"])
        self.class_count = int(spec["class_count"])
        self.image_size = int(spec["image_size"])
        self.device = device
        self.train = load_dataset_bundle(
            dataset, data_root, download=False, cifar100_image_size=self.image_size,
        ).train
        self.input_features = self.image_size * self.image_size
        self.build_key = "resnet18" if self.model_key.startswith("resnet18") else self.model_key
        self._lanes = {}
        self._replay_models = threading.local()

    def task_lanes(self, width):
        from sevc.core.task_lanes import TaskLanes
        if width not in self._lanes:
            self._lanes[width] = TaskLanes(width, getattr(self.device,"name",self.device))
        return self._lanes[width]

    def close_task_lanes(self):
        for lanes in self._lanes.values():
            lanes.close()
        self._lanes.clear()

    def cached_replay_factory(self):
        """One model per lane; canonical verifier resets full state/optimizer."""
        if not hasattr(self._replay_models,"model"):
            self._replay_models.model = self.factory()
        return self._replay_models.model

    def factory(self):
        return build_model(self.build_key, class_count=self.class_count,
                           input_features=self.input_features)


def batches_from_indices(context: ReplayDatasetContext, indices: Sequence[int],
                         labels: Sequence[int]) -> tuple:
    if len(indices) != 8 or len(labels) != 8:
        raise ValueError("short-source recipe requires exactly eight samples")
    samples = [context.train[int(index)] for index in indices]
    observed = [int(target) for _, target in samples]
    if observed != [int(value) for value in labels]:
        raise ValueError("rematerialized sample labels differ from the sealed identity")
    return tuple((torch.stack([samples[i][0], samples[i + 1][0]]),
                  torch.as_tensor(observed[i:i + 2], dtype=torch.long))
                 for i in range(0, len(samples), 2))


def materialize_short_source(context: ReplayDatasetContext, row: Mapping[str, Any]):
    """The existing canonical four-step FP32 trainer; no second training loop."""
    set_global_seed(int(row["source_seed"]))
    batches = batches_from_indices(context, row["sample_indices"], row["sample_labels"])
    return train_short_source(context, batches)


def train_short_source(context: ReplayDatasetContext, batches: tuple):
    """Shared training call; callers retain their historical timer boundaries."""
    device = getattr(context.device, "name", context.device)
    update = produce_worker_update(
        context.factory().to(device), context.factory, batches, WorkerBehavior.NORMAL,
        device=device, learning_rate=0.01, momentum=0.9, local_epochs=1,
        max_batches=4, capture_replay=True,
    )
    if update.replay_proof is None:
        raise RuntimeError("canonical source did not emit a replay proof")
    return update.replay_proof


def materialize_branch_source(context, row, *, initial_state=None, initial_momentum=None, initial_rng=None):
    """Variable frozen workloads and milestone branches use the same trainer."""
    set_global_seed(int(row["source_seed"]))
    indices, labels = row["sample_indices"], row["sample_labels"]
    steps, size = row["steps"], row["batch_size"]
    if len(indices) != steps * size or len(labels) != len(indices):
        raise ValueError("branch source sample count does not match workload")
    samples = [context.train[i] for i in indices]
    if [int(y) for _, y in samples] != list(labels):
        raise ValueError("branch data identity mismatch")
    batches = tuple((torch.stack([x for x, _ in samples[i:i+size]]),
                     torch.tensor(labels[i:i+size], dtype=torch.long))
                    for i in range(0, len(indices), size))
    model = context.factory()
    if initial_state is not None:
        model.load_state_dict(initial_state, strict=True)
    update = produce_worker_update(model.to(context.device.name), context.factory, batches,
        WorkerBehavior.NORMAL, device=context.device.name, learning_rate=.01, momentum=.9,
        max_batches=steps, capture_replay=True, initial_momentum_state=initial_momentum,
        initial_rng_state=initial_rng)
    if update.replay_proof is None:
        raise RuntimeError("branch trainer emitted no complete proof")
    return update.replay_proof
