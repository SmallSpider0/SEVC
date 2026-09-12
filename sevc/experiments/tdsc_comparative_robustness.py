"""Selected canonical scientific routines; deployment wrappers are omitted."""
from __future__ import annotations


from datetime import datetime, timezone

import copy

import hashlib

import math

import random

import time

from typing import Any, Mapping, Sequence

import torch

from torch.utils.data import DataLoader, Subset

from sevc.attacks import WorkerBehavior, apply_checkerboard_trigger, apply_strong_model_replacement, poison_selection, prepare_evasion_projection, project_evasion_aware_update, train_evasion_aware_update

from sevc.core.artifacts import canonical_json_text, sha256_text

from sevc.core.runtime import release_process_memory, set_global_seed

from sevc.data import IndexedSubset, exact_processed_event_indices, indices_sha256

from sevc.evaluation.tdsc_comp_e0_freeze import _root_indices

from sevc.evaluation.trainer_population_scale import trainer_contract_selections

from sevc.incentives import actual_utilities

from sevc.models import build_model, model_state_sha256

from sevc.training import evaluate_model, produce_worker_update

from sevc.verification import SubmittedUpdate, aggregate_submissions

CHANGE_ID = "experiment-tdsc-comparative-robustness-v1"


METHODS = ("none", "multi-krum", "fltrust", "sevc-ctiv-registered-v1")


COMMON_REWARD_RESERVE = 10.92


def derive_comparative_seed(change_id: str, base_seed: int, namespace: str) -> int:
    material = f"{change_id}|{int(base_seed)}|{namespace}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % 2147483647


def _dataset_spec(config: Mapping[str, Any], dataset_key: str) -> Mapping[str, Any]:
    spec = config["datasets"][dataset_key]
    if str(spec["key"]) != dataset_key:
        raise ValueError(f"dataset spec identity drift for {dataset_key}")
    return spec


def _attacker_order(
    trainer_count: int,
    seed: int,
    formal_seeds: Sequence[int],
    *,
    change_id: str = CHANGE_ID,
) -> tuple[int, ...]:
    ordinal = tuple(formal_seeds).index(seed)
    layers = {
        layer: [index for index in range(trainer_count) if index % 5 == layer]
        for layer in range(5)
    }
    for layer, values in layers.items():
        random.Random(
            derive_comparative_seed(
                change_id, seed, f"attack|N={trainer_count}|layer={layer}"
            )
        ).shuffle(values)
    order: list[int] = []
    layer_order = tuple((ordinal + offset) % 5 for offset in range(5))
    while any(layers.values()):
        for layer in layer_order:
            if layers[layer]:
                order.append(layers[layer].pop(0))
    return tuple(order)


def _classification_metrics(truth: Sequence[bool], predicted: Sequence[bool]) -> dict[str, float]:
    if len(truth) != len(predicted) or not truth:
        raise ValueError("classification vectors must be equal and non-empty")
    tp = sum(left and right for left, right in zip(truth, predicted))
    fn = sum(left and not right for left, right in zip(truth, predicted))
    fp = sum(not left and right for left, right in zip(truth, predicted))
    tn = sum(not left and not right for left, right in zip(truth, predicted))
    tpr = 0.0 if tp + fn == 0 else tp / (tp + fn)
    fpr = 0.0 if fp + tn == 0 else fp / (fp + tn)
    return {
        "trainer_TPR": tpr,
        "trainer_FPR": fpr,
        "trainer_balanced_accuracy": 0.5 * (tpr + (1.0 - fpr)),
        "TP": tp,
        "FN": fn,
        "FP": fp,
        "TN": tn,
    }


def _evaluate_triggered_asr(
    model: torch.nn.Module,
    test_dataset: Any,
    device: str,
    *,
    dataset_key: str,
) -> float:
    loader = DataLoader(
        test_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=0,
        pin_memory=device == "cuda",
    )
    numerator = denominator = 0
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        for data, target in loader:
            eligible = target != 0
            if not bool(eligible.any()):
                continue
            triggered = apply_checkerboard_trigger(
                data[eligible].to(device, non_blocking=device == "cuda"),
                dataset_key=dataset_key,
            )
            output = model(triggered)
            logits = output.logits if hasattr(output, "logits") else output
            numerator += int((logits.argmax(dim=1) == 0).sum().cpu())
            denominator += int(eligible.sum())
    return 0.0 if denominator == 0 else numerator / denominator


def _make_loader(
    dataset: Any,
    indices: Sequence[int],
    batch_size: int,
    *,
    pin_memory: bool = False,
) -> DataLoader:
    return DataLoader(
        IndexedSubset(dataset, indices),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=pin_memory,
    )


def _train_honest(
    anchor: torch.nn.Module,
    batches: Sequence[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    *,
    device: str,
    learning_rate: float,
    momentum: float,
    model_key: str,
    class_count: int,
) -> Any:
    update = produce_worker_update(
        anchor,
        lambda: build_model(model_key, class_count=class_count),
        ((data, target) for data, target, _ in batches),
        WorkerBehavior.NORMAL,
        device=device,
        learning_rate=learning_rate,
        momentum=momentum,
        local_epochs=1,
        capture_replay=True,
        replay_tail_batches=1,
    )
    return update


def _train_root(
    anchor: torch.nn.Module,
    batches: Sequence[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    *,
    device: str,
    learning_rate: float,
    momentum: float,
    model_key: str,
    class_count: int,
) -> SubmittedUpdate:
    update = produce_worker_update(
        anchor,
        lambda: build_model(model_key, class_count=class_count),
        ((data, target) for data, target, _ in batches),
        WorkerBehavior.NORMAL,
        device=device,
        learning_rate=learning_rate,
        momentum=momentum,
        local_epochs=1,
        capture_replay=False,
    )
    return SubmittedUpdate(-1, update.model.cpu(), update.samples_seen, None)


def _run_source_unit(
    identity: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    bundle: Any,
    labels: Sequence[int],
    partitions: Mapping[tuple[int, int, str], tuple[tuple[int, ...], ...]],
    device: str,
    change_id: str = CHANGE_ID,
    methods: Sequence[str] = METHODS,
) -> dict[str, Any]:
    started = time.perf_counter()
    dataset_key = str(identity["dataset"])
    spec = _dataset_spec(config, dataset_key)
    model_key = str(spec["model"])
    class_count = int(spec["class_count"])
    global_epochs = int(spec["global_epochs"])
    seed = int(identity["seed"])
    trainer_count = int(identity["trainer_count"])
    regime = str(identity["partition"])
    scenario = str(identity["scenario"])
    trainer_partitions = partitions[(seed, trainer_count, regime)]
    selections, contribution_weights = trainer_contract_selections(config, trainer_count, seed)
    reserve = float(config["contract"].get("common_reward_reserve", 0.0))
    exact_events = bool(config["contract"].get("enforce_exact_processed_events", False))
    if not math.isclose(reserve, COMMON_REWARD_RESERVE, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("COMP-E2 requires the prospectively frozen 10.92 reward reserve")
    if not exact_events:
        raise ValueError("COMP-E2 requires exact processed sample-event enforcement")
    selected_quotas = tuple(int(selection.item.contribution) for selection in selections)
    if any(
        quota <= 0
        or not math.isclose(
            float(selection.item.contribution), float(quota), rel_tol=0.0, abs_tol=1e-12
        )
        for quota, selection in zip(selected_quotas, selections)
    ):
        raise ValueError("COMP-E2 selected contribution D must be a positive integer")
    method_keys = tuple(str(value) for value in methods)
    if not method_keys or "none" not in method_keys:
        raise ValueError("comparative execution requires a non-empty method set containing none")
    attack_order = _attacker_order(
        trainer_count,
        seed,
        config["formal_paired_seeds"],
        change_id=change_id,
    )
    attacked = set(attack_order[: math.floor(0.4 * trainer_count)]) if scenario != "honest" else set()
    attack_scenario = str(
        config["attack"].get("scenario_key", "evasion-aware-poison-40")
    )
    if scenario == "honest":
        behavior = tuple("honest" for _ in range(trainer_count))
    elif scenario == "freerider-40":
        behavior = tuple("freerider" if index in attacked else "honest" for index in range(trainer_count))
    elif scenario == attack_scenario:
        behavior = tuple(
            "strong-model-replacement-poison" if index in attacked else "honest"
            for index in range(trainer_count)
        )
    else:
        raise ValueError(f"unsupported scenario {scenario}")

    set_global_seed(
        derive_comparative_seed(change_id, seed, f"dataset={dataset_key}|initial-model")
    )
    anchor = build_model(model_key, class_count=class_count).cpu()
    evaluation_loader = DataLoader(
        bundle.test,
        batch_size=128,
        shuffle=False,
        num_workers=0,
        pin_memory=device == "cuda",
    )
    root_indices = _root_indices(
        labels,
        seed,
        class_count=class_count,
        per_class=int(spec["trusted_root_per_class"]),
    )
    trusted_validation_loader = DataLoader(
        Subset(bundle.train, tuple(root_indices)),
        batch_size=128,
        shuffle=False,
        num_workers=0,
        pin_memory=device == "cuda",
    )
    trusted_validation_max_batches = int(
        config.get("verification", {}).get("refiner_validation_max_batches", 2)
    )
    if trusted_validation_max_batches <= 0:
        raise ValueError("refiner_validation_max_batches must be positive")

    def trusted_validation_evaluator(model: torch.nn.Module) -> tuple[float, float]:
        model.to(device)
        try:
            limited_batches = (
                batch
                for batch_index, batch in enumerate(trusted_validation_loader)
                if batch_index < trusted_validation_max_batches
            )
            return evaluate_model(
                model,
                limited_batches,
                device=device,
                max_fused_batches=1,
            )
        finally:
            model.cpu()
    batch_size = int(config["training"]["batch_size"])
    learning_rate = float(spec["learning_rate"])
    momentum = float(config["training"]["momentum"])
    poison_sets = {
        trainer_id: poison_selection(
            change_id,
            seed,
            trainer_id,
            trainer_partitions[trainer_id],
            labels,
            target_class=0,
            fraction=0.5,
        )
        for trainer_id in attacked
        if scenario == attack_scenario
    }
    decision_rows: list[dict[str, Any]] = []
    method_last: dict[str, dict[str, Any]] = {}
    bank_hashes: list[str] = []
    for round_index in range(global_epochs):
        attack_active = (
            scenario == attack_scenario
            and (
                str(config["attack"].get("activation_round", "all-rounds"))
                == "all-rounds"
                or (
                    str(config["attack"].get("activation_round"))
                    == "final-global-round-only"
                    and round_index == global_epochs - 1
                )
            )
        )
        if scenario == attack_scenario and str(
            config["attack"].get("activation_round", "all-rounds")
        ) not in {"all-rounds", "final-global-round-only"}:
            raise ValueError("unsupported strong-attack activation round")
        round_behavior = tuple(
            "strong-model-replacement-poison"
            if attack_active and trainer_id in attacked
            else (
                "honest"
                if scenario == attack_scenario and trainer_id in attacked
                else value
            )
            for trainer_id, value in enumerate(behavior)
        )
        raw_updates: list[Any | None] = [None] * trainer_count
        honest_models: list[torch.nn.Module] = []
        processed_events: list[tuple[int, ...]] = [tuple() for _ in range(trainer_count)]
        for trainer_id in range(trainer_count):
            set_global_seed(
                derive_comparative_seed(
                    change_id, seed, f"round={round_index}|trainer={trainer_id}"
                )
            )
            trainer_indices = trainer_partitions[trainer_id]
            if round_behavior[trainer_id] == "freerider":
                raw_updates[trainer_id] = None
                continue
            trainer_indices = exact_processed_event_indices(
                trainer_indices,
                selected_quotas[trainer_id],
                change_id=change_id,
                seed=seed,
                trainer_id=trainer_id,
                round_index=round_index,
            )
            processed_events[trainer_id] = tuple(trainer_indices)
            batches = tuple(
                _make_loader(
                    bundle.train,
                    trainer_indices,
                    batch_size,
                    pin_memory=device == "cuda",
                )
            )
            if round_behavior[trainer_id] == "strong-model-replacement-poison":
                raw_updates[trainer_id] = train_evasion_aware_update(
                    anchor,
                    lambda: build_model(model_key, class_count=class_count),
                    batches,
                    poison_sets[trainer_id],
                    device=device,
                    learning_rate=learning_rate,
                    momentum=momentum,
                    local_epochs=1,
                    target_class=0,
                    proximal_penalty=1e-4,
                    dataset_key=dataset_key,
                )
            else:
                raw_updates[trainer_id] = _train_honest(
                    anchor,
                    batches,
                    device=device,
                    learning_rate=learning_rate,
                    momentum=momentum,
                    model_key=model_key,
                    class_count=class_count,
                )
                raw_updates[trainer_id] = copy.copy(raw_updates[trainer_id])
                raw_updates[trainer_id].model.cpu()
                honest_models.append(raw_updates[trainer_id].model)
        submissions: list[SubmittedUpdate] = []
        evasion_meta: dict[int, dict[str, Any]] = {}
        evasion_context = (
            prepare_evasion_projection(anchor, honest_models)
            if scenario == attack_scenario
            and config["attack"].get("projection_mode") == "honest-median-evasion"
            else None
        )
        for trainer_id, update in enumerate(raw_updates):
            if update is None:
                submissions.append(SubmittedUpdate(trainer_id, copy.deepcopy(anchor), 0, None))
                continue
            model = update.model.cpu()
            if round_behavior[trainer_id] == "strong-model-replacement-poison":
                replacement_scale = float(config["attack"]["raw_model_replacement_scale"])
                if config["attack"].get("projection_mode") == "honest-median-evasion":
                    model, scale, distance, cosine = project_evasion_aware_update(
                        anchor,
                        model,
                        replacement_scale=replacement_scale,
                        projection_context=evasion_context,
                    )
                    evasion_meta[trainer_id] = {
                        "projection_mode": "honest-median-evasion",
                        "selected_scale": scale,
                        "distance_to_honest_median": distance,
                        "cosine_with_honest_median": cosine,
                    }
                elif config["attack"].get("projection_mode") == "none":
                    model = apply_strong_model_replacement(
                        anchor, model, replacement_scale=replacement_scale
                    )
                    evasion_meta[trainer_id] = {
                        "projection_mode": "none",
                        "selected_scale": 1.0,
                        "replacement_scale": replacement_scale,
                    }
                else:
                    raise ValueError("unsupported strong-attack projection mode")
            submissions.append(
                SubmittedUpdate(trainer_id, model, int(update.samples_seen), update.replay_proof)
            )
        root_batches = tuple(
            _make_loader(
                bundle.train,
                root_indices,
                batch_size,
                pin_memory=device == "cuda",
            )
        )
        set_global_seed(
            derive_comparative_seed(change_id, seed, f"round={round_index}|trusted-root")
        )
        root_update = _train_root(
            anchor,
            root_batches,
            device=device,
            learning_rate=learning_rate,
            momentum=momentum,
            model_key=model_key,
            class_count=class_count,
        )
        bank_identity = {
            "round": round_index + 1,
            "trainer_model_sha256": [model_state_sha256(item.model) for item in submissions],
            "samples_seen": [item.samples_seen for item in submissions],
            "trainer_order": list(range(trainer_count)),
            "reuse_count": len(method_keys),
        }
        bank_sha = sha256_text(canonical_json_text(bank_identity))
        bank_hashes.append(bank_sha)
        truth = tuple(index in attacked for index in range(trainer_count))
        results = {}
        for method in method_keys:
            result = aggregate_submissions(
                method,
                anchor=anchor,
                submissions=submissions,
                contribution_weights=contribution_weights,
                root_update=root_update,
                replay_model_factory=lambda: build_model(model_key, class_count=class_count),
                replay_device=device,
                replay_tolerance=float(config["verification"]["replay_tolerance"]),
                distance_chunk_elements=int(config["verification"]["distance_chunk_elements"]),
                vector_compute_device=device,
                trusted_validation_evaluator=trusted_validation_evaluator,
            )
            classification = _classification_metrics(truth, result.predicted_malicious)
            rewards = tuple(
                0.0 if result.predicted_malicious[index] else float(selections[index].item.reward)
                for index in range(trainer_count)
            )
            owner_utility, trainer_utilities = actual_utilities(
                selections,
                [item.samples_seen for item in submissions],
                rewards,
                float(config["contract"]["sigma_1"]),
                float(config["contract"]["sigma_2"]),
            )
            method_last[method] = {
                **classification,
                "owner_utility": owner_utility,
                "honest_trainer_utility": min(
                    (value for index, value in enumerate(trainer_utilities) if index not in attacked),
                    default=0.0,
                ),
                "attacker_utility": max(
                    (value for index, value in enumerate(trainer_utilities) if index in attacked),
                    default=0.0,
                ),
                "verification_wall_seconds": result.verification_wall_seconds,
                "verification_bytes": result.verification_bytes,
                "aggregate": result.aggregate,
                "details": dict(result.details),
            }
            results[method] = result
            for trainer_id in range(trainer_count):
                decision_rows.append(
                    {
                        **dict(identity),
                        "round": round_index + 1,
                        "method": method,
                        "trainer_id": trainer_id,
                        "behavior": round_behavior[trainer_id],
                        "selected_contribution": int(
                            selections[trainer_id].item.contribution
                        ),
                        "processed_sample_events": int(
                            submissions[trainer_id].samples_seen
                        ),
                        "exact_contribution_enforced": (
                            int(submissions[trainer_id].samples_seen) == 0
                            if round_behavior[trainer_id] == "freerider"
                            else int(submissions[trainer_id].samples_seen)
                            == int(selections[trainer_id].item.contribution)
                        ),
                        "processed_event_count": len(processed_events[trainer_id]),
                        "processed_event_indices_sha256": indices_sha256(
                            processed_events[trainer_id]
                        ),
                        "base_contract_reward": float(
                            selections[trainer_id].item.reward - reserve
                        ),
                        "contract_common_reward_reserve": reserve,
                        "contract_reward": float(selections[trainer_id].item.reward),
                        "settled_reward": float(rewards[trainer_id]),
                        "ground_truth_malicious": truth[trainer_id],
                        "predicted_malicious": result.predicted_malicious[trainer_id],
                        "score": result.scores[trainer_id],
                        "update_bank_sha256": bank_sha,
                        "evasion": evasion_meta.get(trainer_id),
                    }
                )
        anchor = results["none"].aggregate.cpu()
        del submissions, raw_updates, root_update, root_batches, results
        release_process_memory(cuda=device == "cuda")

    trajectory_rows = []
    for method in method_keys:
        metrics = dict(method_last[method])
        aggregate = metrics.pop("aggregate")
        aggregate = aggregate.to(device)
        clean_accuracy, clean_loss = evaluate_model(
            aggregate, evaluation_loader, device=device, max_fused_batches=32,
            max_fused_input_bytes=512 * 1024 * 1024,
        )
        triggered_asr = _evaluate_triggered_asr(
            aggregate,
            bundle.test,
            device,
            dataset_key=dataset_key,
        )
        aggregate.cpu()
        details = metrics.pop("details")
        trajectory_rows.append(
            {
                **dict(identity),
                "method": method,
                "logical_trajectory_id": sha256_text(
                    canonical_json_text(
                        {"change_id": change_id, **dict(identity), "method": method}
                    )
                )[:24],
                "round_count": global_epochs,
                "update_bank_reuse_count": len(method_keys),
                "update_bank_chain_sha256": sha256_text(canonical_json_text(bank_hashes)),
                "metrics": {
                    **metrics,
                    "clean_test_accuracy": clean_accuracy,
                    "clean_test_loss": clean_loss,
                    "triggered_ASR": triggered_asr,
                },
                "method_details": details,
            }
        )
        del aggregate
    return {
        "schema_version": "sevc-tdsc-comparative-source-unit-v1",
        "change_id": change_id,
        "status": "COMPLETE",
        "identity": dict(identity),
        "device": device,
        "partition_set_sha256": next(
            row["partition_set_sha256"]
            for row in config["_partition_lock"]["partition_sets"]
            if int(row["formal_seed"]) == seed
            and int(row["trainer_count"]) == trainer_count
            and str(row["mode"]) == regime
        ),
        "attacker_order": list(attack_order),
        "attacked_trainers": sorted(attacked),
        "behavior": list(behavior),
        "trajectory_rows": trajectory_rows,
        "trainer_decision_rows": decision_rows,
        "wall_seconds": time.perf_counter() - started,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
