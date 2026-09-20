"""Replay-tolerance characterization across numerical configurations on one host.

Honest four-step segments and registered gradient-continuation challenges are
produced under the reference configuration (the trainer's commitment) and then
replayed under each registered numerical configuration.  Training, challenge and
replay use the canonical primitives only.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import random
import statistics
import time
from typing import Any, Callable, Iterable, Mapping, Sequence

CHANGE_ID = "experiment-tdsc-replay-tolerance-heterogeneity-v1"
EXPERIMENT_KEY = "tdsc-replay-tolerance-heterogeneity-v1"
TOLERANCE = 1e-5
INVALID_SHIFT = 4e-5
CHALLENGE_SCALE = 0.99
STEPS, BATCH, PREFIX_STEPS = 4, 32, 64
ACTIVATION_KEYS = ("protocol_lock_path", "protocol_lock_sha256", "confirmation_receipt_path",
                   "formal_execution_authorized")

# name -> (device, deterministic, cudnn_benchmark, cudnn_deterministic, matmul_tf32, cudnn_tf32)
NUMERIC_CONFIGS: dict[str, tuple[str, bool, bool, bool, bool, bool]] = {
    "C0-reference": ("cuda:0", True, False, True, False, False),
    "C1-nondeterministic-autotune": ("cuda:0", False, True, False, False, False),
    "C2-tf32": ("cuda:0", True, False, True, True, True),
    "C3-pytorch-defaults": ("cuda:0", False, False, False, False, True),
    "C4-cpu": ("cpu", True, False, True, False, False),
}


def _flags(torch) -> dict[str, Any]:
    return {"deterministic": torch.are_deterministic_algorithms_enabled(),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_tf32": torch.backends.cudnn.allow_tf32,
            "cpu_threads": torch.get_num_threads()}


@contextmanager
def apply_numeric_config(name: str):
    """Set the registered torch numeric flags, yield the device, then restore."""
    import torch
    device, det, bench, cdet, mtf32, ctf32 = NUMERIC_CONFIGS[name]
    before = _flags(torch)
    torch.use_deterministic_algorithms(det, warn_only=False)
    torch.backends.cudnn.benchmark = bench
    torch.backends.cudnn.deterministic = cdet
    torch.backends.cuda.matmul.allow_tf32 = mtf32
    torch.backends.cudnn.allow_tf32 = ctf32
    if device == "cpu":
        torch.set_num_threads(1)
    try:
        yield device, _flags(torch)
    finally:
        torch.use_deterministic_algorithms(before["deterministic"], warn_only=False)
        torch.backends.cudnn.benchmark = before["cudnn_benchmark"]
        torch.backends.cudnn.deterministic = before["cudnn_deterministic"]
        torch.backends.cuda.matmul.allow_tf32 = before["matmul_tf32"]
        torch.backends.cudnn.allow_tf32 = before["cudnn_tf32"]
        torch.set_num_threads(before["cpu_threads"])


def sample_row(context, *, dataset: str, seed: int, partition: Sequence[int], count: int, tag: str) -> dict:
    rng = random.Random(int(hashlib.sha256(f"{CHANGE_ID}|{dataset}|{seed}|{tag}".encode()).hexdigest(), 16))
    indices = rng.sample(range(int(partition[0]), int(partition[1])), count)
    labels = [int(context.train[i][1]) for i in indices]
    return {"source_seed": seed % (2 ** 32), "sample_indices": indices, "sample_labels": labels,
            "steps": count // BATCH, "batch_size": BATCH}


def mid_anchor(context, row: Mapping[str, Any]):
    """Train the registered prefix and return the final model and momentum state."""
    import torch
    from sevc.core.runtime import set_global_seed
    from sevc.training import WorkerBehavior, produce_worker_update
    set_global_seed(int(row["source_seed"]))
    samples = [context.train[i] for i in row["sample_indices"]]
    batches = tuple((torch.stack([x for x, _ in samples[i:i + BATCH]]),
                     torch.tensor(row["sample_labels"][i:i + BATCH], dtype=torch.long))
                    for i in range(0, len(samples), BATCH))
    last: dict[str, Any] = {}

    def keep(step, model, momentum):
        last.update(step=step, state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                    momentum={k: v.detach().cpu().clone() for k, v in momentum.items()})

    produce_worker_update(context.factory().to(context.device.name), context.factory, batches,
                          WorkerBehavior.NORMAL, device=context.device.name, learning_rate=.01,
                          momentum=.9, max_batches=PREFIX_STEPS, milestone_callback=keep,
                          milestone_steps=(PREFIX_STEPS,))
    if last.get("step") != PREFIX_STEPS:
        raise RuntimeError("prefix milestone was not captured")
    return last["state"], last["momentum"]


def build_segments(context, *, dataset: str, seed: int, anchor: str, partition: Sequence[int]):
    """Return (honest proof, [(index, challenge proof)], identity) under the reference config."""
    from sevc.training.replay_sources import materialize_branch_source
    from sevc.verification.public_replay_shortcuts import gradient_continuation_challenge
    from sevc.verification.replay_coupled_probes import proof_component_hashes
    initial = momentum = None
    prefix_row = None
    if anchor == "mid":
        prefix_row = sample_row(context, dataset=dataset, seed=seed, partition=partition,
                                count=PREFIX_STEPS * BATCH, tag="prefix")
        initial, momentum = mid_anchor(context, prefix_row)
    row = sample_row(context, dataset=dataset, seed=seed, partition=partition, count=STEPS * BATCH,
                     tag=f"segment-{anchor}")
    honest = materialize_branch_source(context, row, initial_state=initial, initial_momentum=momentum)
    model = context.factory()
    challenges = []
    for index in range(STEPS):
        proof, _ = gradient_continuation_challenge(honest, model, checkpoint_index=index,
                                                   scale=CHALLENGE_SCALE, device=context.device.name)
        challenges.append((index, proof))
    identity = {"dataset": dataset, "seed": seed, "anchor": anchor,
                "sample_indices_sha256": hashlib.sha256(json.dumps(row["sample_indices"]).encode()).hexdigest(),
                "prefix_indices_sha256": None if prefix_row is None else hashlib.sha256(
                    json.dumps(prefix_row["sample_indices"]).encode()).hexdigest(),
                "honest": proof_component_hashes(honest),
                "challenges": {str(i): proof_component_hashes(p) for i, p in challenges}}
    return honest, challenges, identity


def replay_rows(context, honest, challenges, *, base: Mapping[str, Any], configs: Iterable[str],
                applied: dict | None = None) -> list[dict]:
    from sevc.training.engine import verify_replay_proof
    rows = []
    for name in configs:
        with apply_numeric_config(name) as (device, flags):
            if applied is not None:
                applied.setdefault(name, flags)
            for kind, index, proof in [("honest", None, honest)] + [("challenge", i, p) for i, p in challenges]:
                started = time.perf_counter()
                result = verify_replay_proof(proof, context.factory, device=device, tolerance=TOLERANCE,
                                             comparison_device="cpu")
                rows.append({**base, "config": name, "kind": kind, "challenge_index": index,
                             "max_abs_difference": float(result["max_abs_difference"]),
                             "max_optimizer_abs_difference": float(result.get("max_optimizer_abs_difference", 0.0)),
                             "passed": bool(result["passed"]), "state_complete": bool(result.get("state_complete", False)),
                             "seconds": time.perf_counter() - started})
    return rows


def summarize(rows: Sequence[Mapping[str, Any]], *, tolerance: float = TOLERANCE,
              invalid_shift: float = INVALID_SHIFT) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for dataset in sorted({r["dataset"] for r in rows}):
        per = {}
        for name in NUMERIC_CONFIGS:
            honest = [r["max_abs_difference"] for r in rows if r["dataset"] == dataset
                      and r["config"] == name and r["kind"] == "honest"]
            challenge = [r["max_abs_difference"] for r in rows if r["dataset"] == dataset
                         and r["config"] == name and r["kind"] == "challenge"]
            if not honest or not challenge:
                continue
            h_max, c_min = max(honest), min(challenge)
            per[name] = {
                "honest_n": len(honest), "honest_max": h_max, "honest_median": statistics.median(honest),
                "honest_optimizer_max": max(r["max_optimizer_abs_difference"] for r in rows
                                            if r["dataset"] == dataset and r["config"] == name and r["kind"] == "honest"),
                "honest_false_rejections_at_tolerance": sum(x > tolerance for x in honest),
                "challenge_n": len(challenge), "challenge_min": c_min,
                "challenge_misses_at_tolerance": sum(x <= tolerance for x in challenge),
                "separating_interval": [h_max, c_min] if h_max < c_min else None,
                "tolerance_separates": h_max <= tolerance < c_min,
                "invalid_shift_detectable_at_tolerance": h_max < tolerance <= invalid_shift - h_max,
                "invalid_shift_interval_nonempty": h_max < invalid_shift - h_max,
            }
        out[dataset] = per
    return out


def scientific_configuration_sha256(config: Mapping[str, Any]) -> str:
    body = {k: v for k, v in config.items() if k not in ACTIVATION_KEYS and not k.startswith("_runtime")}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def validate_activation(config: Mapping[str, Any], profile: Mapping[str, Any], output_root) -> dict:
    """Fail before output creation, data access or a CUDA context."""
    from sevc.core.artifacts import sha256_file
    if config.get("change_id") != CHANGE_ID or not config.get("formal_execution_authorized") \
            or profile.get("formal") is not True:
        raise PermissionError("replay-tolerance formal authorization absent")
    lock_path = Path(config["protocol_lock_path"])
    if sha256_file(lock_path) != config["protocol_lock_sha256"]:
        raise PermissionError("protocol lock drifted")
    lock = json.loads(lock_path.read_text())
    if (lock.get("execution_ready") is not True
            or lock.get("scientific_configuration_sha256") != scientific_configuration_sha256(config)
            or lock.get("output_root") != str(output_root) or profile.get("device") != "cuda:0"
            or profile.get("gpu_uuid") != lock.get("gpu_uuid")
            or set(config["datasets"]) != {"mnist", "cifar10", "cifar100"}
            or config["maximum_runner_wall_seconds"] > 4 * 3600):
        raise PermissionError("replay-tolerance lock identity invalid")
    receipt = json.loads(Path(config["confirmation_receipt_path"]).read_text())
    if (receipt.get("status") != "AUTHOR_CONFIRMED" or not receipt.get("author_message")
            or receipt.get("protocol_sha256") != config["protocol_lock_sha256"]
            or receipt.get("output_root") != str(output_root)):
        raise PermissionError("activation receipt does not bind this protocol")
    return lock


def _live_gpu_uuid() -> str:
    import subprocess
    out = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True)
    return dict(line.split(", ") for line in out.strip().splitlines())["0"]


def run(repo_root: Path, config_path: Path, output_root: Path, config: Mapping[str, Any],
        *, context_factory: Callable | None = None) -> dict[str, Any]:
    from sevc.core.artifacts import sha256_file, write_json
    profile_name = config.get("_runtime_profile", config["default_profile"])
    profile = config["profiles"][profile_name]
    output_root = Path(output_root)
    if profile.get("formal"):
        lock = validate_activation(config, profile, output_root)
        if _live_gpu_uuid() != lock["gpu_uuid"]:
            raise PermissionError("live GPU identity differs from the protocol lock")
    if output_root.exists():
        raise FileExistsError("output root must not exist")
    output_root.mkdir(parents=True)
    write_json(output_root / "status.json", {"status": "RUNNING", "change_id": CHANGE_ID})
    import torch
    from sevc.experiments.tdsc_five_rq_evidence import ReplayDevice
    from sevc.training.replay_sources import ReplayDatasetContext
    started = time.monotonic()
    applied: dict[str, Any] = {}
    rows_path, sources_path = output_root / "rows.jsonl", output_root / "sources.jsonl"
    all_rows: list[dict] = []
    for dataset in config["datasets"]:
        if context_factory is not None:
            context = context_factory(dataset)
        else:
            context = ReplayDatasetContext(dataset, config["dataset_specs"][dataset],
                                           Path(profile["data_root"]), ReplayDevice(profile["device"]))
        for seed in config["block_seeds"][dataset]:
            for anchor in config["anchors"]:
                with apply_numeric_config("C0-reference"):
                    honest, challenges, identity = build_segments(
                        context, dataset=dataset, seed=int(seed), anchor=anchor,
                        partition=config["partitions"][dataset])
                with sources_path.open("a") as handle:
                    handle.write(json.dumps(identity, sort_keys=True) + "\n")
                rows = replay_rows(context, honest, challenges,
                                   base={"dataset": dataset, "seed": int(seed), "anchor": anchor},
                                   configs=config["numeric_configs"], applied=applied)
                with rows_path.open("a") as handle:
                    for row in rows:
                        handle.write(json.dumps(row, sort_keys=True) + "\n")
                all_rows.extend(rows)
                del honest, challenges
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    expected = sum(len(config["block_seeds"][d]) for d in config["datasets"]) * len(config["anchors"]) \
        * (1 + STEPS) * len(config["numeric_configs"])
    complete = len(all_rows) == expected and all(r["state_complete"] for r in all_rows)
    write_json(output_root / "configs-applied.json", {
        "flags": applied, "cudnn_version": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None})
    summary = {"change_id": CHANGE_ID, "rows": len(all_rows), "expected_rows": expected,
               "complete": complete, "tolerance": TOLERANCE, "invalid_shift": INVALID_SHIFT,
               "wall_seconds": time.monotonic() - started, "summary": summarize(all_rows)}
    write_json(output_root / "summary.json", summary)
    files = sorted(p for p in output_root.iterdir() if p.is_file() and p.name != "result_index.json")
    write_json(output_root / "result_index.json", {p.name: sha256_file(p) for p in files})
    write_json(output_root / "status.json", {"status": "COMPLETE" if complete else "INCOMPLETE",
                                             "change_id": CHANGE_ID,
                                             "config_sha256": sha256_file(Path(config_path))})
    if not complete:
        raise RuntimeError("replay-tolerance run incomplete")
    return {"status": "COMPLETE", "rows": len(all_rows)}
