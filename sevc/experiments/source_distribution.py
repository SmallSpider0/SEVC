"""Portable configuration and CLI for the canonical source-distribution runner.

This module only handles user-owned inputs and execution identity. All training,
verification, recovery, statistics, and auditing remain in their canonical modules.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import subprocess
import uuid

EXPECTED_PROTOCOL_SHA256 = "58ca16f36cb26644207f6c338ae57560b0ad1023abef3310b890364550c68558"
DATASETS = ("mnist", "cifar10", "cifar100")
DATASET_SPECS = {
    "mnist": {"model": "small-mlp", "class_count": 10, "image_size": 28},
    "cifar10": {"model": "resnet18", "class_count": 10, "image_size": 32},
    "cifar100": {"model": "resnet18", "class_count": 100, "image_size": 32},
}


def protocol_digest(protocol):
    return hashlib.sha256(json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def validate_protocol(protocol):
    if protocol_digest(protocol) != EXPECTED_PROTOCOL_SHA256:
        raise ValueError("protocol differs from this source snapshot; create and document a new study version")
    if set(protocol["science"]["datasets"]) != set(DATASETS):
        raise ValueError("all three datasets are required")
    return protocol


def _read_json(path):
    return json.loads(Path(path).read_text())


def _distribution_roots():
    roots = {Path(__file__).resolve().parents[2]}
    if (Path.cwd()/"public-manifest.json").is_file():
        roots.add(Path.cwd().resolve())
    return roots


def _new_json(path, value, *, private=False):
    path = Path(path).resolve()
    if any(path.is_relative_to(root) for root in _distribution_roots()):
        raise ValueError("local run inputs must be created outside the source repository")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600 if private else 0o644)
    with os.fdopen(fd, "w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def validate_calibration(calibration):
    if set(calibration) != set(DATASETS):
        raise ValueError("calibration must cover mnist, cifar10, and cifar100")
    for dataset, row in calibration.items():
        for key in ("cost_unit_seconds", "timeout_seconds", "deadline_seconds", "owner_reserve_units"):
            value = row.get(key)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"missing or invalid calibration: {dataset}.{key}; see docs/inputs.md")
        if row["timeout_seconds"] < 1 or not math.isclose(row["deadline_seconds"], 12 * row["timeout_seconds"]):
            raise ValueError("deadline must equal 12 timeouts; timeout must be at least one second")
        if row["owner_reserve_units"] < 1:
            raise ValueError("owner reserve must be at least one cost unit")
    return calibration


def validate_streams(path):
    streams = _read_json(path)
    if set(streams) != {"roles", "audits"}:
        raise ValueError("private stream file must contain roles and audits")
    for value in streams.values():
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError("each stream must be a newly generated 32-byte hexadecimal value")
        try:
            bytes.fromhex(value)
        except ValueError as error:
            raise ValueError("invalid private stream encoding") from error
    if streams["roles"] == streams["audits"]:
        raise ValueError("independent roles and audits streams are required")


def prepare_config(protocol_path, data_root, scratch_root, calibration_path, streams_path, *, device="cpu"):
    protocol = validate_protocol(_read_json(protocol_path))
    calibration = validate_calibration(_read_json(calibration_path))
    validate_streams(streams_path)
    data_root, scratch_root = Path(data_root).absolute(), Path(scratch_root).absolute()
    if not data_root.is_dir() or not scratch_root.is_dir():
        raise ValueError("data and scratch directories must already exist; data are not downloaded automatically")
    if device not in {"cpu", "cuda:0"}:
        raise ValueError("supported device values are cpu and cuda:0")
    performance = {
        "cpu_threads": 1, "identity_profile": "fused", "comparison_device": "replay",
        "conditional_preparation_reuse": True, "disk_delivery": "always",
        "joint_proof_digest_only": True, "lazy_payload_read": True,
        "owner_compile_device": device, "owner_compile_lanes": 1,
        "resident_delivery_bytes": 0, "reuse_replay_model": True,
        "selective_cuda_timing": device.startswith("cuda"), "state_transform_device": device,
        "streaming_canonical_replay": True, "task_lanes": 1, "tensor_access": "owned-mmap",
        "trim_cpu_arenas": True, "virtual_fault_wait": True,
        "release_preparation_after_last_use": True, "large_workload_lane_cap": 1,
    }
    return {
        "change_id": "experiment-tdsc-bounded-memory-v1", "experiment_key": "tdsc-bounded-memory-v1",
        "source_distribution": True, "original_study_evidence": False,
        "standalone_run_id": str(uuid.uuid4()), "default_profile": "standalone",
        "scoped_candidate": protocol, "private_streams_path": str(Path(streams_path).absolute()),
        "scratch_parent": str(scratch_root), "datasets": DATASET_SPECS,
        "performance": {"portable-serial": performance},
        "profiles": {"standalone": {
            "full_matrix": True, "namespace": protocol["science"]["namespace"],
            "device": device, "performance_key": "portable-serial", "data_root": str(data_root),
            "calibration": calibration, "sample_ranges": {"mnist": [0, 50000], "cifar10": [0, 40000], "cifar100": [0, 40000]},
        }},
    }


def validate_public_run(config, profile, output_root):
    if config.get("source_distribution") is not True or config.get("original_study_evidence") is not False:
        raise ValueError("use the configure command to create a standalone run configuration")
    if config.get("experiment_key") != "tdsc-bounded-memory-v1" or config.get("change_id") != "experiment-tdsc-bounded-memory-v1":
        raise ValueError("unsupported standalone experiment identity")
    uuid.UUID(config["standalone_run_id"])
    validate_protocol(config["scoped_candidate"])
    if profile.get("full_matrix") is not True:
        raise ValueError("the public study runner executes the complete matrix; use synthetic tests for software checks")
    if config["datasets"] != DATASET_SPECS or profile["sample_ranges"] != {"mnist": [0, 50000], "cifar10": [0, 40000], "cifar100": [0, 40000]}:
        raise ValueError("dataset/model or sample-range drift")
    validate_calibration(profile["calibration"])
    validate_streams(config["private_streams_path"])
    for field, value in (("data_root", profile["data_root"]), ("scratch_parent", config["scratch_parent"])):
        if not Path(value).is_absolute() or not Path(value).is_dir():
            raise ValueError(f"{field} must name an existing absolute directory")
    output = Path(output_root)
    if not output.is_absolute() or output.exists() or output.is_relative_to(Path(profile["data_root"])):
        raise ValueError("output must be a new absolute directory outside the input data")
    if any(output.resolve().is_relative_to(root) for root in _distribution_roots()):
        raise ValueError("run output must be outside the source repository")
    perf = config["performance"][profile["performance_key"]]
    if perf.get("cpu_threads", 0) < 1 or perf.get("task_lanes", 0) < 1:
        raise ValueError("invalid execution resource configuration")


def validate_cuda(profile):
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("the standalone GPU runner requires exactly one visible CUDA device")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise ValueError("set CUBLAS_WORKSPACE_CONFIG=:4096:8 before starting Python")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    if "," in visible:
        raise ValueError("select one CUDA device")
    profile["gpu_uuid"] = subprocess.check_output([
        "nvidia-smi", "-i", visible, "--query-gpu=uuid", "--format=csv,noheader"], text=True).strip()


def main(argv=None):
    parser = argparse.ArgumentParser(description="SEVC source-only prototype and experiment interface")
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect", help="validate and describe the scientific protocol without loading data")
    inspect.add_argument("--protocol", type=Path, required=True)
    streams = commands.add_parser("init-streams", help="create private streams for your own new run outside this repository")
    streams.add_argument("--output", type=Path, required=True)
    configure = commands.add_parser("configure", help="bind the protocol to your own data, calibration and private streams")
    for name in ("protocol", "data-root", "scratch-root", "calibration", "private-streams", "output"):
        configure.add_argument("--"+name, type=Path, required=True)
    configure.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    run = commands.add_parser("run", help="execute the complete study using the canonical runner")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "inspect":
            from sevc.evaluation.scoped_campaign_contract import expand_units
            protocol = validate_protocol(_read_json(args.protocol)); rows = expand_units(protocol)
            print(json.dumps({"datasets": list(DATASETS), "logical_units": len(rows),
                "packages": dict(Counter(r["package"] for r in rows)), "contains_results": False}, indent=2))
        elif args.command == "init-streams":
            _new_json(args.output, {"roles": secrets.token_hex(32), "audits": secrets.token_hex(32)}, private=True)
            print("Created private streams. Keep this file outside version control.")
        elif args.command == "configure":
            config = prepare_config(args.protocol, args.data_root, args.scratch_root,
                args.calibration, args.private_streams, device=args.device)
            _new_json(args.output, config, private=True)
            print("Created standalone configuration; no experiment was started.")
        elif args.command == "run":
            from sevc.experiments.runner import run_experiment
            print(json.dumps(run_experiment(args.config, args.output_root), indent=2))
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
