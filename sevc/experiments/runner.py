"""The single registry-dispatched composition root for SEVC experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .registry import EXPERIMENT_RUNNERS, ensure_default_experiment_runners


def run_experiment(
    config_path: Path,
    output_root: Path,
    *,
    profile: str | None = None,
    max_units: int | None = None,
    action: str = "execute",
    worker_id: str | None = None,
    unit_manifest_path: Path | None = None,
    shard_output_root: Path | None = None,
    physical_gpu_id: int | None = None,
    schedule_path: Path | None = None,
    merge_provenance_policy_path: Path | None = None,
    runtime_estimates_path: Path | None = None,
    resource_class_policy_path: Path | None = None,
    source_schedule_path: Path | None = None,
    host_width: int = 2,
    packed_end_to_end_allowed: bool = False,
    authorization_only: bool = False,
    authorization_receipt_path: Path | None = None,
) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[2]
    config = json.loads(config_path.read_text(encoding="utf-8"))
    change_id = str(config["change_id"])
    experiment_key = str(config.get("experiment_key", change_id))
    if profile is not None:
        config["_runtime_profile"] = profile
    if max_units is not None:
        config["_runtime_max_units"] = max_units
    if action != "execute":
        config["_runtime_action"] = action
    if worker_id is not None:
        config["_runtime_worker_id"] = worker_id
    if unit_manifest_path is not None:
        config["_runtime_unit_manifest_path"] = str(unit_manifest_path)
    if shard_output_root is not None:
        config["_runtime_shard_output_root"] = str(shard_output_root)
    if physical_gpu_id is not None:
        config["_runtime_physical_gpu_id"] = int(physical_gpu_id)
    if schedule_path is not None:
        config["_runtime_schedule_path"] = str(schedule_path)
    if merge_provenance_policy_path is not None:
        config["_runtime_merge_provenance_policy_path"] = str(
            merge_provenance_policy_path
        )
    if runtime_estimates_path is not None:
        config["_runtime_estimates_path"] = str(runtime_estimates_path)
    if resource_class_policy_path is not None:
        config["_runtime_resource_class_policy_path"] = str(
            resource_class_policy_path
        )
    if source_schedule_path is not None:
        config["_runtime_source_schedule_path"] = str(source_schedule_path)
    if host_width != 2:
        config["_runtime_host_width"] = int(host_width)
    if packed_end_to_end_allowed:
        config["_runtime_packed_end_to_end_allowed"] = True
    ensure_default_experiment_runners()
    runner = EXPERIMENT_RUNNERS.get(experiment_key)
    if authorization_only or authorization_receipt_path is not None:
        return runner(
            repo_root,
            config_path,
            output_root,
            config,
            authorization_only=authorization_only,
            authorization_receipt_path=authorization_receipt_path,
        )
    return runner(repo_root, config_path, output_root, config)
