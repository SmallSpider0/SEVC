"""Audit sealed RCMP units for jobs that entered the insufficient-source branch."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from sevc.core.artifacts import capture_environment, ensure_experiment_output_root, sha256_file, write_json

CHANGE_ID = "experiment-tdsc-owner-failure-rule-v1"
LEGACY_STATUS = "INSUFFICIENT_VALID_REFERENCES_SAFE_DEFER"


def entered_branch(unit: Mapping[str, Any]) -> bool:
    preparation = unit.get("preparation") or {}
    return (unit.get("status") == LEGACY_STATUS
            or (isinstance(preparation, Mapping) and preparation.get("status") == LEGACY_STATUS))


def audit_package(root: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    rcmp = 0
    entered, unissued = [], Counter()
    for path in sorted((root / "units").glob("*.json")):
        data = path.read_bytes()
        digest.update(data)
        unit = json.loads(data)
        if not str(unit.get("method", "")).startswith("rcmp-"):
            continue
        rcmp += 1
        if unit.get("issued") is False:
            unissued[str(unit.get("status"))] += 1
        if entered_branch(unit):
            entered.append(unit.get("unit_id", path.stem))
    return {"root": str(root), "units_concat_sha256": digest.hexdigest(), "rcmp_units": rcmp,
            "entered_branch": entered, "unissued_by_status": dict(unissued)}


def run(config_path: Path, output_root: Path, *, repo_root: Path, command: Sequence[str]) -> dict[str, Any]:
    config = json.loads(config_path.read_text())
    out = ensure_experiment_output_root(output_root, repo_root, CHANGE_ID)
    if any(out.iterdir()):
        raise FileExistsError("output root must be empty")
    packages = {name: audit_package(Path(root)) for name, root in config["packages"].items()}
    total = sum(len(p["entered_branch"]) for p in packages.values())
    decision = {"verdict": "PASS" if total == 0 else "ACCEPTED_NEGATIVE", "entered_branch_total": total,
                "rcmp_units_total": sum(p["rcmp_units"] for p in packages.values()),
                "routing": "PAPER_CHANGE_REQUIRED"}
    write_json(out / "summary.json", {"change_id": CHANGE_ID, "packages": packages, "decision": decision})
    code = [Path(__file__), repo_root / "scripts/audit_owner_failure_branch.py", config_path,
            repo_root / "sevc/verification/reference_acquisition.py"]
    write_json(out / "manifest.json", {"change_id": CHANGE_ID, "command": list(command),
        "code_sha256": {str(p.resolve().relative_to(repo_root.resolve())): sha256_file(p) for p in code},
        "outputs": {"summary.json": sha256_file(out / "summary.json")}, "environment": capture_environment()})
    return decision
