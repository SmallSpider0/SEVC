"""Shortcut-majority recombination of recorded DePoL native reports.

v0 is the deviating verifier and v1/v2 are honest in every recorded unit
(`sevc/verification/depol_local.py`).  Two verifiers that independently use the
same deterministic shortcut produce byte-identical reports, so a
shortcut-majority committee is obtained by duplicating v0's report.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from sevc.committee.depol_arbitration import arbitrate_digests, arbitrate_distances
from sevc.core.artifacts import capture_environment, ensure_experiment_output_root, sha256_file, write_json
from sevc.verification.reference_acquisition import commitment

CHANGE_ID = "experiment-tdsc-depol-shared-shortcut-v1"
METHOD = "depol-local-verification-arbitration-v1"
DETERMINISTIC = ("constant-accept", "constant-reject")
COMBINATIONS = {"C2": ("A", "A", "H"), "C3": ("A", "A", "A")}


def reveals(native: Mapping[str, Any]) -> dict[str, Any]:
    commits, opened = {}, {}
    for event in native["native_commitment_events"]:
        if event["phase"] == "COMMITTED":
            commits[event["party"]] = event["commitment"]
        elif event["phase"] == "REVEALED":
            if commitment([event["party"], event["report"]]) != commits[event["party"]]:
                raise ValueError("revealed report does not open its commitment")
            opened[event["party"]] = event["report"]
    return opened


def sampling_order(native: Mapping[str, Any]) -> list[int]:
    return next(e["sampled_interval_indices"] for e in native["protocol_events"] if e["phase"] == "GROUP_SAMPLING")


def _needs_slow(fast: Mapping[str, Any]) -> bool:
    return not fast["trainer_verdict"] or not all(fast["verifier_reward_eligible"])


def reproduce(native: Mapping[str, Any]) -> None:
    opened = reveals(native)
    order = sampling_order(native)
    trainer = [opened["trainer"][i] for i in order]
    fast = arbitrate_digests(trainer, [opened[f"v{i}"] for i in range(3)])
    if fast != native["fast"]:
        raise ValueError("recorded fast arbitration is not reproduced")
    if _needs_slow(fast) != (native["slow"] is not None):
        raise ValueError("slow-path trigger is not reproduced")
    if native["slow"] is not None:
        slow = arbitrate_distances(native["slow"]["distance_matrices"], native["epsilon"])
        for key in ("trainer_verdict", "verifier_reward_eligible"):
            if slow[key] != native["final"][key]:
                raise ValueError("recorded slow arbitration is not reproduced")


def combine_matrices(matrices: Sequence[Sequence[Sequence[float]]], parties: Sequence[str]) -> list[list[list[float]]]:
    """Rebuild per-interval distance rows [trainer, p1, p2, p3] for A=v0 / H=v1 parties.

    Recorded rows are [d(v,trainer), d(v,v0), d(v,v1), d(v,v2)].  Copies of one
    party are at distance 0 from each other.
    """
    source = {"A": 0, "H": 1}
    column = {"A": 1, "H": 2}
    combined = []
    for party in parties:
        rows = []
        for row in matrices[source[party]]:
            rows.append([row[0]] + [0.0 if other == party else row[column[other]] for other in parties])
        combined.append(rows)
    return combined


def arbitrate_combination(native: Mapping[str, Any], parties: Sequence[str]) -> dict[str, Any]:
    opened = reveals(native)
    order = sampling_order(native)
    trainer = [opened["trainer"][i] for i in order]
    report = {"A": opened["v0"], "H": opened["v1"]}
    fast = arbitrate_digests(trainer, [report[p] for p in parties])
    if not _needs_slow(fast):
        final, path = fast, "fast"
    elif native["slow"] is None:
        return {"status": "NOT_RECONSTRUCTIBLE", "path": "slow"}
    else:
        final = arbitrate_distances(combine_matrices(native["slow"]["distance_matrices"], parties), native["epsilon"])
        path = "slow"
    eligible = final["verifier_reward_eligible"]
    return {
        "status": "RECONSTRUCTED",
        "path": path,
        "trainer_verdict": final["trainer_verdict"],
        "deviator_eligible": [bool(e) for e, p in zip(eligible, parties) if p == "A"],
        "honest_eligible": [bool(e) for e, p in zip(eligible, parties) if p == "H"],
    }


def analyse_unit(unit: Mapping[str, Any]) -> dict[str, Any]:
    native = unit["native"]
    reproduce(native)
    row = {
        "unit_id": unit.get("unit_id"), "dataset": unit["dataset"], "invalid": bool(unit["invalid"]),
        "behavior": unit["behavior"], "granularity": native.get("interval_granularity") or "segment-endpoint",
        "recorded_C1": {"trainer_verdict": native["final"]["trainer_verdict"],
                        "verifier_reward_eligible": native["final"]["verifier_reward_eligible"]},
    }
    if unit["behavior"] in DETERMINISTIC:
        for name, parties in COMBINATIONS.items():
            result = arbitrate_combination(native, parties)
            if result["status"] == "RECONSTRUCTED":
                result["trainer_verdict_correct"] = result["trainer_verdict"] == (not row["invalid"])
            row[name] = result
    return row


def _load(root: Path) -> tuple[list[dict[str, Any]], str]:
    paths = sorted((root / "units").glob("*.json"))
    digest = hashlib.sha256()
    units = []
    for path in paths:
        data = path.read_bytes()
        unit = json.loads(data)
        if unit.get("method") == METHOD and unit.get("phase") == "native":
            digest.update(data)
            units.append(unit)
    return units, digest.hexdigest()


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for row in rows:
        if row["behavior"] not in DETERMINISTIC:
            continue
        for name in COMBINATIONS:
            r = row[name]
            key = f"{row['granularity']}|{row['dataset']}|{name}"
            cell = out.setdefault(key, {"units": 0, "reconstructed": 0, "deviators_all_eligible": 0,
                                        "honest_all_ineligible": 0, "trainer_verdict_correct": 0})
            cell["units"] += 1
            if r["status"] != "RECONSTRUCTED":
                continue
            cell["reconstructed"] += 1
            cell["deviators_all_eligible"] += all(r["deviator_eligible"])
            cell["honest_all_ineligible"] += (not any(r["honest_eligible"])) if r["honest_eligible"] else 0
            cell["trainer_verdict_correct"] += r["trainer_verdict_correct"]
    return out


def decide(rows: Sequence[Mapping[str, Any]], datasets: Sequence[str]) -> dict[str, Any]:
    primary = [r for r in rows if r["granularity"] == "per-step" and r["behavior"] in DETERMINISTIC]
    c2 = [r["C2"] for r in primary if r["C2"]["status"] == "RECONSTRUCTED"]
    covered = {r["dataset"] for r in primary if r["C2"]["status"] == "RECONSTRUCTED"}
    supported = (bool(c2) and covered == set(datasets)
                 and all(all(x["deviator_eligible"]) and not any(x["honest_eligible"]) for x in c2))
    return {"verdict": "PASS" if supported else "ACCEPTED_NEGATIVE", "routing": "PAPER_CHANGE_REQUIRED",
            "primary_C2_reconstructed": len(c2), "primary_datasets_covered": sorted(covered)}


def run(config_path: Path, output_root: Path, *, repo_root: Path, command: Sequence[str]) -> dict[str, Any]:
    config = json.loads(config_path.read_text())
    out = ensure_experiment_output_root(output_root, repo_root, CHANGE_ID)
    if any(out.iterdir()):
        raise FileExistsError("output root must be empty")
    rows, inputs = [], {}
    for name, spec in config["inputs"].items():
        units, digest = _load(Path(spec["root"]))
        if digest != spec["concat_sha256"] or len(units) != spec["units"]:
            raise ValueError(f"input identity mismatch: {name}")
        inputs[name] = {"root": spec["root"], "concat_sha256": digest, "units": len(units)}
        rows.extend(analyse_unit(u) for u in units)
    summary = summarize(rows)
    decision = decide(rows, config["datasets"])
    write_json(out / "per-unit.json", rows)
    write_json(out / "summary.json", {"change_id": CHANGE_ID, "summary": summary, "decision": decision})
    code = [Path(__file__), repo_root / "scripts/run_depol_shared_shortcut.py", config_path,
            repo_root / "sevc/committee/depol_arbitration.py"]
    write_json(out / "manifest.json", {
        "change_id": CHANGE_ID, "command": list(command), "inputs": inputs,
        "code_sha256": {str(p.resolve().relative_to(repo_root.resolve())): sha256_file(p) for p in code},
        "outputs": {n: sha256_file(out / n) for n in ("per-unit.json", "summary.json")},
        "environment": capture_environment(),
    })
    return decision
