"""Zero-replay distinguishers over the verifier-visible RCMP task view.

Adversaries receive only fields a verifier can observe (content address,
start-state identity, payload size) and the commitment set visible under a
stated model.  Roles, source hashes and invalid markers are used only by the
scorer.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import gzip
import hashlib
import hmac
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from sevc.core.artifacts import (
    canonical_json_text,
    capture_environment,
    ensure_experiment_output_root,
    sha256_file,
    write_json,
)
from sevc.core.statistical_bounds import clopper_pearson_bound

CHANGE_ID = "experiment-tdsc-verifier-view-linkability-v1"
ADVERSARIES = ("A1-open", "A1-hiding", "A2", "A3")
DECISION_ADVERSARIES = ("A1-hiding", "A2", "A3")
ALPHA = 0.05


@dataclass(frozen=True)
class VisibleTask:
    content_address: str
    start_state: str
    payload_bytes: int


def visible_task(task: Mapping[str, Any]) -> VisibleTask:
    wrapped = task["wrapped_hashes"]
    return VisibleTask(str(wrapped["proof_sha256"]), str(wrapped["checkpoint_sha256"]),
                       int(task["payload_bytes"]))


def hiding_commitment(salt_key: bytes, proof_sha256: str) -> str:
    """Salted commitment opened only to the owner."""
    return hmac.new(salt_key, proof_sha256.encode(), hashlib.sha256).hexdigest()


def adversary_commitment_match(view: Sequence[VisibleTask], visible_commitments: frozenset[str]) -> frozenset[int]:
    """A1: flag tasks whose content address is not a visible commitment."""
    return frozenset(i for i, t in enumerate(view) if t.content_address not in visible_commitments)


def adversary_start_state_repeat(view: Sequence[VisibleTask]) -> frozenset[int]:
    """A2: flag tasks whose start-state identity repeats within the task set."""
    counts = Counter(t.start_state for t in view)
    return frozenset(i for i, t in enumerate(view) if counts[t.start_state] > 1)


def adversary_payload_outlier(view: Sequence[VisibleTask]) -> frozenset[int]:
    """A3: flag tasks whose payload size differs from the task-set mode.

    With tied modes every value except the most frequent ones is flagged; if all
    values tie or all tasks share one value nothing is flagged.
    """
    counts = Counter(t.payload_bytes for t in view)
    top = max(counts.values())
    modes = {value for value, count in counts.items() if count == top}
    if len(modes) == len(counts):
        return frozenset()
    return frozenset(i for i, t in enumerate(view) if t.payload_bytes not in modes)


def score_task_set(flags: frozenset[int], roles: Sequence[str], invalid_index: int | None) -> dict[str, Any]:
    """Score the strategy: reject flagged tasks, accept every other task."""
    challenges = {i for i, r in enumerate(roles) if r == "challenge"}
    controls = {i for i, r in enumerate(roles) if r == "control"}
    probe_pass = challenges <= flags and not (controls & flags)
    return {
        "flagged": len(flags),
        "flagged_challenges": len(flags & challenges),
        "challenges": len(challenges),
        "probe_pass": probe_pass,
        "incorrect_admission": bool(probe_pass and invalid_index is not None and invalid_index not in flags),
    }


def _load_units(root: Path, method: str, phase: str) -> list[dict[str, Any]]:
    units = []
    for path in sorted((root / "units").glob("*.json")):
        row = json.loads(path.read_text())
        if row.get("method") == method and row.get("phase") == phase:
            units.append(row)
    return units


def _load_groups(root: Path) -> dict[str, list[dict[str, Any]]]:
    groups = {}
    with gzip.open(root / "source-task-identities.jsonl.gz", "rt") as handle:
        for line in handle:
            row = json.loads(line)
            if "group_id" in row and "kind" not in row:
                groups[row["group_id"]] = row["sources"]
    return groups


def analyse_unit(unit: Mapping[str, Any], sources: Sequence[Mapping[str, Any]], salt_key: bytes) -> dict[str, Any]:
    tasks = unit["tasks"]
    view = [visible_task(t) for t in tasks]
    roles = [str(t["role"]) for t in tasks]
    mutated = {str(s["proof_sha256"]) for s in sources if s.get("trainer_mutation") is not None}
    invalid = [i for i, t in enumerate(tasks) if t["role"] == "production" and t["source_sha256"] in mutated]
    if len(invalid) > 1:
        raise ValueError("more than one invalid production task in a task set")
    invalid_index = invalid[0] if invalid else None
    if bool(unit["invalid"]) != (invalid_index is not None):
        raise ValueError("invalid marker disagrees with committed trainer mutation")
    open_commitments = frozenset(str(s["proof_sha256"]) for s in sources)
    hidden_commitments = frozenset(hiding_commitment(salt_key, c) for c in open_commitments)
    checkpoint_commitments = frozenset(str(s["checkpoint_sha256"]) for s in sources)
    flags = {
        "A1-open": adversary_commitment_match(view, open_commitments),
        "A1-hiding": adversary_commitment_match(view, hidden_commitments),
        "A2": adversary_start_state_repeat(view),
        "A3": adversary_payload_outlier(view),
    }
    return {
        "unit_id": unit.get("unit_id"),
        "dataset": unit["dataset"],
        "block": unit["block"],
        "context": unit["context"],
        "invalid": invalid_index is not None,
        "start_state_not_in_open_commitments": sum(t.start_state not in checkpoint_commitments for t in view),
        "scores": {name: score_task_set(f, roles, invalid_index) for name, f in flags.items()},
    }


def aggregate(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    by_dataset: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_dataset[row["dataset"]].append(row)
    summary: dict[str, Any] = {}
    for dataset, items in sorted(by_dataset.items()):
        blocks = sorted({r["block"] for r in items}, key=str)
        invalid_items = [r for r in items if r["invalid"]]
        invalid_blocks = sorted({r["block"] for r in invalid_items}, key=str)
        per: dict[str, Any] = {}
        for name in ADVERSARIES:
            s = [r["scores"][name] for r in items]
            flagged = sum(x["flagged"] for x in s)
            hits = sum(x["flagged_challenges"] for x in s)
            total_challenges = sum(x["challenges"] for x in s)
            pass_blocks = {r["block"] for r in items if r["scores"][name]["probe_pass"]}
            wrong_blocks = {r["block"] for r in invalid_items if r["scores"][name]["incorrect_admission"]}
            precision_lower = (clopper_pearson_bound(hits, flagged, side="lower", alpha=ALPHA)
                               if flagged else 0.0)
            per[name] = {
                "task_sets": len(s),
                "flagged_tasks": flagged,
                "challenge_precision": hits / flagged if flagged else None,
                "challenge_precision_lower95": precision_lower,
                "challenge_recall": hits / total_challenges,
                "probe_pass_rate": sum(x["probe_pass"] for x in s) / len(s),
                "probe_pass_task_sets": sum(x["probe_pass"] for x in s),
                "probe_pass_blocks": len(pass_blocks),
                "blocks": len(blocks),
                "probe_pass_block_upper95": clopper_pearson_bound(len(pass_blocks), len(blocks), side="upper", alpha=ALPHA),
                "incorrect_admission_rate": (sum(r["scores"][name]["incorrect_admission"] for r in invalid_items)
                                             / len(invalid_items)) if invalid_items else None,
                "incorrect_admission_blocks": len(wrong_blocks),
                "invalid_blocks": len(invalid_blocks),
                "effective": bool(sum(x["probe_pass"] for x in s) > 0 or precision_lower > 0.1),
            }
        summary[dataset] = per
    return summary


def decide(summary: Mapping[str, Any]) -> dict[str, Any]:
    effective = sorted((d, a) for d, per in summary.items() for a in DECISION_ADVERSARIES if per[a]["effective"])
    return {
        "verdict": "ACCEPTED_NEGATIVE" if effective else "PASS",
        "effective_under_adopted_model": [list(x) for x in effective],
        "counterfactual_A1_open_effective": sorted(d for d, per in summary.items() if per["A1-open"]["effective"]),
        "routing": "PAPER_CHANGE_REQUIRED",
    }


def run(config_path: Path, output_root: Path, *, repo_root: Path, command: Sequence[str]) -> dict[str, Any]:
    config = json.loads(config_path.read_text())
    root = Path(config["evidence_root"])
    for name, expected in config["input_sha256"].items():
        if sha256_file(root / name) != expected:
            raise ValueError(f"input hash mismatch: {name}")
    out = ensure_experiment_output_root(output_root, repo_root, CHANGE_ID)
    if any(out.iterdir()):
        raise FileExistsError("output root must be empty")
    units = _load_units(root, config["method"], config["phase"])
    groups = _load_groups(root)
    expected = config["expected_counts"]
    task_count = sum(len(u["tasks"]) for u in units)
    per_dataset = Counter(u["dataset"] for u in units)
    if (len(units) != expected["task_sets"] or task_count != expected["tasks"]
            or any(per_dataset[d] != expected["task_sets_per_dataset"] for d in config["datasets"])):
        raise ValueError("input counts disagree with the frozen contract")
    salt_key = hashlib.sha256(config["hiding_salt_domain"].encode()).digest()
    rows = [analyse_unit(u, groups[u["source_group"]], salt_key) for u in units]
    summary = aggregate(rows)
    decision = decide(summary)
    with gzip.open(out / "per-task-set.jsonl.gz", "wt") as handle:
        for row in rows:
            handle.write(canonical_json_text(row) + "\n")
    write_json(out / "summary.json", {"change_id": CHANGE_ID, "summary": summary, "decision": decision})
    code = [Path(__file__), repo_root / "scripts/run_verifier_view_linkability.py", config_path]
    write_json(out / "manifest.json", {
        "change_id": CHANGE_ID,
        "command": list(command),
        "config_sha256": sha256_file(config_path),
        "code_sha256": {str(p.relative_to(repo_root) if p.is_relative_to(repo_root) else p): sha256_file(p) for p in code},
        "inputs": {name: {"path": str(root / name), "sha256": value} for name, value in config["input_sha256"].items()},
        "outputs": {name: sha256_file(out / name) for name in ("per-task-set.jsonl.gz", "summary.json")},
        "environment": capture_environment(),
    })
    return decision


# --- WP1b: envelope-length equalization re-test ---------------------------------

REPAIR_CHANGE_ID = "experiment-tdsc-verifier-view-repair-v1"
# len(',"padding":""'): canonical JSON adds this when an empty padding field is inserted.
EMPTY_PADDING_OVERHEAD = len(',"padding":""')


def equalized_payload_lengths(lengths: Sequence[int]) -> list[int]:
    """Serialized lengths after `equalize_public_envelopes`, derived from recorded lengths."""
    target = max(length + EMPTY_PADDING_OVERHEAD for length in lengths)
    return [target] * len(lengths)


def _repair_row(unit: Mapping[str, Any], sources: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    tasks = unit["tasks"]
    recorded = [int(t["payload_bytes"]) for t in tasks]
    padded = equalized_payload_lengths(recorded)
    view = [VisibleTask(str(t["wrapped_hashes"]["proof_sha256"]), str(t["wrapped_hashes"]["checkpoint_sha256"]), n)
            for t, n in zip(tasks, padded)]
    mutated = {str(s["proof_sha256"]) for s in sources if s.get("trainer_mutation") is not None}
    invalid = [i for i, t in enumerate(tasks) if t["role"] == "production" and t["source_sha256"] in mutated]
    score = score_task_set(adversary_payload_outlier(view), [str(t["role"]) for t in tasks],
                           invalid[0] if invalid else None)
    return {
        "dataset": unit["dataset"], "block": unit["block"], "invalid": bool(invalid),
        "scores": {name: score for name in ADVERSARIES},
        "padding_bytes": [p - r for p, r in zip(padded, recorded)],
    }


def run_repair(config_path: Path, output_root: Path, *, repo_root: Path, command: Sequence[str]) -> dict[str, Any]:
    config = json.loads(config_path.read_text())
    root = Path(config["evidence_root"])
    for name, expected in config["input_sha256"].items():
        if sha256_file(root / name) != expected:
            raise ValueError(f"input hash mismatch: {name}")
    out = ensure_experiment_output_root(output_root, repo_root, REPAIR_CHANGE_ID)
    if any(out.iterdir()):
        raise FileExistsError("output root must be empty")
    units = _load_units(root, config["method"], config["phase"])
    if len(units) != config["expected_counts"]["task_sets"]:
        raise ValueError("input counts disagree with the frozen contract")
    groups = _load_groups(root)
    rows = [_repair_row(u, groups[u["source_group"]]) for u in units]
    a3 = {d: per["A3"] for d, per in aggregate(rows).items()}
    padding: dict[str, Any] = {}
    for dataset in sorted({r["dataset"] for r in rows}):
        values = [b for r in rows if r["dataset"] == dataset for b in r["padding_bytes"]]
        padding[dataset] = {"min": min(values), "max": max(values), "mean": sum(values) / len(values)}
    effective = sorted(d for d, v in a3.items() if v["effective"])
    decision = {"verdict": "ACCEPTED_NEGATIVE" if effective else "PASS", "a3_effective_after_repair": effective,
                "routing": "PAPER_CHANGE_REQUIRED",
                "note": "lengths derived from recorded envelopes; delivery not re-executed"}
    write_json(out / "summary.json", {"change_id": REPAIR_CHANGE_ID, "a3_after_repair": a3,
                                      "padding_bytes_per_task": padding, "decision": decision})
    code = [Path(__file__), repo_root / "scripts/run_verifier_view_linkability.py", config_path,
            repo_root / "sevc/verification/replay_coupled_probes.py", repo_root / "sevc/verification/on_demand_service.py"]
    write_json(out / "manifest.json", {
        "change_id": REPAIR_CHANGE_ID, "command": list(command),
        "code_sha256": {str(p.resolve().relative_to(repo_root.resolve())): sha256_file(p) for p in code},
        "inputs": {name: {"path": str(root / name), "sha256": value} for name, value in config["input_sha256"].items()},
        "outputs": {"summary.json": sha256_file(out / "summary.json")},
        "environment": capture_environment(),
    })
    return decision
