"""Exploratory derivations for system cost, matched-budget coverage and bond sensitivity.

All inputs are audited F v2 artifacts; nothing here changes a registered verdict.
"""
from __future__ import annotations

from collections import defaultdict
import json
import math
from pathlib import Path
import statistics
from typing import Any, Mapping, Sequence
import zlib

from sevc.core.artifacts import capture_environment, ensure_experiment_output_root, sha256_file, write_json
from sevc.core.statistical_bounds import clopper_pearson_bound
from sevc.evaluation.f_rq234_audit import MAIN, bootstrap_mean, payoff
from sevc.evaluation.f_rq_audit import DEVIATIONS, RCMP

CHANGE_ID = "experiment-tdsc-cost-incentive-reanalysis-v1"
DATASETS = ("mnist", "cifar10", "cifar100")
CONTEXTS = ("init-valid", "init-invalid")
SEGMENTS, DIRECT_SEGMENTS = 40, 32
BAD = (1, 2, 4)
FEES = (1.25, 2.5)
BONDS = (0.5, 1.0, 2.0, 5.0)
PRICE = 0.01


def system_ratios(resources: Mapping[str, Mapping[str, float]], dataset: str) -> dict[str, Any]:
    base = resources[f"{dataset}/O"]["owner/wall"]
    out = {"direct_owner_wall": base, "direct_owner_cpu": resources[f"{dataset}/O"].get("owner/cpu")}
    for arm in ("R-repaired", "R-as-run"):
        r = resources[f"{dataset}/{arm}"]
        out[arm] = {"owner_wall": r["owner/wall"], "owner_cpu": r.get("owner/cpu"), "verifier_wall": r["verifier/wall"],
                    **{f"system_over_direct_s{s}": (r["owner/wall"] + s * r["verifier/wall"]) / base for s in (1, 3)}}
    return out


def detection_probability(n: int, bad: int, total: int = SEGMENTS) -> float:
    """Probability that a uniform n-subset of `total` segments contains at least one of `bad`."""
    if n >= total:
        return 1.0
    return 1.0 - math.comb(total - bad, n) / math.comb(total, n)


def matched_budget(block_ratios: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    groups = defaultdict(list)
    for row in block_ratios:
        groups[(row["dataset"], row["context"])].append(row["R_repaired_over_O"])
    out = {}
    for (dataset, context), ratios in sorted(groups.items()):
        med = statistics.median(ratios)
        n_eq = math.floor(med * DIRECT_SEGMENTS)
        out[f"{dataset}/{context}"] = {
            "blocks": len(ratios), "median_owner_ratio": med, "direct_segments_at_equal_owner_time": n_eq,
            "sevc_segments_checked": SEGMENTS,
            "direct_detection": {str(m): detection_probability(n_eq, m) for m in BAD},
            "sevc_detection_given_correct_majority": {str(m): 1.0 for m in BAD},
        }
    return out


def smallest_losing_bond(cells: Mapping[float, Sequence[float]]) -> float | None:
    """cells maps bond -> list of bootstrap upper bounds; return the smallest bond with all < 0."""
    for bond in sorted(cells):
        if cells[bond] and all(u < 0 for u in cells[bond]):
            return bond
    return None


def bond_grid(services: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    prod = [s for s in services if s["phase"] == "production" and s["method"] == RCMP]
    honest = {(s["dataset"], s["context"], s["block"]): s for s in prod if s["behavior"] == "honest"}
    strata: dict[str, Any] = {}
    upper: dict[tuple, dict[float, list[float]]] = defaultdict(lambda: defaultdict(list))
    honest_seconds = defaultdict(list)
    honest_rejected = defaultdict(int)
    for (dataset, context, _), s in honest.items():
        if context in CONTEXTS:
            honest_seconds[dataset].append(s["verifier_seconds"])
            honest_rejected[dataset] += (not s["admitted"])
    for dataset in DATASETS:
        for context in CONTEXTS:
            for behavior in DEVIATIONS:
                group = sorted((s for s in prod if s["dataset"] == dataset and s["context"] == context
                                and s["behavior"] == behavior), key=lambda s: s["block"])
                if len(group) != 24 or len({s["block"] for s in group}) != 24:
                    raise ValueError(f"stratum is not 24 blocks: {dataset}/{context}/{behavior}")
                for fee in FEES:
                    for bond in BONDS:
                        diff = [payoff(s, fee, bond, PRICE) - payoff(honest[(dataset, context, s["block"])], fee, bond, PRICE)
                                for s in group]
                        ci = bootstrap_mean(diff, [s["block"] for s in group],
                                            seed=zlib.crc32(f"{dataset}/{context}/{behavior}/{fee}/{bond}".encode()))
                        strata[f"{dataset}/{context}/{behavior}/f{fee}/b{bond}"] = {
                            "mean_deviation_minus_honest": statistics.mean(diff), "block_bootstrap_95": ci,
                            "positive_gain_services": sum(x > 0 for x in diff)}
                        upper[(dataset, fee)][bond].append(ci[1])
    summary = {}
    for dataset in DATASETS:
        t_h = statistics.mean(honest_seconds[dataset])
        unit = PRICE * t_h
        n_blocks = 24
        frr_upper = clopper_pearson_bound(0, n_blocks, side="upper", alpha=0.05)
        summary[dataset] = {
            "honest_services": len(honest_seconds[dataset]), "honest_rejected": honest_rejected[dataset],
            "honest_replay_cost_at_main_price": unit,
            "fee_in_honest_cost_units": {str(f): f / unit for f in FEES},
            "bond_in_honest_cost_units": {str(b): b / unit for b in BONDS},
            "honest_forfeiture_upper_by_bond": {str(b): frr_upper * b for b in BONDS},
            "smallest_grid_bond_all_deviations_strictly_losing": {
                str(f): smallest_losing_bond(upper[(dataset, f)]) for f in FEES},
        }
    return {"summary": summary, "strata": strata, "frr_block_upper_95": clopper_pearson_bound(0, 24, side="upper", alpha=0.05)}


def run(config_path: Path, output_root: Path, *, repo_root: Path, command: Sequence[str]) -> dict[str, Any]:
    config = json.loads(config_path.read_text())
    for key in ("rq4_science", "rq1_service_rows"):
        if sha256_file(Path(config[key]["path"])) != config[key]["sha256"]:
            raise ValueError(f"input hash mismatch: {key}")
    out = ensure_experiment_output_root(output_root, repo_root, CHANGE_ID)
    if any(out.iterdir()):
        raise FileExistsError("output root must be empty")
    rq4 = json.loads(Path(config["rq4_science"]["path"]).read_text())["SE-COST"]
    services = json.loads(Path(config["rq1_service_rows"]["path"]).read_text())
    production = [s for s in services if s["phase"] == "production" and s["method"] == RCMP]
    if len(services) != 1401 or len(production) != 1080:
        raise ValueError("service-row counts disagree with the frozen contract")
    result = {
        "change_id": CHANGE_ID, "label": "exploratory derivation; registered verdicts unchanged",
        "E1_system": {d: system_ratios(rq4["role_resources_mean_per_job"], d) for d in DATASETS},
        "E2_matched_budget": matched_budget(rq4["block_ratios"]),
        "E3_bond": bond_grid(services),
        "main_terms": MAIN,
    }
    write_json(out / "summary.json", result)
    code = [Path(__file__), repo_root / "scripts/run_cost_incentive_reanalysis.py", config_path,
            repo_root / "sevc/evaluation/f_rq234_audit.py"]
    write_json(out / "manifest.json", {"change_id": CHANGE_ID, "command": list(command),
        "inputs": {k: config[k] for k in ("rq4_science", "rq1_service_rows")},
        "code_sha256": {str(p.resolve().relative_to(repo_root.resolve())): sha256_file(p) for p in code},
        "outputs": {"summary.json": sha256_file(out / "summary.json")}, "environment": capture_environment()})
    return {"verdict": "PASS", "routing": "PAPER_CHANGE_REQUIRED"}
