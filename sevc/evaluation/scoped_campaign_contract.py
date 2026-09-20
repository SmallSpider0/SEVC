"""Frozen unit expansion and reuse identities for the shared five-RQ runner."""
from __future__ import annotations

import itertools
from sevc.verification.reference_acquisition import commitment

CHANGE = "experiment-tdsc-scoped-five-rq-confirmation-v1"


def expand_units(candidate):
    if candidate.get("protocol_variant") == "claim-linked-tiny-v1":
        return expand_claim_linked_units(candidate)
    science = candidate["science"]
    methods = science["methods"]
    packages = {p["id"]: p for p in candidate["packages"]}
    rows = []
    def add(package, dataset=None, block=None, method=None, behavior=None, invalid=0,
            steps=4, batch_size=2, **extra):
        seed = science["datasets"][dataset]["seed_start"] + block if dataset else None
        row = {"package": package, "dataset": dataset, "block": block, "seed": seed,
               "method": method, "behavior": behavior, "invalid": invalid,
               "steps": steps, "batch_size": batch_size, **extra}
        row["unit_id"] = commitment([science["namespace"], row])
        rows.append(row)
        return row
    for dataset in science["dataset_order"]:
        for i, m in itertools.product(range(200, 203), (4, 33)):
            add("M0", dataset, i, methods["R"], invalid=m)
        base = {}
        for i in range(packages["M1"]["blocks_per_dataset"]):
            for m, method, behavior in itertools.product((0, 1), (methods["R"], methods["G"]),
                    packages["M1"]["behaviors"]):
                row = add("M1", dataset, i, method, behavior, m)
                base[i, m, method, behavior] = row["unit_id"]
        for i, method, scenario in itertools.product(packages["M2"]["block_indices"], packages["M2"]["methods"], packages["M2"]["scenarios"]):
            behavior, invalid = scenario.rsplit("-m", 1)
            reuse = base.get((i, int(invalid), method, behavior))
            add("M2", dataset, i, method, behavior, int(invalid), reuse_of=reuse)
        for i, method in itertools.product(range(6), packages["M2C"]["methods"]):
            add("M2C", dataset, i, method, "joint-cached-correct", 1)
        for i, m, method in itertools.product(packages["M3"]["block_indices"], (0, 1), (methods["R"], methods["G"])):
            add("M3", dataset, i, method, invalid=m, preparation_from=base[i, m, method, "honest"], replays=8)
        for i, policy in itertools.product(packages["M4L"]["block_indices"], packages["M4L"]["methods"]):
            fault = ("no-missing" if i < 8 else "one-missing" if i < 16 else
                     "correlated-missing" if i < 23 else "insufficient-reserve")
            add("M4L", dataset, i, methods["R"], "honest", policy=policy, fault=fault, graph="all-compatible")
        for i, method, scenario in itertools.product(range(12), packages["M4S"]["admission_methods"], packages["M4S"]["scenarios"]):
            behavior, invalid = scenario.rsplit("-m", 1)
            add("M4S", dataset, i, method, behavior, int(invalid), policy="current-certified-ecs",
                fault="no-missing", graph="all-compatible", colluders=2 if i < 6 else 3)
        for i, graph, policy in itertools.product(range(6), packages["M4X"]["graphs"], packages["M4X"]["methods"]):
            add("M4X", dataset, i, methods["R"], "honest", graph=graph, policy=policy, fault="v0-v3-missing")
        costs = {}
        for i, m in itertools.product(packages["M5"]["block_indices"], (0, 1)):
            arms = packages["M5"]["methods"]
            shift = i % len(arms)
            for method in arms[shift:] + arms[:shift]:
                row = add("M5", dataset, i, method, "honest", m, verifier_count=0 if method == methods["O"] else 3)
                costs[i, m, method] = row["unit_id"]
        for i, workload, method, behavior in itertools.product(packages["M6"]["block_indices"], packages["M6"]["workloads"],
                (methods["R"], methods["G"]), ("honest", "partial-50")):
            baseline = workload == {"steps": 4, "batch_size": 2}
            add("M6", dataset, i, method, behavior, 1, **workload,
                reuse_of=base[i, 1, method, behavior] if baseline else None)
        for i, workload in itertools.product(packages["M6C"]["block_indices"], packages["M6C"]["workloads"]):
            arms=packages["M6C"]["methods"]; shift=i%len(arms)
            for method in arms[shift:]+arms[:shift]:
                add("M6C", dataset, i, method, "honest", 1, **workload,
                    verifier_count=0 if method == methods["O"] else 3,
                    baseline_from=costs[i, 1, method])
        for i, anchor, method, behavior in itertools.product(range(400, 403), (32, 512, 2048),
                (methods["R"], methods["G"]), ("honest", "partial-50")):
            add("M7", dataset, i, method, behavior, 1, anchor=anchor,
                cluster_id=f"{dataset}-trajectory-{i}", replays=8 if behavior == "honest" else 0)
    for graph in packages["M4B"]["graphs"]:
        for size in range(5):
            for missing in itertools.combinations(range(9), size):
                for policy in packages["M4B"]["methods"]:
                    add("M4B", graph=graph, missing=list(missing), policy=policy)
    traces = [r for r in rows if r["package"] in {"M4L", "M4X"}]
    for trace, counterfactual in itertools.product(traces, packages["M4T"]["counterfactuals"]):
        add("M4T", trace["dataset"], trace["block"], trace_from=trace["unit_id"], counterfactual=counterfactual)
    for method, epsilon in itertools.product(packages["M8"]["methods"], packages["M8"]["epsilon"]):
        add("M8", method=method, epsilon=epsilon)
    if len({r["unit_id"] for r in rows}) != len(rows):
        raise ValueError("duplicate frozen unit identity")
    for key, spec in packages.items():
        group = [r for r in rows if r["package"] == key]
        if len(group) != spec["logical_count"]:
            raise ValueError(f"{key} expanded count differs: {len(group)}")
        if "reused_count" in spec and sum(bool(r.get("reuse_of")) for r in group) != spec["reused_count"]:
            raise ValueError(f"{key} reuse count differs")
    return rows


def expand_claim_linked_units(candidate):
    """The reviewed improved-method slice, consumed by the same ScopedStudy."""
    science = candidate["science"]
    if set(science["dataset_order"]) != {"mnist", "cifar10", "cifar100"}:
        raise ValueError("claim-linked evidence requires exactly three datasets")
    methods = science["methods"]
    rows = []
    def add(package, dataset, block, method, behavior="honest", invalid=0, **extra):
        row = {"package": package, "dataset": dataset, "block": block,
               "seed": science["datasets"][dataset]["seed_start"] + block,
               "method": method, "behavior": behavior, "invalid": invalid,
               "steps": 4, "batch_size": 2, **extra}
        row["unit_id"] = commitment([science["namespace"], row])
        rows.append(row)
    for dataset in science["dataset_order"]:
        for block in range(science["blocks_per_dataset"]):
            for method, behavior in itertools.product((methods["R"], methods["G"]),
                                                     ("honest", "partial-50", "partial-90")):
                add("M1", dataset, block, method, behavior, 1, verifier_count=1,
                    reference_check=behavior == "honest")
            for scenario, invalid, fault, policy, flip in (
                ("valid-honest", 0, "no-missing", "all-response-certified-ecs", False),
                ("invalid-honest", 1, "no-missing", "all-response-certified-ecs", False),
                ("valid-one-missing-recovery", 0, "one-missing", "all-response-certified-ecs", False),
                ("valid-one-incorrect-report", 0, "no-missing", "all-response-certified-ecs", True),
                ("valid-one-missing-no-recovery", 0, "one-missing", "no-recovery", False),
            ):
                add("M4L", dataset, block, methods["R"], invalid=invalid, scenario=scenario,
                    fault=fault, policy=policy, graph="all-compatible", conditioned_production_flip=flip)
            for invalid in (0, 1):
                add("M5", dataset, block, methods["O"], invalid=invalid, verifier_count=0)
    if len({r["unit_id"] for r in rows}) != len(rows):
        raise ValueError("duplicate claim-linked unit identity")
    return rows


def technical_units(candidate,namespace):
    """Disjoint engineering calibration and holdout; never scientific samples."""
    if namespace not in ('technical-selection','technical-validation'):
        raise ValueError('unregistered blind namespace')
    selection=namespace=='technical-selection'
    all_rows=expand_units(candidate); selected=[]; identities={}; methods=candidate['science']['methods']
    for row in all_rows:
        p=row['package']; i=row['block']; take=False
        if selection:
            if i==0 and row['invalid']==1:
                take=(p=='M1' and row['behavior'] in ('honest','partial-50') or
                      p=='M2' and row['method']!=methods['R'] and row['behavior']=='joint-targeted-cover' or
                      p in ('M5','M6C') or p=='M6' and not row.get('reuse_of') and row['behavior']=='honest' or p=='M3')
            take |= p=='M0' and i==200
            take |= p=='M4L' and i==23
            take |= p=='M4X' and i==0 and row['policy']=='current-certified-ecs'
            take |= p=='M4S' and i==0 and row['method']==methods['PA'] and row['invalid']==1
            take |= p=='M7' and i==400 and row['method']==methods['R'] and row['behavior']=='honest'
            take |= p=='M4B' and row['missing'] in ([],[0,3])
            take |= p=='M8'
        else:
            take=(i==1 and row['invalid']==1 and (p=='M1' and row['behavior']=='honest' or
                 p=='M5' and row['method']==methods['R'] or
                 p=='M6' and not row.get('reuse_of') and row['behavior']=='honest' and row['method']==methods['R']))
            take |= p=='M4X' and i==1 and row['graph']=='reserve-specialists-j0' and row['policy']=='current-certified-ecs'
            take |= p=='M8' and row['epsilon']==0
            take |= p=='M2' and i==1 and row['invalid']==1 and row['behavior']=='honest' and row['method'] in (methods['A8'],methods['P8'])
            take |= p=='M2C' and i==1
        if take:
            r=dict(row)
            if r['seed'] is not None:
                r['seed']+=900000 if selection else 1800000
            r['namespace']=namespace
            r.pop('preparation_from',None); r.pop('baseline_from',None)
            r['unit_id']=commitment([namespace,{k:v for k,v in r.items() if k!='unit_id'}])
            identities[row['unit_id']]=r['unit_id']
            selected.append(r)
    if not selection:
        for row in all_rows:
            if row['package']=='M4T' and row['trace_from'] in identities:
                r={**row,'namespace':namespace,'trace_from':identities[row['trace_from']]}
                r['seed']+=1800000
                r['unit_id']=commitment([namespace,{k:v for k,v in r.items() if k!='unit_id'}])
                selected.append(r)
    return selected
