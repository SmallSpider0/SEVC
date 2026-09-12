import importlib
import json
from pathlib import Path
import secrets
import threading

import numpy as np
import pytest
import torch
from torch import nn

from sevc.models import build_model
from sevc.training import WorkerBehavior, produce_worker_update, verify_replay_proof
from sevc.training.replay_sources import ReplayDatasetContext
from sevc.experiments.source_distribution import (
    DATASET_SPECS, prepare_config, validate_protocol, validate_public_run,
)
from sevc.evaluation.scoped_campaign_contract import expand_units

ROOT = Path(__file__).resolve().parents[1]


def protocol():
    return json.loads((ROOT/"configs/protocol.json").read_text())


def test_three_dataset_protocol_and_exact_reuse_dependencies():
    cfg = validate_protocol(protocol()); rows = expand_units(cfg)
    identities = {r["unit_id"] for r in rows}
    assert len(rows) == len(identities) == 9968
    assert {r["dataset"] for r in rows if r["dataset"]} == {"mnist", "cifar10", "cifar100"}
    assert {r["package"] for r in rows} == {p["id"] for p in cfg["packages"]}
    assert len(cfg["packages"]) == 15
    assert all(r["reuse_of"] in identities for r in rows if r.get("reuse_of"))
    cfg["science"]["datasets"]["mnist"]["blocks"] -= 1
    with pytest.raises(ValueError, match="differs"):
        validate_protocol(cfg)


@pytest.mark.parametrize("dataset", ["mnist", "cifar10", "cifar100"])
def test_real_model_factory_and_canonical_replay_with_synthetic_inputs(dataset):
    torch.manual_seed(23)
    spec = DATASET_SPECS[dataset]
    shape = (1, 28, 28) if dataset == "mnist" else (3, 32, 32)
    factory = lambda: build_model(spec["model"], class_count=spec["class_count"])
    batches = ((torch.randn(2, *shape), torch.tensor([0, 1])),)
    update = produce_worker_update(factory(), factory, batches, WorkerBehavior.NORMAL,
        device="cpu", learning_rate=.01, momentum=.9, max_batches=1, capture_replay=True)
    assert update.model(batches[0][0]).shape == (2, spec["class_count"])
    assert verify_replay_proof(update.replay_proof, factory, device="cpu", tolerance=1e-5)["passed"]


def test_native_game_primal_dual_certificate():
    from sevc.incentives.native_peer_prediction import solve_score
    spec = next(p for p in protocol()["packages"] if p["id"] == "M8")
    result = solve_score(spec)
    a, b, x, dual = map(np.asarray, (result["A"], result["b"], result["primal"], result["dual"]))
    assert result["status"] == "OPTIMAL"
    assert np.max(a @ x - b) < 1e-8
    assert abs(b @ dual - result["K"]) < 1e-8


def local_inputs(tmp_path):
    data = tmp_path/"datasets"; data.mkdir()
    scratch = tmp_path/"scratch"; scratch.mkdir()
    # Arbitrary software-fixture values, never a calibration observation.
    cal = {d: {"cost_unit_seconds": 1., "timeout_seconds": 1.,
               "deadline_seconds": 12., "owner_reserve_units": 100.} for d in DATASET_SPECS}
    cpath = tmp_path/"calibration.json"; cpath.write_text(json.dumps(cal))
    streams = tmp_path/"streams.json"
    streams.write_text(json.dumps({"roles": secrets.token_hex(32), "audits": secrets.token_hex(32)}))
    return prepare_config(ROOT/"configs/protocol.json", data, scratch, cpath, streams)


def test_standalone_configuration_fails_closed_without_touching_output(tmp_path):
    cfg = local_inputs(tmp_path); prof = cfg["profiles"]["standalone"]; output = tmp_path/"output"
    validate_public_run(cfg, prof, output)
    assert cfg["original_study_evidence"] is False and not output.exists()
    prof["calibration"]["mnist"]["cost_unit_seconds"] = 0
    with pytest.raises(ValueError, match="calibration"):
        validate_public_run(cfg, prof, output)
    assert not output.exists()


class TinyMLP(nn.Module):
    """Small generated-fixture model with the canonical MLP state schema."""
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Flatten(), nn.Linear(4, 8), nn.ReLU(),
                                     nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 2))

    def forward(self, inputs):
        return self.network(inputs)


class SyntheticContext(ReplayDatasetContext):
    """Tiny generated input using the canonical context and training interfaces."""
    def __init__(self):
        from sevc.experiments.tdsc_five_rq_evidence import ReplayDevice
        self.dataset = "mnist"; self.model_key = "small-mlp"; self.class_count = 2
        self.image_size = 2; self.input_features = 4; self.build_key = "small-mlp"
        self.device = ReplayDevice("cpu"); self._lanes = {}; self._replay_models = threading.local()
        generator = torch.Generator().manual_seed(81)
        self.train = torch.utils.data.TensorDataset(torch.randn(80, 1, 2, 2, generator=generator), torch.arange(80) % 2)

    def factory(self):
        return TinyMLP()


def test_shared_service_assembly_and_independent_audit_on_generated_inputs(tmp_path):
    from sevc.core.role_accounting import RoleClock
    from sevc.evaluation.workload_performance import AppendLog, GPUSampler
    from sevc.experiments.scoped_five_rq_units import ScopedStudy
    from sevc.evaluation.scoped_result_audit import audit_and_summarize
    cfg = local_inputs(tmp_path)
    rows = expand_units(cfg["scoped_candidate"])
    chosen = [r for r in rows if r["dataset"] == "mnist" and r["block"] == 0 and r["invalid"] == 1 and (
        r["package"] == "M1" and r["behavior"] == "honest" or
        r["package"] == "M3" or r["package"] == "M5" and r["method"] in {"owner-direct-v2", "plain-delegation-v2", "rcmp-probe-source-v2"})]
    cfg["technical_units"] = {"fixture": chosen}
    prof = {**cfg["profiles"]["standalone"], "full_matrix": False, "namespace": "fixture"}
    output = tmp_path/"generated-fixture"; output.mkdir()
    names = ("phase-timing.jsonl", "assignment-audit.jsonl", "source-task-identities.jsonl", "report-lifecycle.jsonl", "workload-units.jsonl")
    phases, records, sources, commits, events = logs = [AppendLog(output/name) for name in names]
    sampler = GPUSampler(output, "CPU-FIXTURE", False)
    sampler.start()
    clock = RoleClock(lambda: None, phases, cuda=False)
    study = ScopedStudy(cfg, prof, output, clock, records, sources, commits, events)
    context = SyntheticContext()
    try:
        study.run_dataset(context, (0, 80))
        study.run_cpu()
    finally:
        context.close_task_lanes()
        for log in logs:
            log.close()
        sampler.finish(output)
    result = audit_and_summarize(output, cfg, prof)
    audit = json.loads((output/"independent-audit.json").read_text())
    assert audit["status"] == "AUDIT_PASS" and audit["units"] == len(chosen)
    assert len(list((output/"units").glob("*.json"))) == len(chosen)


def test_all_distributed_modules_import_without_original_workspace():
    for path in (ROOT/"sevc").rglob("*.py"):
        name = ".".join(path.relative_to(ROOT).with_suffix("").parts)
        if name.endswith(".__init__"):
            name = name[:-9]
        module = importlib.import_module(name)
        assert Path(module.__file__).resolve().is_relative_to(ROOT)
