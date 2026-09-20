import json

import pytest
import torch

from sevc.experiments import tdsc_replay_tolerance as t


class _Device:
    name = "cpu"


class _Context:
    def __init__(self):
        g = torch.Generator().manual_seed(0)
        self.dataset = "mnist"
        self.device = _Device()
        self.train = [(torch.randn(1, 28, 28, generator=g), i % 10) for i in range(4000)]

    def factory(self):
        from sevc.models import build_model
        return build_model("small-mlp", class_count=10, input_features=784)


def test_numeric_config_is_applied_and_restored():
    before = (torch.are_deterministic_algorithms_enabled(), torch.backends.cudnn.benchmark,
              torch.backends.cuda.matmul.allow_tf32, torch.get_num_threads())
    with t.apply_numeric_config("C1-nondeterministic-autotune") as (device, flags):
        assert device == "cuda:0" and flags["cudnn_benchmark"] and not flags["deterministic"]
    with t.apply_numeric_config("C4-cpu") as (device, flags):
        assert device == "cpu" and flags["cpu_threads"] == 1 and flags["deterministic"]
    after = (torch.are_deterministic_algorithms_enabled(), torch.backends.cudnn.benchmark,
             torch.backends.cuda.matmul.allow_tf32, torch.get_num_threads())
    assert before == after


def test_summary_interval_and_invalid_shift_logic():
    rows = []
    for x in (1e-7, 2e-6):
        rows.append({"dataset": "d", "config": "C4-cpu", "kind": "honest", "max_abs_difference": x,
                     "max_optimizer_abs_difference": 0.0})
    for x in (3e-4, 5e-5):
        rows.append({"dataset": "d", "config": "C4-cpu", "kind": "challenge", "max_abs_difference": x,
                     "max_optimizer_abs_difference": 0.0})
    s = t.summarize(rows)["d"]["C4-cpu"]
    assert s["separating_interval"] == [2e-6, 5e-5] and s["tolerance_separates"]
    assert s["invalid_shift_detectable_at_tolerance"] and s["honest_false_rejections_at_tolerance"] == 0
    rows[1]["max_abs_difference"] = 3e-5
    s = t.summarize(rows)["d"]["C4-cpu"]
    assert not s["tolerance_separates"] and not s["invalid_shift_interval_nonempty"]


def test_formal_activation_refuses_without_lock(tmp_path):
    config = json.loads(open("configs/tdsc_replay_tolerance_heterogeneity_v1.json").read())
    profile = config["profiles"][config["default_profile"]]
    with pytest.raises((PermissionError, FileNotFoundError)):
        t.validate_activation(config, profile, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_scientific_hash_ignores_activation_fields():
    config = json.loads(open("configs/tdsc_replay_tolerance_heterogeneity_v1.json").read())
    a = t.scientific_configuration_sha256(config)
    config["protocol_lock_sha256"] = "x" * 64
    assert t.scientific_configuration_sha256(config) == a
    config["anchors"] = ["init"]
    assert t.scientific_configuration_sha256(config) != a


def test_cpu_end_to_end_block_separates_honest_from_challenges():
    context = _Context()
    with t.apply_numeric_config("C4-cpu"):
        honest, challenges, identity = t.build_segments(context, dataset="mnist", seed=7, anchor="mid",
                                                        partition=(0, 4000))
    assert len(challenges) == 4 and set(identity["challenges"]) == {"0", "1", "2", "3"}
    rows = t.replay_rows(context, honest, challenges, base={"dataset": "mnist"}, configs=["C4-cpu"])
    honest_rows = [r for r in rows if r["kind"] == "honest"]
    challenge_rows = [r for r in rows if r["kind"] == "challenge"]
    assert honest_rows[0]["max_abs_difference"] == 0.0 and honest_rows[0]["passed"]
    assert all(r["max_abs_difference"] > t.TOLERANCE and not r["passed"] for r in challenge_rows)
    assert all(r["state_complete"] for r in rows)
