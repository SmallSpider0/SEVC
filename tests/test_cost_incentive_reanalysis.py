import math

from sevc.evaluation.cost_incentive_reanalysis import (
    detection_probability, matched_budget, smallest_losing_bond, system_ratios)


def test_detection_probability_bounds():
    assert detection_probability(40, 1) == 1.0
    assert math.isclose(detection_probability(20, 1), 0.5)
    assert detection_probability(0, 4) == 0.0


def test_smallest_losing_bond():
    assert smallest_losing_bond({0.5: [0.1, -1], 1.0: [-0.1, -2], 2.0: [-1, -3]}) == 1.0
    assert smallest_losing_bond({0.5: [0.1]}) is None


def test_system_ratio_counts_verifier_s_times():
    res = {"d/O": {"owner/wall": 10.0}, "d/R-repaired": {"owner/wall": 6.0, "verifier/wall": 12.0},
           "d/R-as-run": {"owner/wall": 11.0, "verifier/wall": 12.0}}
    out = system_ratios(res, "d")
    assert out["R-repaired"]["system_over_direct_s3"] == 4.2 and out["R-repaired"]["system_over_direct_s1"] == 1.8


def test_matched_budget_floor():
    rows = [{"dataset": "d", "context": "c", "R_repaired_over_O": r} for r in (0.6, 0.62, 0.64)]
    assert matched_budget(rows)["d/c"]["direct_segments_at_equal_owner_time"] == 19
