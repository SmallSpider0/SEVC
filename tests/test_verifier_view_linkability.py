from sevc.evaluation.verifier_view_linkability import (
    VisibleTask, adversary_commitment_match, adversary_payload_outlier,
    adversary_start_state_repeat, aggregate, decide, hiding_commitment, score_task_set,
)


def _view(n=10, challenge_payload=None, repeat=False):
    view = [VisibleTask(f"p{i}", f"s{i}", 100) for i in range(n)]
    if challenge_payload is not None:
        view[-1] = VisibleTask("c", "s9", challenge_payload)
    if repeat:
        view[1] = VisibleTask("p1", "s0", 100)
    return view


def test_commitment_match_flags_uncommitted_content_only():
    view = _view()
    assert adversary_commitment_match(view, frozenset(f"p{i}" for i in range(9))) == {9}


def test_hiding_commitments_are_not_comparable_to_content():
    view = _view()
    hidden = frozenset(hiding_commitment(b"k", f"p{i}") for i in range(10))
    assert adversary_commitment_match(view, hidden) == set(range(10))


def test_payload_outlier_and_ties():
    assert adversary_payload_outlier(_view(challenge_payload=50)) == {9}
    assert adversary_payload_outlier(_view()) == frozenset()
    tied = [VisibleTask("a", "a", 1), VisibleTask("b", "b", 2)]
    assert adversary_payload_outlier(tied) == frozenset()


def test_start_state_repeat():
    assert adversary_start_state_repeat(_view(repeat=True)) == {0, 1}


def test_score_probe_pass_and_incorrect_admission():
    roles = ["production"] * 6 + ["control", "control", "challenge", "challenge"]
    passing = score_task_set(frozenset({8, 9}), roles, invalid_index=0)
    assert passing["probe_pass"] and passing["incorrect_admission"]
    assert not score_task_set(frozenset({7, 8, 9}), roles, None)["probe_pass"]
    assert not score_task_set(frozenset({8, 9, 0}), roles, 0)["incorrect_admission"]


def test_aggregate_and_decision_ignore_counterfactual():
    roles = ["production"] * 6 + ["control", "control", "challenge", "challenge"]
    good = score_task_set(frozenset({8, 9}), roles, None)
    none = score_task_set(frozenset(), roles, None)
    row = {"dataset": "mnist", "block": 0, "invalid": False,
           "scores": {"A1-open": good, "A1-hiding": none, "A2": none, "A3": none}}
    summary = aggregate([row])
    assert summary["mnist"]["A1-open"]["effective"] and not summary["mnist"]["A3"]["effective"]
    decision = decide(summary)
    assert decision["verdict"] == "PASS" and decision["counterfactual_A1_open_effective"] == ["mnist"]


def test_equalized_envelopes_have_equal_length_and_keep_fields():
    import json
    from sevc.evaluation.verifier_view_linkability import EMPTY_PADDING_OVERHEAD, equalized_payload_lengths
    from sevc.verification.replay_coupled_probes import (
        PADDING_FIELD, equalize_public_envelopes, serialize_public_replay_envelope)
    envelopes = [{"task_id": "a" * 64, "tensor_payload_bytes": 5},
                 {"task_id": "b" * 64, "tensor_payload_bytes": 123456},
                 {"task_id": "c" * 64, "tensor_payload_bytes": 77, "wrapper_descriptor": {"x": [1, 2]}}]
    padded = equalize_public_envelopes(envelopes)
    lengths = {len(serialize_public_replay_envelope(e)) for e in padded}
    assert len(lengths) == 1
    for original, out in zip(envelopes, padded):
        assert set(out[PADDING_FIELD]) <= {" "}
        assert {k: v for k, v in out.items() if k != PADDING_FIELD} == original
        json.loads(serialize_public_replay_envelope(out))
    recorded = [len(serialize_public_replay_envelope(e)) for e in envelopes]
    assert equalized_payload_lengths(recorded) == [lengths.pop()] * 3
    assert EMPTY_PADDING_OVERHEAD == len(serialize_public_replay_envelope({**envelopes[0], PADDING_FIELD: ""})) - recorded[0]


def test_registered_methods_do_not_equalize_by_default():
    from sevc.verification.on_demand_service import METHODS
    assert METHODS.get("rcmp-opaque-gradient-continuation-gpu-equalized-v1").equalize_envelopes
    assert not METHODS.get("rcmp-opaque-gradient-continuation-gpu-v1").equalize_envelopes
