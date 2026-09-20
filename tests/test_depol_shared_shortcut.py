from sevc.committee.depol_arbitration import arbitrate_digests, arbitrate_distances
from sevc.evaluation.depol_shared_shortcut import arbitrate_combination, combine_matrices, reproduce
from sevc.verification.reference_acquisition import commitment


def _native(trainer, v0, v1, v2, matrices=None, epsilon=0.5):
    events = [{"phase": "COMMITTED", "party": p, "commitment": commitment([p, r])}
              for p, r in (("trainer", trainer), ("v0", v0), ("v1", v1), ("v2", v2))]
    events += [{"phase": "REVEALED", "party": p, "report": r}
               for p, r in (("trainer", trainer), ("v0", v0), ("v1", v1), ("v2", v2))]
    order = [1, 0]
    fast = arbitrate_digests([trainer[i] for i in order], [v0, v1, v2])
    slow = None
    final = fast
    if matrices is not None:
        slow = arbitrate_distances(matrices, epsilon)
        final = slow
        slow = {**slow, "distance_matrices": matrices}
    return {"native_commitment_events": events, "protocol_events": [{"phase": "GROUP_SAMPLING", "sampled_interval_indices": order}],
            "fast": fast, "slow": slow, "final": final, "epsilon": epsilon}


def test_combine_matrices_reorders_and_zeroes_copies():
    m0 = [[5.0, 0.0, 7.0, 7.0]]
    m1 = [[0.0, 7.0, 0.0, 0.0]]
    out = combine_matrices([m0, m1, m1], ("A", "A", "H"))
    assert out[0] == [[5.0, 0.0, 0.0, 7.0]] and out[1] == out[0]
    assert out[2] == [[0.0, 7.0, 7.0, 0.0]]


def test_shortcut_majority_takes_eligibility_from_honest_minority():
    trainer = [[1], [2]]
    honest = [[2], [1]]          # sampled order [1, 0]
    lazy = [[0], [0]]
    m_lazy = [[5.0, 0.0, 7.0, 7.0]] * 2
    m_honest = [[0.0, 7.0, 0.0, 0.0]] * 2
    native = _native(trainer, lazy, honest, honest, [m_lazy, m_honest, m_honest])
    reproduce(native)
    c2 = arbitrate_combination(native, ("A", "A", "H"))
    assert c2["status"] == "RECONSTRUCTED" and c2["path"] == "slow"
    assert c2["deviator_eligible"] == [True, True] and c2["honest_eligible"] == [False]
    assert c2["trainer_verdict"] is False


def test_missing_slow_matrices_are_not_inferred():
    trainer = [[1], [2]]
    honest = [[2], [1]]
    native = _native(trainer, honest, honest, honest)
    reproduce(native)
    native = {**native, "native_commitment_events": _native(trainer, [[0], [0]], honest, honest)["native_commitment_events"]}
    assert arbitrate_combination(native, ("A", "A", "H"))["status"] == "NOT_RECONSTRUCTIBLE"
