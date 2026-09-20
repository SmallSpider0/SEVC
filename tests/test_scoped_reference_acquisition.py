import hashlib

import pytest

from sevc.verification.reference_acquisition import JobReferences, acquire_probe_sources


IDS = tuple(hashlib.sha256(str(i).encode()).hexdigest() for i in range(40))


def fixture(valid):
    calls, events = [], []
    def replay(sid):
        calls.append(sid)
        return {"proof_sha256": sid, "state_complete": True, "passed": sid in valid}
    return JobReferences("job", replay, events.append), calls, events


def test_stops_on_eight_actual_valid_references_and_never_prevalidates_population():
    refs, calls, events = fixture(set(IDS))
    out = acquire_probe_sources(IDS, secret_hex="ab" * 32, references=refs)
    assert len(calls) == 8
    assert out["issued"] and len(out["production_ids"]) == 32
    assert set(out["production_ids"]).isdisjoint(out["selected_ids"])
    assert events[0]["event"] == "role-seed-committed"


def test_exhausted_valid_population_rejects_on_owner_failures_and_preserves_attempts():
    refs, calls, events = fixture(set(IDS[:7]))
    out = acquire_probe_sources(IDS, secret_hex="ab" * 32, references=refs)
    assert len(calls) == 40 and len(set(calls)) == 40
    assert out["status"] == "OWNER_REPLAY_REJECT"
    assert out["trainer_terminal"] == "reject" and out["misconduct_finding"] is True
    assert sorted(out["rejecting_attempts"]) == sorted(IDS[7:])
    assert not out["issued"] and len(out["selected_ids"]) == 7
    assert sum(not r["passed"] for r in out["attempts"]) == 33
    assert len([e for e in events if e["event"] == "reference-replay"]) == 40


def test_failed_candidate_stays_in_production_and_exact_cache_is_job_local():
    refs, calls, _ = fixture(set(IDS))
    first = acquire_probe_sources(IDS, secret_hex="ab" * 32, references=refs)
    bad = first["selected_ids"][0]
    second, calls2, events = fixture(set(IDS) - {bad})
    out = acquire_probe_sources(IDS, secret_hex="ab" * 32, references=second)
    assert len(calls2) == 9 and bad in out["production_ids"]
    second.acquire(bad)
    assert len(calls2) == 9 and events[-1]["event"] == "reference-cache-hit"
    third, calls3, _ = fixture(set(IDS))
    third.acquire(bad)
    assert calls3 == [bad]


def test_malformed_receipt_is_technical_failure_not_safe_defer():
    refs = JobReferences("job", lambda sid: {"passed": True, "state_complete": True}, lambda e: None)
    with pytest.raises(ValueError, match="receipt"):
        acquire_probe_sources(IDS, secret_hex="ab" * 32, references=refs)


def test_secret_and_population_validation_precede_replay():
    refs, calls, _ = fixture(set(IDS))
    with pytest.raises(ValueError):
        acquire_probe_sources(IDS, secret_hex="ab", references=refs)
    with pytest.raises(ValueError):
        acquire_probe_sources(IDS[:-1] + (IDS[0],), secret_hex="ab" * 32, references=refs)
    assert not calls
