"""Bounded online reference acquisition over committed, unlabeled inputs.

This controller receives identities and an actual replay operation, never an
evaluation label, a declared-valid stratum, or a paired production target set.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Callable, Mapping


def commitment(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass
class JobReferences:
    job_id: str
    replay: Callable[[str], Mapping]
    emit: Callable[[dict], None]
    _receipts: dict = field(default_factory=dict, init=False, repr=False)

    def acquire(self, source_sha256):
        if len(source_sha256) != 64:
            raise ValueError("source reference requires SHA-256 identity")
        hit = source_sha256 in self._receipts
        if not hit:
            receipt = dict(self.replay(source_sha256))
            if (receipt.get("proof_sha256") != source_sha256
                    or receipt.get("state_complete") is not True
                    or type(receipt.get("passed")) is not bool):
                raise ValueError("actual replay receipt does not bind complete source")
            self._receipts[source_sha256] = receipt
        self.emit({"event": "reference-cache-hit" if hit else "reference-replay",
                   "job_id": self.job_id, "source_sha256": source_sha256,
                   "receipt": self._receipts[source_sha256]})
        return dict(self._receipts[source_sha256])


def acquire_probe_sources(source_ids, *, secret_hex, references: JobReferences):
    """Commit private priority, then stop at eight replay-valid sources or forty.

    Replay failures are ordinary attempts. Malformed receipts are technical
    failures. Unselected identities, including failed attempts, remain production.
    """
    ids = tuple(source_ids)
    if len(ids) != 40 or len(set(ids)) != 40:
        raise ValueError("the frozen source population is forty distinct identities")
    if len(bytes.fromhex(secret_hex)) != 32:
        raise ValueError("private role seed must have 256 bits")
    private_commit = commitment(["scoped-role-v2", references.job_id, secret_hex])
    references.emit({"event": "role-seed-committed", "job_id": references.job_id,
                     "source_ids": list(ids), "commitment": private_commit})
    priority = sorted(ids, key=lambda sid: commitment(["scoped-priority-v2", secret_hex, sid]))
    selected, attempts = [], []
    for sid in priority:
        receipt = references.acquire(sid)
        attempts.append({"source_sha256": sid, "passed": receipt["passed"]})
        if receipt["passed"]:
            selected.append(sid)
        if len(selected) == 8:
            break
    issued = len(selected) == 8
    outcome = {"job_id": references.job_id, "status": "REFERENCES_READY" if issued else
               "INSUFFICIENT_VALID_REFERENCES_SAFE_DEFER", "issued": issued,
               "selected_ids": selected, "production_ids": [s for s in ids if s not in selected],
               "attempts": attempts, "role_commitment": private_commit}
    references.emit({"event": "reference-preparation-outcome", **outcome})
    return outcome
