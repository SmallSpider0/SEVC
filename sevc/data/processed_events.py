"""Deterministic exact-quota sample-event schedules shared by SEVC runners."""

from __future__ import annotations

import hashlib
import random
from typing import Sequence


def _namespace_seed(change_id: str, seed: int, namespace: str) -> int:
    material = f"{change_id}|{int(seed)}|{namespace}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % 2147483647


def exact_processed_event_indices(
    partition: Sequence[int],
    quota: int,
    *,
    change_id: str,
    seed: int,
    trainer_id: int,
    round_index: int,
) -> tuple[int, ...]:
    """Return exactly ``quota`` deterministic sample-events from one partition.

    A sample may appear once per deterministic pass and passes repeat only when
    the quota exceeds the unique partition cardinality.  The schedule is fixed
    entirely by the run identity and never depends on observed results.
    """

    source = tuple(int(value) for value in partition)
    if not source or int(quota) <= 0:
        raise ValueError(
            "exact processed-event schedule requires a non-empty partition and quota"
        )
    events: list[int] = []
    pass_index = 0
    while len(events) < int(quota):
        ordered = list(source)
        random.Random(
            _namespace_seed(
                str(change_id),
                int(seed),
                (
                    f"quota|trainer={int(trainer_id)}|"
                    f"round={int(round_index)}|pass={pass_index}"
                ),
            )
        ).shuffle(ordered)
        events.extend(ordered[: int(quota) - len(events)])
        pass_index += 1
    if len(events) != int(quota):
        raise AssertionError("processed-event quota cardinality drift")
    return tuple(events)


__all__ = ["exact_processed_event_indices"]
