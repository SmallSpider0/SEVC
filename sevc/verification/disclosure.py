"""Owner-controlled publication boundary for related service assignments."""
from __future__ import annotations


class ProbeDisclosureEpoch:
    def __init__(self, epoch_id: str, assignment_ids):
        ids = tuple(assignment_ids)
        if not epoch_id or not ids or len(ids) != len(set(ids)) or any(not i for i in ids):
            raise ValueError("disclosure epoch requires unique planned assignments")
        self.epoch_id = epoch_id
        self._states = dict.fromkeys(ids, "PLANNED")
        self._closed = False
        self._published = False

    def covers(self, assignment_ids):
        return set(assignment_ids).issubset(self._states) and not self._closed

    def begin(self, assignment_id):
        if self._closed or self._published or self._states.get(assignment_id) != "PLANNED":
            raise ValueError("unplanned, repeated or closed-epoch assignment")
        self._states[assignment_id] = "RUNNING"

    def finish(self, assignment_id, status):
        if self._states.get(assignment_id) != "RUNNING":
            raise ValueError("only a running assignment may finish")
        if status not in {"PASS", "FAIL_CONFIRMED", "DROPOUT", "TECHNICAL_FAILURE"}:
            raise ValueError("unknown terminal service status")
        self._states[assignment_id] = status

    def close(self):
        """The owner closes scheduling and cancels only never-started reserves."""
        if "RUNNING" in self._states.values():
            raise ValueError("cannot disclose while related service is running")
        self._closed = True
        self._states = {k: "NOT_ACTIVATED" if v == "PLANNED" else v for k, v in self._states.items()}

    def release(self, answers):
        if not self._closed:
            raise ValueError("reference answers remain private until scheduling closes")
        self._published = True
        return {"epoch_id": self.epoch_id, "assignment_states": dict(self._states), "answers": tuple(answers)}
