"""Opt-in profiler annotations; no timing or behavior change when unselected."""
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar

_active = ContextVar('sevc_operation_probe', default=False)


def operation_scope(name):
    if not _active.get():
        return nullcontext()
    from torch.profiler import record_function
    return record_function('sevc.' + name)


@contextmanager
def operation_probe():
    token = _active.set(True)
    try:
        yield
    finally:
        _active.reset(token)
