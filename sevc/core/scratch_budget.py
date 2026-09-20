"""Opt-in, fail-closed capacity guard for disposable tensor payloads."""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import fields, is_dataclass
from pathlib import Path

_budget = ContextVar('scratch_payload_budget', default=None)


def tensor_bytes(value):
    import torch
    if isinstance(value, torch.Tensor):
        return value.untyped_storage().nbytes()
    if isinstance(value, dict):return sum(tensor_bytes(v) for v in value.values())
    if isinstance(value, (list, tuple)):return sum(tensor_bytes(v) for v in value)
    if is_dataclass(value):return sum(tensor_bytes(getattr(value,f.name)) for f in fields(value))
    return 0


def disk_usage(root):
    files=[p for p in Path(root).rglob('*') if p.is_file()]
    stats=[p.stat() for p in files]
    return {'files':len(stats),'logical_bytes':sum(s.st_size for s in stats),
            'allocated_bytes':sum(s.st_blocks*512 for s in stats)}


@contextmanager
def scratch_budget(root, limit):
    token=_budget.set((Path(root).resolve(),int(limit)))
    try:yield
    finally:_budget.reset(token)


def guarded_tensor_save(value, path):
    import torch
    active=_budget.get()
    if active is None:return torch.save(value,path)
    root,limit=active;path=Path(path)
    if not path.resolve().is_relative_to(root):raise ValueError('scratch write outside guarded root')
    size=tensor_bytes(value);reserve=max(1024**2,size//100)
    if disk_usage(root)['logical_bytes']+size+reserve>limit:
        raise OSError('HOLD_RESOURCE: frozen scratch payload capacity would be exceeded')
    torch.save(value,path)
    if disk_usage(root)['logical_bytes']>limit:
        raise OSError('HOLD_RESOURCE: serialized scratch capacity exceeded; preserve written payload')
