"""Selected public interfaces, loaded without unrelated dependencies."""
from importlib import import_module
_EXPORTS = {'ReplayProof': ('sevc.training.engine', 'ReplayProof'), 'WorkerBehavior': ('sevc.attacks', 'WorkerBehavior'), 'WorkerUpdate': ('sevc.training.engine', 'WorkerUpdate'), 'average_models': ('sevc.training.engine', 'average_models'), 'evaluate_model': ('sevc.training.engine', 'evaluate_model'), 'produce_worker_update': ('sevc.training.engine', 'produce_worker_update'), 'verify_replay_proof': ('sevc.training.engine', 'verify_replay_proof')}
__all__ = list(_EXPORTS)
def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module, attribute = _EXPORTS[name]
    value = getattr(import_module(module), attribute)
    globals()[name] = value
    return value
def __dir__():
    return sorted(set(globals()) | set(__all__))
