"""Selected public interfaces, loaded without unrelated dependencies."""
from importlib import import_module
_EXPORTS = {'EvasionProjectionContext': ('sevc.attacks.evasion_aware', 'EvasionProjectionContext'), 'PreparedWorkerModel': ('sevc.attacks.strategies', 'PreparedWorkerModel'), 'WorkerBehavior': ('sevc.attacks.strategies', 'WorkerBehavior'), 'apply_checkerboard_trigger': ('sevc.attacks.evasion_aware', 'apply_checkerboard_trigger'), 'apply_cifar10_checkerboard_trigger': ('sevc.attacks.evasion_aware', 'apply_cifar10_checkerboard_trigger'), 'apply_strong_model_replacement': ('sevc.attacks.evasion_aware', 'apply_strong_model_replacement'), 'poison_selection': ('sevc.attacks.evasion_aware', 'poison_selection'), 'prepare_evasion_projection': ('sevc.attacks.evasion_aware', 'prepare_evasion_projection'), 'prepare_worker_model': ('sevc.attacks.strategies', 'prepare_worker_model'), 'project_evasion_aware_update': ('sevc.attacks.evasion_aware', 'project_evasion_aware_update'), 'train_evasion_aware_update': ('sevc.attacks.evasion_aware', 'train_evasion_aware_update')}
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
