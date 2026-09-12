"""Selected public interfaces, loaded without unrelated dependencies."""
from importlib import import_module
_EXPORTS = {'EXPERIMENT_RUNNERS': ('sevc.experiments.registry', 'EXPERIMENT_RUNNERS'), 'LocalTrainingConfig': ('sevc.experiments.prototype', 'LocalTrainingConfig'), 'PrototypeRoundResult': ('sevc.experiments.prototype', 'PrototypeRoundResult'), 'SEVCPrototype': ('sevc.experiments.prototype', 'SEVCPrototype'), 'run_experiment': ('sevc.experiments.runner', 'run_experiment')}
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
