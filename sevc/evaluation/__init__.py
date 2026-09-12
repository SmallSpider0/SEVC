"""Selected public interfaces, loaded without unrelated dependencies."""
from importlib import import_module
_EXPORTS = {'clustered_binary_interval': ('sevc.evaluation.statistics', 'clustered_binary_interval'), 'clustered_paired_bootstrap_interval': ('sevc.evaluation.statistics', 'clustered_paired_bootstrap_interval'), 'grouped_auc_interval': ('sevc.evaluation.statistics', 'grouped_auc_interval'), 'paired_bootstrap_interval': ('sevc.evaluation.statistics', 'paired_bootstrap_interval'), 'wilson_interval': ('sevc.evaluation.statistics', 'wilson_interval')}
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
