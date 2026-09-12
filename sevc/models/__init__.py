"""Selected public interfaces, loaded without unrelated dependencies."""
from importlib import import_module
_EXPORTS = {'LargeMLP': ('sevc.models.factory', 'LargeMLP'), 'MediumMLP': ('sevc.models.factory', 'MediumMLP'), 'SmallMLP': ('sevc.models.factory', 'SmallMLP'), 'build_model': ('sevc.models.factory', 'build_model'), 'hf_snapshot_lock': ('sevc.models.factory', 'hf_snapshot_lock'), 'model_parameter_count': ('sevc.models.factory', 'model_parameter_count'), 'model_state_nbytes': ('sevc.models.factory', 'model_state_nbytes'), 'model_state_sha256': ('sevc.models.factory', 'model_state_sha256'), 'resolve_hf_snapshot': ('sevc.models.factory', 'resolve_hf_snapshot')}
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
