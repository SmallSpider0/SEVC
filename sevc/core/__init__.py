"""Selected public interfaces, loaded without unrelated dependencies."""
from importlib import import_module
_EXPORTS = {'Registry': ('sevc.core.registry', 'Registry'), 'configure_torch_thread_caps_from_environment': ('sevc.core.runtime', 'configure_torch_thread_caps_from_environment'), 'release_process_memory': ('sevc.core.runtime', 'release_process_memory'), 'resolve_device': ('sevc.core.runtime', 'resolve_device'), 'set_global_seed': ('sevc.core.runtime', 'set_global_seed'), 'sha256_file': ('sevc.core.artifacts', 'sha256_file'), 'write_json': ('sevc.core.artifacts', 'write_json')}
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
