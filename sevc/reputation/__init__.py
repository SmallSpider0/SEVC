"""Selected public interfaces, loaded without unrelated dependencies."""
from importlib import import_module
_EXPORTS = {'ReputationSimulation': ('sevc.reputation.estimator', 'ReputationSimulation'), 'exponential_decay_reputation': ('sevc.reputation.estimator', 'exponential_decay_reputation'), 'simulate_historical_verifier_groups': ('sevc.reputation.estimator', 'simulate_historical_verifier_groups')}
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
