"""Selected public interfaces, loaded without unrelated dependencies."""
from importlib import import_module
_EXPORTS = {'BetaResponsePrior': ('sevc.committee.reliability', 'BetaResponsePrior'), 'Committee': ('sevc.committee.formation', 'Committee'), 'CommitteeFormationRequest': ('sevc.committee.formation', 'CommitteeFormationRequest'), 'FORMATION_VARIANTS': ('sevc.committee.formation', 'FORMATION_VARIANTS'), 'beta_prior': ('sevc.committee.reliability', 'beta_prior'), 'capacity_independent_probability': ('sevc.committee.reliability', 'capacity_independent_probability'), 'fit_beta_response_prior': ('sevc.committee.reliability', 'fit_beta_response_prior'), 'form_committees': ('sevc.committee.formation', 'form_committees'), 'genetic_committees': ('sevc.committee.formation', 'genetic_committees'), 'greedy_minimal_committees': ('sevc.committee.formation', 'greedy_minimal_committees'), 'legacy_greedy_committees': ('sevc.committee.formation', 'legacy_greedy_committees'), 'majority_success_probability': ('sevc.committee.formation', 'majority_success_probability'), 'posterior_completion_probability': ('sevc.committee.reliability', 'posterior_completion_probability'), 'prior_payload': ('sevc.committee.reliability', 'prior_payload')}
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
