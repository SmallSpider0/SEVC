"""Selected canonical scientific routines; deployment wrappers are omitted."""
from __future__ import annotations


from typing import Any, Callable, Dict

from sevc.core.registry import Registry

ExperimentRunner = Callable[..., Dict[str, Any]]


EXPERIMENT_RUNNERS: Registry[ExperimentRunner] = Registry("experiment runner")


def _tdsc_five_rq_runner(repo_root, config_path, output_root, resolved_config, **kwargs):
    from .tdsc_five_rq_evidence import run_tdsc_five_rq_evidence
    return run_tdsc_five_rq_evidence(repo_root, config_path, output_root, resolved_config, **kwargs)


def _recovery_enumeration_runner(repo_root, config_path, output_root, resolved_config):
    from .recovery_enumeration import run_recovery_enumeration
    return run_recovery_enumeration(repo_root, config_path, output_root, resolved_config)


def _replay_tolerance_runner(repo_root, config_path, output_root, resolved_config):
    from .tdsc_replay_tolerance import run
    return run(repo_root, config_path, output_root, resolved_config)



def ensure_default_experiment_runners():
    for key, runner in (("tdsc-bounded-memory-v1", _tdsc_five_rq_runner),
                        ("tdsc-submission-tiny-v1", _tdsc_five_rq_runner),
                        ("tdsc-rq3-value-preserving-recovery-v1", _recovery_enumeration_runner),
                        ("tdsc-replay-tolerance-heterogeneity-v1", _replay_tolerance_runner)):
        if key not in EXPERIMENT_RUNNERS.keys():
            EXPERIMENT_RUNNERS.add(key, runner)
ensure_default_experiment_runners()
