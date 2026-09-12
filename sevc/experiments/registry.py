"""Selected canonical scientific routines; deployment wrappers are omitted."""
from __future__ import annotations


from typing import Any, Callable, Dict

from sevc.core.registry import Registry

ExperimentRunner = Callable[..., Dict[str, Any]]


EXPERIMENT_RUNNERS: Registry[ExperimentRunner] = Registry("experiment runner")


def _tdsc_five_rq_runner(repo_root, config_path, output_root, resolved_config, **kwargs):
    from .tdsc_five_rq_evidence import run_tdsc_five_rq_evidence
    return run_tdsc_five_rq_evidence(repo_root, config_path, output_root, resolved_config, **kwargs)



def ensure_default_experiment_runners():
    key = "tdsc-bounded-memory-v1"
    if key not in EXPERIMENT_RUNNERS.keys():
        EXPERIMENT_RUNNERS.add(key, _tdsc_five_rq_runner)
ensure_default_experiment_runners()
