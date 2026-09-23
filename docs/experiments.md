# Experiment-to-source map

All study methods reuse the training/replay implementation and shared service
interfaces. `scripts/sevc.py` or the installed `sevc` command dispatches to
`sevc/experiments/runner.py`, its restricted registry, and
`tdsc_five_rq_evidence.py`. `scoped_five_rq_units.py` assembles the 15 packages.

| Question / package | Main implementation |
|---|---|
| Source validity (M0), training-stage branches (M7) | `training/replay_sources.py`, `training/replay_state.py`, `experiments/scoped_five_rq_units.py` |
| RQ1: admission and settlement (M1, M4S) | `verification/on_demand_service.py`, `production_audit.py`, `committee/executed_recovery.py` |
| RQ2: matched mechanisms and references (M1, M2, M3) | `verification/reference_acquisition.py`, `paid_replay_service.py`, `on_demand_service.py` |
| RQ2: native incentive game (M8) | `incentives/native_peer_prediction.py` |
| RQ3: shared information, partial replay and caches (M2, M2C) | `verification/paid_replay_service.py`, `on_demand_service.py` |
| RQ4: recovery and resource constraints (M4L, M4S, M4X, M4B, M4T) | `committee/executed_recovery.py`, `evaluation/recovery_graph_audit.py` |
| RQ5: costs, scaling and utilities (M5, M6, M6C) | `core/role_accounting.py`, `experiments/scoped_five_rq_units.py`, `evaluation/scoped_statistics.py` |
| Independent audit and statistical analysis | `evaluation/scoped_result_audit.py`, `scoped_statistics.py` |

Paths in this table are relative to `sevc/`. M0/M6/M7 supply validity and scale
context across questions. `inspect` lists exact package counts from the frozen
public parameters. Matrix expansion preserves its declared independent units
and exact reuse; reused rows do not create additional independent observations.

## Current four-question evaluation

The manuscript's current evaluation (detection, incentives, settlement and
recovery, overhead) runs through the same runner. The registry exposes
`tdsc-submission-tiny-v1` for the fixed-design main study and its detection,
participation and cost supplements, `tdsc-rq3-value-preserving-recovery-v1` for the
constructed recovery instances, and `tdsc-replay-tolerance-heterogeneity-v1` for
the replay-tolerance study. Their full configurations contain private deployment
fields and are not bundled; `configs/tdsc_replay_tolerance_heterogeneity_v1.json`
is a public projection that runs as a non-formal standalone study after
`data_root` is set.

| Question / analysis | Main implementation |
|---|---|
| End-to-end settlement comparison (existing rules, ablations, SEVC and single-verifier SEVC; reference-replay residuals) | `evaluation/end_to_end_comparison.py`, `committee/executed_recovery.py` |
| RQ1 detection, verifier view and native comparison | `verification/on_demand_service.py`, `verification/replay_coupled_probes.py`, `evaluation/verifier_view_linkability.py`, `committee/depol_arbitration.py`, `verification/depol_local.py`, `evaluation/depol_shared_shortcut.py` |
| RQ2 incentives and bond sensitivity | `evaluation/f_rq234_audit.py`, `evaluation/cost_incentive_reanalysis.py` |
| RQ3 settlement and recovery | `committee/executed_recovery.py`, `experiments/recovery_enumeration.py`, `evaluation/value_preserving_recovery.py` |
| RQ4 overhead and matched-budget coverage | `core/role_accounting.py`, `evaluation/overhead_supplement.py`, `evaluation/cost_incentive_reanalysis.py` |
| Owner-verified source failures | `verification/reference_acquisition.py`, `evaluation/owner_failure_branch_audit.py` |
| Replay tolerance across numerical configurations | `experiments/tdsc_replay_tolerance.py` |

The analysis modules read recorded unit rows supplied by the user; no recorded
rows are bundled. Envelope equalization is available as the method key
`rcmp-opaque-gradient-continuation-gpu-equalized-v1`.

## Earlier main experiments

The following scientific routines are retained from the earlier studies. Their
private deployment, prior-run hash locks, manuscript checks, and server/process
management are omitted. The retained function bodies and their required helpers
come from the same canonical implementation. They are **library interfaces**, not
old server commands relabeled as portable end-to-end experiments.

| Study | Source and entry | Input contract |
|---|---|---|
| Trainer attribution and robust aggregation | `experiments/tdsc_comparative_robustness.py::_run_source_unit`; `verification/robust_aggregation.py` | Dataset/model context, declared worker partitions, method/attack parameters and seed; returns source-unit trajectories and metrics |
| Segment sampling and replay tolerance | `experiments/tdsc_sampling_replay_stress.py::_process_segment`; `verification/sampling_replay_stress.py` | A replay segment, declared sampling/tolerance protocol and dataset/model context |
| Settlement-interface comparisons | `evaluation/tdsc_direct_execution.py` | `DirectExecutionContext` with trace, supplied source-unit metrics and method parameters; includes no-verification, owner-only, PoL, DePoL-style, Refiner-style and SEVC interfaces |
| Reliability and failure simulation | `evaluation/tdsc_failure_scale_simulation.py` | Explicit trace/roster/source inputs and failure law; simulation results are not measured service outcomes |
| Recovery ablations and paired effects | `experiments/tdsc_sevc_ablation_frontier.py` | User-supplied direct/failure rows and versioned parameters for summaries and paired bootstrap |
| Canonical probe construction | `experiments/tdsc_canonical_probe_certificate.py::_build_block_sources` and `_timed_canonical_replay` | Model/data configuration and committed sources; canonical probe logic is in `verification/replay_coupled_probes.py` |
| Adaptive recognizability | `experiments/tdsc_adaptive_recognizer.py`; `verification/adaptive_recognizer.py` | User-supplied task-envelope rows and training/holdout partitions; feature and classifier implementations are retained |
| System-cost instrumentation | `experiments/tdsc_system_overhead.py::_measure_block` and `_uninstrumented_block`; `evaluation/tdsc_system_overhead.py` | Replay source/context, device/timer, protocol and method settings; no historical timing values supplied |
| Probe ablation / incentive sensitivity | `evaluation/tdsc_rcmp_ablation_sensitivity.py` | Source records explicitly supplied by the user, with their input identities |

Consult the preserved signatures, type annotations and dataclasses for the full
input structures. No genuine sample, prior result, source metric, trajectory or
timing observation is bundled to fill these inputs.

## Interpretation boundaries

The native peer-prediction game retains the published four-observation,
two-verifier specification and its mathematical score table. Those constants are
algorithm inputs, not this project's measured results. References are identified
in the public protocol. DePoL-style and Refiner-style comparison interfaces do not
represent complete reimplementations of those external systems.

The source snapshot retains the registered method and threat-model distinctions.
Correct report admission, individual replay effort, coalition incentives, and
availability are separate outcomes. Source distribution does not convert a
pending, limited or negative scientific result into a completed positive claim.
