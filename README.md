# SEVC

SEVC is a research prototype for evidence-supported verification and settlement
in outsourced learning. It combines committed training replay, replay-coupled
service probes, verifier admission and payment, reliability estimation, and
committee recovery. The experiment code examines settlement safety, service
evidence, strategic verification, recovery, and costs.

This repository is a **source-only research snapshot**. It includes the prototype,
main experiment implementations, public protocol parameters, and software tests.
Datasets, model weights, experiment results, plots, logs, and private deployment
information are not included. Preparing this source snapshot does not establish
that the corresponding scientific experiment has completed or passed.

## Project structure

```text
sevc/
  core/           Shared hashing, storage, timing, and runtime utilities
  data/           Dataset loading and partitioning code (no dataset files)
  models/         Model factory
  training/       Canonical training, replay proofs, and state handling
  verification/  Reference acquisition, probes, service admission, and audits
  incentives/    Payment, contracts, and the native incentive comparison
  committee/     Reliability, scheduling, and recovery
  reputation/    Reputation estimation
  experiments/   Shared runner, five-RQ assembly, and selected earlier routines
  evaluation/    Statistical calculations and independent result auditing
configs/
  protocol.json  Public scientific parameters for all three datasets and 15 packages
scripts/
  sevc.py        Thin command entry point
tests/           Offline software tests with runtime-generated synthetic inputs
docs/
  experiments.md Experiment-to-source map and historical interfaces
  inputs.md      Dataset, calibration, and run-input requirements
```

## Installation

Use Python 3.11 or later. Create an environment and install the project from this
directory. Select a compatible PyTorch/torchvision build for your CPU or CUDA
platform before installation if you need a particular accelerator build.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
sevc --help
```

The source snapshot is checked locally with Python 3.11, PyTorch 2.12.0,
torchvision 0.27.0, NumPy 2.4.6, and SciPy 1.17.1 on CPU. This is a software
validation environment, not a claim that published timings were measured there.
The package's lower dependency bounds describe intended compatibility, not a
tested matrix of every intermediate version. CUDA execution has not been tested
as part of preparing this distribution.

## Quick start: no dataset required

```bash
sevc inspect --protocol configs/protocol.json
python -m pytest -q
```

`inspect` validates the public protocol and expands its experiment matrix without
loading samples or running training. The tests generate synthetic tensors in
memory and exercise the existing training/replay, service, and statistical code.
They do not download datasets or reproduce paper results.

## Prototype interfaces

The canonical methods are available as Python APIs:

- `sevc.training.produce_worker_update` and `verify_replay_proof` produce and check
  committed training segments.
- `sevc.verification.on_demand_service.OnlineJob` uses the same service interface
  for owner-direct, plain-delegation, probe, and production-audit variants.
- `sevc.committee.executed_recovery.execute_recovery` runs the shared recovery
  controller.
- `sevc.experiments.prototype.SEVCPrototype` exposes the earlier local
  training/verification/aggregation composition.

See the synthetic tests for executable examples with no data files.

## Main experiments with your own inputs

The full study retains MNIST with an MLP and CIFAR-10/CIFAR-100 with ResNet-18.
Public seeds, sampling rules, registered methods, statistical parameters, and all
15 experiment packages are in `configs/protocol.json`. It is a substantial study;
the quick-start tests are the appropriate installation check.

Prepare the input datasets yourself in torchvision's expected directory layout.
Prepare your own calibration measurements according to [the input contract](docs/inputs.md).
Keep all data, calibration, secrets, scratch files, and run outputs outside this
repository. The following paths are examples for files you create locally:

```bash
mkdir -p ../sevc-local/scratch
sevc init-streams --output ../sevc-local/private-streams.local.json
sevc configure \
  --protocol configs/protocol.json \
  --data-root ../sevc-local/datasets \
  --scratch-root ../sevc-local/scratch \
  --calibration ../sevc-local/calibration.local.json \
  --private-streams ../sevc-local/private-streams.local.json \
  --device cpu \
  --output ../sevc-local/run.local.json
sevc run \
  --config ../sevc-local/run.local.json \
  --output-root "$(cd .. && pwd)/sevc-local/run-001"
```

These commands require the datasets and calibration file to exist. No default
measurements, data downloads, or private run streams are supplied. The runner
fails before execution when required inputs are missing. Output directories must
be new and outside the repository. `run` executes the full matrix through the
shared runner and retains its independent audit; it does not silently shrink the
protocol.

For one CUDA device, set `CUDA_VISIBLE_DEVICES` to that device and
`CUBLAS_WORKSPACE_CONFIG=:4096:8` before Python starts, then configure with
`--device cuda:0`. A run records its own environment and input identity. The
portable default uses one execution lane; changing resource settings changes
timing comparability. Historical wall-clock values must not be attributed to
this execution environment.

## Experiment coverage

The [experiment map](docs/experiments.md) connects each research question to its
implementation and lists selected earlier experiment routines. Old private
server launchers and manuscript-specific checks are omitted. Earlier scientific
routines remain callable with their documented input objects; old deployment
commands and missing original result files are not reconstructed.

## Distribution and licensing

This is a one-time source distribution. The development project remains separate;
there is no automatic synchronization or ongoing development commitment for this
snapshot. `public-manifest.json` records the distributed source files and hashes.

The source in this repository is released under the MIT License; see `LICENSE`.
The license covers this source snapshot only: no dataset, model weight or
experimental result is distributed with it, and no author contact information is
invented by this snapshot.
