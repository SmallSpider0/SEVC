# Local inputs

## Data

The loading code uses torchvision's MNIST, CIFAR-10, and CIFAR-100 training splits
with `download=False`. Obtain the datasets separately from their public sources
and place them in the layout expected by torchvision. Do not add downloaded or
processed data to this repository.

The complete study retains the first 50,000 MNIST training images and the first
40,000 CIFAR training images for its declared sampling ranges. Dataset models,
seeds, sample counts, bootstrap settings, source construction, and all method
parameters are preserved in `configs/protocol.json`.

## Calibration

Supply a JSON object with exactly three keys: `mnist`, `cifar10`, and `cifar100`.
Each value must contain these positive, finite fields:

| Field | Meaning and retained rule |
|---|---|
| `cost_unit_seconds` | Mean honest RCMP M1 verifier-assignment cost from an independent calibration sample |
| `timeout_seconds` | Maximum of one second and 1.5 times the largest measured honest-assignment wall time across the registered methods |
| `deadline_seconds` | Exactly 12 times `timeout_seconds` |
| `owner_reserve_units` | Maximum of one and 10 times the largest honest owner-admission time divided by `cost_unit_seconds` |

These are calibration inputs, not interchangeable free parameters for a paper
comparison. The source package intentionally contains no measured calibration
file. Use independent local measurements, retain their provenance outside this
repository, and freeze them before a study run. Configuration validation checks
shape, finiteness, positivity, and the deadline rule; it cannot certify how a
user obtained the measurements. Do not substitute values from a different host
or claim that supplied values were validated by this source release.

The low-level service APIs and earlier cost-measurement routines are included
for composing local calibration work. This distribution does not run a new
calibration campaign or provide an automatic substitute for its protocol.

## Private streams and runtime configuration

`sevc init-streams` generates independent `roles` and `audits` values for a new
run using the operating system's random source. The output file is created with
owner-only permissions and exclusive creation. It contains private run inputs;
keep it outside version control. No historical stream is shipped.

`sevc configure` combines your protocol, paths, calibration, and streams into a
new local configuration. The canonical algorithm version and scientific fields
stay fixed; a new run UUID distinguishes this execution. The configuration and
all run outputs can include your local paths and device identity, so keep them
outside the source repository.

## Result consumers

Independent auditing and statistics consume the records produced by the shared
runner: unit objects, report commitments/reveals, source/reference receipts,
role-accounting events, and recovery traces. Their schemas are defined by the
corresponding dataclasses and checks in `sevc/verification`, `sevc/core`,
`sevc/evaluation/scoped_result_audit.py`, and `scoped_statistics.py`.

These consumers are supplied as code. There are no recorded result files or
stored “expected paper outputs” in the distribution. Missing inputs must be
provided by the user; they are not interpreted as a favorable or negative result.
