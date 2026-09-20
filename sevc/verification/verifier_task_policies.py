"""Shared task preparation and adjudication policies for verifier incentives."""

from __future__ import annotations

from collections import Counter
import hashlib
import math
import random
from typing import Any, Mapping, Sequence

from sevc.incentives.verifier_protocol import (
    AllocationOutcome,
    AdjudicationRecord,
    CommittedVerifierReport,
    PublicTaskBundle,
    SealedEvaluationTruth,
    SettlementOutcome,
    SettlementRequest,
    TASK_POLICIES,
    SETTLEMENT_POLICIES,
    VerifierSettlement,
    PublicReplaySegment,
)


def tamper_replay_proof(
    proof: Any, checkpoint_index: int, delta: float
) -> Any:
    """Create the canonical invalid replay segment without mutating the proof."""

    import torch

    from sevc.training import ReplayProof

    checkpoints = list(proof.checkpoints)
    if not checkpoints:
        raise ValueError("cannot tamper with an empty replay proof")
    target_index = checkpoint_index % len(checkpoints)
    state = {
        key: value.detach().clone() for key, value in checkpoints[target_index].items()
    }
    tensor_key = next(key for key in sorted(state) if state[key].is_floating_point())
    tensor = state[tensor_key].clone()
    flattened = tensor.reshape(-1)
    flattened[0] = flattened[0] + torch.as_tensor(delta, dtype=flattened.dtype)
    state[tensor_key] = tensor
    checkpoints[target_index] = state
    return ReplayProof(
        initial_state=proof.initial_state,
        batches=proof.batches,
        checkpoints=tuple(checkpoints),
        learning_rate=proof.learning_rate,
        momentum=proof.momentum,
    )


def produce_canonical_replay_proof(
    seed: int, settings: Mapping[str, Any], device: str
) -> tuple[Any, Any]:
    """Use the canonical trainer to produce material consumed by trajectory-replay."""

    from sevc.core.runtime import set_global_seed
    from sevc.data import make_synthetic_bundle, partition_dataset
    from sevc.models import build_model
    from sevc.training import WorkerBehavior, produce_worker_update

    # Protocol block identities are 64-bit SHA-256 prefixes, while NumPy's
    # legacy RandomState (used by the canonical partitioner) accepts uint32.
    # Preserve the full block seed in evidence and derive this execution seed
    # deterministically at the sole canonical replay-generation boundary.
    runtime_seed = int(seed) & 0xFFFFFFFF
    set_global_seed(runtime_seed)
    bundle = make_synthetic_bundle(
        seed=runtime_seed,
        sample_count=int(settings["sample_count"]),
        class_count=int(settings["class_count"]),
    )
    loader = partition_dataset(
        bundle.train,
        [1.0],
        batch_size=int(settings["batch_size"]),
        seed=runtime_seed,
        shuffle_batches=False,
    )[0]
    factory = lambda: build_model(
        str(settings["model"]), class_count=bundle.num_classes
    )
    global_model = factory().to(device)
    update = produce_worker_update(
        global_model,
        factory,
        loader,
        WorkerBehavior.NORMAL,
        device=device,
        learning_rate=float(settings["learning_rate"]),
        momentum=float(settings["momentum"]),
        local_epochs=int(settings["local_epochs"]),
        max_batches=int(settings["max_batches_per_worker"]),
        capture_replay=True,
    )
    if update.replay_proof is None or not update.replay_proof.checkpoints:
        raise RuntimeError("canonical training did not produce a replay proof")
    return update.replay_proof, factory


def produce_canonical_replay_microsegments(
    seed: int,
    settings: Mapping[str, Any],
    device: str,
    *,
    count: int,
) -> tuple[tuple[Any, ...], Any]:
    """Produce independently replayable one-batch proofs for one task block."""

    from sevc.core.runtime import set_global_seed
    from sevc.data import make_synthetic_bundle, partition_dataset
    from sevc.models import build_model
    from sevc.training import WorkerBehavior, produce_worker_update

    if count <= 0:
        raise ValueError("microsegment count must be positive")
    if int(settings["local_epochs"]) != 1:
        raise ValueError("canonical replay microsegments require one local epoch")
    runtime_seed = int(seed) & 0xFFFFFFFF
    set_global_seed(runtime_seed)
    minimum_samples = math.ceil(
        count * int(settings["batch_size"]) / 0.75
    )
    bundle = make_synthetic_bundle(
        seed=runtime_seed,
        sample_count=max(int(settings["sample_count"]), minimum_samples),
        class_count=int(settings["class_count"]),
    )
    loader = partition_dataset(
        bundle.train,
        [1.0],
        batch_size=int(settings["batch_size"]),
        seed=runtime_seed,
        shuffle_batches=False,
    )[0]
    batches = tuple(loader)[:count]
    if len(batches) != count:
        raise RuntimeError(
            f"canonical task block yielded {len(batches)} batches, expected {count}"
        )
    factory = lambda: build_model(
        str(settings["model"]), class_count=bundle.num_classes
    )
    global_model = factory().to(device)
    proofs = []
    for batch in batches:
        update = produce_worker_update(
            global_model,
            factory,
            (batch,),
            WorkerBehavior.NORMAL,
            device=device,
            learning_rate=float(settings["learning_rate"]),
            momentum=float(settings["momentum"]),
            local_epochs=1,
            max_batches=1,
            capture_replay=True,
        )
        proof = update.replay_proof
        if proof is None or len(proof.batches) != 1 or len(proof.checkpoints) != 1:
            raise RuntimeError("canonical one-batch replay proof generation failed")
        proofs.append(proof)
    return tuple(proofs), factory


def _selected_segments(
    public_view: Any,
    segment_ids: Sequence[str],
) -> tuple[Any, ...]:
    by_id = {item.segment_id: item for item in public_view.replay_segments}
    missing = sorted(set(segment_ids) - set(by_id))
    if missing:
        raise ValueError(f"task policy references unknown segments: {missing}")
    return tuple(by_id[segment_id] for segment_id in segment_ids)


def _bundles(
    public_view: Any,
    allocation: AllocationOutcome,
    segment_ids: Sequence[str],
) -> tuple[PublicTaskBundle, ...]:
    if allocation.status != "ALLOCATED":
        return ()
    segments = _selected_segments(public_view, segment_ids)
    return tuple(
        PublicTaskBundle(
            scenario_id=public_view.scenario_id,
            verifier_id=verifier_id,
            segments=segments,
        )
        for verifier_id in allocation.selected_verifiers
    )


@TASK_POLICIES.register("majority-only")
def prepare_majority_only(
    public_view: Any,
    evaluation_bundle: SealedEvaluationTruth,
    allocation: AllocationOutcome,
    parameters: Mapping[str, Any],
) -> tuple[PublicTaskBundle, ...]:
    del parameters
    if (
        public_view.protocol_version
        in {
            "verifier-incentive-screen-v2",
            "verifier-incentive-screen-v3",
            "verifier-incentive-screen-v4",
            "verifier-incentive-screen-v5",
            "verifier-incentive-screen-v6",
            "verifier-incentive-confirmatory-v1",
            "verifier-incentive-scale-v1",
            "verifier-incentive-paper-reconfirmatory-v1",
            "verifier-incentive-system-overhead-v1",
        }
        and allocation.transfer_terms
    ):
        return build_production_assignment_bundles(
            public_view, evaluation_bundle, allocation
        )
    return _bundles(
        public_view,
        allocation,
        tuple(item.segment_id for item in public_view.replay_segments),
    )


def build_production_assignment_bundles(
    public_view: Any,
    evaluation_bundle: SealedEvaluationTruth,
    allocation: AllocationOutcome,
) -> tuple[PublicTaskBundle, ...]:
    if allocation.status != "ALLOCATED" or not allocation.transfer_terms:
        return ()
    public_by_id = {item.segment_id: item for item in public_view.replay_segments}
    production_ids = tuple(
        sorted(
            item.segment_id
            for item in evaluation_bundle.segments
            if not item.is_sentinel
        )
    )
    bundles = []
    for term in sorted(
        allocation.transfer_terms, key=lambda item: (item.job_id, item.verifier_id)
    ):
        source_ids = list(production_ids)
        order_seed = int.from_bytes(
            hashlib.sha256(
                (
                    f"baseline-task-permutation|{public_view.protocol_version}|"
                    f"{public_view.trainer_output_identity}|{term.job_id}|"
                    f"{term.verifier_id}"
                ).encode("utf-8")
            ).digest()[:8],
            "big",
        )
        random.Random(order_seed).shuffle(source_ids)
        segments = tuple(
            _personalized_segment(
                public_by_id[source_id],
                scenario_id=public_view.scenario_id,
                verifier_id=term.verifier_id,
                job_id=term.job_id,
                position=position,
            )
            for position, source_id in enumerate(source_ids)
        )
        bundles.append(
            PublicTaskBundle(
                scenario_id=public_view.scenario_id,
                verifier_id=term.verifier_id,
                job_id=term.job_id,
                segments=segments,
                bundle_nonce=hashlib.sha256(
                    (
                        f"baseline-bundle|{public_view.scenario_id}|"
                        f"{term.job_id}|{term.verifier_id}|{order_seed}"
                    ).encode("utf-8")
                ).hexdigest(),
            )
        )
    return tuple(bundles)


@TASK_POLICIES.register("hidden-audit-v1")
def prepare_hidden_audit_v1(
    public_view: Any,
    evaluation_bundle: SealedEvaluationTruth,
    allocation: AllocationOutcome,
    parameters: Mapping[str, Any],
) -> tuple[PublicTaskBundle, ...]:
    del evaluation_bundle
    audit_rate = float(parameters["audit_rate"])
    if not 0 < audit_rate <= 1:
        raise ValueError("audit_rate must lie in (0, 1]")
    count = max(1, math.ceil(len(public_view.replay_segments) * audit_rate))
    return _bundles(
        public_view,
        allocation,
        tuple(item.segment_id for item in public_view.replay_segments[:count]),
    )


@TASK_POLICIES.register("hidden-sentinel-bundle")
def prepare_hidden_sentinel_bundle(
    public_view: Any,
    evaluation_bundle: SealedEvaluationTruth,
    allocation: AllocationOutcome,
    parameters: Mapping[str, Any],
) -> tuple[PublicTaskBundle, ...]:
    requested = int(parameters["sentinel_count"])
    if (
        public_view.protocol_version
        in {
            "verifier-incentive-screen-v2",
            "verifier-incentive-screen-v3",
            "verifier-incentive-screen-v4",
            "verifier-incentive-screen-v5",
            "verifier-incentive-screen-v6",
            "verifier-incentive-confirmatory-v1",
            "verifier-incentive-scale-v1",
            "verifier-incentive-paper-reconfirmatory-v1",
            "verifier-incentive-system-overhead-v1",
        }
        and allocation.transfer_terms
    ):
        return build_mixed_personalized_bundles(
            public_view,
            evaluation_bundle,
            allocation,
            sentinel_count=requested,
            maximum_jaccard=float(
                parameters.get("maximum_personalized_sentinel_jaccard", 0.75)
            ),
            bundle_domain=(
                ""
                if "recovery_wave_index" not in parameters
                else (
                    f"psrr-wave-{int(parameters['recovery_wave_index'])}|"
                    f"{parameters.get('recovery_roster_sha256', '')}|"
                    f"{parameters.get('recovery_activation_nonce', '')}"
                )
            ),
        )
    sentinel_ids = tuple(
        item.segment_id for item in evaluation_bundle.segments if item.is_sentinel
    )
    if len(sentinel_ids) < requested:
        raise ValueError(
            f"sentinel pool has {len(sentinel_ids)} segments, requires {requested}"
        )
    return _bundles(public_view, allocation, sentinel_ids[:requested])


def _personalized_segment(
    source: PublicReplaySegment,
    *,
    scenario_id: str,
    verifier_id: str,
    job_id: str,
    position: int,
    bundle_domain: str = "",
) -> PublicReplaySegment:
    material = (
        f"sevc-personalized-task-v2|{scenario_id}|{verifier_id}|{job_id}|"
        f"{source.segment_id}|{position}"
        + (f"|{bundle_domain}" if bundle_domain else "")
    )
    segment_id = hashlib.sha256(material.encode("utf-8")).hexdigest()[:24]
    nonce = hashlib.sha256(f"nonce|{material}".encode("utf-8")).hexdigest()[:24]
    features = list(source.public_features)
    if len(features) >= 10:
        features[-3] = int(segment_id[:8], 16) / float(0xFFFFFFFF)
        features[-2] = 1.0
        features[-1] = int(nonce[:8], 16) / float(0xFFFFFFFF)
    return PublicReplaySegment(
        segment_id=segment_id,
        payload_digest=source.payload_digest,
        public_features=tuple(features),
        public_nonce=nonce,
        serialized_size=source.serialized_size,
        tensor_count=source.tensor_count,
        checkpoint_count=source.checkpoint_count,
        batch_count=source.batch_count,
    )


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return 0.0 if not union else len(left & right) / len(union)


def build_mixed_personalized_bundles(
    public_view: Any,
    evaluation_bundle: SealedEvaluationTruth,
    allocation: AllocationOutcome,
    *,
    sentinel_count: int,
    maximum_jaccard: float,
    bundle_domain: str = "",
) -> tuple[PublicTaskBundle, ...]:
    """Build one production+sentinel block for every assignment transfer."""

    if allocation.status != "ALLOCATED" or not allocation.transfer_terms:
        return ()
    if sentinel_count <= 0 or not 0 <= maximum_jaccard <= 1:
        raise ValueError("mixed-bundle parameters are invalid")
    public_by_id = {item.segment_id: item for item in public_view.replay_segments}
    truth_by_id = {item.segment_id: item for item in evaluation_bundle.segments}
    production_ids = tuple(
        sorted(key for key, value in truth_by_id.items() if not value.is_sentinel)
    )
    sentinel_ids = tuple(
        sorted(key for key, value in truth_by_id.items() if value.is_sentinel)
    )
    if not production_ids or len(sentinel_ids) < sentinel_count:
        raise ValueError("mixed task bank lacks production or sentinel capacity")
    prior_by_job: dict[str, list[set[str]]] = {}
    bundles: list[PublicTaskBundle] = []
    for term in sorted(
        allocation.transfer_terms, key=lambda item: (item.job_id, item.verifier_id)
    ):
        selected: tuple[str, ...] | None = None
        for attempt in range(10_000):
            material = (
                f"sentinel-subset|{public_view.protocol_version}|"
                f"{public_view.trainer_output_identity}|{term.job_id}|"
                f"{term.verifier_id}|"
                f"{sentinel_count}|{attempt}"
                + (f"|{bundle_domain}" if bundle_domain else "")
            )
            seed = int.from_bytes(
                hashlib.sha256(material.encode("utf-8")).digest()[:8], "big"
            )
            candidate = tuple(
                sorted(random.Random(seed).sample(sentinel_ids, sentinel_count))
            )
            candidate_set = set(candidate)
            if all(
                _jaccard(candidate_set, previous) <= maximum_jaccard + 1e-12
                for previous in prior_by_job.get(term.job_id, [])
            ):
                selected = candidate
                prior_by_job.setdefault(term.job_id, []).append(candidate_set)
                break
        if selected is None:
            raise RuntimeError("unable to construct the frozen personalized subsets")
        source_ids = [*production_ids, *selected]
        order_seed = int.from_bytes(
            hashlib.sha256(
                (
                    f"task-permutation|{public_view.protocol_version}|"
                    f"{public_view.trainer_output_identity}|{term.job_id}|"
                    f"{term.verifier_id}"
                    + (f"|{bundle_domain}" if bundle_domain else "")
                ).encode("utf-8")
            ).digest()[:8],
            "big",
        )
        random.Random(order_seed).shuffle(source_ids)
        segments = tuple(
            _personalized_segment(
                public_by_id[source_id],
                scenario_id=public_view.scenario_id,
                verifier_id=term.verifier_id,
                job_id=term.job_id,
                position=position,
                bundle_domain=bundle_domain,
            )
            for position, source_id in enumerate(source_ids)
        )
        bundle_nonce = hashlib.sha256(
            (
                f"bundle-nonce|{public_view.scenario_id}|{term.job_id}|"
                f"{term.verifier_id}|{order_seed}"
                + (f"|{bundle_domain}" if bundle_domain else "")
            ).encode("utf-8")
        ).hexdigest()
        bundles.append(
            PublicTaskBundle(
                scenario_id=public_view.scenario_id,
                verifier_id=term.verifier_id,
                job_id=term.job_id,
                segments=segments,
                bundle_nonce=bundle_nonce,
            )
        )
    return tuple(bundles)


def expand_personalized_evaluation_truth(
    bundles: Sequence[PublicTaskBundle],
    source_truth: SealedEvaluationTruth,
) -> SealedEvaluationTruth:
    """Create evaluator-only alias truth after the commit deadline closes."""

    proof_truth = {
        item.proof_sha256: item for item in source_truth.segments
    }
    if len(proof_truth) != len(source_truth.segments):
        raise ValueError("source proof identities must be unique")
    rows = []
    for bundle in bundles:
        for segment in bundle.segments:
            try:
                source = proof_truth[segment.payload_digest]
            except KeyError as exc:
                raise ValueError("bundle payload is absent from sealed source truth") from exc
            rows.append(
                type(source)(
                    segment_id=segment.segment_id,
                    expected_verdict=source.expected_verdict,
                    is_sentinel=source.is_sentinel,
                    technical_failure=False,
                    source_segment_id=source.source_segment_id or source.segment_id,
                    job_id=bundle.job_id,
                    proof_sha256=source.proof_sha256,
                )
            )
    return SealedEvaluationTruth(
        scenario_id=source_truth.scenario_id,
        segments=tuple(rows),
    )


def score_threshold_report(
    report: CommittedVerifierReport,
    evaluation_bundle: SealedEvaluationTruth,
    *,
    failure_threshold: int,
    sentinel_only: bool = False,
    required_probe_ids: Sequence[str] | None = None,
) -> tuple[str, int, int]:
    """Apply technical-failure filtering before threshold integrity scoring."""

    if failure_threshold <= 0:
        raise ValueError("failure_threshold must be positive")
    if not report.committed or (required_probe_ids is not None and not report.revealed):
        return "DROPOUT", 0, 0
    by_id = {item.segment_id: item for item in evaluation_bundle.segments}
    if required_probe_ids is not None:
        required = tuple(required_probe_ids)
        if not required or len(set(required)) != len(required):
            raise ValueError("required probe identities must be nonempty and unique")
        if report.scenario_id != evaluation_bundle.scenario_id:
            raise ValueError("report/reference scenario mismatch")
        # Only owner-authenticated reference failures may remove comparability.
        # An actor's missing answer is a mismatch, not a technical exemption.
        if not set(required).issubset(report.ordered_segment_ids):
            return "FAIL_CONFIRMED", len(set(required) - set(report.ordered_segment_ids)), 0
        unavailable = sum(
            k not in by_id or by_id[k].technical_failure or not by_id[k].is_sentinel
            for k in required
        )
        if unavailable:
            return "TECHNICAL_FAILURE", 0, unavailable
    mismatch_count = 0
    technical_count = 0
    comparable_count = 0
    for segment_id, verdict in zip(report.ordered_segment_ids, report.verdicts):
        try:
            expected = by_id[segment_id]
        except KeyError as exc:
            raise ValueError(f"report references unknown segment {segment_id}") from exc
        if sentinel_only and not expected.is_sentinel:
            continue
        if expected.technical_failure:
            technical_count += 1
            continue
        comparable_count += 1
        mismatch_count += int(verdict is None or bool(verdict) != expected.expected_verdict)
    if comparable_count == 0:
        return "TECHNICAL_FAILURE", mismatch_count, technical_count
    if mismatch_count >= failure_threshold:
        return "FAIL_CONFIRMED", mismatch_count, technical_count
    return "PASS", mismatch_count, technical_count


def adjudicate_threshold_report(
    report: CommittedVerifierReport,
    evaluation_bundle: SealedEvaluationTruth,
    bank: object,
    *,
    failure_threshold: int,
    seed: int,
) -> AdjudicationRecord:
    """Run first-pass sentinel scoring and real canonical re-adjudication."""

    from sevc.verification.verifier_replay_bank import ReplayObservationProvider

    if not isinstance(bank, ReplayObservationProvider):
        raise TypeError("adjudication requires the canonical observation provider")
    if not report.job_id:
        raise ValueError("v2 adjudication requires a job identity")
    if not report.committed:
        return AdjudicationRecord(
            scenario_id=report.scenario_id,
            verifier_id=report.verifier_id,
            job_id=report.job_id,
            initial_status="DROPOUT",
            final_status="DROPOUT",
            first_pass_mismatch_count=0,
            rerun_count=0,
            confirmed_mismatch_count=0,
            technical_failure_count=0,
        )
    by_id = {item.segment_id: item for item in evaluation_bundle.segments}
    mismatches = []
    for segment_id, verdict in zip(report.ordered_segment_ids, report.verdicts):
        try:
            truth = by_id[segment_id]
        except KeyError as exc:
            raise ValueError(f"report references unknown alias {segment_id}") from exc
        if not truth.is_sentinel:
            continue
        if verdict is None or bool(verdict) != truth.expected_verdict:
            mismatches.append(truth)
    if len(mismatches) < failure_threshold:
        return AdjudicationRecord(
            scenario_id=report.scenario_id,
            verifier_id=report.verifier_id,
            job_id=report.job_id,
            initial_status="PASS",
            final_status="PASS",
            first_pass_mismatch_count=len(mismatches),
            rerun_count=0,
            confirmed_mismatch_count=0,
            technical_failure_count=0,
        )
    confirmed = 0
    technical = 0
    cache_keys = []
    for index, truth in enumerate(mismatches):
        try:
            observation = bank.observe(
                truth.proof_sha256,
                execution_plan_key="adjudication-rerun",
                seed=int(seed) + index,
            )
            cache_keys.append(observation.cache_key)
            if observation.verdict == truth.expected_verdict:
                confirmed += 1
            else:
                technical += 1
        except (KeyError, RuntimeError, ValueError):
            technical += 1
    final_status = (
        "FAIL_CONFIRMED"
        if confirmed >= failure_threshold
        else "TECHNICAL_FAILURE"
    )
    return AdjudicationRecord(
        scenario_id=report.scenario_id,
        verifier_id=report.verifier_id,
        job_id=report.job_id,
        initial_status="FAIL",
        final_status=final_status,
        first_pass_mismatch_count=len(mismatches),
        rerun_count=len(mismatches),
        confirmed_mismatch_count=confirmed,
        technical_failure_count=technical,
        rerun_cache_keys=tuple(cache_keys),
    )


def aggregate_settlements(
    rows: Sequence[VerifierSettlement],
    *,
    evaluator_reserve: float,
    diagnostics: tuple[tuple[str, Any], ...] = (),
) -> SettlementOutcome:
    row_values = tuple(rows)
    if not row_values:
        raise ValueError("settlement requires at least one verifier row")
    statuses = {row.status for row in row_values}
    if "FAIL_CONFIRMED" in statuses:
        status = "FAIL_CONFIRMED"
    elif statuses == {"DROPOUT"}:
        status = "DROPOUT"
    elif "TECHNICAL_FAILURE" in statuses and statuses <= {
        "TECHNICAL_FAILURE",
        "DROPOUT",
    }:
        status = "TECHNICAL_FAILURE"
    else:
        status = "PASS"
    return SettlementOutcome(
        status=status,
        service_fee=sum(row.service_fee for row in row_values),
        refundable_bond=sum(row.refundable_bond for row in row_values),
        slashed_bond=sum(row.slashed_bond for row in row_values),
        owner_expenditure=sum(row.owner_expenditure for row in row_values)
        + float(evaluator_reserve),
        verifier_cost=sum(row.verifier_cost for row in row_values),
        effort_fraction=sum(row.effort_fraction for row in row_values)
        / len(row_values),
        committed=sum(int(row.committed) for row in row_values),
        revealed=sum(int(row.revealed) for row in row_values),
        accepted_report=sum(int(row.accepted_report) for row in row_values),
        technical_failure=sum(int(row.technical_failure) for row in row_values),
        abstained=sum(int(row.abstained) for row in row_values),
        rows=row_values,
        diagnostics=diagnostics,
    )


def _request_maps(
    request: SettlementRequest,
) -> tuple[dict[str, float], dict[str, float]]:
    costs = {str(key): float(value) for key, value in request.verifier_costs}
    efforts = {str(key): float(value) for key, value in request.effort_fractions}
    return costs, efforts


@SETTLEMENT_POLICIES.register("majority-match")
def settle_majority_match(
    request: SettlementRequest, parameters: Mapping[str, Any]
) -> SettlementOutcome:
    del parameters
    if request.allocation.transfer_terms:
        report_map = {
            (item.verifier_id, item.job_id): item for item in request.reports
        }
        truth_map = {
            item.segment_id: item for item in request.evaluation_bundle.segments
        }
        assignment_costs = {
            (str(verifier_id), str(job_id)): float(value)
            for verifier_id, job_id, value in request.assignment_costs
        }
        assignment_efforts = {
            (str(verifier_id), str(job_id)): float(value)
            for verifier_id, job_id, value in request.assignment_effort_fractions
        }
        votes: dict[tuple[str, str], list[bool]] = {}
        for report in request.reports:
            if not report.committed:
                continue
            for segment_id, verdict in zip(
                report.ordered_segment_ids, report.verdicts
            ):
                truth = truth_map.get(segment_id)
                if truth is None or truth.is_sentinel or verdict is None:
                    continue
                votes.setdefault(
                    (report.job_id, truth.source_segment_id or truth.segment_id), []
                ).append(bool(verdict))
        majorities = {
            key: Counter(values)[True] >= Counter(values)[False]
            for key, values in votes.items()
            if values
        }
        rows = []
        for term in request.allocation.transfer_terms:
            identity = (term.verifier_id, term.job_id)
            report = report_map.get(identity)
            if report is None or not report.committed:
                accepted = False
                status = "DROPOUT"
            else:
                comparable = []
                for segment_id, verdict in zip(
                    report.ordered_segment_ids, report.verdicts
                ):
                    truth = truth_map.get(segment_id)
                    if truth is None or truth.is_sentinel:
                        continue
                    key = (term.job_id, truth.source_segment_id or truth.segment_id)
                    comparable.append(
                        verdict is not None
                        and key in majorities
                        and bool(verdict) == majorities[key]
                    )
                accepted = bool(comparable) and all(comparable)
                status = "PASS" if accepted else "FAIL_CONFIRMED"
            rows.append(
                VerifierSettlement(
                    verifier_id=term.verifier_id,
                    job_id=term.job_id,
                    status=status,
                    service_fee=term.service_fee if accepted else 0.0,
                    refundable_bond=0.0,
                    slashed_bond=0.0,
                    owner_expenditure=term.service_fee if accepted else 0.0,
                    verifier_cost=assignment_costs.get(identity, 0.0),
                    effort_fraction=assignment_efforts.get(identity, 0.0),
                    committed=bool(report and report.committed),
                    revealed=bool(report and report.revealed),
                    accepted_report=accepted,
                    technical_failure=False,
                    abstained=status == "DROPOUT",
                )
            )
        return aggregate_settlements(
            rows,
            evaluator_reserve=(
                request.allocation.sentinel_generation_cost
                + request.allocation.adjudication_reserve
            ),
            diagnostics=(("per_assignment_transfer_conserved", True),),
        )
    costs, efforts = _request_maps(request)
    selected_count = len(request.allocation.selected_verifiers)
    if selected_count == 0:
        raise ValueError("majority settlement requires an allocated committee")
    fee = request.allocation.service_fee / selected_count
    reserve = max(
        0.0,
        request.allocation.worst_case_transfer - request.allocation.service_fee,
    )
    majorities: dict[str, bool] = {}
    segment_ids = {
        segment_id
        for report in request.reports
        for segment_id in report.ordered_segment_ids
    }
    for segment_id in segment_ids:
        votes = [
            verdict
            for report in request.reports
            for candidate_id, verdict in zip(
                report.ordered_segment_ids, report.verdicts
            )
            if candidate_id == segment_id and verdict is not None
        ]
        if votes:
            counts = Counter(bool(value) for value in votes)
            majorities[segment_id] = counts[True] >= counts[False]
    rows = []
    for report in request.reports:
        accepted = bool(report.committed) and all(
            verdict is not None and bool(verdict) == majorities.get(segment_id)
            for segment_id, verdict in zip(
                report.ordered_segment_ids, report.verdicts
            )
        )
        status = "PASS" if accepted else "DROPOUT" if not report.committed else "FAIL_CONFIRMED"
        rows.append(
            VerifierSettlement(
                verifier_id=report.verifier_id,
                status=status,
                service_fee=fee if accepted else 0.0,
                refundable_bond=0.0,
                slashed_bond=0.0,
                owner_expenditure=fee if accepted else 0.0,
                verifier_cost=costs.get(report.verifier_id, 0.0),
                effort_fraction=efforts.get(report.verifier_id, 0.0),
                committed=report.committed,
                revealed=report.revealed,
                accepted_report=accepted,
                technical_failure=False,
                abstained=not report.committed,
            )
        )
    return aggregate_settlements(rows, evaluator_reserve=reserve)


def settle_threshold_assignment(
    report: CommittedVerifierReport, evaluation_bundle: SealedEvaluationTruth, *,
    failure_threshold: int, fee: float, bond: float, cost: float, effort: float,
    sentinel_only: bool = False,
    required_probe_ids: Sequence[str] | None = None,
) -> VerifierSettlement:
    """Shared threshold service transfer law for one committed assignment."""
    status, mismatches, technical = score_threshold_report(
        report, evaluation_bundle, failure_threshold=failure_threshold,
        sentinel_only=sentinel_only,
        required_probe_ids=required_probe_ids,
    )
    passed = status == "PASS"
    slashed = bond if status in {"FAIL_CONFIRMED", "DROPOUT"} else 0.0
    return VerifierSettlement(
        verifier_id=report.verifier_id, status=status,
        service_fee=fee if passed else 0.0, refundable_bond=bond - slashed,
        slashed_bond=slashed, owner_expenditure=fee if passed else 0.0,
        verifier_cost=cost, effort_fraction=effort, committed=report.committed,
        revealed=report.revealed, accepted_report=passed,
        technical_failure=status == "TECHNICAL_FAILURE", abstained=status == "DROPOUT",
        diagnostics=(("mismatch_count", mismatches), ("technical_segment_count", technical)),
    )


@SETTLEMENT_POLICIES.register("critical-audit-v1")
def settle_critical_audit_v1(
    request: SettlementRequest, parameters: Mapping[str, Any]
) -> SettlementOutcome:
    costs, efforts = _request_maps(request)
    selected_count = len(request.allocation.selected_verifiers)
    if selected_count == 0:
        raise ValueError("critical audit settlement requires an allocation")
    fee = request.allocation.service_fee / selected_count
    bond = request.allocation.refundable_bond / selected_count
    reserve = max(
        0.0,
        request.allocation.worst_case_transfer - request.allocation.service_fee,
    )
    threshold = int(parameters.get("failure_threshold", 1))
    rows = []
    for report in request.reports:
        rows.append(settle_threshold_assignment(
            report, request.evaluation_bundle, failure_threshold=threshold,
            fee=fee, bond=bond, cost=costs.get(report.verifier_id, 0.0),
            effort=efforts.get(report.verifier_id, 0.0),
        ))
    return aggregate_settlements(rows, evaluator_reserve=reserve)
