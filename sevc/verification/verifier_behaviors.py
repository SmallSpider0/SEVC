"""Registered verifier effort behaviors for hidden trajectory-replay audits.

The public behavior input intentionally contains no audit truth, attack label,
gold marker, or audit-decision field.  An independent evaluator may compare a
committed report with trajectory-replay truth only after the report is fixed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import random
import time
from typing import Callable, Mapping

import torch
import torch.nn as nn

from sevc.core.registry import Registry
from sevc.training import ReplayProof
from sevc.incentives.verifier_protocol import (
    CommittedVerifierReport,
    PublicTaskBundle,
)

ModelFactory = Callable[[], nn.Module]


@dataclass(frozen=True)
class VerifierBehaviorInput:
    """Only task material and public execution parameters visible to a verifier."""

    task_id: str
    verifier_id: int
    replay_proof: ReplayProof
    replay_device: str
    replay_tolerance: float
    report_nonce: str
    seed: int
    segment_inputs: tuple[tuple[str, ReplayProof], ...] = ()
    peer_precommit_verdicts: tuple[bool | None, ...] = ()
    adaptive_probe_prediction: bool | None = None

    def __post_init__(self) -> None:
        if not self.task_id or not self.report_nonce:
            raise ValueError("task id and report nonce must be non-empty")
        if self.verifier_id < 0 or self.seed < 0:
            raise ValueError("verifier id and seed must be non-negative")
        if self.replay_tolerance < 0 or not math.isfinite(self.replay_tolerance):
            raise ValueError("replay tolerance must be finite and non-negative")
        segment_ids = tuple(item[0] for item in self.segment_inputs)
        if any(not value for value in segment_ids) or len(set(segment_ids)) != len(
            segment_ids
        ):
            raise ValueError("segment ids must be non-empty and unique")
        if self.peer_precommit_verdicts and len(self.peer_precommit_verdicts) not in {
            1,
            max(1, len(self.segment_inputs)),
        }:
            raise ValueError("peer precommit verdict count must match segment count")

    def to_public_dict(self) -> dict[str, object]:
        """Serialize metadata without tensor contents or hidden audit state."""

        return {
            "task_id": self.task_id,
            "verifier_id": self.verifier_id,
            "checkpoint_count": len(self.replay_proof.checkpoints),
            "batch_count": len(self.replay_proof.batches),
            "segment_ids": [item[0] for item in self.segment_inputs]
            or [self.task_id],
            "segment_checkpoint_counts": [
                len(item[1].checkpoints) for item in self.segment_inputs
            ]
            or [len(self.replay_proof.checkpoints)],
            "peer_precommit_count": len(self.peer_precommit_verdicts),
            "replay_device": self.replay_device,
            "replay_tolerance": self.replay_tolerance,
            "report_nonce_sha256": hashlib.sha256(
                self.report_nonce.encode("utf-8")
            ).hexdigest(),
            "seed": self.seed,
            "adaptive_probe_prediction": self.adaptive_probe_prediction,
        }


@dataclass(frozen=True)
class VerifierBehaviorOutput:
    behavior_key: str
    report_passed: bool | None
    commitment: str
    committed: bool
    revealed: bool
    replay_seconds: float
    batches_replayed: int
    effort_fraction: float
    details: dict[str, object]
    segment_ids: tuple[str, ...] = ()
    segment_verdicts: tuple[bool | None, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


Behavior = Callable[[VerifierBehaviorInput, ModelFactory], VerifierBehaviorOutput]
BEHAVIORS: Registry[Behavior] = Registry("verifier behavior")


@dataclass(frozen=True)
class BehaviorExecutionPlan:
    """Public-only declaration of replayed IDs and guesses for one bundle."""

    behavior_key: str
    scenario_id: str
    verifier_id: str
    job_id: str
    ordered_segment_ids: tuple[str, ...]
    executed_segment_ids: tuple[str, ...]
    planned_verdicts: tuple[bool | None, ...]
    target_effort_fraction: float

    def __post_init__(self) -> None:
        if not self.behavior_key or not self.scenario_id or not self.verifier_id:
            raise ValueError("behavior execution plan identity is incomplete")
        if not self.ordered_segment_ids or len(self.ordered_segment_ids) != len(
            self.planned_verdicts
        ):
            raise ValueError("behavior plan verdicts must match ordered segments")
        if len(set(self.ordered_segment_ids)) != len(self.ordered_segment_ids):
            raise ValueError("behavior plan segment ids must be unique")
        if not set(self.executed_segment_ids).issubset(self.ordered_segment_ids):
            raise ValueError("behavior plan executes an unknown segment")
        if not 0 <= self.target_effort_fraction <= 1:
            raise ValueError("target effort fraction must lie in [0, 1]")


@dataclass(frozen=True)
class ExecutedBehaviorPlan:
    plan: BehaviorExecutionPlan
    report: CommittedVerifierReport
    executed_segment_ids: tuple[str, ...]
    replay_seconds: float
    effort_fraction: float
    cache_hit_count: int
    cache_miss_count: int


def _proof_segments(
    context: VerifierBehaviorInput,
) -> tuple[tuple[str, ReplayProof], ...]:
    return context.segment_inputs or ((context.task_id, context.replay_proof),)


def _commitment(
    context: VerifierBehaviorInput,
    report: bool | None,
    segment_ids: tuple[str, ...],
    segment_verdicts: tuple[bool | None, ...],
) -> str:
    if not context.segment_inputs:
        payload = {
            "domain": "sevc-verifier-report-v1",
            "task_id": context.task_id,
            "verifier_id": context.verifier_id,
            "report": report,
            "nonce": context.report_nonce,
        }
    else:
        payload = {
            "domain": "sevc-verifier-segment-reports-v1",
            "task_id": context.task_id,
            "verifier_id": context.verifier_id,
            "ordered_segment_ids": segment_ids,
            "verdicts": segment_verdicts,
            "nonce": context.report_nonce,
        }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _output(
    key: str,
    context: VerifierBehaviorInput,
    report: bool | None,
    *,
    replay_seconds: float,
    batches_replayed: int,
    effort_fraction: float,
    details: dict[str, object] | None = None,
    segment_verdicts: tuple[bool | None, ...] | None = None,
) -> VerifierBehaviorOutput:
    segment_ids = (
        tuple(item[0] for item in context.segment_inputs)
        if context.segment_inputs
        else (context.task_id,)
    )
    verdicts = segment_verdicts if segment_verdicts is not None else (report,)
    if len(verdicts) != len(segment_ids):
        raise ValueError("segment verdict count must match segment ids")
    committed = any(value is not None for value in verdicts)
    return VerifierBehaviorOutput(
        behavior_key=key,
        report_passed=report,
        commitment=_commitment(context, report, segment_ids, verdicts),
        committed=committed,
        revealed=committed,
        replay_seconds=float(replay_seconds),
        batches_replayed=int(batches_replayed),
        effort_fraction=float(effort_fraction),
        details=details or {},
        segment_ids=segment_ids,
        segment_verdicts=verdicts,
    )


def _run_replay(
    context: VerifierBehaviorInput,
    model_factory: ModelFactory,
    proof: ReplayProof,
) -> tuple[bool, float, dict[str, object]]:
    from sevc.verification.verifier_replay_bank import execute_replay_proof

    report, elapsed, detail = execute_replay_proof(
        proof,
        model_factory,
        device=context.replay_device,
        tolerance=context.replay_tolerance,
        seed=context.seed,
    )
    return report, elapsed, detail


@BEHAVIORS.register("honest")
def _honest(
    context: VerifierBehaviorInput, model_factory: ModelFactory
) -> VerifierBehaviorOutput:
    reports: list[bool] = []
    elapsed = 0.0
    replayed = 0
    results: list[dict[str, object]] = []
    for _, proof in _proof_segments(context):
        report, segment_elapsed, result = _run_replay(
            context, model_factory, proof
        )
        reports.append(report)
        elapsed += segment_elapsed
        replayed += int(result.get("checkpoint_count", len(proof.checkpoints)))
        results.append(result)
    return _output(
        "honest",
        context,
        all(reports),
        replay_seconds=elapsed,
        batches_replayed=replayed,
        effort_fraction=1.0,
        details={"verification_method": "trajectory-replay", "replays": results},
        segment_verdicts=tuple(reports),
    )


@BEHAVIORS.register("always-pass")
def _always_pass(
    context: VerifierBehaviorInput, model_factory: ModelFactory
) -> VerifierBehaviorOutput:
    del model_factory
    verdicts = tuple(True for _ in _proof_segments(context))
    return _output(
        "always-pass",
        context,
        True,
        replay_seconds=0.0,
        batches_replayed=0,
        effort_fraction=0.0,
        segment_verdicts=verdicts,
    )


@BEHAVIORS.register("always-fail")
def _always_fail(
    context: VerifierBehaviorInput, model_factory: ModelFactory
) -> VerifierBehaviorOutput:
    del model_factory
    verdicts = tuple(False for _ in _proof_segments(context))
    return _output(
        "always-fail",
        context,
        False,
        replay_seconds=0.0,
        batches_replayed=0,
        effort_fraction=0.0,
        segment_verdicts=verdicts,
    )


@BEHAVIORS.register("random-report")
def _random_report(
    context: VerifierBehaviorInput, model_factory: ModelFactory
) -> VerifierBehaviorOutput:
    del model_factory
    seed_material = f"{context.seed}|{context.task_id}|random-report".encode("utf-8")
    derived_seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "big")
    rng = random.Random(derived_seed)
    reports = tuple(bool(rng.getrandbits(1)) for _ in _proof_segments(context))
    return _output(
        "random-report",
        context,
        all(reports),
        replay_seconds=0.0,
        batches_replayed=0,
        effort_fraction=0.0,
        segment_verdicts=reports,
    )


def _partial_replay(
    key: str,
    fraction: float,
    context: VerifierBehaviorInput,
    model_factory: ModelFactory,
) -> VerifierBehaviorOutput:
    reports: list[bool] = []
    results: list[dict[str, object]] = []
    total_replayed = 0
    elapsed = 0.0
    for _, source_proof in _proof_segments(context):
        total = len(source_proof.checkpoints)
        if total == 0:
            raise ValueError("partial replay requires a non-empty proof")
        replayed = max(1, min(total, math.ceil(total * fraction)))
        proof = ReplayProof(
            initial_state=source_proof.initial_state,
            batches=source_proof.batches[:replayed],
            checkpoints=source_proof.checkpoints[:replayed],
            learning_rate=source_proof.learning_rate,
            momentum=source_proof.momentum,
        )
        report, segment_elapsed, result = _run_replay(
            context, model_factory, proof
        )
        reports.append(report)
        results.append(result)
        total_replayed += replayed
        elapsed += segment_elapsed
    return _output(
        key,
        context,
        all(reports),
        replay_seconds=elapsed,
        batches_replayed=total_replayed,
        effort_fraction=fraction,
        details={"verification_method": "trajectory-replay", "replays": results},
        segment_verdicts=tuple(reports),
    )


@BEHAVIORS.register("partial-replay-25")
def _partial_replay_25(
    context: VerifierBehaviorInput, model_factory: ModelFactory
) -> VerifierBehaviorOutput:
    return _partial_replay("partial-replay-25", 0.25, context, model_factory)


@BEHAVIORS.register("partial-replay-50")
def _partial_replay_50(
    context: VerifierBehaviorInput, model_factory: ModelFactory
) -> VerifierBehaviorOutput:
    return _partial_replay("partial-replay-50", 0.50, context, model_factory)


@BEHAVIORS.register("minority-collusion")
def _minority_collusion(
    context: VerifierBehaviorInput, model_factory: ModelFactory
) -> VerifierBehaviorOutput:
    del model_factory
    # A pre-coordinated constant report models a minority coalition without
    # exposing the audit truth or waiting for other reveals.
    verdicts = tuple(True for _ in _proof_segments(context))
    return _output(
        "minority-collusion",
        context,
        True,
        replay_seconds=0.0,
        batches_replayed=0,
        effort_fraction=0.0,
        details={"coordination": "pre-committed-constant-report"},
        segment_verdicts=verdicts,
    )


def _collusion_after_replay(
    key: str,
    context: VerifierBehaviorInput,
    model_factory: ModelFactory,
    *,
    withhold_rejections: bool,
) -> VerifierBehaviorOutput:
    """Replay honestly, then apply a preregistered trainer-covering report rule."""

    honest_reports: list[bool] = []
    committed_reports: list[bool | None] = []
    results: list[dict[str, object]] = []
    elapsed = 0.0
    replayed = 0
    for _, proof in _proof_segments(context):
        report, segment_elapsed, result = _run_replay(context, model_factory, proof)
        honest_reports.append(report)
        committed_reports.append(
            report if report else (None if withhold_rejections else True)
        )
        elapsed += segment_elapsed
        replayed += int(result.get("checkpoint_count", len(proof.checkpoints)))
        results.append(result)
    aggregate = (
        all(value is True for value in committed_reports)
        if all(value is not None for value in committed_reports)
        else None
    )
    return _output(
        key,
        context,
        aggregate,
        replay_seconds=elapsed,
        batches_replayed=replayed,
        effort_fraction=1.0,
        details={
            "verification_method": "trajectory-replay",
            "post_replay_rule": (
                "withhold-rejecting-report"
                if withhold_rejections
                else "replace-reject-with-pass"
            ),
            "honest_replay_verdicts": honest_reports,
            "replays": results,
        },
        segment_verdicts=tuple(committed_reports),
    )


@BEHAVIORS.register("collude_cover_trainer")
def _collude_cover_trainer(
    context: VerifierBehaviorInput, model_factory: ModelFactory
) -> VerifierBehaviorOutput:
    return _collusion_after_replay(
        "collude_cover_trainer",
        context,
        model_factory,
        withhold_rejections=False,
    )


@BEHAVIORS.register("collude_withhold")
def _collude_withhold(
    context: VerifierBehaviorInput, model_factory: ModelFactory
) -> VerifierBehaviorOutput:
    return _collusion_after_replay(
        "collude_withhold",
        context,
        model_factory,
        withhold_rejections=True,
    )


@BEHAVIORS.register("adaptive_recognizer")
def _adaptive_recognizer(
    context: VerifierBehaviorInput, model_factory: ModelFactory
) -> VerifierBehaviorOutput:
    """Replay predicted probes; use the registered always-pass action otherwise."""

    if context.adaptive_probe_prediction is None:
        raise ValueError("adaptive recognizer requires a frozen pre-effort prediction")
    if context.adaptive_probe_prediction:
        honest = _honest(context, model_factory)
        return VerifierBehaviorOutput(
            behavior_key="adaptive_recognizer",
            report_passed=honest.report_passed,
            commitment=honest.commitment,
            committed=honest.committed,
            revealed=honest.revealed,
            replay_seconds=honest.replay_seconds,
            batches_replayed=honest.batches_replayed,
            effort_fraction=honest.effort_fraction,
            details={
                **honest.details,
                "adaptive_action": "honest-replay",
                "prediction_source": "pre-effort-registered-discriminator",
            },
            segment_ids=honest.segment_ids,
            segment_verdicts=honest.segment_verdicts,
        )
    passed = _always_pass(context, model_factory)
    return VerifierBehaviorOutput(
        behavior_key="adaptive_recognizer",
        report_passed=passed.report_passed,
        commitment=passed.commitment,
        committed=passed.committed,
        revealed=passed.revealed,
        replay_seconds=0.0,
        batches_replayed=0,
        effort_fraction=0.0,
        details={
            "adaptive_action": "always-pass",
            "prediction_source": "pre-effort-registered-discriminator",
        },
        segment_ids=passed.segment_ids,
        segment_verdicts=passed.segment_verdicts,
    )


@BEHAVIORS.register("copy-peer")
def _copy_peer(
    context: VerifierBehaviorInput, model_factory: ModelFactory
) -> VerifierBehaviorOutput:
    del model_factory
    count = len(_proof_segments(context))
    if not context.peer_precommit_verdicts:
        verdicts = tuple(True for _ in range(count))
        source = "deterministic-no-peer-fallback"
    elif len(context.peer_precommit_verdicts) == 1:
        verdicts = tuple(context.peer_precommit_verdicts[0] for _ in range(count))
        source = "single-precommit-peer"
    else:
        verdicts = tuple(context.peer_precommit_verdicts)
        source = "ordered-precommit-peers"
    return _output(
        "copy-peer",
        context,
        all(value is True for value in verdicts)
        if any(value is not None for value in verdicts)
        else None,
        replay_seconds=0.0,
        batches_replayed=0,
        effort_fraction=0.0,
        details={"copy_source": source, "timing": "pre-commit-public-state-only"},
        segment_verdicts=verdicts,
    )


@BEHAVIORS.register("dropout")
def _dropout(
    context: VerifierBehaviorInput, model_factory: ModelFactory
) -> VerifierBehaviorOutput:
    del model_factory
    verdicts = tuple(None for _ in _proof_segments(context))
    return _output(
        "dropout",
        context,
        None,
        replay_seconds=0.0,
        batches_replayed=0,
        effort_fraction=0.0,
        details={"availability": "no-commit"},
        segment_verdicts=verdicts,
    )


def run_verifier_behavior(
    behavior_key: str,
    context: VerifierBehaviorInput,
    model_factory: ModelFactory,
) -> VerifierBehaviorOutput:
    return BEHAVIORS.get(behavior_key)(context, model_factory)


def plan_behavior_segments(
    behavior_key: str,
    bundle: PublicTaskBundle,
    *,
    seed: int,
    peer_precommit_by_payload: Mapping[str, bool | None] | None = None,
    adaptive_prediction_by_payload: Mapping[str, bool] | None = None,
) -> BehaviorExecutionPlan:
    """Plan effort using only the serialized public task bundle."""

    segments = tuple(bundle.segments)
    ordered_ids = tuple(item.segment_id for item in segments)
    count = len(segments)
    executed: tuple[str, ...]
    verdicts: list[bool | None]
    target = 0.0
    if behavior_key == "honest":
        executed = ordered_ids
        verdicts = [None] * count
        target = 1.0
    elif behavior_key == "always-pass":
        executed = ()
        verdicts = [True] * count
    elif behavior_key == "always-fail":
        executed = ()
        verdicts = [False] * count
    elif behavior_key == "random-report":
        rng = random.Random(int(seed))
        executed = ()
        verdicts = [bool(rng.getrandbits(1)) for _ in segments]
    elif behavior_key in {"partial-replay-25", "partial-replay-50"}:
        target = 0.25 if behavior_key.endswith("25") else 0.50
        replay_count = int(round(count * target))
        if abs(replay_count / count - target) > 1e-12:
            raise ValueError("bundle size cannot realize the exact partial fraction")
        executed = ordered_ids[:replay_count]
        verdicts = [None if index < replay_count else True for index in range(count)]
    elif behavior_key == "copy-peer":
        peer = dict(peer_precommit_by_payload or {})
        executed = ()
        verdicts = [peer.get(item.payload_digest, True) for item in segments]
    elif behavior_key == "minority-collusion":
        executed = ()
        verdicts = [True] * count
    elif behavior_key in {"collude_cover_trainer", "collude_withhold"}:
        executed = ordered_ids
        verdicts = [None] * count
        target = 1.0
    elif behavior_key == "adaptive_recognizer":
        predictions = dict(adaptive_prediction_by_payload or {})
        if set(predictions) != {item.payload_digest for item in segments}:
            raise ValueError("adaptive plan requires one prediction per public payload")
        executed = tuple(
            item.segment_id for item in segments if predictions[item.payload_digest]
        )
        verdicts = [
            None if predictions[item.payload_digest] else True for item in segments
        ]
        target = len(executed) / count
    elif behavior_key == "dropout":
        executed = ()
        verdicts = [None] * count
    else:
        raise KeyError(f"unknown verifier behavior: {behavior_key}")
    return BehaviorExecutionPlan(
        behavior_key=behavior_key,
        scenario_id=bundle.scenario_id,
        verifier_id=bundle.verifier_id,
        job_id=bundle.job_id,
        ordered_segment_ids=ordered_ids,
        executed_segment_ids=executed,
        planned_verdicts=tuple(verdicts),
        target_effort_fraction=float(target),
    )


def execute_behavior_plan(
    plan: BehaviorExecutionPlan,
    bundle: PublicTaskBundle,
    bank: object,
    *,
    seed: int,
) -> ExecutedBehaviorPlan:
    """Execute selected microsegments through the evaluator-owned cache."""

    from sevc.verification.verifier_replay_bank import ReplayObservationProvider

    if not isinstance(bank, ReplayObservationProvider):
        raise TypeError("behavior executor requires the canonical observation provider")
    if (
        plan.scenario_id,
        plan.verifier_id,
        plan.job_id,
    ) != (bundle.scenario_id, bundle.verifier_id, bundle.job_id):
        raise ValueError("behavior plan and bundle identities differ")
    segment_map = {item.segment_id: item for item in bundle.segments}
    executed = set(plan.executed_segment_ids)
    verdicts = list(plan.planned_verdicts)
    replay_seconds = 0.0
    hits = 0
    misses = 0
    for index, segment_id in enumerate(plan.ordered_segment_ids):
        if segment_id not in executed:
            continue
        segment = segment_map[segment_id]
        observation = bank.observe(
            segment.payload_digest,
            execution_plan_key="canonical-full-microsegment",
            seed=int(seed),
        )
        verdicts[index] = observation.verdict
        if plan.behavior_key == "collude_cover_trainer" and verdicts[index] is False:
            verdicts[index] = True
        elif plan.behavior_key == "collude_withhold" and verdicts[index] is False:
            verdicts[index] = None
        replay_seconds += observation.replay_seconds
        hits += int(observation.cache_hit)
        misses += int(not observation.cache_hit)
    nonce = hashlib.sha256(
        (
            f"sevc-rc-phse-report-v2|{plan.scenario_id}|{plan.verifier_id}|"
            f"{plan.job_id}|{plan.behavior_key}"
        ).encode("utf-8")
    ).hexdigest()
    report = CommittedVerifierReport.create(
        scenario_id=plan.scenario_id,
        verifier_id=plan.verifier_id,
        job_id=plan.job_id,
        ordered_segment_ids=plan.ordered_segment_ids,
        verdicts=tuple(verdicts),
        nonce=nonce,
    )
    actual_fraction = len(executed) / len(plan.ordered_segment_ids)
    return ExecutedBehaviorPlan(
        plan=plan,
        report=report,
        executed_segment_ids=tuple(plan.executed_segment_ids),
        replay_seconds=float(replay_seconds),
        effort_fraction=float(actual_fraction),
        cache_hit_count=hits,
        cache_miss_count=misses,
    )


_FORBIDDEN_KEY_FRAGMENTS = (
    "ground_truth",
    "groundtruth",
    "attack_label",
    "attacklabel",
    "is_gold",
    "isgold",
    "audit_decision",
    "auditdecision",
    "is_sentinel",
    "issentinel",
    "expected_verdict",
    "expectedverdict",
    "is_probe",
    "isprobe",
    "sealed_truth",
    "sealedtruth",
)


def forbidden_label_field_count(payload: object) -> int:
    """Count forbidden keys recursively in a type schema or serialized payload."""

    if isinstance(payload, type) and hasattr(payload, "__dataclass_fields__"):
        payload = {name: None for name in payload.__dataclass_fields__}
    count = 0
    if isinstance(payload, dict):
        for key, value in payload.items():
            normalized = str(key).lower().replace("-", "_")
            compact = normalized.replace("_", "")
            if any(
                fragment in normalized or fragment.replace("_", "") in compact
                for fragment in _FORBIDDEN_KEY_FRAGMENTS
            ):
                count += 1
            count += forbidden_label_field_count(value)
    elif isinstance(payload, (list, tuple)):
        count += sum(forbidden_label_field_count(value) for value in payload)
    return count
