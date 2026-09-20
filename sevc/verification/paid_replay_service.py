"""Paid replay task mechanisms with explicit owner and adversary views.

No evaluator truth is an input to effort selection or service settlement.
"""
from __future__ import annotations

from functools import cached_property

from dataclasses import dataclass, asdict, replace, field
import json
import random
import time
import threading
from typing import Callable, Mapping, Any

from sevc.core.artifacts import canonical_json_text, sha256_text
from sevc.core.registry import Registry
from sevc.core.task_lanes import TaskLanes, shared_wall_charges
from sevc.incentives.verifier_protocol import (
    CommittedVerifierReport, SealedSegmentTruth, SealedEvaluationTruth,
)
from sevc.verification.verifier_task_policies import settle_threshold_assignment
from sevc.training import verify_replay_proof
from sevc.training.replay_sources import materialize_short_source
from sevc.verification.replay_coupled_probes import (
    ATOM_KEYS, derive_int, proof_component_hashes, mutate_replay_proof,
    compile_canonical_replay_task, canonicalize_replay_proof,
    canonicalize_replay_proof_with_identity, build_mutation_invalidity_witness,
    verify_mutation_invalidity_witness, verify_wrapped_replay_proof,
)

VERSION = "tdsc-ctiv-canonical-probe-certificate-v4"
WRAPPER_DOMAIN = "five-rq-wrapper-v1"
BEHAVIOR_KEYS = ("honest", "zero-effort-fixed-prior-constant", "partial-50",
                 "joint-view-targeted-cover")
REPAIR_BEHAVIOR_KEYS = BEHAVIOR_KEYS + ("joint-cached-correct",)
SCOPED_BEHAVIOR_KEYS = REPAIR_BEHAVIOR_KEYS + ("constant-accept", "constant-reject",
                                            "joint-targeted-cover", "joint-targeted-false-reject",
                                            "source-binding-shortcut", "sgd-consistency-shortcut", "partial-90",
                                            "softmax-bias-shortcut", "low-rank-gradient-shortcut",
                                            "uniform-k32", "uniform-k39", "prefix-one-step-shortcut", "cheap-recognizer-k32")
TASK_MECHANISMS = Registry("paid replay task mechanism")
TASK_MECHANISMS.add("rcmp-source-coupled", False)
TASK_MECHANISMS.add("independent-hidden-gold", True)


class SourceReferenceNotEstablished(ValueError):
    """The proposed source/reference construction has not established its premise."""


def identity(value):
    return sha256_text(canonical_json_text(value))


def verify_fixture_equivalence(context, partition, clock, performance=None):
    """Known construction on the actual device, before any performance selection."""
    recipe = source_recipe(context, 2026098000, 0, partition, "engineering-equivalence")
    source = materialize_short_source(context, recipe)
    results = []
    for variant in ("valid", "invalid"):
        proof = source
        if variant == "invalid":
            proof, _ = mutate_replay_proof(source,source_id=recipe["source_id"],post_commit_seed=18,
                atom_key="cp4-positive",magnitude=4e-5,protocol_version=VERSION)
        baseline = verify_replay_proof(proof,context.factory,device=context.device.name,tolerance=1e-5)
        candidate = verify_replay_proof(proof,context.factory,device=context.device.name,
                                        tolerance=1e-5,comparison_device="replay")
        if baseline != candidate or baseline["passed"] != (variant == "valid"):
            raise ValueError("device replay equivalence/known-construction check failed")
        hashes = proof_component_hashes(proof)
        identities = []
        for profile in ("reference","fused"):
            tasks, wrapped_hashes = [], []
            for index in range(max(4,(performance or {}).get("task_lanes",1))):
                bundle = compile_canonical_replay_task(proof,context.factory(),context.build_key,
                    source_id=recipe["source_id"],source_commitment=hashes["proof_sha256"],
                    post_commit_seed=19,role="production",atom_key=None,permutation_seed=20+index,
                    protocol_version=VERSION,wrapper_seed_domain=WRAPPER_DOMAIN,tamper_delta=4e-5,
                    validated_source_verdict=baseline["passed"],delivery_profile="compact",identity_profile=profile)
                tasks.append(DeliveredReplay(bundle.sealed.task_id,hashes["proof_sha256"],bundle.wrapped_proof,
                                             bundle.descriptor,bundle.public_envelope))
                wrapped_hashes.append(bundle.component_hashes["wrapped"])
            r, _ = execute_assignment(tuple(tasks),behavior="honest",trainer_hashes=frozenset([hashes["proof_sha256"]]),
                context=context,seed=21,assignment_id="equivalence",clock=clock,
                performance={"identity_profile":profile,"comparison_device":"cpu" if profile=="reference" else "replay",
                             "task_lanes":1 if profile == "reference" else (performance or {}).get("task_lanes",1),
                             "reuse_replay_model":profile != "reference" and (performance or {}).get("reuse_replay_model",False)},
                emit_commit=lambda row:None)
            identities.append((wrapped_hashes,r.commitment))
        if identities[0] != identities[1]:
            raise ValueError("wrapped identity/committed report equivalence failed")
        results.append({"variant":variant,"source_sha256":hashes["proof_sha256"],
                        "comparison_exact":True,"wrapped_and_report_exact":True,
                        "reference_report_commitment":identities[0][1],
                        "candidate_report_commitment":identities[1][1],
                        "task_count":len(tasks)})
    return {"dataset":context.dataset,"device":context.device.name,"passed":True,
            "recipe":recipe,"checks":results,"scientifically_eligible":False}


@dataclass(frozen=True)
class DeliveredReplay:
    task_id: str
    source_commitment: str
    proof: Any
    descriptor: Mapping
    envelope: Mapping


@dataclass(frozen=True)
class StoredReplayProof:
    """Run-owned disposable delivery cache, authenticated before every read."""
    path: str
    sha256: str
    access_profile: str = "strict"
    _reader: object = field(default=None, compare=False, repr=False)

    @cached_property
    def mapped_file(self):
        from sevc.core.tensor_storage import AuthenticatedTensorFile
        if self._reader is not None:
            if str(self._reader.path) != self.path or self._reader.digest != self.sha256:
                raise ValueError("delivery authentication capability mismatch")
            return self._reader
        return AuthenticatedTensorFile(self.path, self.sha256)

    def load(self):
        if self.access_profile in {"authenticated-mmap", "owned-mmap"}:
            return self.mapped_file.load()
        from pathlib import Path
        import torch
        from sevc.core.artifacts import sha256_file
        path = Path(self.path)
        if sha256_file(path) != self.sha256:
            raise ValueError("delivery cache payload identity drift")
        return torch.load(path, map_location="cpu", weights_only=False)


@dataclass(frozen=True)
class OwnerProbeReferences:
    answers: tuple[tuple[str, bool], ...]

    def __post_init__(self):
        if len(self.answers) != 8 or len(dict(self.answers)) != 8:
            raise ValueError("exactly eight unique comparable probe references required")
        if any(type(v) is not bool for _, v in self.answers):
            raise TypeError("probe answers must be validated booleans")


@dataclass(frozen=True)
class OwnerProductionReference:
    """Owner-owned replay receipt, bound to an unmutated wrapped derivative."""
    task_id: str
    source_commitment: str
    wrapped_proof_sha256: str
    descriptor_sha256: str
    source_receipt_json: str
    answer: bool

    def __post_init__(self):
        receipt = json.loads(self.source_receipt_json)
        if (type(self.answer) is not bool or receipt.get("passed") is not self.answer
                or receipt.get("state_complete") is not True
                or receipt.get("proof_sha256") != self.source_commitment):
            raise ValueError("production reference requires a bound actual source replay receipt")
        for value in (self.source_commitment, self.wrapped_proof_sha256, self.descriptor_sha256):
            if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError("production reference identity must be SHA-256")


@dataclass(frozen=True)
class OwnerProductionReferences:
    items: tuple[OwnerProductionReference, ...]

    def __post_init__(self):
        if not self.items or any(type(r) is not OwnerProductionReference for r in self.items):
            raise TypeError("typed owner production receipts required")
        if len({r.task_id for r in self.items}) != len(self.items):
            raise ValueError("duplicate owner production reference")


@dataclass
class PreparedService:
    mechanisms: dict[str, tuple[DeliveredReplay, ...]]
    references: dict[str, OwnerProbeReferences]
    trainer_committed_hashes: frozenset[str]
    audit_rows: list[dict]
    source_rows: list[dict]
    owner_reference_seconds: float
    production_references: OwnerProductionReferences | None = None
    trainer_answer_cache: dict[str, bool] = field(default_factory=dict)


class WorkClock:
    """One monotonic wall timeline, including synchronous CPU and GPU work."""
    def __init__(self, sync: Callable[[], None], emit: Callable[[dict], None]):
        self.sync, self.emit = sync, emit
        self.local = threading.local()

    @property
    def context(self):
        if not hasattr(self.local,"context"):
            self.local.context = {}
        return self.local.context

    @context.setter
    def context(self,value):
        self.local.context = value

    def call(self, phase: str, role: str, fn: Callable, /, *args, **kwargs):
        start = time.monotonic()
        value = fn(*args, **kwargs)
        self.sync()
        end = time.monotonic()
        self.emit({**self.context, "phase": phase, "role": role,
                   "start": start, "end": end, "seconds": end - start})
        return value, end - start


def source_recipe(context, seed: int, index: int, partition: tuple[int, int], namespace: str):
    rng = random.Random(derive_int(namespace, seed, index, "samples"))
    indices = rng.sample(range(*partition), 8)
    labels = [int(context.train[i][1]) for i in indices]
    return {"source_id": identity([namespace, context.dataset, seed, index]),
            "source_index": index, "source_seed": derive_int(namespace, seed, index, "model") % (2**32),
            "sample_indices": indices, "sample_labels": labels,
            "dataset": context.dataset, "block_seed": seed, "namespace": namespace}


def prepare_service(context, *, seed: int, partition: tuple[int, int], namespace: str,
                    clock: WorkClock, performance: Mapping, all_honest: bool = False,
                    mechanisms: tuple[str, ...] | None = None,
                    invalid_source_count: int | None = None,
                    collect_production_references: bool = False):
    """Commit first, pay to validate, then conditionally select valid probe sources."""
    sources, source_rows, originals = [], [], set()
    if invalid_source_count is not None and (type(invalid_source_count) is not int
            or not 0 <= invalid_source_count <= 16 or all_honest):
        raise ValueError("explicit invalid source count must be 0..16 and cannot overlap all_honest")
    invalid_count = (0 if all_honest else 16) if invalid_source_count is None else invalid_source_count
    trainer_answer_cache = {}
    mechanisms = TASK_MECHANISMS.keys() if mechanisms is None else mechanisms
    if not mechanisms or len(set(mechanisms)) != len(mechanisms):
        raise ValueError("task mechanism selection must be nonempty and unique")
    for key in mechanisms:
        TASK_MECHANISMS.get(key)
    has_gold = "independent-hidden-gold" in mechanisms
    reference_seconds = 0.0
    for index in range(48 if has_gold else 40):
        clock.context.update(source_index=index)
        recipe, _ = clock.call("source_recipe", "trainer" if index < 40 else "owner",
                               source_recipe, context, seed, index, partition, namespace)
        source, _ = clock.call("source_materialization", "trainer" if index < 40 else "owner",
                               materialize_short_source, context, recipe)
        mutation = None
        invalid = 24 <= index < 24 + invalid_count
        if invalid:
            (source, mutation), _ = clock.call(
                "trainer_precommit_mutation", "trainer", mutate_replay_proof, source,
                source_id=recipe["source_id"], post_commit_seed=derive_int(seed, "trainer"),
                atom_key=ATOM_KEYS[(index - 24) % 4], magnitude=4e-5, protocol_version=VERSION)
        hashes, _ = clock.call("source_commitment_hash", "trainer" if index < 40 else "owner",
                               proof_component_hashes, source)
        # All source commitments exist before any roles are selected.
        if index < 40:
            originals.add(hashes["proof_sha256"])
            # This knowledge precedes owner validation: the trainer performed the
            # SGD work and knows its own deliberate precommit corruption.
            trainer_answer_cache[hashes["proof_sha256"]] = not invalid
        sources.append((source, hashes, recipe, invalid))
        source_rows.append({**recipe, **hashes, "trainer_mutation": mutation,
                            "commitment": hashes["proof_sha256"], "commit_order": index})
    lanes = context.task_lanes(performance.get("task_lanes",1))
    def validate_source(item):
        index, (source, hashes, recipe, invalid) = item
        clock.context.update(source_index=index)
        result, seconds = clock.call("source_reference_replay", "owner", verify_replay_proof,
                                      source, context.cached_replay_factory if performance.get("reuse_replay_model") else context.factory, device=context.device.name,
                                      tolerance=1e-5, comparison_device=performance.get("comparison_device", "cpu"))
        return result, seconds
    reference_start = time.monotonic()
    results = lanes.map(validate_source,enumerate(sources),clock)
    reference_seconds = time.monotonic()-reference_start
    validated = []
    for index, ((source, hashes, recipe, invalid),(result,seconds)) in enumerate(zip(sources,results)):
        actual = bool(result["passed"])
        source_rows[index]["owner_reference"] = {**result, "seconds": seconds,
                                                   "proof_sha256": hashes["proof_sha256"]}
        if actual != (not invalid) or not result["state_complete"]:
            raise SourceReferenceNotEstablished("actual source replay does not establish the frozen construction")
        validated.append(actual)
    # First 24 are the predeclared valid stratum even in the all-honest control.
    selected = sorted(range(24), key=lambda i: derive_int(seed, "postcommit-probe", source_rows[i]["commitment"]))[:8]
    role_map = {i: ("control" if rank < 4 else "challenge", None if rank < 4 else ATOM_KEYS[rank-4])
                for rank, i in enumerate(selected)}
    pools = {key: [] for key in mechanisms}
    answers = {key: [] for key in pools}
    audit_rows = []
    production_receipts = []
    model = context.factory().cpu()
    compile_jobs = []
    for index in range(40):
        variants = [("shared", index)] if index not in role_map else [
            (key, index if key == "rcmp-source-coupled" else 40 + selected.index(index))
            for key in mechanisms]
        for mechanism, source_index in variants:
            compile_jobs.append((index,mechanism,source_index))
    if len({job[2] for job in compile_jobs}) != len(compile_jobs):
        raise ValueError("source lifetime requires a unique wrapper consumer")
    def compile_task(job):
        index,mechanism,source_index = job
        source, hashes, recipe, _ = sources[source_index]
        role, atom = role_map.get(index, ("production", None))
        clock.context.update(source_index=source_index, mechanism=mechanism)
        bundle, _ = clock.call(
            "task_compile", "owner", compile_canonical_replay_task, source, model, context.build_key,
            source_id=recipe["source_id"], source_commitment=hashes["proof_sha256"],
            post_commit_seed=derive_int(seed, "owner-atom", source_index), role=role, atom_key=atom,
            permutation_seed=derive_int(seed, "wrapper", source_index), protocol_version=VERSION,
            wrapper_seed_domain=WRAPPER_DOMAIN, tamper_delta=4e-5,
            source_component_hashes=hashes, delivery_profile="compact",
            identity_profile=performance["identity_profile"], schema_validator_profile="shared",
            validated_source_verdict=validated[source_index])
        witness = None
        if atom is not None:
            candidate_hashes = bundle.component_hashes.get("candidate")
            if candidate_hashes is None:
                candidate_hashes, _ = clock.call("challenge_identity", "owner", proof_component_hashes,
                                                 bundle.canonical_candidate)
            witness, _ = clock.call("challenge_witness", "owner", build_mutation_invalidity_witness,
                source, bundle.canonical_candidate, bundle.mutation, source_id=recipe["source_id"],
                source_commitment=hashes["proof_sha256"], source_proof_sha256=hashes["proof_sha256"],
                mutated_proof_sha256=candidate_hashes["proof_sha256"], descriptor=bundle.descriptor,
                replay_tolerance=1e-5, source_replay_valid=validated[source_index])
            if not verify_mutation_invalidity_witness(witness):
                raise SourceReferenceNotEstablished("challenge invalidity witness failed")
        delivered = DeliveredReplay(bundle.sealed.task_id, hashes["proof_sha256"],
                                    bundle.wrapped_proof, bundle.descriptor, bundle.public_envelope)
        # Validation, wrapper and witness have finished for this sole consumer.
        # Keep the frozen identity/recipe but release the large temporary source
        # as wrapped tasks accumulate, rather than doubling the whole bank.
        sources[source_index] = None
        return delivered, bundle.sealed.expected_verdict, witness, bundle.component_hashes["wrapped"]
    for (index,mechanism,source_index),(delivered,expected,witness,wrapped_hashes) in zip(compile_jobs,lanes.map(compile_task,compile_jobs,clock)):
        recipe = hashes = source_rows[source_index]
        role,_ = role_map.get(index,("production",None))
        keys = tuple(pools) if mechanism == "shared" else (mechanism,)
        for key in keys:
            pools[key].append(delivered)
            if role != "production":
                answers[key].append((delivered.task_id, expected))
        audit_rows.append({"task_id": delivered.task_id, "source_id": recipe["source_id"],
                           "mechanism": mechanism, "role": role,
                           "expected": expected,
                           "source_commitment": hashes["proof_sha256"],
                           "wrapped_hashes": wrapped_hashes, "witness": witness})
        if collect_production_references and role == "production":
            # Online references do not read the evaluator's expected field.
            # Compilation has validated the state-complete, verdict-preserving map.
            production_receipts.append(OwnerProductionReference(
                delivered.task_id, delivered.source_commitment, wrapped_hashes["proof_sha256"],
                identity(delivered.descriptor), canonical_json_text(source_rows[source_index]["owner_reference"]),
                validated[source_index]))
    clock.context.pop("source_index", None)
    clock.context.pop("mechanism", None)
    return PreparedService(
        {key: tuple(sorted(items, key=lambda t: derive_int(seed, "public-order", t.task_id)))
         for key, items in pools.items()},
        {key: OwnerProbeReferences(tuple(value)) for key, value in answers.items()},
        frozenset(originals), audit_rows, source_rows, reference_seconds,
        OwnerProductionReferences(tuple(production_receipts)) if collect_production_references else None,
        trainer_answer_cache if collect_production_references else {})


def execute_assignment(tasks: tuple[DeliveredReplay, ...], *, behavior: str, trainer_hashes: frozenset[str],
                       context, seed: int, assignment_id: str, clock: WorkClock, performance: Mapping,
                       emit_commit: Callable[[dict], None], job_id: str | None = None,
                       verifier_id: str | None = None,
                       trainer_answer_cache: Mapping[str, bool] | None = None,
                       precommit_fault_task_id: str | None = None):
    """Effort decisions see only public delivered tasks and trainer-owned commitments."""
    if behavior not in SCOPED_BEHAVIOR_KEYS:
        raise KeyError(behavior)
    if behavior == "joint-cached-correct":
        if (trainer_answer_cache is None or set(trainer_answer_cache) != set(trainer_hashes)
                or any(type(v) is not bool for v in trainer_answer_cache.values())):
            raise ValueError("cached-correct strategy requires the trainer's own complete answer cache")
    fraction = .9 if behavior == "partial-90" else .5
    count = int(behavior.removeprefix("uniform-k")) if behavior.startswith("uniform-k") else int(len(tasks)*fraction)
    if behavior.startswith("uniform-k") and count > len(tasks):
        raise ValueError("uniform effort exceeds delivered task count")
    selected = set(t.task_id for t in sorted(tasks, key=lambda t: derive_int(seed, "partial", t.task_id))[:count])
    if behavior == "cheap-recognizer-k32":
        selector = performance.get("cheap_selector")
        if selector is None:
            raise PermissionError("frozen disclosed-history recognizer is required")
        selected, _ = clock.call("recognizer_selection", "verifier", selector, tasks, clock)
        if len(selected) != 32 or not set(selected) <= {t.task_id for t in tasks}:
            raise ValueError("recognizer replay quota or task identity drift")
    false_reject_target = None
    if behavior == "joint-targeted-false-reject":
        # A source handle can identify a trainer-known accepting source, but it
        # does not reveal its hidden role. A transformed challenge may fail the
        # later canonical-match test; do not substitute evaluator role truth.
        candidates = [t.task_id for t in tasks if (trainer_answer_cache or {}).get(t.source_commitment) is True]
        false_reject_target = min(candidates) if candidates else None
    schema_model = context.factory().cpu()
    def execute_task(task):
        clock.context.update(task_id=task.task_id, behavior=behavior)
        replay = behavior == "honest" or ((behavior in {"partial-50", "partial-90", "cheap-recognizer-k32"} or behavior.startswith("uniform-k")) and task.task_id in selected)
        verdict, matched, canonical = behavior != "constant-reject", False, None
        start = time.monotonic()
        proof = task.proof
        shortcut = behavior in {"source-binding-shortcut", "sgd-consistency-shortcut", "softmax-bias-shortcut",
                                "low-rank-gradient-shortcut", "prefix-one-step-shortcut"}
        shortcut_receipt = None
        needs_payload = replay or shortcut or behavior.startswith("joint-") or not performance.get("lazy_payload_read", False)
        if needs_payload and isinstance(proof, StoredReplayProof):
            selector=performance.get('cheap_selector') if behavior=='cheap-recognizer-k32' else None
            if selector is not None and hasattr(selector,'load_for_replay'):
                proof, _ = clock.call("verifier_payload_read", "verifier", selector.load_for_replay, task)
            else:
                proof, _ = clock.call("verifier_payload_read", "verifier", proof.load)
        if shortcut:
            if behavior == "prefix-one-step-shortcut":
                from sevc.training.engine import _tensor_sequence_sha256
                (canonical, _), _ = clock.call("prefix_canonicalize", "verifier", canonicalize_replay_proof,
                    proof, schema_model, task.descriptor, protocol_version=VERSION, wrapper_seed_domain=WRAPPER_DOMAIN)
                prefix = replace(canonical, batches=canonical.batches[:1], checkpoints=canonical.checkpoints[:1],
                                 optimizer_checkpoints=canonical.optimizer_checkpoints[:1],
                                 data_order_sha256=_tensor_sequence_sha256(canonical.batches[:1]))
                shortcut_receipt, _ = clock.call("prefix_replay", "verifier", verify_replay_proof,
                    prefix, context.cached_replay_factory if performance.get("reuse_replay_model") else context.factory,
                    device=context.device.name, tolerance=1e-5,
                    comparison_device=performance.get("comparison_device", "cpu"))
                shortcut_receipt = {**shortcut_receipt, "prefix_steps": 1, "full_replay": False}
                verdict = bool(shortcut_receipt["passed"])
            elif behavior == "source-binding-shortcut":
                (canonical, _, hashes, _), _ = clock.call("public_binding_canonicalize_hash", "verifier",
                    canonicalize_replay_proof_with_identity, proof, schema_model, task.descriptor,
                    protocol_version=VERSION, wrapper_seed_domain=WRAPPER_DOMAIN)
                verdict = hashes["proof_sha256"] == task.source_commitment
                shortcut_receipt = {"canonical_matches_public_binding": verdict}
            else:
                from sevc.verification.public_replay_shortcuts import sgd_public_consistency, softmax_bias_gradient_consistency
                # SGD identities commute with the legal public channel permutation.
                shortcut_receipt, _ = clock.call("public_sgd_consistency", "verifier",
                    sgd_public_consistency, proof, tolerance=1e-5)
                if behavior in {"softmax-bias-shortcut", "low-rank-gradient-shortcut"}:
                    bias, _ = clock.call("public_softmax_bias", "verifier",
                        softmax_bias_gradient_consistency, proof, tolerance=1e-5)
                    shortcut_receipt = {**shortcut_receipt, "softmax_bias": bias,
                                        "passed": shortcut_receipt["passed"] and bias["passed"]}
                if behavior == "low-rank-gradient-shortcut":
                    from sevc.verification.public_replay_shortcuts import low_rank_gradient_consistency
                    rank, _ = clock.call("public_gradient_rank", "verifier",
                        low_rank_gradient_consistency, proof, tolerance=1e-5)
                    shortcut_receipt = {**shortcut_receipt, "gradient_rank": rank,
                                        "passed": shortcut_receipt["passed"] and rank["passed"]}
                verdict = shortcut_receipt["passed"]
        if behavior in {"joint-view-targeted-cover", "joint-cached-correct",
                        "joint-targeted-cover", "joint-targeted-false-reject"}:
            # Commitment lookup alone does not label the delivered derivative.
            known_source = task.source_commitment in trainer_hashes
            if performance["identity_profile"] == "reference":
                (canonical, _), _ = clock.call("joint_canonicalize", "verifier", canonicalize_replay_proof,
                    proof, schema_model, task.descriptor, protocol_version=VERSION, wrapper_seed_domain=WRAPPER_DOMAIN)
                hashes, _ = clock.call("joint_hash", "verifier", proof_component_hashes, canonical)
            else:
                (canonical, _, hashes, _), _ = clock.call("joint_canonicalize_hash", "verifier",
                    canonicalize_replay_proof_with_identity, proof, schema_model, task.descriptor,
                    protocol_version=VERSION, wrapper_seed_domain=WRAPPER_DOMAIN,
                    transform_device=performance.get("state_transform_device"),
                    include_checkpoint_digest=not performance.get("joint_proof_digest_only", False))
            matched = known_source and hashes["proof_sha256"] in trainer_hashes
            replay = not matched
            if matched and behavior == "joint-cached-correct":
                verdict = trainer_answer_cache[hashes["proof_sha256"]]
            if matched and behavior in {"joint-targeted-cover", "joint-targeted-false-reject"}:
                if trainer_answer_cache is None:
                    raise ValueError("targeted strategy requires the actual trainer cache")
                verdict = trainer_answer_cache[hashes["proof_sha256"]]
                if behavior == "joint-targeted-cover" and verdict is False:
                    verdict = True
                elif behavior == "joint-targeted-false-reject" and verdict is True and task.task_id == false_reject_target:
                    verdict = False
        if replay:
            factory = context.cached_replay_factory if performance.get("reuse_replay_model") else context.factory
            if canonical is None and performance.get("streaming_canonical_replay", False):
                result, _ = clock.call("verifier_replay", "verifier", verify_wrapped_replay_proof,
                    proof, schema_model, task.descriptor, factory,
                    protocol_version=VERSION, wrapper_seed_domain=WRAPPER_DOMAIN,
                    device=context.device.name, tolerance=1e-5,
                    comparison_device=performance.get("comparison_device", "cpu"))
            else:
                if canonical is None:
                    (canonical, _), _ = clock.call("verifier_canonicalize", "verifier", canonicalize_replay_proof,
                        proof, schema_model, task.descriptor, protocol_version=VERSION, wrapper_seed_domain=WRAPPER_DOMAIN)
                result, _ = clock.call("verifier_replay", "verifier", verify_replay_proof,
                    canonical, factory, device=context.device.name, tolerance=1e-5,
                    comparison_device=performance.get("comparison_device", "cpu"))
            verdict = bool(result["passed"])
        end = time.monotonic()
        detail = {"task_id": task.task_id, "replayed": replay, "matched_trainer_commitment": matched,
                  "start":start,"end":end,"elapsed_seconds":end-start}
        if shortcut_receipt is not None:
            detail["public_shortcut"] = shortcut_receipt
        return verdict, detail
    rows = context.task_lanes(performance.get("task_lanes",1)).map(execute_task,tasks,clock)
    verdicts, execution = [r[0] for r in rows], [r[1] for r in rows]
    if precommit_fault_task_id is not None:
        # Explicit laboratory fault injection, never an inferred adversarial
        # frequency or an answer supplied to ordinary effort strategies.
        ids = [t.task_id for t in tasks]
        if behavior != "honest" or precommit_fault_task_id not in ids:
            raise ValueError("invalid pre-commit report fault fixture")
        position = ids.index(precommit_fault_task_id)
        verdicts[position] = not verdicts[position]
        execution[position]["precommit_injected_flip"] = True
    charges = shared_wall_charges([(r["start"],r["end"]) for r in execution])
    for row, charge in zip(execution,charges):
        row["seconds"] = charge
    clock.context.pop("task_id", None)
    report, _ = clock.call("report_commit", "verifier", CommittedVerifierReport.create,
        scenario_id=str(seed), verifier_id=verifier_id or assignment_id, job_id=job_id or assignment_id,
        ordered_segment_ids=[t.task_id for t in tasks], verdicts=verdicts,
        nonce=identity([seed, assignment_id, "nonce"]))
    # Persist the commitment before any owner probe references enter the scoring call.
    report = replace(report, revealed=False)
    clock.call("commit_flush", "verifier", emit_commit, {"assignment_id": assignment_id,
                "commitment": report.commitment, "phase": "COMMITTED"})
    report = replace(report, revealed=True)
    clock.call("reveal_flush", "verifier", emit_commit,
               {"assignment_id": assignment_id, "phase": "REVEALED", "report": asdict(report)})
    return report, execution


def settle_service(report: CommittedVerifierReport, references: OwnerProbeReferences, *,
                   cost_seconds: float, effort_fraction: float, fee: float = 1.25, bond: float = .5,
                   failure_threshold: int = 2, require_complete_probes: bool = False):
    """Only the eight owner probe answers enter this live service gate."""
    if type(references) is not OwnerProbeReferences:
        raise TypeError("live scoring requires the restricted owner probe reference view")
    by_id = dict(zip(report.ordered_segment_ids, report.verdicts))
    if not require_complete_probes and not set(dict(references.answers)).issubset(by_id):
        raise ValueError("committed report omitted required probe identities")
    answers = dict(references.answers)
    if require_complete_probes and sum(answers.values()) != 4:
        raise ValueError("complete service requires four controls and four challenges")
    # Production placeholders carry no audit truth and the shared scorer skips them.
    score_view = SealedEvaluationTruth(report.scenario_id, tuple(
        SealedSegmentTruth(k, answers.get(k, False), k in answers)
        for k in report.ordered_segment_ids))
    if not report.revealed:
        report = replace(report, committed=False, revealed=False)
    result = settle_threshold_assignment(report, score_view, failure_threshold=failure_threshold,
        fee=fee, bond=bond, cost=cost_seconds, effort=effort_fraction, sentinel_only=True,
        required_probe_ids=tuple(answers) if require_complete_probes else None)
    return replace(result, job_id=report.job_id)
