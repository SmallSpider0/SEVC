"""Versioned owner reference and delivery modules for the shared paid runner.

Raw source generation is shared input. Each OnlineJob owns a fresh reference
cache and independently pays its online suffix. No evaluator label is accepted.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, replace, field
import random
from pathlib import Path
from functools import cached_property
from copy import copy, deepcopy

from sevc.core.registry import Registry
from sevc.training import verify_replay_proof
from sevc.training.replay_sources import materialize_branch_source
from sevc.verification.reference_acquisition import JobReferences, acquire_probe_sources
from sevc.verification.paid_replay_service import (
    identity, VERSION, WRAPPER_DOMAIN, DeliveredReplay, OwnerProbeReferences,
    execute_assignment, settle_service, StoredReplayProof,
)
from sevc.verification.replay_coupled_probes import (
    derive_int, ATOM_KEYS, proof_component_hashes, mutate_replay_proof,
    compile_canonical_replay_task, build_mutation_invalidity_witness,
    verify_mutation_invalidity_witness,
)
from sevc.verification.production_audit import audit_sample
from sevc.incentives.verifier_protocol import SealedEvaluationTruth, SealedSegmentTruth
from sevc.verification.verifier_task_policies import settle_threshold_assignment


@dataclass(frozen=True)
class Method:
    probes: str | None
    audit_count: int
    owner_direct: bool = False
    source_binding: str = "raw"
    failure_threshold: int = 2
    fee: float = 1.25
    bond: float = .5
    mutation_profile: str = "legacy-suffix"
    wrapper_profile: str = "permuted"
    require_complete_probes: bool = False
    challenge_backend: str = "cpu"
    equalize_envelopes: bool = False


METHODS = Registry("scoped paid replay method v2")
for _key, _spec in (
    ("rcmp-probe-source-v2", Method("source", 0)),
    ("hidden-gold-paid-v2", Method("gold", 0)),
    ("rcmp-audit-8-v2", Method("source", 8)),
    ("rcmp-audit-all-v2", Method("source", 32)),
    ("production-audit-8-v2", Method(None, 8)),
    ("production-audit-all-v2", Method(None, 32)),
    ("owner-direct-v2", Method(None, 32, True)),
    ("plain-delegation-v2", Method(None, 0)),
    ("depol-local-verification-arbitration-v1", Method(None, 0)),
    ("rcmp-opaque-source-binding-v1", Method("source", 0, source_binding="opaque",
                                           failure_threshold=1, fee=2.5)),
    ("rcmp-opaque-coherent-identity-v2", Method("source", 0, source_binding="opaque",
        failure_threshold=1, fee=2.5, mutation_profile="coherent-all-checkpoints-v1", wrapper_profile="identity")),
    ("rcmp-opaque-gradient-continuation-v3", Method("source", 0, source_binding="opaque",
        failure_threshold=1, fee=2.5, mutation_profile="gradient-continuation-v3", wrapper_profile="identity")),
    ("rcmp-opaque-gradient-continuation-complete-v1", Method("source", 0, source_binding="opaque",
        failure_threshold=1, fee=2.5, mutation_profile="gradient-continuation-v3", wrapper_profile="identity",
        require_complete_probes=True)),
    ("hidden-gold-gradient-continuation-complete-v1", Method("gold", 0, source_binding="opaque",
        failure_threshold=1, fee=2.5, mutation_profile="gradient-continuation-v3", wrapper_profile="identity",
        require_complete_probes=True)),
    ("rcmp-opaque-gradient-continuation-gpu-v1", Method("source", 0, source_binding="opaque",
        failure_threshold=1, fee=2.5, mutation_profile="gradient-continuation-v3", wrapper_profile="identity",
        require_complete_probes=True, challenge_backend="replay")),
    ("hidden-gold-gradient-continuation-gpu-v1", Method("gold", 0, source_binding="opaque",
        failure_threshold=1, fee=2.5, mutation_profile="gradient-continuation-v3", wrapper_profile="identity",
        require_complete_probes=True, challenge_backend="replay")),
    ("rcmp-opaque-gradient-continuation-gpu-equalized-v1", Method("source", 0, source_binding="opaque",
        failure_threshold=1, fee=2.5, mutation_profile="gradient-continuation-v3", wrapper_profile="identity",
        require_complete_probes=True, challenge_backend="replay", equalize_envelopes=True)),
):
    METHODS.add(_key, _spec)


@dataclass(frozen=True)
class Source:
    _proof: object
    hashes: dict
    recipe: dict
    path: str | None = None
    file_sha256: str | None = None

    access_profile: str = "strict"
    _reader: object = field(default=None, compare=False, repr=False)

    compact_header: object = field(default=None, compare=False, repr=False)
    compact_header_seconds: float = 0.0

    @cached_property
    def mapped_file(self):
        from sevc.core.tensor_storage import AuthenticatedTensorFile
        if self._reader is not None:
            if str(self._reader.path) != self.path or self._reader.digest != self.file_sha256:
                raise ValueError("source authentication capability mismatch")
            return self._reader
        return AuthenticatedTensorFile(self.path, self.file_sha256)

    @property
    def proof(self):
        if self.path is None:
            return self._proof
        if self.access_profile in {"authenticated-mmap", "owned-mmap"}:
            return self.mapped_file.load()
        import torch
        from sevc.core.artifacts import sha256_file
        path = Path(self.path)
        if sha256_file(path) != self.file_sha256:
            raise ValueError("scratch source file integrity mismatch")
        return torch.load(path, map_location="cpu", weights_only=False)


def build_source_bank(context, *, seed, namespace, partition, steps, batch_size,
                      invalid_count, clock, initial_state=None, initial_momentum=None,
                      count=40, source_offset=0, role="trainer", scratch_dir=None, initial_rng=None,
                      access_profile="strict", capture_compact_headers=False):
    if not 0 <= invalid_count <= 40:
        raise ValueError("invalid source corruption count")
    invalid_ids = {(24 + j) % 40 for j in range(invalid_count)}
    bank, rows, trainer_cache = [], [], {}
    for index in range(source_offset, source_offset + count):
        clock.context["source_index"] = index
        def recipe():
            rng = random.Random(derive_int(namespace, seed, index, steps, batch_size, "data"))
            indices = rng.sample(range(*partition), steps * batch_size)
            return {"source_id": identity([namespace, context.dataset, seed, index, steps, batch_size]),
                    "source_index": index, "source_seed": derive_int(namespace, seed, index, "model") % 2**32,
                    "sample_indices": indices, "sample_labels": [int(context.train[i][1]) for i in indices],
                    "steps": steps, "batch_size": batch_size, "dataset": context.dataset,
                    "block_seed": seed, "namespace": namespace}
        rec, _ = clock.call("source_recipe", role, recipe)
        proof, _ = clock.call("source_materialization", role, materialize_branch_source,
            context, rec, initial_state=initial_state, initial_momentum=initial_momentum, initial_rng=initial_rng)
        mutation = None
        if index in invalid_ids:
            (proof, mutation), _ = clock.call("trainer_precommit_mutation", role,
                mutate_replay_proof, proof, source_id=rec["source_id"],
                post_commit_seed=derive_int(seed, "trainer"), atom_key=ATOM_KEYS[index % 4],
                magnitude=4e-5, protocol_version=VERSION, final_two_checkpoints=True)
        hashes, _ = clock.call("source_commitment_hash", role, proof_component_hashes, proof)
        environment_id = getattr(context, 'replay_environment_id', None)
        if environment_id is not None:
            from sevc.core.replay_environment import bind_source
            rec.update(bind_source(hashes['proof_sha256'], environment_id))
        source_path = source_file_hash = source_reader = None
        if scratch_dir is not None:
            import torch
            from sevc.core.artifacts import sha256_file
            source_path = Path(scratch_dir) / f"source-{index}.pt"
            if source_path.exists():
                raise FileExistsError("scratch source may not overwrite another identity")
            from sevc.core.scratch_budget import guarded_tensor_save
            clock.call("source_payload_materialization", role, guarded_tensor_save, proof, source_path)
            if access_profile == "owned-mmap":
                from sevc.core.tensor_storage import AuthenticatedTensorFile
                source_reader = AuthenticatedTensorFile.inspect_trusted_owned(source_path)
            elif access_profile == "authenticated-mmap":
                from sevc.core.tensor_storage import AuthenticatedTensorFile
                source_reader, _ = clock.call("source_payload_hash", role, AuthenticatedTensorFile.inspect_owned, source_path)
                source_file_hash = source_reader.digest
            else:
                source_file_hash, _ = clock.call("source_payload_hash", role, sha256_file, source_path)
        header, header_seconds = None, 0.0
        if capture_compact_headers:
            from sevc.verification.replay_coupled_probes import CompactProofHeader
            header, header_seconds = clock.call("source_compact_header_capture", "owner",
                CompactProofHeader.capture, proof, hashes["proof_sha256"])
        bank.append(Source(proof if source_path is None else None, hashes, rec,
                           str(source_path) if source_path else None, source_file_hash, access_profile, source_reader, header, header_seconds))
        # Trainer-created information only. Owner controller never receives it.
        trainer_cache[hashes["proof_sha256"]] = index not in invalid_ids
        rows.append({**rec, **hashes, "trainer_mutation": mutation, "scratch_sha256": source_file_hash})
        del proof
    clock.context.pop("source_index", None)
    return tuple(bank), rows, trainer_cache


def complete_calibration_observation(*, ledger, assignment_id, report, settlement,
                                     production, references, clock, timely=True,
                                     owner_cost_per_second=0.):
    """Audit an already-completed reserved service through actual owner replay.

    The same operation supports immediate calibration and post-epoch observation.
    It never grants qualification retroactively or accepts evaluator truth.
    """
    correct, receipt_hash, elapsed, receipts = None, None, 0., {}
    if settlement.accepted_report:
        def audit():
            rs = {k: references.acquire(sid) for k, sid in production.items()}
            verdicts = dict(zip(report.ordered_segment_ids, report.verdicts))
            return rs, all(verdicts.get(k) == r['passed'] for k,r in rs.items())
        (receipts, correct), elapsed = clock.call('calibration_production_reference', 'owner', audit)
        receipt_hash = identity(receipts)
    ledger.record(assignment_id, settlement, timely=timely, reference_correct=correct,
                  reference_receipt_sha256=receipt_hash, owner_cost=elapsed * owner_cost_per_second)
    return {'receipts':receipts, 'correct':correct, 'receipt_sha256':receipt_hash,
            'owner_wall_seconds':elapsed}


def replay_owner_proof(proof, *, context, clock, performance, phase):
    """Apply the same backend policy to owner references and challenges."""
    result, seconds = clock.call(phase, "owner", verify_replay_proof, proof,
        context.cached_replay_factory if performance.get("reuse_replay_model") else context.factory,
        device=context.device.name, tolerance=1e-5,
        comparison_device=performance.get("comparison_device", "cpu"))
    environment_id = getattr(context, 'replay_environment_id', None)
    if environment_id is not None:
        result = {**result, 'replay_environment_id':environment_id}
    return result, seconds


class OnlineJob:
    def __init__(self, *, bank, context, clock, performance, job_id, method_key,
                 role_secret, emit, paired_production=None, gold_factory=None, delivery_scratch_dir=None,
                 prepare_only=False):
        self.context, self.clock, self.performance = context, clock, performance
        self.job_id, self.method = job_id, METHODS.get(method_key)
        self.method_key, self.emit = method_key, emit
        self.header_capture_charge = (sum(s.compact_header_seconds for s in bank)
            if performance.get("compact_source_headers") and not self.method.owner_direct else 0.0)
        self.disclosure_epoch = None
        from sevc.core.replay_environment import require_sources
        require_sources(bank, getattr(context, 'replay_environment_id', None), getattr(context, 'replay_environment_bridge', None))
        self.bank = {s.hashes["proof_sha256"]: s for s in bank}
        if len(self.bank) != 40:
            raise ValueError("committed source bank is not forty unique proofs")
        self.references = JobReferences(job_id, self._replay, emit)
        self.probes = None
        self.task_rows, self.tasks, self.production = [], (), {}
        selected = []
        if self.method.probes == "source":
            self.preparation = acquire_probe_sources(tuple(self.bank), secret_hex=role_secret,
                                                      references=self.references)
            if not self.preparation["issued"]:
                self.issued = False
                return
            selected = self.preparation["selected_ids"]
            production_ids = self.preparation["production_ids"]
            if paired_production is not None:
                raise ValueError("RCMP controller must not receive paired production IDs")
        else:
            if paired_production is None or len(set(paired_production)) != 32:
                raise ValueError("comparison receives exactly 32 target identities")
            production_ids = list(paired_production)
            if not set(production_ids) <= self.bank.keys():
                raise ValueError("paired targets are not committed inputs")
            self.preparation = {"issued": True, "status": "PAIRED_TARGETS_READY",
                                "production_ids": production_ids, "selected_ids": []}
        self.production_source_ids = tuple(production_ids)
        self.issued = True
        if prepare_only:
            return
        if self.method.owner_direct:
            self.production = {sid: sid for sid in production_ids}
            return
        probe_sources = []
        if self.method.probes == "source":
            probe_sources = [self.bank[s] for s in selected]
        elif self.method.probes == "gold":
            if gold_factory is None:
                raise ValueError("hidden gold requires separately paid source generation")
            gold = gold_factory()
            require_sources(gold, getattr(context, 'replay_environment_id', None), getattr(context, 'replay_environment_bridge', None))
            for source in gold:
                sid = source.hashes["proof_sha256"]
                self.bank[sid] = source
                if not self.references.acquire(sid)["passed"]:
                    self.issued = False
            if not self.issued:
                self.preparation.update(issued=False, status="GOLD_REFERENCE_SAFE_DEFER")
                return
            probe_sources = list(gold)
        model = context.factory().cpu()
        from sevc.core.tensor_storage import ResidentDeliveryBudget
        resident_budget = ResidentDeliveryBudget(performance.get("resident_delivery_bytes", 0))
        tasks, answers = [], []
        def compile_one(item):
            rank, source = item
            sid = source.hashes["proof_sha256"]
            borrowed_header = (rank < 36 and self.method.wrapper_profile == "identity"
                and performance.get("compact_source_headers", False)
                and performance.get("identity_disk_delivery", False)
                and source.compact_header is not None and source.path is not None)
            if borrowed_header:
                clock.call("source_header_identity_check", "owner", source.mapped_file.check_identity)
                source_proof = source.compact_header
            else:
                source_proof, _ = clock.call("source_payload_read", "owner", lambda: source.proof)
            if not borrowed_header and performance.get("tensor_access") in {"authenticated-mmap", "owned-mmap"}:
                # A tiny batch view otherwise pins the entire torch mmap file
                # after the transformed model/optimizer states have been copied.
                source_proof = replace(source_proof, batches=tuple(
                    (x.clone(), y.clone()) for x, y in source_proof.batches))
            role = "production" if rank < 32 else "control" if rank < 36 else "challenge"
            atom = ATOM_KEYS[rank - 36] if role == "challenge" else None
            from sevc.verification.public_replay_shortcuts import opaque_source_handle
            public_binding = (opaque_source_handle(role_secret, job_id, sid)
                              if self.method.source_binding == "opaque" else sid)
            bundle, _ = clock.call("task_compile", "owner", compile_canonical_replay_task,
                source_proof, model, context.build_key, source_id=source.recipe["source_id"],
                source_commitment=sid, post_commit_seed=derive_int(role_secret, sid, "atom"),
                role=role, atom_key=atom, permutation_seed=derive_int(role_secret, sid, "wrapper"),
                protocol_version=VERSION, wrapper_seed_domain=WRAPPER_DOMAIN, tamper_delta=4e-5,
                source_component_hashes=source.hashes, delivery_profile="compact",
                identity_profile=performance["identity_profile"], schema_validator_profile="shared",
                validated_source_verdict=True if role != "production" else None,
                final_two_checkpoints=True, transform_device=performance.get("state_transform_device"),
                public_source_commitment=public_binding, mutation_profile=self.method.mutation_profile,
                wrapper_profile=self.method.wrapper_profile,
                challenge_device=(context.device.name if self.method.challenge_backend == "replay" else "cpu"))
            witness = None
            if atom is not None and self.method.mutation_profile == "gradient-continuation-v3":
                outcome, seconds = replay_owner_proof(bundle.canonical_candidate,
                    context=context, clock=clock, performance=performance,
                    phase="challenge_full_replay_validation")
                if outcome["passed"]:
                    raise ValueError("continuation challenge unexpectedly passes full replay")
                witness = {"kind": "measured-canonical-replay-v3", "result": outcome,
                           "seconds": seconds, "candidate_sha256": bundle.component_hashes["candidate"]["proof_sha256"]}
            elif atom is not None:
                mutated = bundle.component_hashes.get("candidate")
                if mutated is None:
                    mutated, _ = clock.call("challenge_hash", "owner", proof_component_hashes, bundle.canonical_candidate)
                witness, _ = clock.call("challenge_witness", "owner", build_mutation_invalidity_witness,
                    source_proof, bundle.canonical_candidate, bundle.mutation,
                    source_id=source.recipe["source_id"], source_commitment=sid,
                    source_proof_sha256=sid, mutated_proof_sha256=mutated["proof_sha256"],
                    descriptor=bundle.descriptor, replay_tolerance=1e-5, source_replay_valid=True)
                if not verify_mutation_invalidity_witness(witness):
                    raise ValueError("invalid challenge witness")
            delivered_proof = bundle.wrapped_proof
            cache_hash = None
            # Identity delivery borrows the run-owned, immutable source mappings.
            # Its challenge adds only copy-on-write tensor patches; serializing
            # forty full derivatives would duplicate the entire source bank.
            borrowed_source = (self.method.wrapper_profile == "identity" and atom is None
                               and performance.get("identity_disk_delivery", False))
            if borrowed_source:
                if source.path is None:
                    raise ValueError("bounded identity delivery requires file-backed sources")
                delivered_proof = StoredReplayProof(source.path, source.file_sha256,
                    source.access_profile, source.mapped_file)
            if (not borrowed_source and (self.method.wrapper_profile != "identity"
                    or performance.get("identity_disk_delivery", False)) and delivery_scratch_dir is not None
                    and not resident_budget.reserve(delivered_proof)):
                import torch
                from sevc.core.artifacts import sha256_file
                cache_path = Path(delivery_scratch_dir) / f"task-{rank}.pt"
                if cache_path.exists():
                    raise FileExistsError("delivery cache overwrite")
                from sevc.core.scratch_budget import guarded_tensor_save
                clock.call("delivery_payload_write", "owner", guarded_tensor_save, delivered_proof, cache_path)
                reader = None
                if performance.get("tensor_access") == "owned-mmap":
                    from sevc.core.tensor_storage import AuthenticatedTensorFile
                    reader = AuthenticatedTensorFile.inspect_trusted_owned(cache_path)
                elif performance.get("tensor_access") == "authenticated-mmap":
                    from sevc.core.tensor_storage import AuthenticatedTensorFile
                    reader, _ = clock.call("delivery_payload_hash", "owner", AuthenticatedTensorFile.inspect_owned, cache_path)
                    cache_hash = reader.digest
                else:
                    cache_hash, _ = clock.call("delivery_payload_hash", "owner", sha256_file, cache_path)
                delivered_proof = StoredReplayProof(str(cache_path), cache_hash, performance.get("tensor_access", "strict"), reader)
            task = DeliveredReplay(bundle.sealed.task_id, public_binding, delivered_proof,
                                   bundle.descriptor, bundle.public_envelope)
            task_row = {"task_id": task.task_id, "source_sha256": sid, "role": role,
                                  "probe_answer": role == "control" if role != "production" else None,
                                  "wrapped_hashes": bundle.component_hashes["wrapped"], "witness": witness,
                                  "payload_bytes": len(bundle.public_envelope_bytes),
                                  "delivery_cache_sha256": cache_hash,
                                  "delivery_storage": "scratch" if isinstance(delivered_proof, StoredReplayProof) else "resident",
                                  "scratch_integrity": performance.get("tensor_access", "strict"),
                                  "compile_seconds": bundle.compile_seconds}
            return task, role, task_row
        from sevc.core.task_lanes import TaskLanes
        lanes = TaskLanes(performance.get("owner_compile_lanes", 1), performance.get("owner_compile_device", "cpu"))
        try:
            compiled = lanes.map(compile_one, enumerate([self.bank[s] for s in production_ids] + probe_sources), clock)
        finally:
            lanes.close()
        if self.method.equalize_envelopes:
            from dataclasses import replace as _replace
            from sevc.verification.replay_coupled_probes import (
                equalize_public_envelopes, serialize_public_replay_envelope)
            padded = clock.call("envelope_equalization", "owner", equalize_public_envelopes,
                                [task.envelope for task, _, _ in compiled])[0]
            compiled = [(_replace(task, envelope=envelope), role,
                         {**task_row, "payload_bytes": len(serialize_public_replay_envelope(envelope))})
                        for (task, role, task_row), envelope in zip(compiled, padded)]
        for task, role, task_row in compiled:
            tasks.append(task)
            self.task_rows.append(task_row)
            if role == "production":
                self.production[task.task_id] = task_row["source_sha256"]
            else:
                answers.append((task.task_id, role == "control"))
        self.tasks = tuple(sorted(tasks, key=lambda t: derive_int(role_secret, "order", t.task_id)))
        self.probes = OwnerProbeReferences(tuple(answers)) if answers else None
        emit({"event": "job-delivery-ready", "job_id": job_id, "method": method_key,
              "tasks": self.task_rows, "production_source_ids": production_ids})

    def preparation_template(self):
        """Capture before actor execution; later audit answers cannot enter it."""
        template = copy(self)
        template._prepared_receipts = deepcopy(self.references._receipts)
        return template

    def reuse_preparation(self, job_id):
        """Explicit conditional study view; complete-cost jobs never call this."""
        result = copy(self)
        result.job_id = job_id
        result.preparation = {**self.preparation, "job_id": job_id,
                              "conditional_preparation_from": self.job_id}
        result.references = JobReferences(job_id, result._replay, self.emit)
        result.references._receipts = deepcopy(self._prepared_receipts)
        self.emit({"event": "conditional-preparation-reused", "job_id": job_id,
                   "donor_job_id": self.job_id, "method": self.method_key,
                   "reference_receipts": result.references._receipts})
        if self.issued:
            self.emit({"event": "job-delivery-ready", "job_id": job_id,
                       "method": self.method_key, "tasks": self.task_rows,
                       "production_source_ids": list(self.production_source_ids),
                       "conditional_preparation_from": self.job_id})
        return result

    def _replay(self, sid):
        source = self.bank[sid]
        proof, _ = self.clock.call("source_payload_read", "owner", lambda: source.proof)
        result, seconds = replay_owner_proof(proof, context=self.context,
            clock=self.clock, performance=self.performance, phase="source_reference_replay")
        return {**result, "proof_sha256": sid, "seconds": seconds}

    def begin_service_epoch(self, assignment_ids, *, shared_epoch=None):
        from sevc.verification.disclosure import ProbeDisclosureEpoch
        if self.disclosure_epoch is not None:
            raise ValueError("job disclosure schedule is already frozen")
        if shared_epoch is not None and not shared_epoch.covers(assignment_ids):
            raise ValueError("shared disclosure epoch does not cover this job")
        self.disclosure_epoch = shared_epoch or ProbeDisclosureEpoch(self.job_id, assignment_ids)

    def close_service_epoch(self):
        if self.disclosure_epoch is None:
            raise ValueError("no disclosure schedule was registered")
        self.disclosure_epoch.close()
        result = self.disclosure_epoch.release(self.probes.answers if self.probes else ())
        self.emit({"event": "reference-answers-published", **result})
        return result

    def calibrate(self, *, ledger, independent_block_id, timely=True, owner_cost_per_second=0., **service):
        """Pay through the normal service and audit production without evaluator labels."""
        assignment_id = service["assignment_id"]
        if ledger.fee != self.method.fee or service.get("verifier_id") != ledger.verifier_id:
            raise ValueError("calibration and service terms must match")
        ledger.reserve(assignment_id, independent_block_id)
        report, settled, details = self.serve(**service)
        complete_calibration_observation(ledger=ledger, assignment_id=assignment_id,
            report=report, settlement=settled, production=self.production, references=self.references,
            clock=self.clock, timely=timely, owner_cost_per_second=owner_cost_per_second)
        return report, settled, details

    def serve(self, *, behavior, seed, assignment_id, audit_secret, trainer_cache, commits,
              verifier_id=None, job_id=None, precommit_fault_task_id=None):
        if not self.issued:
            return None
        if precommit_fault_task_id is not None and precommit_fault_task_id not in self.production:
            raise ValueError("conditioned report fault must target a production task")
        if self.method.require_complete_probes:
            if self.disclosure_epoch is None:
                raise ValueError("complete service requires a frozen disclosure schedule")
            self.disclosure_epoch.begin(assignment_id)
        self.emit({"event": "audit-seed-committed", "assignment_id": assignment_id,
                   "commitment": identity([assignment_id, audit_secret])})
        shared_cache=trainer_cache if behavior.startswith('joint-') else None
        self.emit({'event':'assignment-information-view','assignment_id':assignment_id,
                   'view':'trainer-plus-verifier' if shared_cache is not None else 'verifier-delivered-only',
                   'trainer_cache_size':len(shared_cache) if shared_cache is not None else 0})
        (report, details), verifier_wall = self.clock.call('verifier-service-interface','verifier',execute_assignment,self.tasks, behavior=behavior,
            trainer_hashes=frozenset(shared_cache or {}), trainer_answer_cache=shared_cache,
            context=self.context, seed=seed, assignment_id=assignment_id,
            clock=self.clock, performance=self.performance, emit_commit=commits,
            verifier_id=verifier_id, job_id=job_id, precommit_fault_task_id=precommit_fault_task_id)
        settled, admission_seconds = self.clock.call('owner-service-admission', 'owner', self._admit,
            report, details, assignment_id, audit_secret, verifier_wall)
        settled = replace(settled, diagnostics=settled.diagnostics + (("owner_admission_seconds", admission_seconds),))
        if self.method.require_complete_probes:
            self.disclosure_epoch.finish(assignment_id, settled.status)
        return report, settled, details

    def _admit(self, report, details, assignment_id, audit_secret, verifier_wall):
        selected = audit_sample(audit_secret, report.commitment, self.production, self.method.audit_count)
        self.emit({"event": "audit-selected", "assignment_id": assignment_id,
                   "secret": audit_secret, "selected": list(selected), "report_commitment": report.commitment})
        def acquire(k):
            return k,self.references.acquire(self.production[k])["passed"]
        answers = dict(self.context.task_lanes(self.performance.get('task_lanes',1)).map(acquire,selected,self.clock))
        cost = verifier_wall
        effort = sum(d["replayed"] for d in details) / len(details)
        if self.probes:
            settled = settle_service(report, self.probes, cost_seconds=cost, effort_fraction=effort,
                fee=self.method.fee, bond=self.method.bond, failure_threshold=self.method.failure_threshold,
                require_complete_probes=self.method.require_complete_probes)
        else:
            truth = SealedEvaluationTruth(report.scenario_id, tuple(
                # Plain delegation admits a well-formed report without checking
                # truth. This self-consistency view provides no safety claim.
                SealedSegmentTruth(k, v, True) for k, v in
                zip(report.ordered_segment_ids, report.verdicts)))
            settled = settle_threshold_assignment(report, truth, failure_threshold=1, fee=self.method.fee,
                bond=self.method.bond, cost=cost, effort=effort, sentinel_only=True)
        if selected:
            truth = SealedEvaluationTruth(report.scenario_id, tuple(
                SealedSegmentTruth(k, answers.get(k, False), k in answers) for k in report.ordered_segment_ids))
            audited = settle_threshold_assignment(report, truth, failure_threshold=1, fee=self.method.fee,
                bond=self.method.bond, cost=cost, effort=effort, sentinel_only=True)
            if settled.status == "PASS":
                settled = audited
        settled = replace(settled, job_id=report.job_id)
        return settled
