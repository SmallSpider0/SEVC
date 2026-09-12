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
                      access_profile="strict"):
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
        source_path = source_file_hash = source_reader = None
        if scratch_dir is not None:
            import torch
            from sevc.core.artifacts import sha256_file
            source_path = Path(scratch_dir) / f"source-{index}.pt"
            if source_path.exists():
                raise FileExistsError("scratch source may not overwrite another identity")
            clock.call("source_payload_materialization", role, torch.save, proof, source_path)
            if access_profile == "owned-mmap":
                from sevc.core.tensor_storage import AuthenticatedTensorFile
                source_reader = AuthenticatedTensorFile.inspect_trusted_owned(source_path)
            elif access_profile == "authenticated-mmap":
                from sevc.core.tensor_storage import AuthenticatedTensorFile
                source_reader, _ = clock.call("source_payload_hash", role, AuthenticatedTensorFile.inspect_owned, source_path)
                source_file_hash = source_reader.digest
            else:
                source_file_hash, _ = clock.call("source_payload_hash", role, sha256_file, source_path)
        bank.append(Source(proof if source_path is None else None, hashes, rec,
                           str(source_path) if source_path else None, source_file_hash, access_profile, source_reader))
        # Trainer-created information only. Owner controller never receives it.
        trainer_cache[hashes["proof_sha256"]] = index not in invalid_ids
        rows.append({**rec, **hashes, "trainer_mutation": mutation, "scratch_sha256": source_file_hash})
        del proof
    clock.context.pop("source_index", None)
    return tuple(bank), rows, trainer_cache


class OnlineJob:
    def __init__(self, *, bank, context, clock, performance, job_id, method_key,
                 role_secret, emit, paired_production=None, gold_factory=None, delivery_scratch_dir=None,
                 prepare_only=False):
        self.context, self.clock, self.performance = context, clock, performance
        self.job_id, self.method = job_id, METHODS.get(method_key)
        self.method_key, self.emit = method_key, emit
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
            source_proof, _ = clock.call("source_payload_read", "owner", lambda: source.proof)
            if performance.get("tensor_access") in {"authenticated-mmap", "owned-mmap"}:
                # A tiny batch view otherwise pins the entire torch mmap file
                # after the transformed model/optimizer states have been copied.
                source_proof = replace(source_proof, batches=tuple(
                    (x.clone(), y.clone()) for x, y in source_proof.batches))
            role = "production" if rank < 32 else "control" if rank < 36 else "challenge"
            atom = ATOM_KEYS[rank - 36] if role == "challenge" else None
            bundle, _ = clock.call("task_compile", "owner", compile_canonical_replay_task,
                source_proof, model, context.build_key, source_id=source.recipe["source_id"],
                source_commitment=sid, post_commit_seed=derive_int(role_secret, sid, "atom"),
                role=role, atom_key=atom, permutation_seed=derive_int(role_secret, sid, "wrapper"),
                protocol_version=VERSION, wrapper_seed_domain=WRAPPER_DOMAIN, tamper_delta=4e-5,
                source_component_hashes=source.hashes, delivery_profile="compact",
                identity_profile=performance["identity_profile"], schema_validator_profile="shared",
                validated_source_verdict=True if role != "production" else None,
                final_two_checkpoints=True, transform_device=performance.get("state_transform_device"))
            witness = None
            if atom is not None:
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
            if delivery_scratch_dir is not None and not resident_budget.reserve(delivered_proof):
                import torch
                from sevc.core.artifacts import sha256_file
                cache_path = Path(delivery_scratch_dir) / f"task-{rank}.pt"
                if cache_path.exists():
                    raise FileExistsError("delivery cache overwrite")
                clock.call("delivery_payload_write", "owner", torch.save, delivered_proof, cache_path)
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
            task = DeliveredReplay(bundle.sealed.task_id, sid, delivered_proof,
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
        for task, role, task_row in compiled:
            tasks.append(task)
            self.task_rows.append(task_row)
            if role == "production":
                self.production[task.task_id] = task.source_commitment
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
        result, seconds = self.clock.call("source_reference_replay", "owner", verify_replay_proof,
            proof, self.context.cached_replay_factory if self.performance.get("reuse_replay_model") else self.context.factory,
            device=self.context.device.name, tolerance=1e-5,
            comparison_device=self.performance.get("comparison_device", "cpu"))
        return {**result, "proof_sha256": sid, "seconds": seconds}

    def serve(self, *, behavior, seed, assignment_id, audit_secret, trainer_cache, commits,
              verifier_id=None, job_id=None):
        if not self.issued:
            return None
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
            verifier_id=verifier_id, job_id=job_id)
        settled, admission_seconds = self.clock.call('owner-service-admission', 'owner', self._admit,
            report, details, assignment_id, audit_secret, verifier_wall)
        settled = replace(settled, diagnostics=settled.diagnostics + (("owner_admission_seconds", admission_seconds),))
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
            settled = settle_service(report, self.probes, cost_seconds=cost, effort_fraction=effort)
        else:
            truth = SealedEvaluationTruth(report.scenario_id, tuple(
                # Plain delegation admits a well-formed report without checking
                # truth. This self-consistency view provides no safety claim.
                SealedSegmentTruth(k, v, True) for k, v in
                zip(report.ordered_segment_ids, report.verdicts)))
            settled = settle_threshold_assignment(report, truth, failure_threshold=1, fee=1.25,
                bond=.5, cost=cost, effort=effort, sentinel_only=True)
        if selected:
            truth = SealedEvaluationTruth(report.scenario_id, tuple(
                SealedSegmentTruth(k, answers.get(k, False), k in answers) for k in report.ordered_segment_ids))
            audited = settle_threshold_assignment(report, truth, failure_threshold=1, fee=1.25,
                bond=.5, cost=cost, effort=effort, sentinel_only=True)
            if settled.status == "PASS":
                settled = audited
        settled = replace(settled, job_id=report.job_id)
        return settled
