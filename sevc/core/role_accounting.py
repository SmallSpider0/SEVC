"""Nested phase accounting on one controlled worker, with exclusive wall/CPU."""
from __future__ import annotations

import threading
import time
import uuid


class RoleClock:
    def __init__(self, sync, emit, *, cuda=False, selective_cuda=False, resource_accounting=None):
        self.sync, self.emit, self.cuda = sync, emit, cuda
        self.local = threading.local()
        self.selective_cuda = selective_cuda
        if resource_accounting not in (None, "absolute-v1"):
            raise ValueError("unknown resource accounting")
        self.resource_accounting = resource_accounting
        self.profiling = False

    @property
    def context(self):
        if not hasattr(self.local, "context"):
            self.local.context = {}
        return self.local.context

    @context.setter
    def context(self, value):
        self.local.context = value

    def call(self, phase, role, fn, /, *args, **kwargs):
        if not hasattr(self.local, "stack"):
            self.local.stack = []
        stack = self.local.stack
        event = {"event_id": uuid.uuid4().hex, "parent_id": stack[-1]["event_id"] if stack else None,
                 "child_wall": 0., "child_cpu": 0.}
        gpu_start = gpu_end = None
        gpu_phase = not self.selective_cuda or phase in {
            "source_materialization", "source_reference_replay", "verifier_replay",
            "task_compile", "target-reference", "target-challenge",
            "trajectory-training", "diagnostic-probe-replay", "challenge_full_replay_validation",
            "depol-estimator-replay", "depol-native-recompute", "calibration_owner_replay"}
        if self.resource_accounting and phase == "formal_owner_reference_audit":
            gpu_phase = True
        if self.cuda and gpu_phase:
            import torch
            gpu_start, gpu_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            gpu_start.record()
        resource_start = None
        if self.resource_accounting:
            from sevc.core.absolute_resources import phase_start
            resource_start = phase_start()
        if self.profiling:
            from torch.profiler import record_function
            profile_range = record_function('sevc|'+role+'|'+phase)
            profile_range.__enter__()
        else:
            profile_range = None
        start, cpu_start = time.monotonic(), time.thread_time()
        stack.append(event)
        failed = False
        try:
            return_value = fn(*args, **kwargs)
        except BaseException:
            failed = True
            raise
        finally:
            if not self.selective_cuda or gpu_phase:
                self.sync()
            gpu_seconds = None
            if gpu_end is not None:
                gpu_end.record(); gpu_end.synchronize()
                gpu_seconds = gpu_start.elapsed_time(gpu_end) / 1000
            end, cpu_end = time.monotonic(), time.thread_time()
            stack.pop()
            wall, cpu = end - start, cpu_end - cpu_start
            if stack:
                stack[-1]["child_wall"] += wall; stack[-1]["child_cpu"] += cpu
            extra = {}
            if resource_start is not None:
                from sevc.core.absolute_resources import phase_end
                extra = phase_end(resource_start, event, stack[-1] if stack else None, self.cuda)
            if profile_range is not None:
                profile_range.__exit__(None, None, None)
            self.emit({**extra, **self.context, "event_id": event["event_id"], "parent_id": event["parent_id"],
                "phase": phase, "role": role, "start": start, "end": end, "seconds": wall,
                "cpu_thread_seconds": cpu, "exclusive_seconds": max(0., wall-event["child_wall"]),
                "exclusive_cpu_thread_seconds": max(0., cpu-event["child_cpu"]),
                "cuda_stream_elapsed_seconds": gpu_seconds,
                "cuda_measurement": "stream event span including host gaps; not kernel busy demand",
                "technical_failure": failed})
        return return_value, wall
