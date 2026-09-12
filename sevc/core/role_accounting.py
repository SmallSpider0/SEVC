"""Nested phase accounting on one controlled worker, with exclusive wall/CPU."""
from __future__ import annotations

import threading
import time
import uuid


class RoleClock:
    def __init__(self, sync, emit, *, cuda=False, selective_cuda=False):
        self.sync, self.emit, self.cuda = sync, emit, cuda
        self.local = threading.local()
        self.selective_cuda = selective_cuda

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
            "trajectory-training", "diagnostic-probe-replay"}
        if self.cuda and gpu_phase:
            import torch
            gpu_start, gpu_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            gpu_start.record()
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
            self.emit({**self.context, "event_id": event["event_id"], "parent_id": event["parent_id"],
                "phase": phase, "role": role, "start": start, "end": end, "seconds": wall,
                "cpu_thread_seconds": cpu, "exclusive_seconds": max(0., wall-event["child_wall"]),
                "exclusive_cpu_thread_seconds": max(0., cpu-event["child_cpu"]),
                "cuda_stream_elapsed_seconds": gpu_seconds,
                "cuda_measurement": "stream event span including host gaps; not kernel busy demand",
                "technical_failure": failed})
        return return_value, wall
