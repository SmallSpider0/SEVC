"""Selected canonical scientific routines; deployment wrappers are omitted."""
from __future__ import annotations


import csv

from datetime import datetime, timezone

import json

import os

from pathlib import Path

import subprocess

import threading

import time

from typing import Mapping, Sequence

class AppendLog:
    def __init__(self, path: Path):
        self.path = path
        self.handle = path.open("x", encoding="utf-8")
        self.lock = threading.Lock()

    def __call__(self, value):
        encoded = json.dumps(value, sort_keys=True, allow_nan=False)+"\n"
        with self.lock:
            self.handle.write(encoded)
            self.handle.flush()

    def close(self):
        self.handle.flush()
        os.fsync(self.handle.fileno())
        self.handle.close()


def integrate_utilization(samples: Sequence[Mapping], start: float, end: float):
    """Left-held one-second samples clipped to the entire requested interval."""
    if end <= start:
        raise ValueError("empty utilization window")
    rows = sorted(samples, key=lambda x: x["monotonic"])
    if not rows or rows[0]["monotonic"] > start or rows[-1]["monotonic"] < end:
        raise ValueError("utilization samples do not bracket the full workload")
    area = 0.
    max_gap = 0.
    for left, right in zip(rows, rows[1:]):
        lo, hi = max(start, left["monotonic"]), min(end, right["monotonic"])
        if hi <= lo:
            continue
        gap = right["monotonic"]-left["monotonic"]
        max_gap = max(max_gap, gap)
        value = float(left["utilization_percent"])
        if not 0 <= value <= 100:
            raise ValueError("invalid utilization sample")
        area += value*(hi-lo)
    return {"average_percent": area/(end-start), "wall_seconds": end-start,
            "max_sample_gap_seconds": max_gap, "sampling_valid": max_gap <= 2.5}


class GPUSampler:
    def __init__(self, root: Path, uuid: str, enabled: bool):
        self.uuid, self.enabled = uuid, enabled
        self.rows, self.errors = [], []
        self.log = AppendLog(root/"gpu-utilization.jsonl")
        self.stop_event = threading.Event()
        self.thread = None

    def sample(self):
        if not self.enabled:
            return
        value = subprocess.check_output([
            "nvidia-smi", "-i", self.uuid, "--query-gpu=uuid,utilization.gpu,memory.used",
            "--format=csv,noheader,nounits"], text=True, timeout=5).strip().split(",")
        uuid, utilization, memory = [x.strip() for x in value]
        if uuid != self.uuid:
            raise ValueError("GPU UUID changed during measurement")
        pid_rows = subprocess.check_output([
            "nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
            text=True, timeout=5).strip()
        memory_fields = {}
        status_path = Path("/proc/self/status")
        if status_path.exists():
            for line in status_path.read_text().splitlines():
                if line.startswith(("VmRSS:","VmSwap:")):
                    key,value = line.split(":",1)
                    memory_fields[key] = int(value.split()[0])
        limit_path = Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")
        row = {"monotonic": time.monotonic(), "utc": datetime.now(timezone.utc).isoformat(),
               "gpu_uuid": uuid, "utilization_percent": float(utilization),
               "memory_used_mib": float(memory), "runner_pid": os.getpid(),
               "compute_pid_inventory": pid_rows,
               "runner_rss_kib":memory_fields.get("VmRSS"),
               "runner_swap_kib":memory_fields.get("VmSwap"),
               "effective_memory_limit_bytes":int(limit_path.read_text()) if limit_path.exists() else None}
        for key,name in (('cgroup_memory_usage_bytes','memory.usage_in_bytes'),
                         ('cgroup_memory_failcnt','memory.failcnt')):
            path=Path('/sys/fs/cgroup/memory')/name
            row[key]=int(path.read_text()) if path.exists() else None
        self.rows.append(row); self.log(row)

    def start(self):
        self.sample()
        def loop():
            while not self.stop_event.wait(1.):
                try:
                    self.sample()
                except Exception as exc:
                    self.errors.append(str(exc))
        self.thread = threading.Thread(target=loop, daemon=True)
        self.thread.start()

    def finish(self, root: Path):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=15)
        self.sample(); self.log.close()
        with (root/"gpu-utilization.csv").open("x", newline="") as handle:
            fields = ("monotonic", "utc", "gpu_uuid", "utilization_percent", "memory_used_mib",
                      "runner_pid", "compute_pid_inventory", "runner_rss_kib", "runner_swap_kib",
                      "effective_memory_limit_bytes", "cgroup_memory_usage_bytes", "cgroup_memory_failcnt")
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader(); writer.writerows(self.rows)
