"""Bounded independent task lanes inside one process and one visible GPU."""
from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
import threading
import time


class TaskLanes:
    def __init__(self, width=1, device="cpu"):
        if width not in (1, 2, 4, 8, 16):
            raise ValueError("unreviewed task lane count")
        self.width, self.device = width, device
        self.local = threading.local()
        self.executor = None

    def close(self):
        if self.executor is not None:
            self.executor.shutdown(wait=True, cancel_futures=True)
            self.executor = None

    def map(self, fn, items, clock):
        """Bound in-flight memory and preserve input/report order."""
        if self.width == 1:
            return [fn(item) for item in items]
        parent = dict(clock.context)
        def invoke(item):
            clock.context = dict(parent)
            if self.device.startswith("cuda"):
                import torch
                if not hasattr(self.local, "stream"):
                    self.local.stream = torch.cuda.Stream()
                with torch.cuda.stream(self.local.stream):
                    value = fn(item)
                    self.local.stream.synchronize()
                    return value
            return fn(item)
        # Reuse stream-local allocator pools across assignments. Recreating the
        # executor/streams would strand otherwise reusable CUDA cached blocks.
        if self.executor is None:
            self.executor = ThreadPoolExecutor(max_workers=self.width,thread_name_prefix="sevc-lane")
        executor = self.executor
        try:
            pending, result = deque(), []
            iterator = iter(items)
            for _ in range(self.width):
                try:
                    pending.append(executor.submit(invoke,next(iterator)))
                except StopIteration:
                    break
            while pending:
                result.append(pending.popleft().result())
                try:
                    pending.append(executor.submit(invoke,next(iterator)))
                except StopIteration:
                    pass
        except BaseException:
            self.close()  # no detached futures survive failed work
            raise
        return result


def shared_wall_charges(intervals):
    """Share overlapping wall intervals equally among active tasks, exactly once."""
    edges = sorted({v for pair in intervals for v in pair})
    charges = [0.]*len(intervals)
    for start,end in zip(edges,edges[1:]):
        active = [i for i,(lo,hi) in enumerate(intervals) if lo <= start and hi >= end]
        for index in active:
            charges[index] += (end-start)/len(active)
    return charges
