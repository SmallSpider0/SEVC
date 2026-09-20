"""Opt-in process counters and unit sampling; no experiment or outcome policy."""
from contextlib import nullcontext
from pathlib import Path
import json
import resource
import sys
import threading
import time


IO_KEYS = ('read_bytes', 'write_bytes', 'rchar', 'wchar')


def parse_proc_io(text):
    """Strict parse; None if the snapshot is torn (a counter grew between read syscalls)."""
    values = {}
    for line in text.splitlines():
        key, sep, value = line.partition(':')
        if not sep or not value.strip().isdigit():
            return None
        values[key.strip()] = int(value)
    return {k: values[k] for k in IO_KEYS} if all(k in values for k in IO_KEYS) else None


def process_io(attempts=5):
    """One read syscall per attempt: a buffered second read of this seq file can append a
    fragment of regenerated, longer text (e.g. "0") when counters change during the read."""
    import os
    path = '/proc/self/io'
    if not os.path.exists(path):
        return None
    for _ in range(attempts):
        fd = os.open(path, os.O_RDONLY)
        try:
            parsed = parse_proc_io(os.read(fd, 65536).decode())
        finally:
            os.close(fd)
        if parsed is not None:
            return parsed
    raise RuntimeError('unreadable /proc/self/io after repeated single-syscall reads')


def rss_bytes():
    try:
        import os
        return int(Path('/proc/self/statm').read_text().split()[1]) * os.sysconf('SC_PAGE_SIZE')
    except (OSError, IndexError):
        try:
            import psutil
            return psutil.Process().memory_info().rss
        except ImportError:
            return None


def lifetime_rss():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)


def phase_start():
    return {'cpu': time.process_time(), 'io': process_io(), 'rss': rss_bytes()}


def phase_end(start, event, parent, cuda):
    cpu = max(0., time.process_time() - start['cpu'])
    end_io = process_io()
    io = ({k: max(0, end_io[k] - start['io'][k]) for k in end_io}
          if end_io is not None and start['io'] is not None else None)
    child_io = event.get('child_io', {})
    result = {'resource_accounting': 'absolute-v1', 'cpu_process_seconds': cpu,
              'exclusive_cpu_process_seconds': max(0., cpu - event.get('child_process_cpu', 0.)),
              'io_bytes': io, 'exclusive_io_bytes': None if io is None else
                  {k: max(0, v-child_io.get(k, 0)) for k,v in io.items()},
              'rss_start_bytes': start['rss'], 'rss_end_bytes': rss_bytes(),
              'process_lifetime_peak_rss_bytes_upper_bound': lifetime_rss(),
              'cuda_allocated_bytes_at_phase_end': None,
              'cuda_unit_peak_allocated_bytes_so_far': None}
    if cuda:
        import torch
        result.update(cuda_allocated_bytes_at_phase_end=torch.cuda.memory_allocated(),
                      cuda_unit_peak_allocated_bytes_so_far=torch.cuda.max_memory_allocated())
    if parent is not None:
        parent['child_process_cpu'] = parent.get('child_process_cpu', 0.) + cpu
        totals = parent.setdefault('child_io', {})
        for k,v in (io or {}).items():
            totals[k] = totals.get(k, 0) + v
    return result


class UnitResources:
    """Sampled RSS is a lower bound on true peak; CUDA allocator peak is reset per unit."""
    def __init__(self, clock, folder, uid, *, cuda, profile=False, interval=.01):
        self.clock, self.folder, self.uid = clock, Path(folder), uid
        self.cuda, self.profile, self.interval = cuda, profile, interval
        self.stop = threading.Event()
        self.peak, self.samples = 0, 0
        self.profiler = None
        self.result = None

    def sample(self):
        value = rss_bytes()
        if value is not None:
            self.peak = max(self.peak, value)
            self.samples += 1

    def poll(self):
        while not self.stop.wait(self.interval):
            self.sample()

    def __enter__(self):
        self.sample()
        self.thread = threading.Thread(target=self.poll, daemon=True)
        if self.cuda:
            import torch
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        if self.profile:
            import torch
            if not self.cuda:
                raise ValueError('formal kernel profiling requires CUDA')
            self.profiler = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                              torch.profiler.ProfilerActivity.CUDA])
            self.profiler.__enter__()
            self.clock.profiling = True
        self.thread.start()
        return self

    def finish(self):
        if self.result is not None:
            return self.result
        self.stop.set(); self.thread.join(); self.sample()
        self.result = {'rss_sampled_unit_peak_bytes': self.peak if self.samples else None,
                       'rss_samples': self.samples, 'rss_sample_interval_seconds': self.interval,
                       'rss_peak_semantics': 'sampled lower bound; process includes shared inputs',
                       'process_lifetime_peak_rss_bytes_upper_bound': lifetime_rss(),
                       'cuda_unit_peak_allocated_bytes': None, 'profiler_sample': self.profile}
        if self.cuda:
            import torch
            torch.cuda.synchronize()
            self.result['cuda_unit_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        if self.profiler is not None:
            self.clock.profiling = False
            self.profiler.__exit__(None, None, None)
            self.folder.mkdir(exist_ok=True)
            trace = self.folder / (self.uid+'.trace.json')
            self.profiler.export_chrome_trace(str(trace))
            # Device self-time belongs to CPU events launching kernels; use nearest
            # registered ancestor and retain unattributed work explicitly.
            totals = {}
            for event in self.profiler.events():
                if 'CPU' not in str(event.device_type):
                    continue
                seconds = float(event.self_device_time_total) / 1e6
                if seconds <= 0:
                    continue
                parent = event
                while parent is not None and not parent.name.startswith('sevc|'):
                    parent = parent.cpu_parent
                key = parent.name if parent else 'unattributed'
                totals[key] = totals.get(key, 0.) + seconds
            self.result['kernel_self_seconds_by_role_phase'] = totals
            self.result['kernel_measurement'] = 'CUDA profiler sampled kernel self time; separate units only'
            self.result['trace_file'] = str(trace.name)
        return self.result

    def __exit__(self, exc_type, exc, tb):
        self.finish()


def instrumentation_probe(folder, *, cuda=False, iterations=1000):
    """Engineering-only no-op overhead and a tiny CUDA profiler capability check."""
    import statistics
    from sevc.core.role_accounting import RoleClock
    from sevc.core.artifacts import write_json
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=False)
    timing = {False: [], True: []}
    for repetition in range(5):
        for enabled in ((False, True) if repetition % 2 == 0 else (True, False)):
            clock = RoleClock(lambda:None, lambda row:None,
                              resource_accounting='absolute-v1' if enabled else None)
            start = time.perf_counter()
            for _ in range(iterations):
                clock.call('noop', 'engineering', lambda:None)
            timing[enabled].append((time.perf_counter()-start)/iterations)
    off, on = statistics.median(timing[False]), statistics.median(timing[True])
    result = {'scope':'non-scientific no-op instrumentation benchmark', 'iterations_per_repeat':iterations,
              'repetitions':5, 'seconds_per_call_default':off, 'seconds_per_call_absolute':on,
              'incremental_seconds_per_call':on-off, 'raw_seconds_per_call':{str(k):v for k,v in timing.items()},
              'cuda_requested':cuda,'formal_cells_measured':0,
              'limitations':'CPU no-op benchmark; not a correction factor for real jobs; profiler cells isolated'}
    if cuda:
        import torch
        torch.cuda.synchronize()
        log=[];clock=RoleClock(torch.cuda.synchronize,log.append,cuda=True,resource_accounting='absolute-v1')
        with UnitResources(clock,folder,'engineering-profiler',cuda=True,profile=True) as unit:
            a=torch.ones((128,128),device='cuda')
            clock.call('engineering-matmul','engineering',lambda: a@a)
            resources=unit.finish()
        if sum(resources.get('kernel_self_seconds_by_role_phase',{}).values())<=0:
            raise RuntimeError('CUDA profiler returned no kernel timing')
        result.update(cuda_profiler=resources,cuda_phase_records=log)
    write_json(folder/'instrumentation.json',result)
    return result
