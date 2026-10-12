import time
from collections import defaultdict
from contextlib import contextmanager

import torch


class TimingRecorder:
    """Accumulate nested CPU or CUDA phase timings without per-phase syncs."""

    def __init__(self, *, enabled, device):
        self.enabled = enabled
        self.device = torch.device(device)
        self._active = {}
        self._cpu_elapsed = defaultdict(float)
        self._cuda_events = defaultdict(list)

    def start(self, name):
        if not self.enabled:
            return
        if name in self._active:
            raise RuntimeError(f"Timing phase already active: {name}")

        if self.device.type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            start.record(torch.cuda.current_stream(self.device))
        else:
            start = time.perf_counter()
        self._active[name] = start

    def stop(self, name):
        if not self.enabled:
            return
        if name not in self._active:
            raise RuntimeError(f"Timing phase is not active: {name}")

        start = self._active.pop(name)
        if self.device.type == "cuda":
            end = torch.cuda.Event(enable_timing=True)
            end.record(torch.cuda.current_stream(self.device))
            self._cuda_events[name].append((start, end))
        else:
            self._cpu_elapsed[name] += time.perf_counter() - start

    def switch(self, current_name, next_name):
        """End one phase and start the next with a shared timestamp."""
        if not self.enabled:
            return
        if current_name not in self._active:
            raise RuntimeError(f"Timing phase is not active: {current_name}")
        if next_name in self._active:
            raise RuntimeError(f"Timing phase already active: {next_name}")

        start = self._active.pop(current_name)
        if self.device.type == "cuda":
            boundary = torch.cuda.Event(enable_timing=True)
            boundary.record(torch.cuda.current_stream(self.device))
            self._cuda_events[current_name].append((start, boundary))
        else:
            boundary = time.perf_counter()
            self._cpu_elapsed[current_name] += boundary - start
        self._active[next_name] = boundary

    @contextmanager
    def record(self, name, *, replacing=None):
        """Record a phase, optionally pausing and then resuming another."""
        if replacing is None:
            self.start(name)
        else:
            self.switch(replacing, name)
        try:
            yield
        finally:
            if replacing is None:
                self.stop(name)
            else:
                self.switch(name, replacing)

    def elapsed(self):
        if not self.enabled:
            return {}
        if self._active:
            active = ", ".join(sorted(self._active))
            raise RuntimeError(f"Cannot read timings with active phases: {active}")

        elapsed = dict(self._cpu_elapsed)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            for name, event_pairs in self._cuda_events.items():
                elapsed[name] = sum(
                    start.elapsed_time(end) for start, end in event_pairs
                ) / 1000.0
        return elapsed


def reduce_max_timings(timings, device):
    """Return slowest-rank timings, which determine distributed wall time."""
    if not timings:
        return {}

    names = sorted(timings)
    values = torch.tensor(
        [timings[name] for name in names], dtype=torch.float64, device=device
    )
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(values, op=torch.distributed.ReduceOp.MAX)
    return dict(zip(names, values.cpu().tolist()))
