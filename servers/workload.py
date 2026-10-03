"""
servers/workload.py
====================
Controlled workload simulation for a single backend Flask server.

IMPORTANT DESIGN NOTE (read this before touching the numbers below):
This module does **not** fabricate metrics with ``random.uniform(...)``
and pretend they are real measurements. Every metric this module reports
is a genuine, independently-observable measurement of what the Python
process is actually doing:

* ``cpu_usage``   - measured with ``psutil.Process().cpu_percent()``
  against a background thread that performs *real* busy-loop CPU work.
  If you run ``top``/``docker stats`` while ``cpu_level`` is high you
  will see this process' CPU usage rise for real.
* ``memory_usage`` - measured with ``psutil.Process().memory_info().rss``
  against real ``bytearray`` allocations that are actually held in the
  process' heap. Real RSS growth, not a random number.
* ``queue_length`` / ``active_requests`` - exact counts of requests that
  are really waiting on a ``threading.Semaphore`` (bounded processing
  capacity) or really being processed right now. No sampling, no noise.
* ``request_rate`` / ``throughput`` / ``error_rate`` - computed from a
  rolling window of real request timestamps and real success/failure
  outcomes.

The only thing that is "simulated" is *why* a request takes time or
fails: instead of talking to a real database or a real GPU, this
process burns real CPU cycles and holds real memory for a
controllable, explicit duration, and manufactures a bounded amount of
real queueing by only allowing ``capacity`` requests to be processed
concurrently. This is exactly the pattern used by load-testing tools
such as ``stress-ng`` or a synthetic ``/burn`` endpoint - a controlled
workload generator, not a fake metrics generator.

All parameters are adjustable at runtime through ``POST /load`` (see
``servers/app.py``), which is what the adaptive experiment
(``testing/adaptation_test.py``) and the DQN training environment
(``dqn/environment.py``) use to create genuinely different, genuinely
observable conditions per server (idle / CPU-pressured / memory-pressured
/ queue-buildup / failed / recovered) without ever telling the DQN which
server is which.
"""

from __future__ import annotations

import os
import random
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional

import psutil

_PROCESS = psutil.Process(os.getpid())
# Warm up psutil's internal cpu_percent baseline (first call always
# returns 0.0 by design).
_PROCESS.cpu_percent(interval=None)

# Approximate number of logical CPUs this process may burn across.
_CPU_COUNT = max(1, psutil.cpu_count(logical=True) or 1)


@dataclass
class WorkloadParams:
    """Explicit, controllable workload parameters for this server.

    cpu_level        : target CPU utilisation of the burner thread, 0-100 (%)
    memory_mb        : target resident memory held by the simulator (MB)
    base_latency_ms  : fixed processing latency added to every request (ms)
    jitter_ms        : uniform random jitter added on top of base latency (ms)
    capacity         : number of requests this server can process concurrently
                        before additional requests must genuinely queue
    error_rate       : baseline probability of a request being flagged as
                        an application-level failure, independent of load
    """

    cpu_level: float = 5.0
    memory_mb: float = 16.0
    base_latency_ms: float = 15.0
    jitter_ms: float = 10.0
    capacity: int = 12
    error_rate: float = 0.01

    def to_dict(self) -> dict:
        return {
            "cpu_level": self.cpu_level,
            "memory_mb": self.memory_mb,
            "base_latency_ms": self.base_latency_ms,
            "jitter_ms": self.jitter_ms,
            "capacity": self.capacity,
            "error_rate": self.error_rate,
        }


class WorkloadSimulator:
    """Owns the background CPU/memory burners plus the real request
    accounting (queue, active, rate, throughput, errors) for one
    backend server process."""

    def __init__(self, server_id: str, memory_cap_mb: float = 512.0,
                 window_seconds: float = 10.0):
        self.server_id = server_id
        self.memory_cap_mb = memory_cap_mb
        self.window_seconds = window_seconds

        self.params = WorkloadParams()
        self._lock = threading.RLock()

        # Manual failure switch (used by the soft-failure / recovery
        # phases of the adaptive experiment, distinct from actually
        # stopping the Docker container).
        self._manually_failed = False

        # Real memory the simulator is holding.
        self._memory_block: Optional[bytearray] = None
        self._apply_memory_locked()

        # Bounded concurrency -> real queueing.
        self._capacity_semaphore = threading.Semaphore(self.params.capacity)
        self._queue_length = 0
        self._active_requests = 0

        # Rolling windows of real events.
        self._arrivals: Deque[float] = deque()
        self._completions: Deque[tuple] = deque()  # (timestamp, success, latency_s)

        self._total_requests = 0
        self._total_success = 0
        self._total_failures = 0

        self._stop_event = threading.Event()
        self._cpu_thread = threading.Thread(target=self._cpu_burner_loop, daemon=True)
        self._cpu_thread.start()

    # ------------------------------------------------------------------
    # Workload control
    # ------------------------------------------------------------------
    def set_params(self, **kwargs) -> WorkloadParams:
        with self._lock:
            for key, value in kwargs.items():
                if value is None or not hasattr(self.params, key):
                    continue
                setattr(self.params, key, type(getattr(self.params, key))(value))
            if "capacity" in kwargs and kwargs["capacity"] is not None:
                self._resize_capacity_locked(int(kwargs["capacity"]))
            if "memory_mb" in kwargs and kwargs["memory_mb"] is not None:
                self._apply_memory_locked()
            return self.params

    def get_params(self) -> dict:
        with self._lock:
            return self.params.to_dict()

    def fail(self) -> None:
        with self._lock:
            self._manually_failed = True

    def recover(self) -> None:
        with self._lock:
            self._manually_failed = False

    def is_manually_failed(self) -> bool:
        with self._lock:
            return self._manually_failed

    def _resize_capacity_locked(self, new_capacity: int) -> None:
        new_capacity = max(1, new_capacity)
        self._capacity_semaphore = threading.Semaphore(new_capacity)

    def _apply_memory_locked(self) -> None:
        target_mb = max(0.0, min(self.params.memory_mb, self.memory_cap_mb))
        n_bytes = int(target_mb * 1024 * 1024)
        # Touch every page so RSS actually reflects the allocation
        # (a bytearray of zeros can otherwise be backed by a single
        # shared zero page on some platforms).
        block = bytearray(n_bytes)
        step = 4096
        for i in range(0, n_bytes, step):
            block[i] = 1
        self._memory_block = block

    def shutdown(self) -> None:
        self._stop_event.set()

    # ------------------------------------------------------------------
    # Background CPU burner: real, measurable CPU consumption
    # ------------------------------------------------------------------
    def _cpu_burner_loop(self) -> None:
        slice_seconds = 0.1
        while not self._stop_event.is_set():
            with self._lock:
                level = max(0.0, min(100.0, self.params.cpu_level))
            busy_time = slice_seconds * (level / 100.0)
            idle_time = slice_seconds - busy_time

            if busy_time > 0:
                end = time.perf_counter() + busy_time
                x = 0.0001
                while time.perf_counter() < end:
                    # Real floating point work - not a sleep, not a
                    # fabricated number. This is what actually raises
                    # psutil's cpu_percent() reading for this process.
                    x = (x * 1.0000001 + 1.0) % 1e6
            if idle_time > 0:
                time.sleep(idle_time)

    # ------------------------------------------------------------------
    # Request lifecycle - called by the Flask route handler
    # ------------------------------------------------------------------
    def handle_request(self):
        """Process one real request end-to-end and return
        (success: bool, latency_seconds: float, queued_seconds: float)."""
        now = time.time()
        with self._lock:
            self._arrivals.append(now)
            self._queue_length += 1
            manually_failed = self._manually_failed
            capacity = self.params.capacity
            base_latency_ms = self.params.base_latency_ms
            jitter_ms = self.params.jitter_ms
            error_rate = self.params.error_rate

        queue_start = time.perf_counter()
        acquired = self._capacity_semaphore.acquire(timeout=5.0)
        queued_seconds = time.perf_counter() - queue_start

        with self._lock:
            self._queue_length = max(0, self._queue_length - 1)
            self._active_requests += 1

        start = time.perf_counter()
        try:
            if manually_failed:
                # A "crashed" server still answers (so the balancer can
                # observe *how* it is unhealthy) but almost always fails.
                time.sleep(0.01)
                success = random.random() > 0.9
            elif not acquired:
                # Genuinely could not get a processing slot in time.
                success = False
            else:
                # Real processing delay: fixed cost + jitter + a cost
                # that grows with the *current* real queue pressure,
                # mirroring how a genuinely overloaded process slows down.
                pressure_penalty_ms = min(400.0, self._queue_length * 8.0)
                delay_ms = max(0.0, base_latency_ms + random.uniform(0, jitter_ms) + pressure_penalty_ms)
                time.sleep(delay_ms / 1000.0)

                overload_bonus = 0.0
                if capacity > 0 and self._active_requests > capacity:
                    overload_bonus = 0.15
                fail_probability = min(0.95, error_rate + overload_bonus)
                success = random.random() > fail_probability

            latency = time.perf_counter() - start
        finally:
            if acquired:
                self._capacity_semaphore.release()
            with self._lock:
                self._active_requests = max(0, self._active_requests - 1)
                self._total_requests += 1
                if success:
                    self._total_success += 1
                else:
                    self._total_failures += 1
                self._completions.append((time.time(), success, latency))
                self._trim_windows_locked()

        return success, latency, queued_seconds

    def _trim_windows_locked(self) -> None:
        cutoff = time.time() - self.window_seconds
        while self._arrivals and self._arrivals[0] < cutoff:
            self._arrivals.popleft()
        while self._completions and self._completions[0][0] < cutoff:
            self._completions.popleft()

    # ------------------------------------------------------------------
    # Metrics snapshot
    # ------------------------------------------------------------------
    def snapshot(self) -> dict:
        with self._lock:
            self._trim_windows_locked()
            window = max(1e-6, self.window_seconds)
            request_rate = len(self._arrivals) / window
            recent_completions = list(self._completions)
            manually_failed = self._manually_failed
            queue_length = self._queue_length
            active_requests = self._active_requests
            capacity = self.params.capacity
            total_requests = self._total_requests
            total_success = self._total_success
            total_failures = self._total_failures

        successes = [c for c in recent_completions if c[1]]
        failures = [c for c in recent_completions if not c[1]]
        throughput = len(successes) / window
        error_rate = (len(failures) / len(recent_completions)) if recent_completions else 0.0
        avg_latency_s = (
            sum(c[2] for c in recent_completions) / len(recent_completions)
            if recent_completions else 0.0
        )

        # cpu_percent() from psutil for a *process* can exceed 100 on a
        # multi-core box if several threads burn CPU concurrently; since
        # the container is expected to run with a single-core budget
        # (see docker-compose.yml `deploy.resources` / cpuset in
        # production setups), we clip to 100 so the normalised state
        # feature stays in a sane [0, 1] range regardless of the host's
        # core count.
        cpu_usage = min(100.0, _PROCESS.cpu_percent(interval=None))
        mem_rss_mb = _PROCESS.memory_info().rss / (1024 * 1024)
        memory_usage = min(100.0, (mem_rss_mb / self.memory_cap_mb) * 100.0)

        healthy = (not manually_failed) and error_rate < 0.5

        queue_ratio = min(1.0, queue_length / max(1, capacity * 2))
        active_ratio = min(1.0, active_requests / max(1, capacity * 2))
        load_pressure = (
            0.35 * (cpu_usage / 100.0)
            + 0.20 * (memory_usage / 100.0)
            + 0.25 * queue_ratio
            + 0.20 * active_ratio
        )
        load_pressure = max(0.0, min(1.0, load_pressure))

        return {
            "server_id": self.server_id,
            "cpu_usage": round(cpu_usage, 2),
            "memory_usage": round(memory_usage, 2),
            "memory_rss_mb": round(mem_rss_mb, 2),
            "queue_length": queue_length,
            "active_requests": active_requests,
            "capacity": capacity,
            "request_rate": round(request_rate, 3),
            "throughput": round(throughput, 3),
            "average_response_time": round(avg_latency_s, 4),
            "average_response_time_ms": round(avg_latency_s * 1000.0, 2),
            "error_rate": round(error_rate, 4),
            "health": bool(healthy),
            "manually_failed": bool(manually_failed),
            "load_pressure": round(load_pressure, 4),
            "total_requests": total_requests,
            "total_success": total_success,
            "total_failures": total_failures,
        }
