"""
dqn/environment.py
===================
The "genuine dynamic environment" (Section 4 of the spec).

This environment does **not** simulate the backend servers inside
Python with random numbers - it drives real HTTP requests against the
real Flask backend servers (``servers/app.py``) and reads their real
``/metrics``. What *is* controlled/scripted here is the workload each
backend server is asked to simulate (via ``POST /load``, ``POST /fail``,
``POST /recover`` - see ``servers/workload.py``), which is exactly the
"controlled workload simulation" the spec asks for instead of pretending
random numbers are real measurements.

``SCENARIOS`` gives every backend server one of several workload
presets (idle -> overloaded). ``apply_phase`` can give *different*
servers different presets at the same time, which is what lets the
five-phase adaptive experiment in ``testing/adaptation_test.py``
overload server1, then overload server3, then recover server1, etc.,
while the DQN (or a static baseline) is actively routing real traffic
against it.
"""

from __future__ import annotations

import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Sequence

import numpy as np
import requests

from common.settings import get_settings
from dqn.reward import RewardBreakdown, compute_reward
from dqn.state import build_state

SETTINGS = get_settings()

# Named workload presets applied to a backend server via POST /load.
# These numbers were chosen so that "low" keeps a server almost idle
# and "overload" reliably pushes cpu/queue/latency/error metrics close
# to their normalization ceilings defined in common/settings.py.
SCENARIOS: Dict[str, Dict] = {
    "idle": {"cpu_level": 3, "memory_mb": 12, "base_latency_ms": 8, "jitter_ms": 4, "capacity": 20, "error_rate": 0.005},
    "low": {"cpu_level": 8, "memory_mb": 20, "base_latency_ms": 12, "jitter_ms": 6, "capacity": 18, "error_rate": 0.01},
    "medium": {"cpu_level": 30, "memory_mb": 70, "base_latency_ms": 35, "jitter_ms": 15, "capacity": 12, "error_rate": 0.02},
    "high": {"cpu_level": 55, "memory_mb": 150, "base_latency_ms": 65, "jitter_ms": 25, "capacity": 8, "error_rate": 0.04},
    "cpu_pressure": {"cpu_level": 85, "memory_mb": 60, "base_latency_ms": 40, "jitter_ms": 20, "capacity": 10, "error_rate": 0.03},
    "memory_pressure": {"cpu_level": 25, "memory_mb": 400, "base_latency_ms": 40, "jitter_ms": 20, "capacity": 10, "error_rate": 0.03},
    "queue_buildup": {"cpu_level": 30, "memory_mb": 60, "base_latency_ms": 30, "jitter_ms": 10, "capacity": 3, "error_rate": 0.02},
    "high_latency": {"cpu_level": 20, "memory_mb": 40, "base_latency_ms": 250, "jitter_ms": 100, "capacity": 10, "error_rate": 0.02},
    "overload": {"cpu_level": 92, "memory_mb": 420, "base_latency_ms": 180, "jitter_ms": 80, "capacity": 4, "error_rate": 0.15},
}


class LoadBalancerEnvironment:
    """Drives real traffic + real workload control against the real
    backend servers over HTTP. Used by both ``dqn/train.py`` and
    ``testing/adaptation_test.py``."""

    def __init__(self, server_urls: Optional[Sequence[str]] = None,
                 request_timeout: Optional[float] = None,
                 metrics_timeout: Optional[float] = None):
        self.server_urls = list(server_urls) if server_urls else list(SETTINGS.server_urls)
        self.n_servers = len(self.server_urls)
        self.request_timeout = request_timeout or SETTINGS.request_timeout
        self.metrics_timeout = metrics_timeout or SETTINGS.metrics_timeout
        self._executor = ThreadPoolExecutor(max_workers=max(4, self.n_servers))

    # ------------------------------------------------------------------
    # Metrics / state
    # ------------------------------------------------------------------
    def _fetch_metrics(self, url: str) -> Optional[Dict]:
        try:
            r = requests.get(f"{url}/metrics", timeout=self.metrics_timeout)
            if r.status_code == 200:
                return r.json()
        except requests.RequestException:
            pass
        return None

    def get_all_metrics(self) -> List[Optional[Dict]]:
        return list(self._executor.map(self._fetch_metrics, self.server_urls))

    def get_state(self):
        metrics = self.get_all_metrics()
        return build_state(metrics), metrics

    # ------------------------------------------------------------------
    # Acting in the environment
    # ------------------------------------------------------------------
    def step(self, action: int):
        """Send one real request to server[action], observe the real
        outcome, compute the next state and the multi-objective reward.

        Returns (next_state, reward_breakdown, info) where info contains
        success/response_time/selected server metrics for logging.
        """
        url = self.server_urls[action]
        start = time.perf_counter()
        success = False
        try:
            r = requests.get(url, timeout=self.request_timeout)
            success = r.status_code == 200
        except requests.RequestException:
            success = False
        response_time = time.perf_counter() - start

        next_metrics = self.get_all_metrics()
        next_state = build_state(next_metrics)
        chosen_metrics = next_metrics[action] or {}

        breakdown: RewardBreakdown = compute_reward(
            success=success,
            response_time_s=response_time,
            chosen_server_metrics=chosen_metrics,
            all_server_metrics=next_metrics,
        )

        info = {
            "success": success,
            "response_time": response_time,
            "server_metrics": next_metrics,
            "chosen_server_metrics": chosen_metrics,
        }
        return next_state, breakdown, info

    # ------------------------------------------------------------------
    # Controlled workload / scenario management
    # ------------------------------------------------------------------
    def set_workload(self, server_index: int, **params) -> None:
        url = self.server_urls[server_index]
        try:
            requests.post(f"{url}/load", json=params, timeout=self.request_timeout)
        except requests.RequestException:
            pass

    def apply_preset(self, server_index: int, preset_name: str,
                     randomize: float = 0.0, rng: Optional[random.Random] = None) -> Dict:
        """Apply a named workload preset.

        ``randomize`` > 0 enables DOMAIN RANDOMISATION: every numeric
        parameter of the preset is scaled by an independent factor in
        [1 - randomize, 1 + randomize]. Used only by dqn/train.py so the
        network learns the *shape* of "a degraded server" (high latency /
        queue / errors / cpu) instead of memorising the exact numbers of
        nine fixed presets - a major source of overfitting in the v2
        training run. Evaluation always uses the exact presets.
        """
        params = dict(SCENARIOS[preset_name])
        if randomize > 0:
            r = rng or random
            for k, v in list(params.items()):
                f = 1.0 + r.uniform(-randomize, randomize)
                if k == "capacity":
                    params[k] = max(1, int(round(v * f)))
                elif k == "error_rate":
                    params[k] = max(0.0, min(0.9, v * f))
                elif k == "cpu_level":
                    params[k] = max(0.0, min(100.0, v * f))
                else:
                    params[k] = max(0.0, v * f)
        self.set_workload(server_index, **params)
        return params

    def apply_phase(self, presets: Dict[int, str], randomize: float = 0.0,
                    rng: Optional[random.Random] = None) -> None:
        """presets: {server_index: preset_name}. Any server not listed
        keeps its current workload."""
        for idx, preset in presets.items():
            self.apply_preset(idx, preset, randomize=randomize, rng=rng)

    def apply_uniform(self, preset_name: str) -> None:
        for i in range(self.n_servers):
            self.apply_preset(i, preset_name)

    def fail_server(self, server_index: int) -> None:
        url = self.server_urls[server_index]
        try:
            requests.post(f"{url}/fail", timeout=self.request_timeout)
        except requests.RequestException:
            pass

    def recover_server(self, server_index: int) -> None:
        url = self.server_urls[server_index]
        try:
            requests.post(f"{url}/recover", timeout=self.request_timeout)
        except requests.RequestException:
            pass

    def reset_all(self) -> None:
        for url in self.server_urls:
            try:
                requests.post(f"{url}/reset", timeout=self.request_timeout)
            except requests.RequestException:
                pass

    def servers_reachable(self) -> List[bool]:
        def _check(url):
            try:
                r = requests.get(f"{url}/health", timeout=self.metrics_timeout)
                return r.status_code in (200, 503)
            except requests.RequestException:
                return False
        return list(self._executor.map(_check, self.server_urls))

    def close(self) -> None:
        self._executor.shutdown(wait=False)


class BackgroundTraffic:
    """Concurrent "other clients" traffic used during TRAINING.

    The v2 model was trained with exactly one request in flight at a time,
    so queue_length / active_requests were almost always 0 during
    training - but at evaluation time 10-40 req/s arrive concurrently.
    That train/test distribution shift is a form of overfitting to an idle
    system. This helper keeps a configurable, randomly varying amount of
    real concurrent load on the real backend servers while the agent
    learns, so the agent experiences genuine queueing and contention.

    Requests are spread uniformly at random over the servers (it is
    deliberately NOT a smart policy - it is just ambient load).
    """

    def __init__(self, server_urls: Sequence[str], timeout: float = 5.0, max_workers: int = 32):
        self.server_urls = list(server_urls)
        self.timeout = timeout
        self.rate = 0.0
        self._stop = threading.Event()
        self._pool = ThreadPoolExecutor(max_workers=max_workers)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def set_rate(self, rps: float) -> None:
        self.rate = max(0.0, float(rps))

    def _hit(self, url: str) -> None:
        try:
            requests.get(url, timeout=self.timeout)
        except requests.RequestException:
            pass

    def _loop(self) -> None:
        while not self._stop.is_set():
            if self.rate <= 0:
                time.sleep(0.05)
                continue
            self._pool.submit(self._hit, random.choice(self.server_urls))
            time.sleep(random.expovariate(self.rate))  # Poisson arrivals

    def close(self) -> None:
        self._stop.set()
        self._pool.shutdown(wait=False)
