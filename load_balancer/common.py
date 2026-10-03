"""
load_balancer/common.py
========================
Shared helpers used by all four balancer Flask services: metric
collection from the real backend servers, real request forwarding
with measured latency, and a thread-safe active-connection tracker for
Least Connections / ACO.

IMPORTANT: servers are always identified by their **index / logical
name** (``server1`` .. ``server4``), never by parsing a port number out
of the URL. Inside Docker every backend listens on the same internal
port (5000) and is only distinguished by its service DNS name
(``http://server1:5000`` .. ``http://server4:5000``); the original
project's `server.split(":")[-1]` port-parsing trick only worked for
host-side URLs like ``http://localhost:5001`` and would silently
mislabel every server the moment it ran inside docker-compose. Every
balancer in this project uses index-based naming instead so metrics and
logs are correct in both environments.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple

import requests
from prometheus_client import Counter, Gauge, Histogram

from common.settings import get_settings
from dqn.reward import compute_reward  # pure-python, no torch dependency

SETTINGS = get_settings()


def server_name(index: int) -> str:
    if index < len(SETTINGS.server_names):
        return SETTINGS.server_names[index]
    return f"server{index + 1}"


class MetricsCollector:
    """Fetches `/metrics` from every backend server in parallel."""

    def __init__(self, server_urls: Optional[List[str]] = None, timeout: Optional[float] = None):
        self.server_urls = list(server_urls) if server_urls else list(SETTINGS.server_urls)
        self.timeout = timeout or SETTINGS.metrics_timeout
        self._executor = ThreadPoolExecutor(max_workers=max(4, len(self.server_urls)))

    def _fetch_one(self, url: str) -> Optional[Dict]:
        try:
            r = requests.get(f"{url}/metrics", timeout=self.timeout)
            if r.status_code == 200:
                return r.json()
        except requests.RequestException:
            pass
        return None

    def fetch_all(self) -> List[Optional[Dict]]:
        return list(self._executor.map(self._fetch_one, self.server_urls))

    def health_all(self) -> List[bool]:
        def _check(url: str) -> bool:
            try:
                r = requests.get(f"{url}/health", timeout=self.timeout)
                return r.status_code == 200
            except requests.RequestException:
                return False
        return list(self._executor.map(_check, self.server_urls))


class ActiveConnectionTracker:
    """Thread-safe per-server in-flight request counter, used by Least
    Connections and (for pheromone reinforcement) ACO."""

    def __init__(self, n_servers: int):
        self._lock = threading.Lock()
        self._counts = [0] * n_servers

    def increment(self, index: int) -> None:
        with self._lock:
            self._counts[index] += 1

    def decrement(self, index: int) -> None:
        with self._lock:
            self._counts[index] = max(0, self._counts[index] - 1)

    def snapshot(self) -> List[int]:
        with self._lock:
            return list(self._counts)

    def get(self, index: int) -> int:
        with self._lock:
            return self._counts[index]


def forward_request(url: str, timeout: Optional[float] = None) -> Tuple[bool, float, Optional[dict], int]:
    """Forward one real request to a backend server and measure the
    real wall-clock response time. Returns
    (success, response_time_seconds, response_json_or_none, status_code).
    """
    t = timeout or SETTINGS.request_timeout
    start = time.perf_counter()
    try:
        r = requests.get(url, timeout=t)
        elapsed = time.perf_counter() - start
        try:
            body = r.json()
        except ValueError:
            body = None
        return (r.status_code == 200), elapsed, body, r.status_code
    except requests.RequestException:
        elapsed = time.perf_counter() - start
        return False, elapsed, None, 0


def metrics_by_index(all_metrics: List[Optional[Dict]], index: int) -> Dict:
    m = all_metrics[index]
    if not m:
        return {
            "server_id": server_name(index), "cpu_usage": 100.0, "memory_usage": 100.0,
            "queue_length": SETTINGS.queue_norm_ref, "active_requests": SETTINGS.active_norm_ref,
            "average_response_time_ms": SETTINGS.latency_norm_ref_ms, "request_rate": 0.0,
            "throughput": 0.0, "error_rate": 1.0, "health": False, "load_pressure": 1.0,
        }
    return m


def build_response(
    algorithm: str,
    index: int,
    success: bool,
    response_time: float,
    all_metrics_before: List[Optional[Dict]],
    all_metrics_after: Optional[List[Optional[Dict]]] = None,
    q_values: Optional[List[float]] = None,
    decision_time: Optional[float] = None,
    extra: Optional[Dict] = None,
) -> Dict:
    """Uniform response JSON shared by all four balancers, so
    ``testing/compare_algorithms.py`` can evaluate every algorithm the
    same way. ``reward`` is computed identically for every algorithm
    purely as an evaluation/comparison signal - only the DQN balancer
    actually stores it and learns from it.
    """
    after = all_metrics_after if all_metrics_after is not None else all_metrics_before
    chosen_after = metrics_by_index(after, index)
    breakdown = compute_reward(
        success=success,
        response_time_s=response_time,
        chosen_server_metrics=chosen_after,
        all_server_metrics=after,
    )

    server_metrics_out = {
        server_name(i): metrics_by_index(after, i) for i in range(len(after))
    }

    response = {
        "algorithm": algorithm,
        "selected_server": server_name(index),
        "selected_action": index,
        "success": success,
        "response_time": round(response_time, 4),
        "response_time_ms": round(response_time * 1000.0, 2),
        "reward": round(breakdown.total, 4),
        "reward_breakdown": breakdown.as_dict(),
        "server_metrics": server_metrics_out,
        # Additive field (v3): the metrics the decision was BASED ON
        # (collected before routing). Used by the MyCamu UI to show
        # exactly which server states the algorithm analysed.
        "server_metrics_before": {
            server_name(i): metrics_by_index(all_metrics_before, i) for i in range(len(all_metrics_before))
        },
    }
    if q_values is not None:
        response["q_values"] = {server_name(i): round(float(q), 4) for i, q in enumerate(q_values)}
    if decision_time is not None:
        response["decision_time_ms"] = round(decision_time * 1000.0, 3)
    if extra:
        response.update(extra)
    return response


class StaticBalancerMetrics:
    """Standard Prometheus instrumentation shared by the three static
    balancers (Round Robin / Least Connections / ACO). Each balancer
    runs in its own process on its own metrics port, so there is no
    registry collision between them; the ``algorithm`` label lets
    Grafana/PromQL compare them directly (e.g.
    ``sum by (algorithm) (rate(lb_requests_total{outcome="success"}[1m]))``).
    """

    def __init__(self, algorithm: str):
        self.algorithm = algorithm
        self.requests_total = Counter(
            "lb_requests_total", "Total requests handled by a static load balancer",
            ["algorithm", "server", "outcome"],
        )
        self.response_time = Histogram(
            "lb_response_time_seconds", "Response time of forwarded requests",
            ["algorithm"],
        )
        self.selected_server_total = Counter(
            "lb_selected_server_total", "Number of times each server was selected",
            ["algorithm", "server"],
        )
        self.reward_gauge = Gauge(
            "lb_reward", "Most recent per-request reward (evaluation-only; not used for learning)",
            ["algorithm"],
        )
        self.active_connections = Gauge(
            "lb_active_connections", "Current in-flight requests per server",
            ["algorithm", "server"],
        )

    def record(self, index: int, success: bool, response_time: float, reward: float,
               active_connections: Optional[List[int]] = None) -> None:
        name = server_name(index)
        outcome = "success" if success else "failure"
        self.requests_total.labels(algorithm=self.algorithm, server=name, outcome=outcome).inc()
        self.response_time.labels(algorithm=self.algorithm).observe(response_time)
        self.selected_server_total.labels(algorithm=self.algorithm, server=name).inc()
        self.reward_gauge.labels(algorithm=self.algorithm).set(reward)
        if active_connections is not None:
            for i, count in enumerate(active_connections):
                self.active_connections.labels(algorithm=self.algorithm, server=server_name(i)).set(count)
