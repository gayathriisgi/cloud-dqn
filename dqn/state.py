"""
dqn/state.py
============
Builds the normalized multi-metric DQN state vector described in
``dqn/config.py``.

The state is constructed from the *real* JSON returned by each backend
server's ``/metrics`` endpoint (see ``servers/app.py`` /
``servers/workload.py``). If a server cannot be reached at all, it is
treated as fully overloaded/unhealthy (worst-case values) rather than
silently dropped from the state - a load balancer that cannot see a
server should treat it exactly as it would treat a maxed-out one, and
this also gives the network a clean, consistent input even during a
hard failure test (``docker compose stop server2``).

This module is imported by:
* ``dqn/environment.py``   (training)
* ``dqn/train.py``          (training)
* ``load_balancer/dqn_balancer.py``  (inference)
* ``testing/*``             (to assert state sensitivity)

so training and inference are structurally guaranteed to build the
state the same way.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from common.settings import get_settings
from dqn.config import FEATURES_PER_SERVER, GLOBAL_FEATURES, compute_state_dim

SETTINGS = get_settings()

STATE_DIM = compute_state_dim(SETTINGS.n_servers)

PER_SERVER_FEATURE_NAMES = [
    "cpu_usage_norm",
    "memory_usage_norm",
    "queue_length_norm",
    "active_requests_norm",
    "latency_norm",
    "request_rate_norm",
    "throughput_norm",
    "error_rate_norm",
    "health",
    "load_pressure",
]

GLOBAL_FEATURE_NAMES = [
    "total_request_rate_norm",
    "total_active_norm",
    "overall_latency_norm",
    "overall_throughput_norm",
    "global_error_rate",
    "traffic_intensity",
]

assert len(PER_SERVER_FEATURE_NAMES) == FEATURES_PER_SERVER
assert len(GLOBAL_FEATURE_NAMES) == GLOBAL_FEATURES


def feature_names(n_servers: int = SETTINGS.n_servers) -> List[str]:
    names = []
    for i in range(n_servers):
        names.extend(f"server{i + 1}_{f}" for f in PER_SERVER_FEATURE_NAMES)
    names.extend(GLOBAL_FEATURE_NAMES)
    return names


def _worst_case_metrics(server_name: str) -> Dict:
    """Metrics assumed for a server that could not be reached at all."""
    return {
        "server_id": server_name,
        "cpu_usage": 100.0,
        "memory_usage": 100.0,
        "queue_length": SETTINGS.queue_norm_ref,
        "active_requests": SETTINGS.active_norm_ref,
        "average_response_time_ms": SETTINGS.latency_norm_ref_ms,
        "request_rate": 0.0,
        "throughput": 0.0,
        "error_rate": 1.0,
        "health": False,
        "load_pressure": 1.0,
    }


def _normalize_server_metrics(m: Dict) -> List[float]:
    s = SETTINGS
    cpu_norm = s.clamp01(float(m.get("cpu_usage", 100.0)) / 100.0)
    mem_norm = s.clamp01(float(m.get("memory_usage", 100.0)) / 100.0)
    queue_norm = s.clamp01(float(m.get("queue_length", s.queue_norm_ref)) / s.queue_norm_ref)
    active_norm = s.clamp01(float(m.get("active_requests", s.active_norm_ref)) / s.active_norm_ref)
    latency_ms = m.get("average_response_time_ms")
    if latency_ms is None:
        latency_ms = float(m.get("average_response_time", 0.0)) * 1000.0
    latency_norm = s.clamp01(float(latency_ms) / s.latency_norm_ref_ms)
    rate_norm = s.clamp01(float(m.get("request_rate", 0.0)) / s.rate_norm_ref)
    throughput_norm = s.clamp01(float(m.get("throughput", 0.0)) / s.throughput_norm_ref)
    error_norm = s.clamp01(float(m.get("error_rate", 1.0)))
    health = 1.0 if m.get("health", False) else 0.0
    load_pressure = s.clamp01(float(m.get("load_pressure", 1.0)))

    return [
        cpu_norm, mem_norm, queue_norm, active_norm, latency_norm,
        rate_norm, throughput_norm, error_norm, health, load_pressure,
    ]


def build_state(server_metrics: Sequence[Dict]) -> np.ndarray:
    """Build the full normalized state vector from a list of per-server
    metrics dicts (in server order: server1, server2, server3, server4).

    ``server_metrics[i]`` may be ``None`` (or missing keys) if that
    server could not be reached - it is replaced with worst-case
    metrics rather than crashing or silently shrinking the state.
    """
    s = SETTINGS
    n = len(server_metrics)
    per_server_rows = []
    for i, m in enumerate(server_metrics):
        if not m:
            m = _worst_case_metrics(f"server{i + 1}")
        per_server_rows.append(_normalize_server_metrics(m))

    flat = [v for row in per_server_rows for v in row]

    request_rates = [float((server_metrics[i] or {}).get("request_rate", 0.0)) for i in range(n)]
    actives = [float((server_metrics[i] or {}).get("active_requests", s.active_norm_ref)) for i in range(n)]
    latencies_ms = []
    for i in range(n):
        m = server_metrics[i] or {}
        lm = m.get("average_response_time_ms")
        if lm is None:
            lm = float(m.get("average_response_time", 0.0)) * 1000.0
        latencies_ms.append(float(lm))
    throughputs = [float((server_metrics[i] or {}).get("throughput", 0.0)) for i in range(n)]
    error_norms = [row[7] for row in per_server_rows]

    total_request_rate_norm = s.clamp01(sum(request_rates) / (n * s.rate_norm_ref))
    total_active_norm = s.clamp01(sum(actives) / (n * s.active_norm_ref))
    overall_latency_norm = s.clamp01((sum(latencies_ms) / n) / s.latency_norm_ref_ms)
    overall_throughput_norm = s.clamp01(sum(throughputs) / (n * s.throughput_norm_ref))
    global_error_rate = s.clamp01(sum(error_norms) / n)
    traffic_intensity = s.clamp01(sum(actives) / (n * s.active_norm_ref))

    flat.extend([
        total_request_rate_norm,
        total_active_norm,
        overall_latency_norm,
        overall_throughput_norm,
        global_error_rate,
        traffic_intensity,
    ])

    state = np.array(flat, dtype=np.float32)
    assert state.shape[0] == STATE_DIM, (
        f"Built state has {state.shape[0]} dims, expected {STATE_DIM}. "
        "This should never happen - check FEATURES_PER_SERVER/GLOBAL_FEATURES."
    )
    return state


def load_pressure_series(server_metrics: Sequence[Dict]) -> List[float]:
    """Convenience helper used by the reward function to compute the
    system-wide load-imbalance penalty."""
    out = []
    for i, m in enumerate(server_metrics):
        if not m:
            m = _worst_case_metrics(f"server{i + 1}")
        out.append(float(m.get("load_pressure", 1.0)))
    return out
