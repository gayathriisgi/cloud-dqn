#!/usr/bin/env python3
"""
load_balancer/aco_balancer.py
===============================
Static Ant Colony Optimization-inspired baseline. Servers are chosen
probabilistically, weighted by a pheromone value per server; pheromone
evaporates every request and is reinforced on the server that was just
used (more reinforcement when that server currently has few active
connections). This is a fixed swarm-intelligence heuristic - it adapts
its *probabilities* mechanically from connection counts, but it never
learns a value function from reward, so it remains a static baseline
(not RL) for comparison purposes.
"""

from __future__ import annotations

import os
import random
import sys
import threading

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from flask import Flask, jsonify  # noqa: E402
from prometheus_client import Gauge, start_http_server  # noqa: E402

from common.settings import get_settings  # noqa: E402
from load_balancer.common import (ActiveConnectionTracker, MetricsCollector,  # noqa: E402
                                   StaticBalancerMetrics, build_response,
                                   forward_request, server_name)

SETTINGS = get_settings()
ALGORITHM = "aco"
EVAPORATION_RATE = float(os.environ.get("ACO_EVAPORATION_RATE", 0.1))
# Max-Min Ant System-style pheromone floor/ceiling. Without a floor, a
# server that loses one early race to zero-active-connections can be
# evaporated toward zero pheromone and then mathematically can never
# win again (a small weight only ever shrinks further under repeated
# evaporation) - the swarm collapses onto a single server, which is a
# known instability of naive ACO, not "learning" anything. Clamping
# keeps the intended STATIC, connection-aware-but-probabilistic
# behaviour instead of a degenerate monopoly.
PHEROMONE_MIN = float(os.environ.get("ACO_PHEROMONE_MIN", 0.2))
PHEROMONE_MAX = float(os.environ.get("ACO_PHEROMONE_MAX", 5.0))

app = Flask(__name__)
collector = MetricsCollector()
prom = StaticBalancerMetrics(ALGORITHM)
tracker = ActiveConnectionTracker(len(SETTINGS.server_urls))

_lock = threading.Lock()
_pheromone = [1.0 for _ in SETTINGS.server_urls]

PHEROMONE_GAUGE = Gauge("aco_pheromone", "Current pheromone level per server", ["server"])


def select_index() -> int:
    with _lock:
        total = sum(_pheromone)
        weights = [p / total for p in _pheromone]
    return random.choices(range(len(SETTINGS.server_urls)), weights=weights, k=1)[0]


def update_pheromone(index: int) -> None:
    with _lock:
        for i in range(len(_pheromone)):
            _pheromone[i] *= (1 - EVAPORATION_RATE)
        active = tracker.get(index)
        _pheromone[index] += 1.0 / (active + 1)
        for i in range(len(_pheromone)):
            _pheromone[i] = max(PHEROMONE_MIN, min(PHEROMONE_MAX, _pheromone[i]))
        for i, p in enumerate(_pheromone):
            PHEROMONE_GAUGE.labels(server=server_name(i)).set(p)


@app.route("/")
def route_request():
    before_metrics = collector.fetch_all()
    index = select_index()
    url = SETTINGS.server_urls[index]

    tracker.increment(index)
    try:
        success, response_time, _, _ = forward_request(url)
    finally:
        tracker.decrement(index)
        update_pheromone(index)

    after_metrics = collector.fetch_all()
    response = build_response(ALGORITHM, index, success, response_time, before_metrics, after_metrics)
    prom.record(index, success, response_time, response["reward"], active_connections=tracker.snapshot())
    with _lock:
        response["pheromone"] = {server_name(i): round(p, 4) for i, p in enumerate(_pheromone)}
    return jsonify(response), (200 if success else 502)


@app.route("/status")
def status():
    with _lock:
        pher = list(_pheromone)
    return jsonify({
        "algorithm": ALGORITHM,
        "pheromone": {server_name(i): round(p, 4) for i, p in enumerate(pher)},
        "active_connections": {server_name(i): c for i, c in enumerate(tracker.snapshot())},
        "port": SETTINGS.aco_port,
        "metrics_port": SETTINGS.aco_metrics_port,
    })


@app.route("/health")
def health():
    return jsonify({"status": "healthy", "algorithm": ALGORITHM})


if __name__ == "__main__":
    start_http_server(SETTINGS.aco_metrics_port)
    print(f"[{ALGORITHM}] Load balancer running on http://0.0.0.0:{SETTINGS.aco_port}")
    print(f"[{ALGORITHM}] Prometheus metrics on http://0.0.0.0:{SETTINGS.aco_metrics_port}/metrics")
    app.run(host="0.0.0.0", port=SETTINGS.aco_port, threaded=True)
