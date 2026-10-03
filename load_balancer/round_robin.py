#!/usr/bin/env python3
"""
load_balancer/round_robin.py
=============================
Static Round Robin baseline. Fixed, stateless decision rule - always
picks the next server in a cycle, completely independent of load. This
is intentionally NOT reinforcement learning; it is one of the three
static baselines the DQN is compared against (Section 7 / Section 9).
"""

from __future__ import annotations

import itertools
import os
import sys
import threading

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from flask import Flask, jsonify  # noqa: E402
from prometheus_client import start_http_server  # noqa: E402

from common.settings import get_settings  # noqa: E402
from load_balancer.common import (MetricsCollector, StaticBalancerMetrics,  # noqa: E402
                                   build_response, forward_request, server_name)

SETTINGS = get_settings()
ALGORITHM = "round_robin"

app = Flask(__name__)
collector = MetricsCollector()
prom = StaticBalancerMetrics(ALGORITHM)

_lock = threading.Lock()
_cycle = itertools.cycle(range(len(SETTINGS.server_urls)))


def select_index() -> int:
    with _lock:
        return next(_cycle)


@app.route("/")
def route_request():
    before_metrics = collector.fetch_all()
    index = select_index()
    url = SETTINGS.server_urls[index]

    success, response_time, _, _ = forward_request(url)
    after_metrics = collector.fetch_all()

    response = build_response(ALGORITHM, index, success, response_time, before_metrics, after_metrics)
    prom.record(index, success, response_time, response["reward"])
    return jsonify(response), (200 if success else 502)


@app.route("/status")
def status():
    return jsonify({
        "algorithm": ALGORITHM,
        "servers": SETTINGS.server_urls,
        "port": SETTINGS.round_robin_port,
        "metrics_port": SETTINGS.round_robin_metrics_port,
    })


@app.route("/health")
def health():
    return jsonify({"status": "healthy", "algorithm": ALGORITHM})


if __name__ == "__main__":
    start_http_server(SETTINGS.round_robin_metrics_port)
    print(f"[{ALGORITHM}] Load balancer running on http://0.0.0.0:{SETTINGS.round_robin_port}")
    print(f"[{ALGORITHM}] Prometheus metrics on http://0.0.0.0:{SETTINGS.round_robin_metrics_port}/metrics")
    app.run(host="0.0.0.0", port=SETTINGS.round_robin_port, threaded=True)
