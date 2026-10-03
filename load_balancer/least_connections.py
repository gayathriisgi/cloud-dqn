#!/usr/bin/env python3
"""
load_balancer/least_connections.py
====================================
Static Least Connections baseline. Picks the server with the fewest
currently in-flight requests (a live, but purely rule-based signal -
no learning, no memory of outcomes beyond the current connection
count).
"""

from __future__ import annotations

import os
import random
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from flask import Flask, jsonify  # noqa: E402
from prometheus_client import start_http_server  # noqa: E402

from common.settings import get_settings  # noqa: E402
from load_balancer.common import (ActiveConnectionTracker, MetricsCollector,  # noqa: E402
                                   StaticBalancerMetrics, build_response,
                                   forward_request, server_name)

SETTINGS = get_settings()
ALGORITHM = "least_connections"

app = Flask(__name__)
collector = MetricsCollector()
prom = StaticBalancerMetrics(ALGORITHM)
tracker = ActiveConnectionTracker(len(SETTINGS.server_urls))


def select_index() -> int:
    """Pick the server with the fewest in-flight connections. Ties
    (e.g. every server idle, which happens constantly with sub-100ms
    backend responses) are broken RANDOMLY among the tied servers
    rather than always favouring the lowest index - a deterministic
    tie-break would otherwise make "least connections" collapse into a
    hidden, unfair preference for server1 every time the system drains
    to idle between bursts. This is still a purely static, rule-based
    decision (no learning, no memory of past outcomes) - just a
    standard, herding-resistant implementation of the same rule.
    """
    counts = tracker.snapshot()
    minimum = min(counts)
    tied = [i for i, c in enumerate(counts) if c == minimum]
    return random.choice(tied)


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

    after_metrics = collector.fetch_all()
    response = build_response(ALGORITHM, index, success, response_time, before_metrics, after_metrics)
    prom.record(index, success, response_time, response["reward"], active_connections=tracker.snapshot())
    return jsonify(response), (200 if success else 502)


@app.route("/status")
def status():
    return jsonify({
        "algorithm": ALGORITHM,
        "active_connections": {server_name(i): c for i, c in enumerate(tracker.snapshot())},
        "port": SETTINGS.least_conn_port,
        "metrics_port": SETTINGS.least_conn_metrics_port,
    })


@app.route("/health")
def health():
    return jsonify({"status": "healthy", "algorithm": ALGORITHM})


if __name__ == "__main__":
    start_http_server(SETTINGS.least_conn_metrics_port)
    print(f"[{ALGORITHM}] Load balancer running on http://0.0.0.0:{SETTINGS.least_conn_port}")
    print(f"[{ALGORITHM}] Prometheus metrics on http://0.0.0.0:{SETTINGS.least_conn_metrics_port}/metrics")
    app.run(host="0.0.0.0", port=SETTINGS.least_conn_port, threaded=True)
