"""
servers/app.py
===============
Upgraded backend Flask server.

Preserves the original architecture (a small Flask app per backend,
one process per container, exposing ``/`` and ``/metrics``) while
replacing the placeholder ``random.randint(10, 50)`` CPU number and the
missing memory/queue/latency/throughput/error/health metrics with real,
measured values produced by ``servers/workload.py``.

Endpoints
---------
GET  /            real request: waits for a processing slot (genuine
                   queueing), does real work, returns success/failure.
GET  /health      liveness/health probe used by Docker healthchecks and
                   by the load balancers to mark a server unavailable.
GET  /metrics     JSON metrics consumed by the DQN state builder and by
                   the static balancers (Round Robin / Least Connections
                   / ACO all read this to make their decisions).
GET  /prometheus  Prometheus text-exposition metrics for direct backend
                   scraping (independent of the balancer-side exporters).
GET  /load        current controlled-workload parameters.
POST /load        set controlled-workload parameters (cpu_level,
                   memory_mb, base_latency_ms, jitter_ms, capacity,
                   error_rate) - this is the explicit, documented
                   workload simulation mechanism described in the
                   project README, used by the DQN training
                   environment and by the adaptive/failure experiments.
POST /fail        soft-fail this server (simulate a crash without
                   stopping the container) - used by the phased
                   adaptive experiment.
POST /recover     clear the soft-fail flag.
POST /reset       reset all counters (useful between test runs).
"""

from __future__ import annotations

import os
import time

from flask import Flask, jsonify, request
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, generate_latest

from workload import WorkloadSimulator

SERVER_ID = os.environ.get("SERVER_ID", "1")
SERVER_NAME = f"server{SERVER_ID}"
PORT = int(os.environ.get("PORT", 5000))
MEMORY_CAP_MB = float(os.environ.get("MEMORY_CAP_MB", 512.0))
WINDOW_SECONDS = float(os.environ.get("METRICS_WINDOW_SECONDS", 10.0))

app = Flask(__name__)
sim = WorkloadSimulator(SERVER_NAME, memory_cap_mb=MEMORY_CAP_MB, window_seconds=WINDOW_SECONDS)

# ----------------------------------------------------------------------
# Prometheus - direct backend instrumentation (in addition to the
# balancer-side `dqn_state_*` exporters; useful for verifying that the
# balancer's view of a server matches the server's own view).
# ----------------------------------------------------------------------
BACKEND_REQUESTS_TOTAL = Counter(
    "backend_requests_total", "Total requests handled by this backend server",
    ["server", "outcome"],
)
BACKEND_CPU_USAGE = Gauge("backend_cpu_usage_percent", "Measured CPU usage percent", ["server"])
BACKEND_MEMORY_USAGE = Gauge("backend_memory_usage_percent", "Measured memory usage percent", ["server"])
BACKEND_QUEUE_LENGTH = Gauge("backend_queue_length", "Current queue length", ["server"])
BACKEND_ACTIVE_REQUESTS = Gauge("backend_active_requests", "Current active requests", ["server"])
BACKEND_REQUEST_RATE = Gauge("backend_request_rate", "Requests/sec, rolling window", ["server"])
BACKEND_THROUGHPUT = Gauge("backend_throughput", "Successful requests/sec, rolling window", ["server"])
BACKEND_ERROR_RATE = Gauge("backend_error_rate", "Error rate, rolling window (0-1)", ["server"])
BACKEND_LOAD_PRESSURE = Gauge("backend_load_pressure", "Composite load pressure (0-1)", ["server"])
BACKEND_HEALTH = Gauge("backend_health", "1 if healthy else 0", ["server"])


def _update_prometheus_gauges(m: dict) -> None:
    BACKEND_CPU_USAGE.labels(server=SERVER_NAME).set(m["cpu_usage"])
    BACKEND_MEMORY_USAGE.labels(server=SERVER_NAME).set(m["memory_usage"])
    BACKEND_QUEUE_LENGTH.labels(server=SERVER_NAME).set(m["queue_length"])
    BACKEND_ACTIVE_REQUESTS.labels(server=SERVER_NAME).set(m["active_requests"])
    BACKEND_REQUEST_RATE.labels(server=SERVER_NAME).set(m["request_rate"])
    BACKEND_THROUGHPUT.labels(server=SERVER_NAME).set(m["throughput"])
    BACKEND_ERROR_RATE.labels(server=SERVER_NAME).set(m["error_rate"])
    BACKEND_LOAD_PRESSURE.labels(server=SERVER_NAME).set(m["load_pressure"])
    BACKEND_HEALTH.labels(server=SERVER_NAME).set(1.0 if m["health"] else 0.0)


@app.route("/")
def handle_request():
    start = time.time()
    success, latency, queued_seconds = sim.handle_request()
    outcome = "success" if success else "failure"
    BACKEND_REQUESTS_TOTAL.labels(server=SERVER_NAME, outcome=outcome).inc()

    payload = {
        "server_id": SERVER_NAME,
        "status": "OK" if success else "ERROR",
        "success": success,
        "response_time_ms": round(latency * 1000.0, 2),
        "queued_ms": round(queued_seconds * 1000.0, 2),
        "wall_time_ms": round((time.time() - start) * 1000.0, 2),
        "message": "Request processed" if success else "Request failed",
    }
    return jsonify(payload), (200 if success else 500)


@app.route("/health")
def health():
    m = sim.snapshot()
    status_code = 200 if m["health"] else 503
    return jsonify({
        "server_id": SERVER_NAME,
        "healthy": m["health"],
        "status": "healthy" if m["health"] else "unhealthy",
        "manually_failed": m["manually_failed"],
        "error_rate": m["error_rate"],
    }), status_code


@app.route("/metrics")
def metrics():
    """Full JSON metrics snapshot - the canonical source for the DQN
    state builder (dqn/state.py) and for the Round Robin / Least
    Connections / ACO balancers."""
    m = sim.snapshot()
    _update_prometheus_gauges(m)
    return jsonify(m)


@app.route("/prometheus")
def prometheus_metrics():
    m = sim.snapshot()
    _update_prometheus_gauges(m)
    return generate_latest(), 200, {"Content-Type": CONTENT_TYPE_LATEST}


@app.route("/load", methods=["GET", "POST"])
def load():
    """Controlled workload simulation control surface (see
    servers/workload.py module docstring for why this is explicit
    controlled simulation and not fabricated metrics)."""
    if request.method == "GET":
        return jsonify(sim.get_params())

    data = request.get_json(silent=True) or request.form.to_dict() or {}
    updated = sim.set_params(
        cpu_level=data.get("cpu_level"),
        memory_mb=data.get("memory_mb"),
        base_latency_ms=data.get("base_latency_ms"),
        jitter_ms=data.get("jitter_ms"),
        capacity=data.get("capacity"),
        error_rate=data.get("error_rate"),
    )
    return jsonify({"updated": True, "params": updated.to_dict()})


@app.route("/fail", methods=["POST"])
def fail():
    sim.fail()
    return jsonify({"server_id": SERVER_NAME, "manually_failed": True})


@app.route("/recover", methods=["POST"])
def recover():
    sim.recover()
    return jsonify({"server_id": SERVER_NAME, "manually_failed": False})


@app.route("/reset", methods=["POST"])
def reset():
    global sim
    sim.shutdown()
    sim = WorkloadSimulator(SERVER_NAME, memory_cap_mb=MEMORY_CAP_MB, window_seconds=WINDOW_SECONDS)
    return jsonify({"server_id": SERVER_NAME, "reset": True})


if __name__ == "__main__":
    print(f"[{SERVER_NAME}] Backend server starting on 0.0.0.0:{PORT}")
    print(f"[{SERVER_NAME}] Endpoints: /  /health  /metrics  /prometheus  /load  /fail  /recover  /reset")
    app.run(host="0.0.0.0", port=PORT, threaded=True)
