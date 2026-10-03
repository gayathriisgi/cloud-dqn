#!/usr/bin/env python3
"""
load_balancer/dqn_balancer.py
===============================
The learned load balancer. Implements the full request flow from
Section 6 of the spec:

    Incoming Request
      -> collect real /metrics from all 4 backend servers
      -> build normalized 46-dim state (dqn/state.py)
      -> DQN inference -> Q-values for 4 servers (dqn/inference.py)
      -> select server (argmax, or epsilon-greedy if DQN_INFERENCE_EPSILON > 0)
      -> forward the real request, measure the real outcome
      -> collect the real next state
      -> compute the multi-objective reward (dqn/reward.py)
      -> record Prometheus metrics
      -> (optional) online learning step, off by default
      -> return JSON with real q_values / reward / server_metrics

PROMETHEUS METRIC CONVENTION (documented once, applied consistently):
the ``dqn_state_*`` gauges below expose **raw, unnormalized** values in
the same units as the backend `/metrics` endpoint (percent, ms,
requests/sec) because that is what a human reads in Grafana. The
network itself only ever sees the **normalized** [0, 1] version built
by ``dqn/state.py`` - normalization is an internal detail of the model,
not something you want to stare at on a dashboard.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from collections import deque

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from flask import Flask, jsonify, request  # noqa: E402
from prometheus_client import Counter, Gauge, Histogram, start_http_server  # noqa: E402

from common.formulas import jain_fairness_index  # noqa: E402
from common.settings import get_settings  # noqa: E402
from dqn.config import DQN_HYPERPARAMS, MODEL_PATH, ONLINE_LEARNING_ENABLED  # noqa: E402
from dqn.inference import DQNInferenceService  # noqa: E402
from dqn.reward import compute_reward, load_variance  # noqa: E402
from dqn.state import build_state  # noqa: E402
from load_balancer.common import (MetricsCollector, build_response,  # noqa: E402
                                   forward_request, metrics_by_index, server_name)

SETTINGS = get_settings()
ALGORITHM = "DQN"
ONLINE_SAVE_EVERY = int(os.environ.get("DQN_ONLINE_SAVE_EVERY", 200))
FAIRNESS_WINDOW = int(os.environ.get("DQN_FAIRNESS_WINDOW", 200))

app = Flask(__name__)
collector = MetricsCollector()
inference = DQNInferenceService(model_path=MODEL_PATH)

_lock = threading.Lock()
_action_window = deque(maxlen=FAIRNESS_WINDOW)
_request_counter_since_save = 0

# ----------------------------------------------------------------------
# Prometheus instrumentation (Section 10)
# ----------------------------------------------------------------------
DQN_REQUESTS_TOTAL = Counter("dqn_requests_total", "Total requests routed by the DQN balancer", ["server", "outcome"])
DQN_RESPONSE_TIME = Histogram("dqn_response_time_seconds", "Response time of DQN-routed requests")
DQN_REWARD = Gauge("dqn_reward", "Most recent reward")
DQN_Q_VALUE = Gauge("dqn_q_value", "Current Q-value estimate per server", ["server"])
DQN_SELECTED_SERVER_TOTAL = Counter("dqn_selected_server_total", "Number of times each server was selected", ["server"])
DQN_STATE_CPU = Gauge("dqn_state_cpu", "Raw CPU usage percent as seen by the DQN state builder", ["server"])
DQN_STATE_MEMORY = Gauge("dqn_state_memory", "Raw memory usage percent", ["server"])
DQN_STATE_QUEUE = Gauge("dqn_state_queue", "Raw queue length", ["server"])
DQN_STATE_ACTIVE_REQUESTS = Gauge("dqn_state_active_requests", "Raw active request count", ["server"])
DQN_STATE_REQUEST_RATE = Gauge("dqn_state_request_rate", "Raw request rate (req/s)", ["server"])
DQN_STATE_THROUGHPUT = Gauge("dqn_state_throughput", "Raw throughput (successful req/s)", ["server"])
DQN_STATE_ERROR_RATE = Gauge("dqn_state_error_rate", "Raw error rate (0-1)", ["server"])
DQN_STATE_LATENCY = Gauge("dqn_state_latency", "Raw average response time (ms)", ["server"])
DQN_SERVER_LOAD_PRESSURE = Gauge("dqn_server_load_pressure", "Composite load pressure (0-1)", ["server"])
DQN_DECISION_TIME = Histogram("dqn_decision_time_seconds", "Time spent on state build + forward pass + action selection")
DQN_EXPLORATION_TOTAL = Counter("dqn_exploration_total", "Number of epsilon-greedy random (exploration) decisions")
DQN_EXPLOITATION_TOTAL = Counter("dqn_exploitation_total", "Number of argmax(Q) (exploitation) decisions")
DQN_SUCCESS_TOTAL = Counter("dqn_success_total", "Total successful DQN-routed requests")
DQN_FAILURE_TOTAL = Counter("dqn_failure_total", "Total failed DQN-routed requests")
DQN_LOAD_VARIANCE = Gauge("dqn_load_variance", "Variance of load_pressure across servers, post-action")
DQN_FAIRNESS_INDEX = Gauge("dqn_fairness_index", "Jain fairness index of server selection over a rolling window")
DQN_EPSILON = Gauge("dqn_epsilon", "Current inference epsilon (exploration probability)")
DQN_TRAINING_LOSS = Gauge("dqn_online_training_loss", "Most recent online-learning loss (0 if online learning disabled)")
DQN_MODEL_LOADED = Gauge("dqn_model_loaded", "1 if a trained checkpoint is currently loaded, else 0")


def _update_state_gauges(all_metrics) -> None:
    for i, m in enumerate(all_metrics):
        m = metrics_by_index(all_metrics, i)
        name = server_name(i)
        DQN_STATE_CPU.labels(server=name).set(m.get("cpu_usage", 0.0))
        DQN_STATE_MEMORY.labels(server=name).set(m.get("memory_usage", 0.0))
        DQN_STATE_QUEUE.labels(server=name).set(m.get("queue_length", 0.0))
        DQN_STATE_ACTIVE_REQUESTS.labels(server=name).set(m.get("active_requests", 0.0))
        DQN_STATE_REQUEST_RATE.labels(server=name).set(m.get("request_rate", 0.0))
        DQN_STATE_THROUGHPUT.labels(server=name).set(m.get("throughput", 0.0))
        DQN_STATE_ERROR_RATE.labels(server=name).set(m.get("error_rate", 0.0))
        lat = m.get("average_response_time_ms")
        if lat is None:
            lat = float(m.get("average_response_time", 0.0)) * 1000.0
        DQN_STATE_LATENCY.labels(server=name).set(lat)
        DQN_SERVER_LOAD_PRESSURE.labels(server=name).set(m.get("load_pressure", 0.0))


@app.route("/")
def route_request():
    global _request_counter_since_save

    decision_start = time.perf_counter()
    metrics_before = collector.fetch_all()
    state = build_state(metrics_before)
    action, q_values, exploring, tie_candidates = inference.select(state)
    decision_time = time.perf_counter() - decision_start

    url = SETTINGS.server_urls[action]
    success, response_time, _, _ = forward_request(url)

    metrics_after = collector.fetch_all()
    next_state = build_state(metrics_after)
    chosen_after = metrics_by_index(metrics_after, action)

    breakdown = compute_reward(
        success=success, response_time_s=response_time,
        chosen_server_metrics=chosen_after, all_server_metrics=metrics_after,
    )

    online_loss = None
    if ONLINE_LEARNING_ENABLED:
        inference.agent.store(state, action, breakdown.total, next_state, False)
        online_loss = inference.agent.learn()
        inference.agent.step_done()
        _request_counter_since_save += 1
        if _request_counter_since_save >= ONLINE_SAVE_EVERY:
            inference.agent.save(MODEL_PATH)
            _request_counter_since_save = 0

    with _lock:
        _action_window.append(action)
        fairness = jain_fairness_index([
            _action_window.count(i) for i in range(len(SETTINGS.server_urls))
        ])

    variance = load_variance(metrics_after)

    # --- Prometheus ---
    name = server_name(action)
    outcome = "success" if success else "failure"
    DQN_REQUESTS_TOTAL.labels(server=name, outcome=outcome).inc()
    DQN_RESPONSE_TIME.observe(response_time)
    DQN_REWARD.set(breakdown.total)
    for i, q in enumerate(q_values):
        DQN_Q_VALUE.labels(server=server_name(i)).set(float(q))
    DQN_SELECTED_SERVER_TOTAL.labels(server=name).inc()
    _update_state_gauges(metrics_after)
    DQN_DECISION_TIME.observe(decision_time)
    (DQN_EXPLORATION_TOTAL if exploring else DQN_EXPLOITATION_TOTAL).inc()
    (DQN_SUCCESS_TOTAL if success else DQN_FAILURE_TOTAL).inc()
    DQN_LOAD_VARIANCE.set(variance)
    DQN_FAIRNESS_INDEX.set(fairness)
    DQN_EPSILON.set(inference.epsilon)
    DQN_TRAINING_LOSS.set(online_loss if online_loss is not None else 0.0)
    DQN_MODEL_LOADED.set(1.0 if inference.model_loaded else 0.0)

    response = build_response(
        ALGORITHM, action, success, response_time, metrics_before, metrics_after,
        q_values=q_values.tolist(), decision_time=decision_time,
        extra={
            "exploration": exploring,
            "tie_candidates": [server_name(i) for i in tie_candidates],
            "tie_margin": inference.tie_margin,
            "model_loaded": inference.model_loaded,
            "epsilon": inference.epsilon,
            "load_variance": round(variance, 5),
            "fairness_index": round(fairness, 4),
            "online_learning": ONLINE_LEARNING_ENABLED,
            "online_loss": online_loss,
        },
    )
    return jsonify(response), (200 if success else 502)


@app.route("/status")
def status():
    with _lock:
        window = list(_action_window)
    return jsonify({
        **inference.status(),
        "algorithm": ALGORITHM,
        "online_learning_enabled": ONLINE_LEARNING_ENABLED,
        "recent_action_counts": {
            server_name(i): window.count(i) for i in range(len(SETTINGS.server_urls))
        },
        "port": SETTINGS.dqn_port,
        "metrics_port": SETTINGS.dqn_metrics_port,
    })


@app.route("/health")
def health():
    return jsonify({"status": "healthy", "algorithm": ALGORITHM, "model_loaded": inference.model_loaded})


@app.route("/predict")
def predict():
    """Read-only: build the CURRENT state from the real /metrics and return
    the real Q-values and the server the policy WOULD pick - without
    forwarding any request or changing anything. Used by MyCamu's DQN
    Monitor to show live Q-values between requests."""
    t0 = time.perf_counter()
    metrics_now = collector.fetch_all()
    state = build_state(metrics_now)
    q = inference.agent.q_values(state)
    action = int(q.argmax())
    best = float(q.max())
    ties = [server_name(i) for i in range(len(q)) if q[i] >= best - inference.tie_margin]
    return jsonify({
        "algorithm": ALGORITHM,
        "model_loaded": inference.model_loaded,
        "q_values": {server_name(i): round(float(v), 4) for i, v in enumerate(q)},
        "would_select": server_name(action),
        "tie_candidates": ties,
        "tie_margin": inference.tie_margin,
        "server_metrics": {server_name(i): metrics_by_index(metrics_now, i) for i in range(len(metrics_now))},
        "state_vector": [round(float(x), 4) for x in state],
        "decision_time_ms": round((time.perf_counter() - t0) * 1000.0, 3),
    })


@app.route("/reset", methods=["POST"])
def reset_counters():
    """Reset the balancer's rolling counters (recent action window).
    Does NOT touch the trained model or its weights."""
    with _lock:
        _action_window.clear()
    return jsonify({"reset": True, **inference.status()})


@app.route("/reload_model", methods=["POST"])
def reload_model():
    """Hot-reload the checkpoint from disk (e.g. after retraining)
    without restarting the container. Still enforces the same strict
    state-dimension check as startup."""
    inference.reload()
    return jsonify(inference.status())


if __name__ == "__main__":
    start_http_server(SETTINGS.dqn_metrics_port)
    print(f"[{ALGORITHM}] hyperparams: {DQN_HYPERPARAMS}")
    print(f"[{ALGORITHM}] model_loaded={inference.model_loaded} path={MODEL_PATH}")
    print(f"[{ALGORITHM}] Load balancer running on http://0.0.0.0:{SETTINGS.dqn_port}")
    print(f"[{ALGORITHM}] Prometheus metrics on http://0.0.0.0:{SETTINGS.dqn_metrics_port}/metrics")
    app.run(host="0.0.0.0", port=SETTINGS.dqn_port, threaded=True)
