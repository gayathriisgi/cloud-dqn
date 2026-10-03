#!/usr/bin/env python3
"""
testing/query_prometheus_snapshot.py
=======================================
NEW, minimal addition. Queries the real, locally-running Prometheus
server's HTTP API (/api/v1/query_range) for the exact same PromQL
expressions used by the project's own
monitoring/grafana/dashboards/dqn_dashboard.json panels, and dumps the
real returned time series to results/prometheus_snapshot.json.

This does not touch Prometheus, Grafana, or any balancer/backend code.
It exists because this sandbox cannot download the real Grafana binary
(network access to dl.grafana.com / Docker Hub / GitHub release assets
for Grafana is blocked here -- see the chat message for the exact
errors), so the real, already-collected Prometheus data is rendered
into an equivalent static dashboard instead, as a disclosed substitute
for actually running Grafana.

Usage:
    python testing/query_prometheus_snapshot.py --start <unix> --end <unix>
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request

PROM_URL = "http://localhost:9090"

PANELS = [
    ("Incoming RPS - Round Robin", 'sum(rate(lb_requests_total{algorithm="round_robin"}[30s]))'),
    ("Incoming RPS - Least Connections", 'sum(rate(lb_requests_total{algorithm="least_connections"}[30s]))'),
    ("Incoming RPS - ACO", 'sum(rate(lb_requests_total{algorithm="aco"}[30s]))'),
    ("Incoming RPS - DQN", 'sum(rate(dqn_requests_total[30s]))'),
    ("Avg response time - Round Robin (ms)",
     'rate(lb_response_time_seconds_sum{algorithm="round_robin"}[30s]) / rate(lb_response_time_seconds_count{algorithm="round_robin"}[30s]) * 1000'),
    ("Avg response time - Least Connections (ms)",
     'rate(lb_response_time_seconds_sum{algorithm="least_connections"}[30s]) / rate(lb_response_time_seconds_count{algorithm="least_connections"}[30s]) * 1000'),
    ("Avg response time - ACO (ms)",
     'rate(lb_response_time_seconds_sum{algorithm="aco"}[30s]) / rate(lb_response_time_seconds_count{algorithm="aco"}[30s]) * 1000'),
    ("Avg response time - DQN (ms)",
     'rate(dqn_response_time_seconds_sum[30s]) / rate(dqn_response_time_seconds_count[30s]) * 1000'),
    ("P95 latency - Round Robin (ms)",
     'histogram_quantile(0.95, sum(rate(lb_response_time_seconds_bucket{algorithm="round_robin"}[30s])) by (le)) * 1000'),
    ("P95 latency - Least Connections (ms)",
     'histogram_quantile(0.95, sum(rate(lb_response_time_seconds_bucket{algorithm="least_connections"}[30s])) by (le)) * 1000'),
    ("P95 latency - ACO (ms)",
     'histogram_quantile(0.95, sum(rate(lb_response_time_seconds_bucket{algorithm="aco"}[30s])) by (le)) * 1000'),
    ("P95 latency - DQN (ms)",
     'histogram_quantile(0.95, sum(rate(dqn_response_time_seconds_bucket[30s])) by (le)) * 1000'),
    ("CPU usage per server, % (during DQN run)", 'dqn_state_cpu{server=~"server.*"}'),
    ("Memory usage per server, % (during DQN run)", 'dqn_state_memory{server=~"server.*"}'),
    ("Queue length per server (during DQN run)", 'dqn_state_queue{server=~"server.*"}'),
    ("Q-value per server (DQN)", 'dqn_q_value{server=~"server.*"}'),
    ("Reward, most recent per request (DQN)", 'dqn_reward'),
    ("Jain fairness index, rolling window (DQN)", 'dqn_fairness_index'),
    ("Success rate (DQN, req/s)", 'rate(dqn_success_total[30s])'),
    ("Failure rate (DQN, req/s)", 'rate(dqn_failure_total[30s])'),
]


def query_range(expr: str, start: float, end: float, step: str = "5s"):
    params = urllib.parse.urlencode({"query": expr, "start": start, "end": end, "step": step})
    url = f"{PROM_URL}/api/v1/query_range?{params}"
    with urllib.request.urlopen(url, timeout=10) as r:
        body = json.loads(r.read().decode())
    if body.get("status") != "success":
        return {"error": body}
    return body["data"]["result"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=float, required=True)
    ap.add_argument("--end", type=float, required=True)
    ap.add_argument("--output", type=str, default="results/prometheus_snapshot.json")
    args = ap.parse_args()

    out = {"queried_at": time.time(), "range_start": args.start, "range_end": args.end, "panels": {}}
    for title, expr in PANELS:
        try:
            result = query_range(expr, args.start, args.end)
        except Exception as e:
            result = {"error": str(e)}
        out["panels"][title] = {"expr": expr, "result": result}
        n_series = len(result) if isinstance(result, list) else 0
        n_points = sum(len(s.get("values", [])) for s in result) if isinstance(result, list) else 0
        print(f"{title}: {n_series} series, {n_points} real datapoints")

    with open(args.output, "w") as f:
        json.dump(out, f)
    print(f"\nSaved real Prometheus query_range results: {args.output}")


if __name__ == "__main__":
    main()
