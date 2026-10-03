#!/usr/bin/env python3
"""
testing/compare_algorithms.py
===============================
Head-to-head comparison of Round Robin, Least Connections, ACO and DQN
(Section 9). All four are evaluated against the SAME backend servers,
the SAME workload conditions, and the SAME multi-segment traffic
scenario (Section 7: fair comparison), one after another.

Usage:
    python testing/compare_algorithms.py

Prerequisites: the 4 backend servers AND the 4 balancers must all be
running (docker compose up -d, or the equivalent local `python
servers/app.py` / `python load_balancer/*.py` processes - see
README.md).

Writes: results/round_robin_results.json, results/least_connections_results.json,
results/aco_results.json, results/dqn_results.json, results/comparison_results.json
"""

from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from common.settings import get_settings  # noqa: E402
from dqn.config import RESULTS_DIR  # noqa: E402
from dqn.environment import LoadBalancerEnvironment  # noqa: E402
from testing.metrics import evaluate_records, run_multi_segment  # noqa: E402
from testing.scenarios import (COMPARISON_BACKEND_WORKLOAD,  # noqa: E402
                                EVALUATION_TRAFFIC_SCENARIO, evaluation_scenario_duration)

SETTINGS = get_settings()

ALGORITHMS = [
    ("round_robin", f"http://localhost:{SETTINGS.round_robin_port}"),
    ("least_connections", f"http://localhost:{SETTINGS.least_conn_port}"),
    ("aco", f"http://localhost:{SETTINGS.aco_port}"),
    ("dqn", f"http://localhost:{SETTINGS.dqn_port}"),
]


def check_balancer(url: str) -> bool:
    import requests
    try:
        r = requests.get(f"{url}/health", timeout=3)
        return r.status_code == 200
    except requests.RequestException:
        return False


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    env = LoadBalancerEnvironment()

    reachable = env.servers_reachable()
    if not all(reachable):
        print("ERROR: not all backend servers are reachable:")
        for url, ok in zip(env.server_urls, reachable):
            print(f"  {url} -> {'OK' if ok else 'UNREACHABLE'}")
        sys.exit(1)

    for name, url in ALGORITHMS:
        if not check_balancer(url):
            print(f"ERROR: {name} balancer not reachable at {url}. "
                  f"Start it, e.g.: docker compose up -d {name.replace('_', '-')}")
            sys.exit(1)

    duration = evaluation_scenario_duration()
    print(f"Evaluation scenario: {EVALUATION_TRAFFIC_SCENARIO} (total {duration:.0f}s per algorithm)")

    all_results = {}
    for name, url in ALGORITHMS:
        print(f"\n=== Evaluating {name} at {url} ===")
        env.reset_all()
        env.apply_phase(COMPARISON_BACKEND_WORKLOAD)
        time.sleep(1.0)

        records = run_multi_segment(url, EVALUATION_TRAFFIC_SCENARIO)
        metrics = evaluate_records(records, SETTINGS.server_names)
        metrics["algorithm"] = name
        metrics["scenario"] = EVALUATION_TRAFFIC_SCENARIO
        metrics["backend_workload"] = {SETTINGS.server_names[i]: v for i, v in COMPARISON_BACKEND_WORKLOAD.items()}
        metrics["generated_at"] = time.time()
        metrics["note"] = ("Example output; actual values depend on execution. "
                            "'Adaptation time' and 'Overload avoidance rate' are computed "
                            "separately by testing/adaptation_test.py, which is the "
                            "experiment specifically designed to exercise changing conditions.")

        out_path = os.path.join(RESULTS_DIR, f"{name}_results.json")
        with open(out_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"  total={metrics['total_requests']} success_rate={metrics['success_rate']*100:.1f}% "
              f"avg_rt={metrics['avg_response_time_ms']:.1f}ms p95={metrics['p95_response_time_ms']:.1f}ms "
              f"fairness={metrics['jain_fairness_index']:.3f} throughput={metrics['throughput_rps']:.2f}rps")
        print(f"  Saved: {out_path}")

        all_results[name] = metrics

    comparison_path = os.path.join(RESULTS_DIR, "comparison_results.json")
    with open(comparison_path, "w") as f:
        json.dump({
            "generated_at": time.time(),
            "scenario": EVALUATION_TRAFFIC_SCENARIO,
            "backend_workload": {SETTINGS.server_names[i]: v for i, v in COMPARISON_BACKEND_WORKLOAD.items()},
            "results": all_results,
            "note": "Example output; actual values depend on execution. No numbers here are hard-coded.",
        }, f, indent=2)
    print(f"\nComparison saved: {comparison_path}")
    print("Next: python visualization/generate_plots.py")
    env.close()


if __name__ == "__main__":
    main()
