#!/usr/bin/env python3
"""
testing/adaptation_test.py
============================
The dedicated adaptive-behaviour experiment (Section 8). Drives real
traffic at the DQN balancer through five real, controlled workload
phases:

    Phase 1: all servers healthy
    Phase 2: server1 overloaded
    Phase 3: server1 AND server3 overloaded
    Phase 4: server1 recovered (server3 still overloaded)
    Phase 5: burst traffic (all servers healthy)

and records, per real request: timestamp, phase, selected server,
Q-values, reward, and every backend server's cpu/memory/queue/active
requests/latency/request rate/throughput/error rate at that moment.

From that real, captured data it computes the adaptive-behaviour
evidence metrics: decision change rate, adaptation time (how long
after overload starts before the DQN meaningfully avoids that server),
overload avoidance rate, recovery adaptation time, Q-value sensitivity
(how much the overloaded server's Q-value actually dropped), the
dynamic rebalancing score (how much the selection distribution shifts
between phases) and the load-response correlation (does higher
load_pressure on a server correlate with fewer selections?).

Usage:
    python testing/adaptation_test.py [--target-name dqn]

Saves:
    results/adaptation_records.json    (every raw request record)
    results/adaptation_summary.json    (phase metrics + adaptive-behaviour evidence)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from common.settings import get_settings  # noqa: E402
from dqn.config import RESULTS_DIR  # noqa: E402
from dqn.environment import LoadBalancerEnvironment  # noqa: E402
from testing.metrics import drive_traffic, evaluate_records  # noqa: E402
from testing.scenarios import ADAPTATION_PHASES  # noqa: E402

SETTINGS = get_settings()

TARGET_BY_NAME = {
    "round_robin": f"http://localhost:{SETTINGS.round_robin_port}",
    "least_connections": f"http://localhost:{SETTINGS.least_conn_port}",
    "aco": f"http://localhost:{SETTINGS.aco_port}",
    "dqn": f"http://localhost:{SETTINGS.dqn_port}",
}


def rolling_share(server_sequence: List[str], target: str, window: int) -> List[float]:
    shares = []
    for i in range(len(server_sequence)):
        lo = max(0, i - window + 1)
        chunk = server_sequence[lo:i + 1]
        shares.append(chunk.count(target) / len(chunk) if chunk else 0.0)
    return shares


def decision_change_rate(server_sequence: List[str]) -> float:
    if len(server_sequence) < 2:
        return 0.0
    changes = sum(1 for a, b in zip(server_sequence, server_sequence[1:]) if a != b)
    return changes / (len(server_sequence) - 1)


def overload_avoidance_rate(records: List[Dict], overloaded_servers: List[str]) -> float:
    if not records:
        return 1.0
    avoided = sum(1 for r in records if r.get("selected_server") not in overloaded_servers)
    return avoided / len(records)


def time_to_threshold(records: List[Dict], target_server: str, window: int,
                       threshold: float, direction: str) -> Optional[float]:
    """Returns seconds from the first record in `records` until the
    rolling selection share of `target_server` first crosses
    `threshold` in the given `direction` ('below' or 'above'). Returns
    None if it never crosses within this phase's records (an honest
    "did not adapt within this phase" result, not a fabricated number).
    """
    if not records:
        return None
    seq = [r.get("selected_server") for r in records]
    shares = rolling_share(seq, target_server, window)
    t0 = records[0]["timestamp"]
    warmup = min(window, len(records))
    for i in range(warmup, len(records)):
        if direction == "below" and shares[i] <= threshold:
            return records[i]["timestamp"] - t0
        if direction == "above" and shares[i] >= threshold:
            return records[i]["timestamp"] - t0
    return None


def mean_q_for_server(records: List[Dict], server_name: str) -> Optional[float]:
    vals = [r["q_values"][server_name] for r in records
            if r.get("q_values") and server_name in r["q_values"]]
    return float(np.mean(vals)) if vals else None


def avg_load_pressure(records: List[Dict], server_name: str) -> Optional[float]:
    vals = []
    for r in records:
        sm = r.get("server_metrics")
        if sm and server_name in sm and "load_pressure" in sm[server_name]:
            vals.append(sm[server_name]["load_pressure"])
    return float(np.mean(vals)) if vals else None


def compute_adaptive_metrics(records: List[Dict], phases: List[Dict], server_names: List[str]) -> Dict:
    by_phase = {p["name"]: [r for r in records if r["phase"] == p["name"]] for p in phases}

    seq_all = [r.get("selected_server") for r in records]
    overall_decision_change_rate = decision_change_rate(seq_all)

    p1, p2, p3, p4, p5 = [by_phase[p["name"]] for p in phases]

    server1, server3 = server_names[0], server_names[2]

    overload_avoidance_phase2 = overload_avoidance_rate(p2, [server1])
    overload_avoidance_phase3 = overload_avoidance_rate(p3, [server1, server3])

    adaptation_time_phase2 = time_to_threshold(p2, server1, window=8, threshold=0.15, direction="below")
    recovery_time_phase4 = time_to_threshold(p4, server1, window=8, threshold=0.15, direction="above")

    q_server1_phase1 = mean_q_for_server(p1, server1)
    q_server1_phase2 = mean_q_for_server(p2, server1)
    q_sensitivity_server1 = (
        (q_server1_phase1 - q_server1_phase2)
        if (q_server1_phase1 is not None and q_server1_phase2 is not None) else None
    )

    q_server1_phase3 = mean_q_for_server(p3, server1)
    q_server1_phase4 = mean_q_for_server(p4, server1)
    q_recovery_rebound = (
        (q_server1_phase4 - q_server1_phase3)
        if (q_server1_phase3 is not None and q_server1_phase4 is not None) else None
    )

    # Dynamic rebalancing score: how much the per-server selection
    # share vector shifts between consecutive phases (L1 distance,
    # averaged over transitions). A balancer that ignores load (like
    # Round Robin) will show a near-zero score here even while the
    # backend workload changes dramatically between phases.
    phase_shares = []
    for p in phases:
        recs = by_phase[p["name"]]
        seq = [r.get("selected_server") for r in recs]
        total = len(seq) or 1
        phase_shares.append(np.array([seq.count(s) / total for s in server_names]))

    rebalancing_deltas = [
        float(np.sum(np.abs(phase_shares[i + 1] - phase_shares[i])))
        for i in range(len(phase_shares) - 1)
    ]
    dynamic_rebalancing_score = float(np.mean(rebalancing_deltas)) if rebalancing_deltas else 0.0

    # Load-response relationship: pool (avg_load_pressure, selection_share)
    # across every (phase, server) pair and compute the Pearson
    # correlation. A well-adapted balancer should show a NEGATIVE
    # correlation (higher load -> lower selection share).
    pooled_pressure, pooled_share = [], []
    for p, shares in zip(phases, phase_shares):
        recs = by_phase[p["name"]]
        for i, s in enumerate(server_names):
            lp = avg_load_pressure(recs, s)
            if lp is not None:
                pooled_pressure.append(lp)
                pooled_share.append(shares[i])
    load_response_correlation = None
    if len(pooled_pressure) >= 3 and np.std(pooled_pressure) > 1e-9 and np.std(pooled_share) > 1e-9:
        load_response_correlation = float(np.corrcoef(pooled_pressure, pooled_share)[0, 1])

    return {
        "overall_decision_change_rate": overall_decision_change_rate,
        "overload_avoidance_rate_phase2_server1_overloaded": overload_avoidance_phase2,
        "overload_avoidance_rate_phase3_server1_and_3_overloaded": overload_avoidance_phase3,
        "adaptation_time_seconds_phase2": adaptation_time_phase2,
        "recovery_time_seconds_phase4": recovery_time_phase4,
        "q_value_sensitivity_server1_phase1_to_phase2": q_sensitivity_server1,
        "q_value_rebound_server1_phase3_to_phase4": q_recovery_rebound,
        "dynamic_rebalancing_score": dynamic_rebalancing_score,
        "load_response_correlation": load_response_correlation,
        "phase_selection_shares": {
            p["name"]: {s: float(phase_shares[i][j]) for j, s in enumerate(server_names)}
            for i, p in enumerate(phases)
        },
        "interpretation": {
            "adaptation_time_seconds_phase2": (
                "Seconds after server1 becomes overloaded until the DQN's rolling "
                "selection share of server1 drops to <=15%. null means it never "
                "dropped that low during this phase's duration - try a longer phase "
                "or more training."
            ),
            "q_value_sensitivity_server1_phase1_to_phase2": (
                "Mean Q(server1) in phase 1 minus mean Q(server1) in phase 2. "
                "Positive = Q genuinely dropped when server1 became overloaded, "
                "which is the direct evidence that the DQN's value function is "
                "sensitive to real load, not just outputting a fixed preference."
            ),
            "load_response_correlation": (
                "Pearson correlation between a server's average load_pressure and "
                "its selection share, pooled across all 5 phases and all 4 servers. "
                "Negative = the balancer routes away from higher-pressure servers, "
                "which is the core evidence of adaptive (not static) behaviour."
            ),
        },
    }


def run(target_name: str, output_dir: str) -> None:
    url = TARGET_BY_NAME[target_name]
    env = LoadBalancerEnvironment()

    reachable = env.servers_reachable()
    if not all(reachable):
        print("ERROR: not all backend servers are reachable. Start them first.")
        sys.exit(1)

    import requests
    try:
        requests.get(f"{url}/health", timeout=3)
    except requests.RequestException:
        print(f"ERROR: balancer '{target_name}' not reachable at {url}.")
        sys.exit(1)

    print(f"Running 5-phase adaptive experiment against {target_name} at {url}")
    print(f"Total duration: {sum(p['duration'] for p in ADAPTATION_PHASES):.0f}s")

    env.reset_all()
    all_records: List[Dict] = []
    phase_boundaries = []
    t0 = time.time()

    for phase in ADAPTATION_PHASES:
        print(f"\n--- {phase['name']} ---  workload={phase['workload']} "
              f"traffic={phase['traffic_pattern']}@{phase['traffic_rate']}rps for {phase['duration']}s")
        env.apply_phase(phase["workload"])
        time.sleep(1.0)  # let cpu/memory/queue settle to the new target before scoring

        phase_start_rel = time.time() - t0
        records = drive_traffic(url, phase["traffic_pattern"], phase["duration"], phase["traffic_rate"])
        for r in records:
            r["phase"] = phase["name"]
        all_records.extend(records)

        phase_metrics = evaluate_records(records, SETTINGS.server_names)
        phase_boundaries.append({
            "name": phase["name"], "start_seconds": phase_start_rel,
            "duration": phase["duration"], "workload": phase["workload"],
            "metrics": phase_metrics,
        })
        print(f"  requests={phase_metrics['total_requests']} "
              f"success_rate={phase_metrics['success_rate']*100:.1f}% "
              f"avg_rt={phase_metrics['avg_response_time_ms']:.1f}ms "
              f"server_counts={phase_metrics['server_request_counts']}")

    adaptive_metrics = compute_adaptive_metrics(all_records, ADAPTATION_PHASES, SETTINGS.server_names)

    os.makedirs(output_dir, exist_ok=True)
    records_path = os.path.join(output_dir, "adaptation_records.json")
    summary_path = os.path.join(output_dir, "adaptation_summary.json")

    with open(records_path, "w") as f:
        json.dump(all_records, f, indent=2, default=str)

    with open(summary_path, "w") as f:
        json.dump({
            "target": target_name,
            "generated_at": time.time(),
            "phases": phase_boundaries,
            "adaptive_behaviour_metrics": adaptive_metrics,
            "note": "Example output; actual values depend on execution and on how much dqn/train.py has trained the model.",
        }, f, indent=2, default=str)

    print(f"\nSaved raw records: {records_path}")
    print(f"Saved summary: {summary_path}")
    print("Next: python visualization/generate_plots.py")
    env.close()


def main():
    parser = argparse.ArgumentParser(description="Run the 5-phase adaptive-behaviour experiment.")
    parser.add_argument("--target-name", type=str, default="dqn", choices=list(TARGET_BY_NAME.keys()))
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()
    from dqn.config import RESULTS_DIR
    run(args.target_name, args.output_dir or RESULTS_DIR)


if __name__ == "__main__":
    main()
