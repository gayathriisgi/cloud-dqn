#!/usr/bin/env python3
"""
testing/failure_test.py
=========================
Failure / recovery experiment (Section 17).

Two ways to fail a server:

1. Soft failure (default, works with or without Docker): calls
   ``POST /fail`` on the target backend server (see
   servers/workload.py) - the process keeps running and keeps
   answering ``/metrics``/``/health`` but almost every request fails,
   which is exactly what a balancer has to detect and route around.
2. Real container failure (``--use-docker``): actually runs
   ``docker compose stop <service>`` / ``docker compose start <service>``,
   matching the spec's exact workflow. This only works when the stack
   is running via Docker Compose on the same host this script runs on.

Usage:
    python testing/failure_test.py --target-name dqn
    python testing/failure_test.py --target-name dqn --use-docker --server-index 1

Saves:
    results/failure_test_records.json
    results/failure_test_summary.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from typing import List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from common.settings import get_settings  # noqa: E402
from dqn.config import RESULTS_DIR  # noqa: E402
from dqn.environment import LoadBalancerEnvironment  # noqa: E402
from testing.adaptation_test import (overload_avoidance_rate,  # noqa: E402
                                      time_to_threshold)
from testing.metrics import drive_traffic, evaluate_records  # noqa: E402
from testing.scenarios import FAILURE_TEST_CONFIG  # noqa: E402

SETTINGS = get_settings()

TARGET_BY_NAME = {
    "round_robin": f"http://localhost:{SETTINGS.round_robin_port}",
    "least_connections": f"http://localhost:{SETTINGS.least_conn_port}",
    "aco": f"http://localhost:{SETTINGS.aco_port}",
    "dqn": f"http://localhost:{SETTINGS.dqn_port}",
}


def docker_stop(service: str) -> None:
    subprocess.run(["docker", "compose", "stop", service], check=False)


def docker_start(service: str) -> None:
    subprocess.run(["docker", "compose", "start", service], check=False)


def run(target_name: str, server_index: int, use_docker: bool, output_dir: str) -> None:
    url = TARGET_BY_NAME[target_name]
    env = LoadBalancerEnvironment()
    server_name = SETTINGS.server_names[server_index]
    compose_service = server_name  # docker-compose.yml service names match server1..server4

    cfg = FAILURE_TEST_CONFIG
    env.reset_all()
    env.apply_uniform("low")
    time.sleep(1.0)

    print(f"Failure/recovery experiment against {target_name} ({url}); target={server_name}; "
          f"mode={'docker compose stop/start' if use_docker else 'soft /fail /recover'}")

    all_records = []
    t0 = time.time()

    print(f"\n--- pre_failure ({cfg['pre_failure_duration']}s) ---")
    pre = drive_traffic(url, cfg["traffic_pattern"], cfg["pre_failure_duration"], cfg["traffic_rate"])
    for r in pre:
        r["phase"] = "pre_failure"
    all_records.extend(pre)

    print(f"\n--- failing {server_name} ---")
    failure_trigger_time = time.time()
    if use_docker:
        docker_stop(compose_service)
    else:
        env.fail_server(server_index)

    failure_records = drive_traffic(url, cfg["traffic_pattern"], cfg["failure_duration"], cfg["traffic_rate"])
    for r in failure_records:
        r["phase"] = "failure"
    all_records.extend(failure_records)

    print(f"\n--- recovering {server_name} ---")
    recovery_trigger_time = time.time()
    if use_docker:
        docker_start(compose_service)
        time.sleep(3.0)  # give the container a moment to actually come back up
    else:
        env.recover_server(server_index)

    recovery_records = drive_traffic(url, cfg["traffic_pattern"], cfg["post_recovery_duration"], cfg["traffic_rate"])
    for r in recovery_records:
        r["phase"] = "post_recovery"
    all_records.extend(recovery_records)

    pre_metrics = evaluate_records(pre, SETTINGS.server_names)
    failure_metrics = evaluate_records(failure_records, SETTINGS.server_names)
    recovery_metrics = evaluate_records(recovery_records, SETTINGS.server_names)

    routing_change = overload_avoidance_rate(failure_records, [server_name])
    failure_detection_time = time_to_threshold(failure_records, server_name, window=8, threshold=0.15, direction="below")
    pre_failure_share = pre_metrics["server_request_counts"].get(server_name, 0) / max(1, pre_metrics["total_requests"])
    recovery_threshold = max(0.10, pre_failure_share * 0.5)
    recovery_detection_time = time_to_threshold(recovery_records, server_name, window=8, threshold=recovery_threshold, direction="above")
    post_recovery_share = recovery_metrics["server_request_counts"].get(server_name, 0) / max(1, recovery_metrics["total_requests"])

    summary = {
        "target": target_name,
        "failed_server": server_name,
        "use_docker": use_docker,
        "generated_at": time.time(),
        "config": cfg,
        "pre_failure_metrics": pre_metrics,
        "failure_metrics": failure_metrics,
        "recovery_metrics": recovery_metrics,
        "pre_failure_share_of_failed_server": pre_failure_share,
        "post_recovery_share_of_failed_server": post_recovery_share,
        "routing_change_avoidance_rate_during_failure": routing_change,
        "failure_detection_time_seconds": failure_detection_time,
        "recovery_detection_time_seconds": recovery_detection_time,
        "note": ("Example output; actual values depend on execution. "
                 "failure_detection_time_seconds/recovery_detection_time_seconds are null "
                 "if the rolling selection share never crossed the threshold within the "
                 "configured phase duration - lengthen --failure-duration / "
                 "--post-recovery-duration or train the DQN more if that happens."),
    }

    os.makedirs(output_dir, exist_ok=True)
    records_path = os.path.join(output_dir, "failure_test_records.json")
    summary_path = os.path.join(output_dir, "failure_test_summary.json")
    with open(records_path, "w") as f:
        json.dump(all_records, f, indent=2, default=str)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nPre-failure share of {server_name}: {pre_failure_share*100:.1f}%")
    print(f"Routing-avoidance rate during failure: {routing_change*100:.1f}%")
    print(f"Post-recovery share of {server_name}: {post_recovery_share*100:.1f}%")
    print(f"Saved: {records_path}\nSaved: {summary_path}")
    env.close()


def main():
    parser = argparse.ArgumentParser(description="Failure/recovery experiment.")
    parser.add_argument("--target-name", type=str, default="dqn", choices=list(TARGET_BY_NAME.keys()))
    parser.add_argument("--server-index", type=int, default=FAILURE_TEST_CONFIG["server_index"])
    parser.add_argument("--use-docker", action="store_true",
                         help="Use `docker compose stop/start` instead of the soft /fail /recover endpoints.")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()
    run(args.target_name, args.server_index, args.use_docker, args.output_dir or RESULTS_DIR)


if __name__ == "__main__":
    main()
