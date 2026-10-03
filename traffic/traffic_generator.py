#!/usr/bin/env python3
"""
traffic/traffic_generator.py
=============================
Upgraded traffic generator (Section 16). Supports the workload
patterns required by the spec and logs every request's outcome so the
same run can be fed into ``testing/metrics.py`` / ``testing/compare_algorithms.py``.

Usage:
    python traffic/traffic_generator.py --target-name dqn --pattern burst --duration 60 --rate 20
    python traffic/traffic_generator.py --target http://localhost:8000 --pattern ramp_up --duration 30 --rate 15

Patterns:
    low            constant, ~0.3x --rate
    normal         constant, --rate
    high           constant, ~2x --rate
    burst          alternates short high-rate bursts with low-rate lulls
    spike          mostly low rate with brief, sharp random spikes
    sustained_high constant, ~3x --rate, for the whole duration
    ramp_up        linearly increases from ~1 req/s to ~2x --rate
    ramp_down      linearly decreases from ~2x --rate to ~1 req/s

Log columns: timestamp, request_id, response_time_ms, success,
selected_server, algorithm, target_rate, pattern.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import requests  # noqa: E402

from common.settings import get_settings  # noqa: E402

SETTINGS = get_settings()

TARGET_BY_NAME = {
    "round_robin": f"http://localhost:{SETTINGS.round_robin_port}",
    "least_connections": f"http://localhost:{SETTINGS.least_conn_port}",
    "aco": f"http://localhost:{SETTINGS.aco_port}",
    "dqn": f"http://localhost:{SETTINGS.dqn_port}",
}

PATTERNS = [
    "low", "normal", "high", "burst", "spike",
    "sustained_high", "ramp_up", "ramp_down",
]


def rate_at(pattern: str, elapsed: float, duration: float, base_rate: float) -> float:
    """Requests-per-second target at time `elapsed` into the run."""
    frac = 0.0 if duration <= 0 else min(1.0, elapsed / duration)

    if pattern == "low":
        return max(0.5, base_rate * 0.3)
    if pattern == "normal":
        return base_rate
    if pattern == "high":
        return base_rate * 2.0
    if pattern == "sustained_high":
        return base_rate * 3.0
    if pattern == "ramp_up":
        return max(0.5, 1.0 + frac * (base_rate * 2.0 - 1.0))
    if pattern == "ramp_down":
        return max(0.5, base_rate * 2.0 - frac * (base_rate * 2.0 - 1.0))
    if pattern == "burst":
        cycle = 5.0
        phase = elapsed % cycle
        return base_rate * 4.0 if phase < 2.0 else base_rate * 0.4
    if pattern == "spike":
        # Deterministic-ish spikes every ~7s (pseudo-random but
        # reproducible per run via the module-level RNG seed).
        cycle = 7.0
        phase = elapsed % cycle
        return base_rate * 6.0 if phase < 0.7 else base_rate * 0.5
    return base_rate


def send_one(target: str, request_id: int, pattern: str, target_rate: float, writer_lock, writer, results) -> None:
    start = time.time()
    success = False
    selected_server = ""
    algorithm = ""
    try:
        r = requests.get(target, timeout=10)
        elapsed = time.time() - start
        success = r.status_code == 200
        try:
            body = r.json()
            selected_server = body.get("selected_server") or body.get("forwarded_to", "")
            algorithm = body.get("algorithm", "")
        except ValueError:
            pass
    except requests.RequestException:
        elapsed = time.time() - start

    row = {
        "timestamp": start,
        "request_id": request_id,
        "response_time_ms": round(elapsed * 1000.0, 2),
        "success": success,
        "selected_server": selected_server,
        "algorithm": algorithm,
        "target_rate": round(target_rate, 3),
        "pattern": pattern,
    }
    with writer_lock:
        writer.writerow(row)
        results.append(row)


def run(target: str, pattern: str, duration: float, base_rate: float, log_path: str, seed: int) -> list:
    random.seed(seed)
    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)

    fieldnames = ["timestamp", "request_id", "response_time_ms", "success",
                  "selected_server", "algorithm", "target_rate", "pattern"]
    results = []
    writer_lock = threading.Lock()

    print(f"Traffic generator -> target={target} pattern={pattern} duration={duration}s base_rate={base_rate}/s")
    print(f"Logging to {log_path}")

    with open(log_path, "w", newline="") as f, ThreadPoolExecutor(max_workers=64) as executor:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        start_time = time.time()
        request_id = 0
        futures = []

        while True:
            elapsed = time.time() - start_time
            if elapsed >= duration:
                break

            target_rate = max(0.1, rate_at(pattern, elapsed, duration, base_rate))
            interval = 1.0 / target_rate

            request_id += 1
            futures.append(executor.submit(
                send_one, target, request_id, pattern, target_rate, writer_lock, writer, results
            ))
            time.sleep(interval)

        for fut in futures:
            fut.result()

    total = len(results)
    success_count = sum(1 for r in results if r["success"])
    avg_ms = sum(r["response_time_ms"] for r in results) / total if total else 0.0
    print(f"\nDone. {total} requests sent, {success_count} succeeded "
          f"({(success_count / total * 100.0) if total else 0:.1f}%), avg response {avg_ms:.1f}ms.")
    return results


def main():
    parser = argparse.ArgumentParser(description="Traffic generator for the cloud load balancer project.")
    parser.add_argument("--target", type=str, default=None, help="Full balancer URL, e.g. http://localhost:8003")
    parser.add_argument("--target-name", type=str, default="dqn", choices=list(TARGET_BY_NAME.keys()),
                         help="Named balancer (used if --target is not given).")
    parser.add_argument("--pattern", type=str, default="normal", choices=PATTERNS)
    parser.add_argument("--duration", type=float, default=60.0, help="Duration in seconds.")
    parser.add_argument("--rate", type=float, default=10.0, help="Base requests/sec.")
    parser.add_argument("--output", type=str, default=None, help="CSV log path.")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    target = args.target or TARGET_BY_NAME[args.target_name]
    output = args.output or os.path.join(
        os.path.dirname(__file__), "..", "logs",
        f"traffic_{args.target_name}_{args.pattern}_{int(time.time())}.csv",
    )
    run(target, args.pattern, args.duration, args.rate, output, args.seed)


if __name__ == "__main__":
    main()
