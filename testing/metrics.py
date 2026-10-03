"""
testing/metrics.py
===================
Evaluation formulas (Section 9) plus the shared "drive real traffic at
a balancer and capture every response" helper used by
``compare_algorithms.py``, ``adaptation_test.py``, ``failure_test.py``
and ``run_tests.py``.

Unlike ``traffic/traffic_generator.py`` (which is the interactive,
CLI-facing tool from Section 16 and logs to CSV), the ``drive_traffic``
helper here keeps the **full JSON response** from every request
in-memory (including ``reward``, ``q_values`` and ``server_metrics``),
because the automated evaluation/experiment scripts need that detail
to compute cumulative reward, Q-value traces and per-server load - not
just response time and success/failure.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

import requests

from common.formulas import (jain_fairness_index, load_balance_degree,  # noqa: F401
                              load_variance, moving_average, percentile,
                              summarize_latencies)
from common.settings import get_settings
from traffic.traffic_generator import rate_at

SETTINGS = get_settings()


def _send_one(url: str, timeout: float = 10.0) -> Dict:
    start = time.time()
    try:
        r = requests.get(url, timeout=timeout)
        elapsed = time.time() - start
        try:
            body = r.json()
        except ValueError:
            body = {}
        return {
            "timestamp": start,
            "response_time_s": elapsed,
            "success": r.status_code == 200 and bool(body.get("success", r.status_code == 200)),
            "http_status": r.status_code,
            "selected_server": body.get("selected_server", ""),
            "algorithm": body.get("algorithm", ""),
            "reward": body.get("reward"),
            "q_values": body.get("q_values"),
            "exploration": body.get("exploration"),
            "server_metrics": body.get("server_metrics"),
            # backend-side processing time reported by the balancer (excludes
            # the balancer's own overhead) - lets reports separate routing
            # quality from balancer compute cost
            "backend_time_ms": body.get("response_time_ms"),
        }
    except requests.RequestException:
        elapsed = time.time() - start
        return {
            "timestamp": start, "response_time_s": elapsed, "success": False,
            "http_status": 0, "selected_server": "", "algorithm": "",
            "reward": None, "q_values": None, "exploration": None, "server_metrics": None,
        }


def drive_traffic(url: str, pattern: str, duration: float, rate: float,
                   max_workers: int = 64) -> List[Dict]:
    """Send real, paced requests at `url` for `duration` seconds
    following the named traffic `pattern`, and return every response
    record (see `_send_one`)."""
    results: List[Dict] = []
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        while True:
            elapsed = time.time() - start_time
            if elapsed >= duration:
                break
            target_rate = max(0.1, rate_at(pattern, elapsed, duration, rate))
            interval = 1.0 / target_rate
            futures.append(executor.submit(_send_one, url))
            time.sleep(interval)
        for fut in futures:
            results.append(fut.result())
    return results


def run_multi_segment(url: str, segments: List[Dict]) -> List[Dict]:
    """Run several {pattern, duration, rate} segments back to back
    against the same balancer and concatenate all records."""
    all_records: List[Dict] = []
    for seg in segments:
        all_records.extend(drive_traffic(url, seg["pattern"], seg["duration"], seg["rate"]))
    return all_records


def evaluate_records(records: List[Dict], server_names: Optional[List[str]] = None) -> Dict:
    """Compute the full Section-9 evaluation metric set from a list of
    response records produced by `drive_traffic`."""
    server_names = server_names or SETTINGS.server_names
    total = len(records)
    successes = [r for r in records if r["success"]]
    failures = [r for r in records if not r["success"]]
    response_times_s = [r["response_time_s"] for r in records]
    lat = summarize_latencies(response_times_s)

    timestamps = [r["timestamp"] for r in records]
    makespan = (max(timestamps) - min(timestamps)) if timestamps else 0.0

    server_counts = {name: 0 for name in server_names}
    for r in records:
        s = r.get("selected_server")
        if s in server_counts:
            server_counts[s] += 1
    count_values = list(server_counts.values())

    rewards = [r["reward"] for r in records if isinstance(r.get("reward"), (int, float))]

    cpu_vals, mem_vals, queue_vals = [], [], []
    for r in records:
        sm = r.get("server_metrics")
        sel = r.get("selected_server")
        if sm and sel and sel in sm:
            m = sm[sel]
            if "cpu_usage" in m:
                cpu_vals.append(m["cpu_usage"])
            if "memory_usage" in m:
                mem_vals.append(m["memory_usage"])
            if "queue_length" in m:
                queue_vals.append(m["queue_length"])

    duration = makespan if makespan > 0 else 1.0

    return {
        "total_requests": total,
        "success_count": len(successes),
        "failure_count": len(failures),
        "success_rate": (len(successes) / total) if total else 0.0,
        "error_rate": (len(failures) / total) if total else 0.0,
        "avg_response_time_ms": lat["avg_ms"],
        "median_response_time_ms": lat["median_ms"],
        "p95_response_time_ms": lat["p95_ms"],
        "p99_response_time_ms": lat["p99_ms"],
        "throughput_rps": (len(successes) / duration) if duration else 0.0,
        "makespan_seconds": makespan,
        "server_request_counts": server_counts,
        "jain_fairness_index": jain_fairness_index(count_values),
        "load_balance_degree": load_balance_degree(count_values),
        "request_count_variance": load_variance(count_values),
        "cumulative_reward": float(sum(rewards)) if rewards else None,
        "avg_reward": (float(sum(rewards)) / len(rewards)) if rewards else None,
        "avg_cpu_of_selected": (sum(cpu_vals) / len(cpu_vals)) if cpu_vals else None,
        "avg_memory_of_selected": (sum(mem_vals) / len(mem_vals)) if mem_vals else None,
        "avg_queue_of_selected": (sum(queue_vals) / len(queue_vals)) if queue_vals else None,
    }
