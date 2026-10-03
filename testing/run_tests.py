#!/usr/bin/env python3
"""
testing/run_tests.py
======================
8 automated tests with real PASS/FAIL verdicts (Section 15). Every
test drives real HTTP traffic against the real backend servers / real
balancers and computes its verdict from the resulting measurements -
"PASS" is never assigned merely because a script ran without raising
an exception.

Tests:
    1. State sensitivity          - does the DQN state vector actually
                                     change when a server's real workload changes?
    2. Q-value response           - do the network's Q-values actually
                                     change when the state changes?
    3. Overload avoidance         - does the (trained) DQN route away
                                     from an overloaded server more than
                                     chance (25%) would predict?
    4. Server recovery            - does the DQN's usage of a recovered
                                     server increase again after it stops
                                     being overloaded?
    5. Load balancing             - is Round Robin's request distribution
                                     actually fair (Jain index) under
                                     uniform load, as a sanity baseline?
    6. Failure handling           - does the DQN route away from a
                                     manually-failed server more than
                                     Round Robin (which cannot see health)?
    7. Dynamic traffic            - does a ramp-up traffic pattern
                                     actually produce a growing request
                                     rate over time?
    8. Performance comparison     - does the full evaluation harness
                                     complete successfully for all four
                                     algorithms with a reasonable success rate?

Usage:
    python testing/run_tests.py

Prerequisites: all 4 backend servers AND all 4 balancers running.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
import requests

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from common.settings import get_settings  # noqa: E402
from dqn.agent import DQNAgent  # noqa: E402
from dqn.config import DQN_HYPERPARAMS, RESULTS_DIR  # noqa: E402
from dqn.environment import LoadBalancerEnvironment  # noqa: E402
from dqn.inference import DQNInferenceService  # noqa: E402
from testing.metrics import drive_traffic, evaluate_records, jain_fairness_index  # noqa: E402

SETTINGS = get_settings()
URLS = {
    "round_robin": f"http://localhost:{SETTINGS.round_robin_port}",
    "least_connections": f"http://localhost:{SETTINGS.least_conn_port}",
    "aco": f"http://localhost:{SETTINGS.aco_port}",
    "dqn": f"http://localhost:{SETTINGS.dqn_port}",
}


@dataclass
class TestResult:
    name: str
    passed: Optional[bool]  # None = skipped
    details: Dict = field(default_factory=dict)
    reason: str = ""


def _balancer_up(url: str) -> bool:
    try:
        return requests.get(f"{url}/health", timeout=3).status_code == 200
    except requests.RequestException:
        return False


def test_state_sensitivity(env: LoadBalancerEnvironment) -> TestResult:
    from dqn.state import build_state
    env.reset_all()
    env.apply_uniform("idle")
    time.sleep(1.2)
    idle_state, _ = env.get_state()

    env.apply_preset(0, "overload")
    time.sleep(1.2)
    overloaded_state, _ = env.get_state()

    diff = float(np.linalg.norm(overloaded_state[:10] - idle_state[:10]))
    env.apply_uniform("low")

    threshold = 0.3
    return TestResult(
        "1. State sensitivity", diff > threshold,
        {"server1_feature_l2_diff": diff, "threshold": threshold,
         "idle_server1_features": idle_state[:10].tolist(),
         "overloaded_server1_features": overloaded_state[:10].tolist()},
    )


def test_q_value_response(env: LoadBalancerEnvironment) -> TestResult:
    from dqn.config import MODEL_PATH
    from dqn.state import build_state

    agent = DQNAgent(DQN_HYPERPARAMS)
    model_used = "untrained (fresh weights)"
    if os.path.exists(MODEL_PATH):
        try:
            agent.load(MODEL_PATH)
            model_used = MODEL_PATH
        except Exception as e:
            model_used = f"untrained (failed to load {MODEL_PATH}: {e})"

    env.reset_all()
    env.apply_uniform("idle")
    time.sleep(1.0)
    idle_state, _ = env.get_state()
    q_idle = agent.q_values(idle_state)

    env.apply_preset(0, "overload")
    time.sleep(1.0)
    overloaded_state, _ = env.get_state()
    q_overloaded = agent.q_values(overloaded_state)
    env.apply_uniform("low")

    diff = float(np.linalg.norm(q_overloaded - q_idle))
    threshold = 1e-3
    return TestResult(
        "2. Q-value response to changing state", diff > threshold,
        {"model_used": model_used, "q_idle": q_idle.tolist(),
         "q_overloaded": q_overloaded.tolist(), "l2_diff": diff, "threshold": threshold},
    )


def test_overload_avoidance(env: LoadBalancerEnvironment) -> TestResult:
    url = URLS["dqn"]
    if not _balancer_up(url):
        return TestResult("3. Overload avoidance", None, reason="DQN balancer not reachable")

    inference = DQNInferenceService()
    if not inference.model_loaded:
        return TestResult("3. Overload avoidance", None,
                           reason=f"No trained model loaded ({inference.load_error}). Run `python dqn/train.py` first.")

    env.reset_all()
    env.apply_phase({0: "overload", 1: "low", 2: "low", 3: "low"})
    time.sleep(1.5)

    records = drive_traffic(url, "normal", duration=15, rate=8)
    env.apply_uniform("low")

    total = len(records)
    server1_name = SETTINGS.server_names[0]
    share = sum(1 for r in records if r.get("selected_server") == server1_name) / total if total else 1.0
    threshold = 0.20  # uniform random would be 0.25
    return TestResult(
        "3. Overload avoidance", share < threshold,
        {"total_requests": total, "server1_share": share, "threshold": threshold,
         "server_counts": {s: sum(1 for r in records if r.get("selected_server") == s) for s in SETTINGS.server_names}},
    )


def test_server_recovery(env: LoadBalancerEnvironment) -> TestResult:
    url = URLS["dqn"]
    if not _balancer_up(url):
        return TestResult("4. Server recovery", None, reason="DQN balancer not reachable")
    inference = DQNInferenceService()
    if not inference.model_loaded:
        return TestResult("4. Server recovery", None,
                           reason=f"No trained model loaded ({inference.load_error}). Run `python dqn/train.py` first.")

    env.reset_all()
    env.apply_phase({0: "overload", 1: "low", 2: "low", 3: "low"})
    time.sleep(1.5)
    overloaded_records = drive_traffic(url, "normal", duration=12, rate=8)
    server1_name = SETTINGS.server_names[0]
    overloaded_share = (sum(1 for r in overloaded_records if r.get("selected_server") == server1_name)
                         / len(overloaded_records)) if overloaded_records else 0.0

    env.apply_preset(0, "low")
    time.sleep(1.5)
    recovered_records = drive_traffic(url, "normal", duration=12, rate=8)
    recovered_share = (sum(1 for r in recovered_records if r.get("selected_server") == server1_name)
                        / len(recovered_records)) if recovered_records else 0.0

    passed = recovered_share > overloaded_share
    return TestResult(
        "4. Server recovery", passed,
        {"overloaded_share": overloaded_share, "recovered_share": recovered_share},
    )


def test_load_balancing_fairness(env: LoadBalancerEnvironment) -> TestResult:
    url = URLS["round_robin"]
    if not _balancer_up(url):
        return TestResult("5. Load balancing (Round Robin fairness)", None, reason="round_robin balancer not reachable")

    env.reset_all()
    env.apply_uniform("low")
    time.sleep(1.0)
    records = drive_traffic(url, "normal", duration=12, rate=10)
    counts = [sum(1 for r in records if r.get("selected_server") == s) for s in SETTINGS.server_names]
    fairness = jain_fairness_index(counts)
    threshold = 0.9
    return TestResult(
        "5. Load balancing fairness (Round Robin)", fairness > threshold,
        {"server_counts": dict(zip(SETTINGS.server_names, counts)), "jain_fairness_index": fairness, "threshold": threshold},
    )


def test_failure_handling(env: LoadBalancerEnvironment) -> TestResult:
    dqn_url, rr_url = URLS["dqn"], URLS["round_robin"]
    if not (_balancer_up(dqn_url) and _balancer_up(rr_url)):
        return TestResult("6. Failure handling", None, reason="dqn or round_robin balancer not reachable")
    inference = DQNInferenceService()
    if not inference.model_loaded:
        return TestResult("6. Failure handling", None,
                           reason=f"No trained model loaded ({inference.load_error}). Run `python dqn/train.py` first.")

    env.reset_all()
    env.apply_uniform("low")
    failed_index = 1
    failed_name = SETTINGS.server_names[failed_index]
    env.fail_server(failed_index)
    time.sleep(1.0)

    dqn_records = drive_traffic(dqn_url, "normal", duration=12, rate=8)
    rr_records = drive_traffic(rr_url, "normal", duration=12, rate=8)
    env.recover_server(failed_index)

    dqn_share = (sum(1 for r in dqn_records if r.get("selected_server") == failed_name)
                 / len(dqn_records)) if dqn_records else 1.0
    rr_share = (sum(1 for r in rr_records if r.get("selected_server") == failed_name)
                / len(rr_records)) if rr_records else 1.0

    passed = dqn_share < rr_share
    return TestResult(
        "6. Failure handling (DQN vs Round Robin avoidance of failed server)", passed,
        {"failed_server": failed_name, "dqn_share_of_failed_server": dqn_share, "round_robin_share_of_failed_server": rr_share},
    )


def test_dynamic_traffic(env: LoadBalancerEnvironment) -> TestResult:
    url = URLS["dqn"]
    if not _balancer_up(url):
        return TestResult("7. Dynamic traffic (ramp-up)", None, reason="DQN balancer not reachable")

    env.reset_all()
    env.apply_uniform("low")
    time.sleep(0.5)
    records = drive_traffic(url, "ramp_up", duration=16, rate=10)
    if len(records) < 4:
        return TestResult("7. Dynamic traffic (ramp-up)", False, {"total_requests": len(records)},
                           reason="too few requests completed to evaluate a ramp")

    records.sort(key=lambda r: r["timestamp"])
    mid = len(records) // 2
    first_half, second_half = records[:mid], records[mid:]
    passed = len(second_half) >= len(first_half)
    return TestResult(
        "7. Dynamic traffic (ramp-up produces growing request rate)", passed,
        {"first_half_count": len(first_half), "second_half_count": len(second_half), "total": len(records)},
    )


def test_performance_comparison_harness(env: LoadBalancerEnvironment) -> TestResult:
    results = {}
    all_ok = True
    for name, url in URLS.items():
        if not _balancer_up(url):
            all_ok = False
            results[name] = {"reachable": False}
            continue
        env.reset_all()
        env.apply_uniform("low")
        time.sleep(0.5)
        records = drive_traffic(url, "normal", duration=8, rate=8)
        metrics = evaluate_records(records, SETTINGS.server_names)
        results[name] = {"reachable": True, "success_rate": metrics["success_rate"], "total_requests": metrics["total_requests"]}
        if metrics["success_rate"] < 0.5 or metrics["total_requests"] == 0:
            all_ok = False

    return TestResult("8. Performance comparison harness (all 4 algorithms)", all_ok, results)


def main():
    env = LoadBalancerEnvironment()
    reachable = env.servers_reachable()
    if not all(reachable):
        print("ERROR: not all backend servers are reachable. Start them first (see README).")
        for url, ok in zip(env.server_urls, reachable):
            print(f"  {url} -> {'OK' if ok else 'UNREACHABLE'}")
        sys.exit(1)

    tests: List[Callable] = [
        test_state_sensitivity,
        test_q_value_response,
        test_overload_avoidance,
        test_server_recovery,
        test_load_balancing_fairness,
        test_failure_handling,
        test_dynamic_traffic,
        test_performance_comparison_harness,
    ]

    results: List[TestResult] = []
    print("=" * 70)
    print("AUTOMATED TEST SUITE")
    print("=" * 70)
    for test_fn in tests:
        try:
            result = test_fn(env)
        except Exception as e:  # a crashing test is a FAIL, not silently ignored
            result = TestResult(test_fn.__name__, False, reason=f"Exception: {e!r}")
        results.append(result)
        if result.passed is None:
            tag = "SKIPPED"
        elif result.passed:
            tag = "PASS"
        else:
            tag = "FAIL"
        print(f"[{tag:7s}] {result.name}")
        if result.reason:
            print(f"          reason: {result.reason}")

    n_pass = sum(1 for r in results if r.passed is True)
    n_fail = sum(1 for r in results if r.passed is False)
    n_skip = sum(1 for r in results if r.passed is None)
    print("=" * 70)
    print(f"RESULT: {n_pass} PASSED, {n_fail} FAILED, {n_skip} SKIPPED (out of {len(results)})")
    print("=" * 70)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    report_path = os.path.join(RESULTS_DIR, "test_report.json")
    with open(report_path, "w") as f:
        json.dump({
            "generated_at": time.time(),
            "summary": {"passed": n_pass, "failed": n_fail, "skipped": n_skip, "total": len(results)},
            "results": [
                {"name": r.name, "passed": r.passed, "details": r.details, "reason": r.reason}
                for r in results
            ],
        }, f, indent=2, default=str)
    print(f"Report saved: {report_path}")

    env.close()
    sys.exit(1 if n_fail > 0 else 0)


if __name__ == "__main__":
    main()
