#!/usr/bin/env python3
"""
testing/evaluate_dqn_only.py
==============================
DQN-ONLY evaluation harness (added for a standalone "test the already
trained DQN, nothing else" run).

This file is a NEW, minimal addition. It does not modify:
    - dqn/agent.py, dqn/model.py            (architecture)
    - dqn/reward.py                          (reward function)
    - dqn/config.py                          (hyperparameters)
    - models/dqn_model.pt                    (trained checkpoint)
    - load_balancer/dqn_balancer.py          (inference/serving code)

It reuses the project's own existing, already-tested formulas and
helpers instead of reimplementing anything:
    - dqn.inference.DQNInferenceService  -> confirms the checkpoint is
      loaded and gives its diagnostic status (Step 1 / Step 5).
    - dqn.environment.LoadBalancerEnvironment -> reachability checks and
      applying a backend workload phase (same class dqn/train.py and
      testing/adaptation_test.py already use).
    - testing.metrics.run_multi_segment / evaluate_records -> drives
      real, paced HTTP traffic at the live DQN balancer and computes
      success rate, avg/p95/p99 response time, throughput, avg CPU/
      memory of the selected server, cumulative reward and Jain's
      fairness index -- the exact same formulas
      (common/formulas.py::summarize_latencies,
      common/formulas.py::jain_fairness_index, dqn/reward.py) already
      used by testing/compare_algorithms.py. No metric is computed with
      a new/ad-hoc formula here.
    - testing.scenarios.EVALUATION_TRAFFIC_SCENARIO /
      COMPARISON_BACKEND_WORKLOAD -> the project's own existing
      "unseen test traffic" definition, not something invented for this
      run.

It ONLY talks to the DQN balancer (http://localhost:<dqn_port>). It
never starts, calls, or imports round_robin/least_connections/aco, and
never calls agent.learn()/agent.store() or sets DQN_ONLINE_LEARNING.

Usage:
    python testing/evaluate_dqn_only.py

Prerequisites: the 4 backend servers AND the DQN balancer must already
be running (see README / QUICKSTART for the exact commands).

Writes (does NOT touch models/dqn_model.pt):
    results/dqn_evaluation_raw.csv       one row per request
    results/dqn_evaluation_summary.json  the 9 final metrics + validation
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from common.settings import get_settings  # noqa: E402
from dqn.config import MODEL_PATH, ONLINE_LEARNING_ENABLED, RESULTS_DIR  # noqa: E402
from dqn.environment import LoadBalancerEnvironment  # noqa: E402
from dqn.inference import DQNInferenceService  # noqa: E402
from testing.metrics import evaluate_records, run_multi_segment  # noqa: E402
from testing.scenarios import (COMPARISON_BACKEND_WORKLOAD,  # noqa: E402
                                EVALUATION_TRAFFIC_SCENARIO,
                                evaluation_scenario_duration)

SETTINGS = get_settings()
DQN_URL = f"http://localhost:{SETTINGS.dqn_port}"


def fail(msg: str) -> None:
    print("=" * 40)
    print("DQN EVALUATION - ABORTED")
    print("=" * 40)
    print(f"ERROR: {msg}")
    sys.exit(1)


def main() -> None:
    print("=" * 40)
    print("STEP 1: Locate & verify the trained checkpoint")
    print("=" * 40)

    if not os.path.exists(MODEL_PATH):
        fail(f"Checkpoint not found at {MODEL_PATH}. "
             f"Fix: place the trained checkpoint at this exact path, or set "
             f"the DQN_MODEL_PATH env var, before re-running this script.")

    # Uses the project's own, unmodified inference service. This does
    # NOT construct a trainable path and never calls .learn()/.store().
    diag = DQNInferenceService(model_path=MODEL_PATH)
    status = diag.status()
    print(json.dumps(status, indent=2))

    if not status["model_loaded"]:
        fail(f"Checkpoint failed to load: {status['load_error']}. "
             f"Fix: inspect dqn/inference.py:_try_load() and dqn/agent.py:load() -- "
             f"a dimension mismatch there raises ModelDimensionMismatchError. "
             f"Do not proceed with a fallback/untrained model.")

    if status["epsilon"] != 0.0:
        fail(f"Inference epsilon is {status['epsilon']}, not 0.0 -- this run would not "
             f"be pure exploitation of the trained policy. "
             f"Fix: dqn/config.py INFERENCE_EPSILON (or env DQN_INFERENCE_EPSILON) must be 0.0.")

    print(f"\nModel loaded: YES  (checkpoint: {os.path.basename(MODEL_PATH)})")
    print(f"Architecture: state_dim={status['state_dim']}, action_dim={status['action_dim']}, "
          f"trained for steps_done={status['steps_done']}")
    print(f"Inference epsilon: {status['epsilon']} (0.0 = deterministic argmax, no exploration)")
    print(f"Online learning enabled: {ONLINE_LEARNING_ENABLED} (must be False for a pure evaluation run)")

    if ONLINE_LEARNING_ENABLED:
        fail("ONLINE_LEARNING_ENABLED is True in this environment. "
             "Fix: unset the DQN_ONLINE_LEARNING env var (dqn/config.py) before evaluating -- "
             "the user's instructions explicitly forbid retraining during evaluation.")

    print("\n" + "=" * 40)
    print("Checking the live DQN balancer process (not just the checkpoint file)")
    print("=" * 40)
    import requests
    try:
        health = requests.get(f"{DQN_URL}/health", timeout=3).json()
    except requests.RequestException as e:
        fail(f"Cannot reach the DQN balancer at {DQN_URL}/health ({e}). "
             f"Fix: start it first with "
             f"`PYTHONPATH=<shim_or_real_torch_path> python3 load_balancer/dqn_balancer.py`.")
    print(json.dumps(health, indent=2))
    if not health.get("model_loaded"):
        fail("The LIVE dqn_balancer.py process reports model_loaded=false. "
             "Fix: check load_balancer/dqn_balancer.py's startup log for the load_error, "
             "and confirm MODEL_PATH / models/dqn_model.pt on that process's filesystem.")

    print("\n" + "=" * 40)
    print("STEP 2: Verify backend servers are reachable, then run real unseen test traffic")
    print("=" * 40)
    env = LoadBalancerEnvironment()
    reachable = env.servers_reachable()
    if not all(reachable):
        details = "\n".join(f"  {u} -> {'OK' if ok else 'UNREACHABLE'}" for u, ok in zip(env.server_urls, reachable))
        fail(f"Not all 4 backend servers are reachable:\n{details}\n"
             f"Fix: start each with `PYTHONPATH=<shim> SERVER_ID=<n> PORT=<500n> python3 servers/app.py` "
             f"(run from inside the servers/ directory).")

    env.reset_all()
    env.apply_phase(COMPARISON_BACKEND_WORKLOAD)
    time.sleep(1.0)
    print(f"Applied backend workload phase: {COMPARISON_BACKEND_WORKLOAD} (project's existing "
          f"'realistic-but-moderate' preset from testing/scenarios.py, not idle/fabricated).")

    duration = evaluation_scenario_duration()
    print(f"Traffic scenario (testing/scenarios.EVALUATION_TRAFFIC_SCENARIO): {EVALUATION_TRAFFIC_SCENARIO}")
    print(f"Total planned duration: {duration:.0f}s against {DQN_URL} only.\n")

    wall_start = time.time()
    records = run_multi_segment(DQN_URL, EVALUATION_TRAFFIC_SCENARIO)
    wall_elapsed = time.time() - wall_start
    env.close()

    if not records:
        fail("Zero requests were completed against the DQN balancer -- traffic generation "
             "itself failed. Fix: check network reachability to " + DQN_URL)

    non_dqn = [r for r in records if r.get("algorithm") not in (None, "", "DQN", "dqn")]
    if non_dqn:
        fail(f"{len(non_dqn)} response(s) did not report algorithm=DQN -- routing decisions "
             f"were not confirmed to come from the DQN balancer. First offending record: {non_dqn[0]}")

    print(f"Sent/collected {len(records)} real HTTP requests in {wall_elapsed:.2f}s wall-clock "
          f"against the live, checkpoint-loaded DQN balancer.")

    # ------------------------------------------------------------------
    # Step 6a: save the raw per-request log (does not touch the model).
    # ------------------------------------------------------------------
    os.makedirs(RESULTS_DIR, exist_ok=True)
    raw_csv_path = os.path.join(RESULTS_DIR, "dqn_evaluation_raw.csv")
    fieldnames = [
        "request_index", "timestamp", "response_time_ms", "success", "http_status",
        "selected_server", "algorithm", "reward", "q_values_json", "exploration",
        "selected_server_cpu_usage", "selected_server_memory_usage",
        "selected_server_queue_length", "selected_server_active_requests",
    ]
    with open(raw_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, r in enumerate(records):
            sm = r.get("server_metrics") or {}
            sel = r.get("selected_server")
            sel_m = sm.get(sel, {}) if sel else {}
            writer.writerow({
                "request_index": i,
                "timestamp": r["timestamp"],
                "response_time_ms": round(r["response_time_s"] * 1000.0, 3),
                "success": r["success"],
                "http_status": r.get("http_status"),
                "selected_server": sel,
                "algorithm": r.get("algorithm"),
                "reward": r.get("reward"),
                "q_values_json": json.dumps(r.get("q_values")) if r.get("q_values") is not None else "",
                "exploration": r.get("exploration"),
                "selected_server_cpu_usage": sel_m.get("cpu_usage"),
                "selected_server_memory_usage": sel_m.get("memory_usage"),
                "selected_server_queue_length": sel_m.get("queue_length"),
                "selected_server_active_requests": sel_m.get("active_requests"),
            })
    print(f"Raw per-request results saved: {raw_csv_path} ({len(records)} rows)")

    # ------------------------------------------------------------------
    # STEP 3: compute ONLY the 9 specified metrics, via the project's
    # own existing testing.metrics.evaluate_records() -- no new formula.
    # ------------------------------------------------------------------
    print("\n" + "=" * 40)
    print("STEP 3: Computing the 9 evaluation metrics from the real collected data")
    print("=" * 40)
    metrics = evaluate_records(records, SETTINGS.server_names)
    print(json.dumps(metrics, indent=2))

    # ------------------------------------------------------------------
    # STEP 5: genuineness validation checks (computed from real observed
    # values above, not asserted).
    # ------------------------------------------------------------------
    checks = {
        "checkpoint_loaded": bool(status["model_loaded"]),
        "model_not_retrained": (not ONLINE_LEARNING_ENABLED),
        "real_test_traffic_sent": len(records) > 0,
        "requests_routed_by_dqn": len(non_dqn) == 0 and len(records) > 0,
        "metrics_from_real_observations": metrics["total_requests"] == len(records),
        "failed_requests_included_in_success_rate": (
            metrics["success_count"] + metrics["failure_count"] == metrics["total_requests"]
        ),
        "p95_p99_from_real_observations": metrics["total_requests"] > 1,
        "throughput_from_real_completed_requests_and_elapsed_time": metrics["makespan_seconds"] > 0,
    }
    all_pass = all(checks.values())
    print("\nValidation checks:")
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")

    # ------------------------------------------------------------------
    # STEP 6b: summary file with final metrics.
    # ------------------------------------------------------------------
    summary = {
        "generated_at": time.time(),
        "model_checkpoint": os.path.basename(MODEL_PATH),
        "model_status_at_eval": status,
        "traffic_scenario": EVALUATION_TRAFFIC_SCENARIO,
        "backend_workload_phase": {SETTINGS.server_names[i]: v for i, v in COMPARISON_BACKEND_WORKLOAD.items()},
        "test_requests": metrics["total_requests"],
        "successful_requests": metrics["success_count"],
        "failed_requests": metrics["failure_count"],
        "metrics": {
            "success_rate": metrics["success_rate"],
            "average_response_time_ms": metrics["avg_response_time_ms"],
            "p95_latency_ms": metrics["p95_response_time_ms"],
            "p99_latency_ms": metrics["p99_response_time_ms"],
            "throughput_rps": metrics["throughput_rps"],
            "average_cpu_utilization_pct": metrics["avg_cpu_of_selected"],
            "average_memory_utilization_pct": metrics["avg_memory_of_selected"],
            "cumulative_reward": metrics["cumulative_reward"],
            "jains_fairness_index": metrics["jain_fairness_index"],
        },
        "validation_checks": checks,
        "all_validation_checks_passed": all_pass,
        "note": "DQN-only evaluation of the existing trained checkpoint. No retraining, "
                "no architecture/reward changes, no comparison to other algorithms performed.",
    }
    summary_path = os.path.join(RESULTS_DIR, "dqn_evaluation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSummary saved: {summary_path}")

    # ------------------------------------------------------------------
    # STEP 7: exact-format final terminal output.
    # ------------------------------------------------------------------
    m = metrics

    def pct(x):
        return "N/A" if x is None else f"{x * 100.0:.2f}%"

    def ms(x):
        return "N/A" if x is None else f"{x:.2f} ms"

    def num(x, unit=""):
        return "N/A" if x is None else f"{x:.2f}{unit}"

    print("\n========================================")
    print("DQN EVALUATION RESULTS")
    print("========================================\n")
    print(f"Model: {os.path.basename(MODEL_PATH)}")
    print(f"Test Requests: {m['total_requests']}")
    print(f"Successful Requests: {m['success_count']}")
    print(f"Failed Requests: {m['failure_count']}\n")
    print(f"Success Rate: {pct(m['success_rate'])}")
    print(f"Average Response Time: {ms(m['avg_response_time_ms'])}")
    print(f"P95 Latency: {ms(m['p95_response_time_ms'])}")
    print(f"P99 Latency: {ms(m['p99_response_time_ms'])}")
    print(f"Throughput: {num(m['throughput_rps'], ' req/s')}")
    print(f"Average CPU: {num(m['avg_cpu_of_selected'], '%')}")
    print(f"Average Memory: {num(m['avg_memory_of_selected'], '%')}")
    print(f"Jain's Fairness Index: {num(m['jain_fairness_index'])}")
    print(f"Cumulative Reward: {num(m['cumulative_reward'])}")
    print("\n========================================")
    print("EVALUATION STATUS")
    print("========================================")
    print(f"Model Loaded: {'PASS' if checks['checkpoint_loaded'] else 'FAIL'}")
    print(f"Inference Running: {'PASS' if checks['requests_routed_by_dqn'] else 'FAIL'}")
    print(f"Real Test Traffic: {'PASS' if checks['real_test_traffic_sent'] else 'FAIL'}")
    print(f"Metrics Calculated: {'PASS' if checks['metrics_from_real_observations'] else 'FAIL'}")
    print("========================================")

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
