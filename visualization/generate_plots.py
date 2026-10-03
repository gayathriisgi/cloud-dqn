#!/usr/bin/env python3
"""
visualization/generate_plots.py
=================================
Generates every plot referenced in the README from REAL, already-saved
experiment data:

    logs/training_history.csv            (dqn/train.py)
    results/adaptation_summary.json      (testing/adaptation_test.py)
    results/adaptation_records.json      (testing/adaptation_test.py)
    results/{round_robin,least_connections,aco,dqn}_results.json
    results/comparison_results.json      (testing/compare_algorithms.py)

Every function here is defensive: if a data source doesn't exist yet
(e.g. you haven't run training or the adaptation experiment), it
prints a one-line notice and skips that figure instead of crashing or
inventing numbers. Run the relevant script first (see README) to
produce the real data, then re-run this script.

Usage:
    python visualization/generate_plots.py
"""

from __future__ import annotations

import csv
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from common.formulas import moving_average  # noqa: E402
from common.settings import get_settings  # noqa: E402
from dqn.config import RESULTS_DIR, TRAINING_LOG_DIR  # noqa: E402

SETTINGS = get_settings()
os.makedirs(RESULTS_DIR, exist_ok=True)

PALETTE = {
    "server1": "#4C72B0", "server2": "#DD8452", "server3": "#55A868", "server4": "#C44E52",
}


def _save(fig, name: str) -> None:
    path = os.path.join(RESULTS_DIR, name)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def _skip(reason: str) -> None:
    print(f"Skipped: {reason}")


# ----------------------------------------------------------------------
# Training plots (Section 14)
# ----------------------------------------------------------------------
def load_training_history():
    path = os.path.join(TRAINING_LOG_DIR, "training_history.csv")
    if not os.path.exists(path):
        return None
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def plot_training(rows) -> None:
    if not rows:
        _skip("logs/training_history.csv not found - run `python dqn/train.py` first")
        return

    episodes = [int(r["episode"]) for r in rows]

    def col(name):
        return [float(r[name]) for r in rows]

    window = max(3, len(episodes) // 15)

    # Reward
    fig, ax = plt.subplots(figsize=(10, 5))
    reward = col("total_reward")
    ax.plot(episodes, reward, color="#b0c4de", linewidth=0.8, alpha=0.6, label="Raw reward")
    ax.plot(episodes, moving_average(reward, window), color="#1f77b4", linewidth=2.2, label="Smoothed reward")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_title("DQN Training - Total Reward per Episode")
    ax.set_xlabel("Episode"); ax.set_ylabel("Total reward"); ax.legend(); ax.grid(alpha=0.3)
    _save(fig, "training_reward.png")

    # Loss
    fig, ax = plt.subplots(figsize=(10, 5))
    loss = col("loss")
    ax.plot(episodes, loss, color="#e07b39", linewidth=1.2)
    ax.set_title("DQN Training - Loss per Episode (Huber loss, mean over episode)")
    ax.set_xlabel("Episode"); ax.set_ylabel("Loss"); ax.grid(alpha=0.3)
    _save(fig, "training_loss.png")

    # Response time
    fig, ax = plt.subplots(figsize=(10, 5))
    rt = col("avg_response_ms")
    p95 = col("p95_latency_ms")
    ax.plot(episodes, rt, color="#2ca02c", linewidth=1.5, label="Avg response time (ms)")
    ax.plot(episodes, p95, color="#d62728", linewidth=1.2, linestyle="--", label="P95 latency (ms)")
    ax.set_title("DQN Training - Response Time per Episode")
    ax.set_xlabel("Episode"); ax.set_ylabel("ms"); ax.legend(); ax.grid(alpha=0.3)
    _save(fig, "training_response_time.png")

    # Throughput
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(episodes, col("throughput"), color="#9467bd", linewidth=1.5)
    ax.set_title("DQN Training - System Throughput per Episode")
    ax.set_xlabel("Episode"); ax.set_ylabel("Requests/sec (sum over servers)"); ax.grid(alpha=0.3)
    _save(fig, "training_throughput.png")

    # Epsilon
    fig, ax = plt.subplots(figsize=(10, 4))
    eps = col("epsilon")
    ax.plot(episodes, eps, color="#2ca02c", linewidth=2)
    ax.fill_between(episodes, eps, alpha=0.15, color="#2ca02c")
    ax.set_title("Epsilon Decay (Exploration -> Exploitation)")
    ax.set_xlabel("Episode"); ax.set_ylabel("epsilon"); ax.grid(alpha=0.3)
    _save(fig, "training_epsilon.png")

    # Load variance + fairness
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(episodes, col("load_variance"), color="#8c564b")
    axes[0].set_title("Load Variance per Episode (lower = more balanced)")
    axes[0].set_xlabel("Episode"); axes[0].set_ylabel("Variance of load_pressure"); axes[0].grid(alpha=0.3)
    axes[1].plot(episodes, col("fairness"), color="#17becf")
    axes[1].set_title("Jain Fairness Index per Episode (by requests routed)")
    axes[1].set_xlabel("Episode"); axes[1].set_ylabel("Fairness (0-1)"); axes[1].set_ylim(0, 1.05); axes[1].grid(alpha=0.3)
    _save(fig, "training_load_variance_fairness.png")

    # Q-value statistics
    fig, ax = plt.subplots(figsize=(10, 5))
    q_mean, q_min, q_max = col("q_mean"), col("q_min"), col("q_max")
    ax.plot(episodes, q_mean, color="#1f77b4", label="Mean Q")
    ax.fill_between(episodes, q_min, q_max, alpha=0.15, color="#1f77b4", label="Min-Max Q range")
    ax.set_title("DQN Training - Q-value Statistics per Episode")
    ax.set_xlabel("Episode"); ax.set_ylabel("Q-value"); ax.legend(); ax.grid(alpha=0.3)
    _save(fig, "training_q_values.png")


# ----------------------------------------------------------------------
# Adaptive-behaviour experiment plots (Section 8)
# ----------------------------------------------------------------------
def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def plot_adaptation(records, summary) -> None:
    if not records or not summary:
        _skip("results/adaptation_records.json / adaptation_summary.json not found - "
              "run `python testing/adaptation_test.py` first")
        return

    records = sorted(records, key=lambda r: r["timestamp"])
    t0 = records[0]["timestamp"]
    t_rel = [r["timestamp"] - t0 for r in records]
    phase_bounds = summary["phases"]
    server_names = SETTINGS.server_names

    def phase_lines(ax):
        acc = 0.0
        for p in phase_bounds:
            ax.axvline(acc, color="gray", linestyle=":", linewidth=1)
            ax.text(acc + 0.3, ax.get_ylim()[1] * 0.95, p["name"].replace("_", " "),
                    rotation=90, fontsize=7, va="top", color="dimgray")
            acc += p["duration"]

    # Selected server over time
    fig, ax = plt.subplots(figsize=(13, 4.5))
    server_to_y = {s: i for i, s in enumerate(server_names)}
    ys = [server_to_y.get(r.get("selected_server"), -1) for r in records]
    colors = [PALETTE.get(r.get("selected_server"), "#999999") for r in records]
    ax.scatter(t_rel, ys, c=colors, s=10, alpha=0.7)
    ax.set_yticks(range(len(server_names)))
    ax.set_yticklabels(server_names)
    ax.set_title("Selected Server Over Time - Adaptive Experiment (5 phases)")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Selected server")
    phase_lines(ax)
    _save(fig, "adaptation_selected_server_over_time.png")

    # Q-values over time (only meaningful for DQN target)
    if any(r.get("q_values") for r in records):
        fig, ax = plt.subplots(figsize=(13, 4.5))
        for s in server_names:
            qs = [r["q_values"].get(s) if r.get("q_values") else None for r in records]
            xs = [t for t, q in zip(t_rel, qs) if q is not None]
            ys = [q for q in qs if q is not None]
            if ys:
                ax.plot(xs, moving_average(ys, max(3, len(ys) // 30)), label=s, color=PALETTE.get(s))
        ax.set_title("DQN Q-values Over Time - Adaptive Experiment")
        ax.set_xlabel("Time (s)"); ax.set_ylabel("Q-value (smoothed)"); ax.legend()
        phase_lines(ax)
        _save(fig, "adaptation_q_values_over_time.png")
    else:
        _skip("no q_values present in adaptation_records.json (target was not the DQN balancer)")

    # Response time over time
    fig, ax = plt.subplots(figsize=(13, 4.5))
    rt_ms = [r["response_time_s"] * 1000.0 for r in records]
    ax.plot(t_rel, rt_ms, color="#d62728", linewidth=0.8, alpha=0.5)
    ax.plot(t_rel, moving_average(rt_ms, max(3, len(rt_ms) // 30)), color="#8b0000", linewidth=2)
    ax.set_title("Response Time Over Time - Adaptive Experiment")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Response time (ms)")
    phase_lines(ax)
    _save(fig, "adaptation_response_time_over_time.png")

    # CPU / memory / queue over time (averaged across all 4 servers, from server_metrics)
    def series(field_name):
        out = []
        for r in records:
            sm = r.get("server_metrics")
            if sm:
                vals = [sm[s][field_name] for s in server_names if s in sm and field_name in sm[s]]
                out.append(float(np.mean(vals)) if vals else None)
            else:
                out.append(None)
        return out

    for field_name, ylabel, fname in [
        ("cpu_usage", "CPU usage (%)", "adaptation_cpu_over_time.png"),
        ("memory_usage", "Memory usage (%)", "adaptation_memory_over_time.png"),
        ("queue_length", "Queue length", "adaptation_queue_over_time.png"),
    ]:
        vals = series(field_name)
        xs = [t for t, v in zip(t_rel, vals) if v is not None]
        ys = [v for v in vals if v is not None]
        if not ys:
            _skip(f"no {field_name} data in adaptation_records.json")
            continue
        fig, ax = plt.subplots(figsize=(13, 4.5))
        ax.plot(xs, ys, color="#555555", linewidth=0.8, alpha=0.5)
        ax.plot(xs, moving_average(ys, max(3, len(ys) // 30)), color="#222222", linewidth=2)
        ax.set_title(f"Average {ylabel} Across All Servers Over Time - Adaptive Experiment")
        ax.set_xlabel("Time (s)"); ax.set_ylabel(ylabel)
        phase_lines(ax)
        _save(fig, fname)

    # Reward over time
    if any(r.get("reward") is not None for r in records):
        fig, ax = plt.subplots(figsize=(13, 4.5))
        rewards = [r.get("reward") for r in records]
        xs = [t for t, rw in zip(t_rel, rewards) if rw is not None]
        ys = [rw for rw in rewards if rw is not None]
        ax.plot(xs, ys, color="#1f77b4", linewidth=0.8, alpha=0.4)
        ax.plot(xs, moving_average(ys, max(3, len(ys) // 30)), color="#0b3d91", linewidth=2)
        ax.set_title("Reward Over Time - Adaptive Experiment")
        ax.set_xlabel("Time (s)"); ax.set_ylabel("Reward")
        phase_lines(ax)
        _save(fig, "adaptation_reward_over_time.png")

    # Server selection distribution per phase (stacked bar)
    fig, ax = plt.subplots(figsize=(11, 5))
    phase_names = [p["name"] for p in phase_bounds]
    bottoms = np.zeros(len(phase_names))
    for s in server_names:
        shares = [p["metrics"]["server_request_counts"].get(s, 0) / max(1, p["metrics"]["total_requests"])
                  for p in phase_bounds]
        ax.bar(phase_names, shares, bottom=bottoms, label=s, color=PALETTE.get(s))
        bottoms += np.array(shares)
    ax.set_title("Server Selection Distribution per Phase")
    ax.set_ylabel("Share of requests"); ax.legend()
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    _save(fig, "adaptation_server_selection_distribution.png")


# ----------------------------------------------------------------------
# Algorithm comparison plots (Section 9)
# ----------------------------------------------------------------------
def plot_comparison(comparison) -> None:
    if not comparison or not comparison.get("results"):
        _skip("results/comparison_results.json not found - run `python testing/compare_algorithms.py` first")
        return

    results = comparison["results"]
    algos = list(results.keys())
    labels = [a.replace("_", " ").title() for a in algos]
    colors = ["#aec7e8", "#ffbb78", "#98df8a", "#1f77b4"]

    def metric(name):
        return [results[a].get(name, 0) or 0 for a in algos]

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    axes[0, 0].bar(labels, metric("avg_response_time_ms"), color=colors)
    axes[0, 0].set_title("Average Response Time (ms) - lower is better")
    axes[0, 0].set_ylabel("ms")

    axes[0, 1].bar(labels, metric("throughput_rps"), color=colors)
    axes[0, 1].set_title("Throughput (req/s) - higher is better")
    axes[0, 1].set_ylabel("requests/sec")

    axes[1, 0].bar(labels, metric("success_rate"), color=colors)
    axes[1, 0].set_title("Success Rate - higher is better")
    axes[1, 0].set_ylabel("fraction"); axes[1, 0].set_ylim(0, 1.05)

    axes[1, 1].bar(labels, metric("jain_fairness_index"), color=colors)
    axes[1, 1].set_title("Jain Fairness Index - higher is better")
    axes[1, 1].set_ylabel("fairness (0-1)"); axes[1, 1].set_ylim(0, 1.05)

    for ax in axes.flat:
        ax.grid(alpha=0.3)
        plt.setp(ax.get_xticklabels(), rotation=15, ha="right")

    fig.suptitle("Load Balancing Algorithm Comparison (real measured data)", fontsize=14, fontweight="bold")
    _save(fig, "comparison_algorithms.png")

    # P95/P99 latency
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(algos))
    width = 0.35
    ax.bar(x - width / 2, metric("p95_response_time_ms"), width, label="P95", color="#4C72B0")
    ax.bar(x + width / 2, metric("p99_response_time_ms"), width, label="P99", color="#C44E52")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_title("P95 / P99 Response Time by Algorithm")
    ax.set_ylabel("ms"); ax.legend(); ax.grid(alpha=0.3)
    _save(fig, "comparison_latency_percentiles.png")


def main():
    rows = load_training_history()
    plot_training(rows)

    records = load_json(os.path.join(RESULTS_DIR, "adaptation_records.json"))
    summary = load_json(os.path.join(RESULTS_DIR, "adaptation_summary.json"))
    plot_adaptation(records, summary)

    comparison = load_json(os.path.join(RESULTS_DIR, "comparison_results.json"))
    plot_comparison(comparison)

    print("\nDone. All available plots were saved to the results/ directory.")


if __name__ == "__main__":
    main()
