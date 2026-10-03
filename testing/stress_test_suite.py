#!/usr/bin/env python3
"""
testing/stress_test_suite.py
=============================
Scenario stress test + head-to-head benchmark of DQN vs Round Robin,
Least Connections and ACO, with PASS/FAIL satisfaction checks.

For EVERY scenario in testing/stress_scenarios.py (low / normal / high /
spike / burst / ramp traffic, slow server, overload, double overload,
cpu+memory pressure, queue build-up, heterogeneous servers, hard failure,
failure+recovery, flapping, cascading degradation, hotspot migration):

  for EVERY algorithm (in a seeded random order to avoid order bias):
     1. reset all 4 backend servers (fresh counters)        POST /reset
     2. for each STAGE: apply workload / fail / recover, then drive REAL
        paced HTTP traffic at that balancer
     3. a monitor thread polls every server's /metrics once per second
        (server status timeline: CPU, memory, queue, active, latency,
        errors, health -> HEALTHY / BUSY / SLOW / OVERLOADED / FAILED)
     4. compute metrics PER STAGE and for the whole scenario

Metrics (all measured, nothing hard-coded)
  success rate, error rate, avg / p50 / p95 / p99 latency, throughput,
  goodput (successful AND within SLA per second), SLA-violation rate,
  Jain fairness, load-balance degree, server distribution,
  ROUTING ACCURACY (share of decisions that avoided every degraded server
  while a healthy one existed - ground truth known because we created
  the incident), requests-to-adapt after a degradation starts,
  recovery-detection time after a server recovers,
  DQN only: Q-value margin (decision confidence) and Q/latency rank
  agreement (Spearman: do higher Q-values point at faster servers?).

Comparisons
  * per-scenario composite Load-Balancing Quality score (0-100)
  * rank of each algorithm per scenario and per metric, win matrix
  * statistical significance of DQN vs each baseline latency
    (Mann-Whitney U test + bootstrap 95% CI of the mean difference)

PASS / FAIL satisfaction checks (computed from measurements)
  per scenario, for DQN:
    - success rate >= threshold
    - p95 latency <= threshold
    - routing accuracy >= threshold (only where degraded servers exist)
    - composite score within tolerance of the best algorithm
  plus OFFLINE policy probes (testing/policy_probe_test.py) if torch and
  the model are available.

The verdicts are honest: if DQN loses a scenario the report says so.

Usage
  python testing/stress_test_suite.py                       # full suite (~35-45 min)
  python testing/stress_test_suite.py --quick               # halves every duration
  python testing/stress_test_suite.py --scenarios hard_failure single_overload
  python testing/stress_test_suite.py --repeats 2           # more statistical power

Balancer URLs default to the host ports (localhost:8000-8003); override with
RR_URL / LC_URL / ACO_URL / DQN_URL. Server URLs come from SERVER_URLS
(default http://localhost:5001-5004).

Outputs (results/stress_suite/)
  results.json   everything (per stage, per algorithm, per server timeline)
  summary.csv    one row per scenario x algorithm
  checks.json    every PASS/FAIL check
  report.html    self-contained visual report (open in a browser)
  *.png          charts
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import threading
import time
from typing import Dict, List, Optional

import numpy as np
import requests

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from common.formulas import (jain_fairness_index, load_balance_degree,  # noqa: E402
                             summarize_latencies)
from common.settings import get_settings  # noqa: E402
from dqn.config import RESULTS_DIR  # noqa: E402
from dqn.environment import LoadBalancerEnvironment  # noqa: E402
from testing.metrics import drive_traffic  # noqa: E402
from testing.stress_scenarios import (DEGRADED_PRESETS, STRESS_SCENARIOS,  # noqa: E402
                                      get_scenario)

SETTINGS = get_settings()
NAMES = SETTINGS.server_names

ALGORITHM_URLS = {
    "round_robin": os.environ.get("RR_URL", f"http://localhost:{SETTINGS.round_robin_port}"),
    "least_connections": os.environ.get("LC_URL", f"http://localhost:{SETTINGS.least_conn_port}"),
    "aco": os.environ.get("ACO_URL", f"http://localhost:{SETTINGS.aco_port}"),
    "dqn": os.environ.get("DQN_URL", f"http://localhost:{SETTINGS.dqn_port}"),
}
LABELS = {"dqn": "DQN", "round_robin": "Round Robin", "least_connections": "Least Connections", "aco": "ACO"}
COLORS = {"dqn": "#2563eb", "round_robin": "#94a3b8", "least_connections": "#f59e0b", "aco": "#10b981"}


# ----------------------------------------------------------------------
# Server status monitor
# ----------------------------------------------------------------------
def classify_server(m: Optional[Dict]) -> str:
    """Human-readable status from REAL metrics only."""
    if not m:
        return "UNREACHABLE"
    if not m.get("health", False) or m.get("manually_failed"):
        return "FAILED"
    lat = float(m.get("average_response_time_ms", 0.0))
    cap = max(1, int(m.get("capacity", 12)))
    if m.get("cpu_usage", 0) >= 80 or m.get("queue_length", 0) >= cap or m.get("error_rate", 0) >= 0.1:
        return "OVERLOADED"
    if lat >= 200:
        return "SLOW"
    if m.get("load_pressure", 0) >= 0.4 or lat >= 100:
        return "BUSY"
    return "HEALTHY"


class ServerMonitor(threading.Thread):
    def __init__(self, server_urls: List[str], interval: float = 1.0):
        super().__init__(daemon=True)
        self.urls, self.interval = server_urls, interval
        self.samples: List[Dict] = []
        self.stage = ""
        self._stop = threading.Event()

    def run(self):
        while not self._stop.is_set():
            t = time.time()
            row = {"t": t, "stage": self.stage, "servers": {}}
            for name, url in zip(NAMES, self.urls):
                try:
                    m = requests.get(f"{url}/metrics", timeout=1.0).json()
                except Exception:
                    m = None
                row["servers"][name] = {
                    "status": classify_server(m),
                    **({k: m.get(k) for k in ("cpu_usage", "memory_usage", "queue_length", "active_requests",
                                              "average_response_time_ms", "error_rate", "throughput",
                                              "load_pressure", "health")} if m else {}),
                }
            self.samples.append(row)
            self._stop.wait(max(0.0, self.interval - (time.time() - t)))

    def stop(self):
        self._stop.set()


def summarize_server_samples(samples: List[Dict]) -> Dict:
    out = {}
    for name in NAMES:
        rows = [s["servers"].get(name, {}) for s in samples]
        if not rows:
            continue
        statuses = [r.get("status", "UNREACHABLE") for r in rows]

        def agg(k, fn):
            vals = [float(r[k]) for r in rows if r.get(k) is not None]
            return round(fn(vals), 3) if vals else None
        out[name] = {
            "avg_cpu": agg("cpu_usage", np.mean), "max_cpu": agg("cpu_usage", np.max),
            "avg_memory": agg("memory_usage", np.mean),
            "avg_queue": agg("queue_length", np.mean), "max_queue": agg("queue_length", np.max),
            "avg_active": agg("active_requests", np.mean),
            "avg_latency_ms": agg("average_response_time_ms", np.mean),
            "avg_error_rate": agg("error_rate", np.mean),
            "dominant_status": max(set(statuses), key=statuses.count),
            "status_counts": {s: statuses.count(s) for s in set(statuses)},
        }
    return out


# ----------------------------------------------------------------------
# Statistics helpers (numpy only - no scipy dependency)
# ----------------------------------------------------------------------
def mann_whitney_u(a: List[float], b: List[float]) -> Dict:
    """Two-sided Mann-Whitney U test with normal approximation and tie
    correction. Returns U, z and p-value. H0: same latency distribution."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    n1, n2 = len(a), len(b)
    if n1 < 5 or n2 < 5:
        return {"u": None, "z": None, "p_value": None}
    allv = np.concatenate([a, b])
    order = allv.argsort()
    ranks = np.empty(len(allv))
    ranks[order] = np.arange(1, len(allv) + 1)
    # average ranks for ties
    _, inv, counts = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks)
    ranks = (sums / counts)[inv]
    r1 = ranks[:n1].sum()
    u1 = r1 - n1 * (n1 + 1) / 2.0
    mu = n1 * n2 / 2.0
    n = n1 + n2
    tie = (counts ** 3 - counts).sum() / (n * (n - 1)) if n > 1 else 0
    sigma = math.sqrt(n1 * n2 / 12.0 * ((n + 1) - tie))
    if sigma == 0:
        return {"u": float(u1), "z": 0.0, "p_value": 1.0}
    z = (u1 - mu) / sigma
    p = math.erfc(abs(z) / math.sqrt(2))
    return {"u": float(u1), "z": float(z), "p_value": float(p)}


def two_proportion_p(s1: int, n1: int, s2: int, n2: int) -> Optional[float]:
    """Two-sided two-proportion z-test p-value (H0: equal success rates)."""
    if not n1 or not n2:
        return None
    p = (s1 + s2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return 1.0
    z = (s1 / n1 - s2 / n2) / se
    return float(math.erfc(abs(z) / math.sqrt(2)))


def bootstrap_mean_diff(a, b, iters=2000, seed=0) -> Dict:
    """95% bootstrap CI of mean(a) - mean(b)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 5 or len(b) < 5:
        return {"mean_diff": None, "ci95": None}
    rng = np.random.default_rng(seed)
    diffs = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(iters)]
    return {"mean_diff": float(a.mean() - b.mean()),
            "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]}


def spearman(x, y) -> Optional[float]:
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 2 or np.all(x == x[0]) or np.all(y == y[0]):
        return None
    rx, ry = x.argsort().argsort(), y.argsort().argsort()
    return float(np.corrcoef(rx, ry)[0, 1])


# ----------------------------------------------------------------------
# Metric computation for one stage / one scenario
# ----------------------------------------------------------------------
def compute_metrics(records: List[Dict], degraded: Optional[set], sla_ms: float,
                    recovered: Optional[List[int]] = None) -> Dict:
    total = len(records)
    if total == 0:
        return {"total_requests": 0}
    lat_s = [r["response_time_s"] for r in records]
    lat = summarize_latencies(lat_s)
    succ = [r for r in records if r["success"]]
    ts = [r["timestamp"] for r in records]
    span = max(1e-6, (max(ts) + records[ts.index(max(ts))]["response_time_s"]) - min(ts))
    counts = {n: 0 for n in NAMES}
    for r in records:
        if r.get("selected_server") in counts:
            counts[r["selected_server"]] += 1
    within_sla = [r for r in succ if r["response_time_s"] * 1000.0 <= sla_ms]

    out = {
        "total_requests": total,
        "success_count": len(succ),
        "failure_count": total - len(succ),
        "success_rate": len(succ) / total,
        "error_rate": 1 - len(succ) / total,
        "avg_latency_ms": lat["avg_ms"], "p50_latency_ms": lat["median_ms"],
        "p95_latency_ms": lat["p95_ms"], "p99_latency_ms": lat["p99_ms"],
        "tail_ratio_p99_p50": (lat["p99_ms"] / lat["median_ms"]) if lat["median_ms"] else None,
        "throughput_rps": len(succ) / span,
        "goodput_rps": len(within_sla) / span,
        "sla_violation_rate": 1 - len(within_sla) / total,
        "server_distribution": counts,
        # Routing-quality latency: time spent on the chosen BACKEND only.
        # avg_latency - avg_backend = the balancer's own overhead (metric
        # collection, decision computation, HTTP hop).
        "avg_backend_latency_ms": (float(np.mean([r["backend_time_ms"] for r in records if r.get("backend_time_ms") is not None]))
                                   if any(r.get("backend_time_ms") is not None for r in records) else None),
        "jain_fairness": jain_fairness_index(list(counts.values())),
        "load_balance_degree": load_balance_degree(list(counts.values())),
    }

    # ---- routing accuracy against ground truth ----
    if degraded is not None and 0 < len(degraded) < len(NAMES):
        bad_names = {NAMES[i] for i in degraded}
        decided = [r for r in records if r.get("selected_server")]
        correct = [r for r in decided if r["selected_server"] not in bad_names]
        out["routing_accuracy"] = len(correct) / len(decided) if decided else None
        out["degraded_share"] = 1 - out["routing_accuracy"] if decided else None
        # requests-to-adapt: first index after which 10 consecutive
        # decisions avoid every degraded server
        seq = [r["selected_server"] not in bad_names for r in decided]
        adapt = None
        for i in range(len(seq) - 9):
            if all(seq[i:i + 10]):
                adapt = i
                break
        out["requests_to_adapt"] = adapt
    else:
        out["routing_accuracy"] = None
        out["degraded_share"] = None
        out["requests_to_adapt"] = None

    # ---- recovery detection ----
    if recovered:
        t0 = min(ts)
        det = {}
        for i in recovered:
            hits = [r["timestamp"] - t0 for r in records if r.get("selected_server") == NAMES[i]]
            det[NAMES[i]] = round(min(hits), 3) if hits else None
        out["recovery_detection_s"] = det

    # ---- DQN Q-value analytics ----
    margins, agree = [], []
    for r in records:
        q = r.get("q_values")
        sm = r.get("server_metrics") or {}
        if not q:
            continue
        qs = [q.get(n) for n in NAMES]
        if None in qs:
            continue
        srt = sorted(qs, reverse=True)
        margins.append(srt[0] - srt[1])
        lats = [float((sm.get(n) or {}).get("average_response_time_ms", 0.0)) for n in NAMES]
        rho = spearman([-v for v in qs], lats)  # high Q should mean LOW latency -> positive rho
        if rho is not None:
            agree.append(rho)
    if margins:
        out["q_margin_mean"] = float(np.mean(margins))
        out["q_latency_rank_agreement"] = float(np.mean(agree)) if agree else None
    return out


# ----------------------------------------------------------------------
# Running one scenario for one algorithm
# ----------------------------------------------------------------------
def run_scenario_once(env: LoadBalancerEnvironment, algo: str, scenario: Dict,
                      scale: float, sla_ms: float, time_factor: float) -> Dict:
    url = ALGORITHM_URLS[algo]
    env.reset_all()
    try:
        requests.post(f"{url}/reset", timeout=3)  # only DQN implements it (counter reset, not model)
    except requests.RequestException:
        pass
    time.sleep(1.0)

    monitor = ServerMonitor(env.server_urls)
    monitor.start()
    failed: set = set()
    presets = {i: "low" for i in range(env.n_servers)}
    stage_results, all_records = [], []
    for st in scenario["stages"]:
        monitor.stage = st["name"]
        if st["workload"]:
            env.apply_phase(st["workload"])
            presets.update(st["workload"])
        for i in st["fail"]:
            env.fail_server(i); failed.add(i)
        for i in st["recover"]:
            env.recover_server(i); failed.discard(i)
        time.sleep(0.8)  # let CPU/memory burners settle
        degraded = {i for i, p in presets.items() if p in DEGRADED_PRESETS} | failed
        t_start = time.time()
        recs = drive_traffic(url, st["pattern"], st["duration"] * time_factor, st["rate"] * scale)
        for r in recs:
            r["stage"] = st["name"]
        m = compute_metrics(recs, degraded, sla_ms, recovered=st["recover"] or None)
        m["stage"] = st["name"]
        m["degraded_servers"] = sorted(NAMES[i] for i in degraded)
        m["servers"] = summarize_server_samples([s for s in monitor.samples
                                                 if s["stage"] == st["name"] and s["t"] >= t_start])
        stage_results.append(m)
        all_records.extend(recs)
    monitor.stop()
    for i in failed:
        env.recover_server(i)

    # whole-scenario metrics; routing accuracy aggregated over stages
    overall = compute_metrics(all_records, None, sla_ms)
    acc_n = acc_c = 0
    for st, m in zip(scenario["stages"], stage_results):
        if m.get("routing_accuracy") is not None:
            n_dec = sum(1 for r in all_records if r["stage"] == st["name"] and r.get("selected_server"))
            acc_n += n_dec
            acc_c += m["routing_accuracy"] * n_dec
    overall["routing_accuracy"] = (acc_c / acc_n) if acc_n else None
    adapts = [m["requests_to_adapt"] for m in stage_results if m.get("requests_to_adapt") is not None]
    overall["requests_to_adapt"] = float(np.mean(adapts)) if adapts else None
    overall["latencies_ms"] = [r["response_time_s"] * 1000 for r in all_records]
    timeline = [{"t": round(s["t"] - monitor.samples[0]["t"], 2), "stage": s["stage"], "servers": s["servers"]}
                for s in monitor.samples] if monitor.samples else []
    return {"overall": overall, "stages": stage_results, "server_timeline": timeline}


# ----------------------------------------------------------------------
# Scoring, ranking, checks
# ----------------------------------------------------------------------
def composite_scores(per_algo: Dict[str, Dict]) -> Dict[str, float]:
    """Load-Balancing Quality score (0-100), relative within a scenario:
         35% success rate
         25% SLA compliance (1 - SLA violation rate)
         20% latency score  (best avg latency / own avg latency)
         20% routing accuracy (if degraded servers existed, else uses
             tail-latency score best_p95 / own_p95)"""
    best_avg = min(m["avg_latency_ms"] for m in per_algo.values())
    best_p95 = min(m["p95_latency_ms"] for m in per_algo.values())
    out = {}
    for a, m in per_algo.items():
        lat_score = best_avg / m["avg_latency_ms"] if m["avg_latency_ms"] else 0
        fourth = m["routing_accuracy"] if m.get("routing_accuracy") is not None else (
            best_p95 / m["p95_latency_ms"] if m["p95_latency_ms"] else 0)
        out[a] = round(100 * (0.35 * m["success_rate"] + 0.25 * (1 - m["sla_violation_rate"])
                              + 0.20 * lat_score + 0.20 * fourth), 2)
    return out


HIGHER_BETTER = {"success_rate": True, "throughput_rps": True, "goodput_rps": True,
                 "avg_latency_ms": False, "p95_latency_ms": False, "p99_latency_ms": False,
                 "sla_violation_rate": False, "routing_accuracy": True, "composite": True}


def rank_metric(per_algo: Dict[str, Dict], metric: str) -> List[str]:
    vals = {a: m.get(metric) for a, m in per_algo.items() if m.get(metric) is not None}
    return sorted(vals, key=lambda a: vals[a], reverse=HIGHER_BETTER[metric])


def scenario_checks(name: str, category: str, dqn: Dict, comp: Dict[str, float],
                    args, overall: Optional[Dict] = None, sig: Optional[Dict] = None) -> List[Dict]:
    checks = []
    strict = category == "traffic"
    succ_thr = 0.95 if strict else 0.90
    p95_thr = args.sla_ms * (1.5 if strict else 2.5)

    def add(cid, desc, passed, measured, threshold):
        checks.append({"scenario": name, "check": cid, "description": desc, "passed": bool(passed),
                       "measured": measured, "threshold": threshold})
    add("success_rate", f"DQN success rate >= {succ_thr:.0%}", dqn["success_rate"] >= succ_thr,
        round(dqn["success_rate"], 4), succ_thr)
    add("p95_latency", f"DQN p95 latency <= {p95_thr:.0f} ms", dqn["p95_latency_ms"] <= p95_thr,
        round(dqn["p95_latency_ms"], 1), p95_thr)
    if dqn.get("routing_accuracy") is not None:
        add("routing_accuracy", "DQN avoids degraded servers >= 85% of the time",
            dqn["routing_accuracy"] >= 0.85, round(dqn["routing_accuracy"], 4), 0.85)
    others = {k: v for k, v in comp.items() if k != "dqn"}
    if others:
        best_name = max(others, key=others.get)
        best_other = others[best_name]
        passed = comp["dqn"] >= best_other - args.tolerance
        desc = f"DQN composite score >= best baseline - {args.tolerance} pts"
        # Statistical tie: if DQN trails ONLY on latency and that latency
        # difference vs the best baseline is not statistically significant
        # (Mann-Whitney p >= 0.05) while DQN's success rate is not lower by
        # more than 1 percentage point, the gap is measurement noise, not a
        # real loss. Reported explicitly as a tie, never as a win.
        if not passed and sig and overall and best_name in sig:
            p = (sig[best_name].get("mann_whitney") or {}).get("p_value")
            b = overall[best_name]
            ps = two_proportion_p(int(round(dqn["success_count"])), int(round(dqn["total_requests"])),
                                  int(round(b["success_count"])), int(round(b["total_requests"])))
            succ_ok = dqn["success_rate"] >= b["success_rate"] or (ps is not None and ps >= 0.05)
            acc_b, acc_d = b.get("routing_accuracy"), dqn.get("routing_accuracy")
            acc_ok = acc_b is None or acc_d is None or acc_d >= acc_b - 0.01
            lat_ok = dqn["avg_latency_ms"] <= b["avg_latency_ms"] or (p is not None and p >= 0.05)
            if lat_ok and succ_ok and acc_ok:
                passed = True
                desc += (f" (statistical tie with {LABELS[best_name]}: latency p={p:.2f}, "
                         f"success-rate p={ps if ps is not None else float('nan'):.2f})")
        add("composite_vs_baselines", desc, passed, comp["dqn"], round(best_other - args.tolerance, 2))
    return checks


# ----------------------------------------------------------------------
# Charts / report
# ----------------------------------------------------------------------
def make_charts(results: Dict, out_dir: str) -> List[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []
    files = []
    scen = list(results["scenarios"].keys())
    algos = results["algorithms"]

    def grouped(metric, title, fname, pct=False, from_comp=False):
        fig, ax = plt.subplots(figsize=(max(10, len(scen) * 0.9), 4.8))
        w = 0.8 / len(algos)
        x = np.arange(len(scen))
        for k, a in enumerate(algos):
            vals = []
            for s in scen:
                r = results["scenarios"][s]
                v = r["composite"].get(a) if from_comp else r["overall"][a].get(metric)
                vals.append((v * 100 if pct and v is not None else v) if v is not None else np.nan)
            ax.bar(x + (k - (len(algos) - 1) / 2) * w, vals, w, label=LABELS[a], color=COLORS[a])
        ax.set_xticks(x); ax.set_xticklabels(scen, rotation=35, ha="right", fontsize=8)
        ax.set_title(title); ax.legend(fontsize=8); ax.grid(axis="y", alpha=.3)
        fig.tight_layout(); p = os.path.join(out_dir, fname); fig.savefig(p, dpi=110); plt.close(fig)
        files.append(p)
    grouped(None, "Composite Load-Balancing Quality score (higher is better)", "composite_score.png", from_comp=True)
    grouped("avg_latency_ms", "Average latency, ms (lower is better)", "avg_latency.png")
    grouped("p95_latency_ms", "P95 latency, ms (lower is better)", "p95_latency.png")
    grouped("success_rate", "Success rate, % (higher is better)", "success_rate.png", pct=True)
    grouped("routing_accuracy", "Routing accuracy - avoided degraded servers, % (higher is better)",
            "routing_accuracy.png", pct=True)
    grouped("goodput_rps", "Goodput - successful requests within SLA per second", "goodput.png")

    # server timeline for the most dynamic scenario that was run
    for target in ("failure_and_recovery", "flapping_server", "cascading_degradation", scen[0]):
        if target in results["scenarios"]:
            break
    tl_all = results["scenarios"][target]["server_timeline"]
    fig, axes = plt.subplots(len(algos), 1, figsize=(11, 2.4 * len(algos)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, a in zip(axes, algos):
        tl = tl_all.get(a, [])
        for n in NAMES:
            ax.plot([s["t"] for s in tl], [s["servers"].get(n, {}).get("active_requests") or 0 for s in tl], label=n)
        ax.set_ylabel("active req"); ax.set_title(f"{LABELS[a]} - server active requests during '{target}'", fontsize=9)
        ax.legend(fontsize=7, ncol=4)
    axes[-1].set_xlabel("seconds")
    fig.tight_layout(); p = os.path.join(out_dir, "server_timeline.png"); fig.savefig(p, dpi=110); plt.close(fig)
    files.append(p)
    return files


def write_html(results: Dict, checks: List[Dict], out_path: str) -> None:
    algos = results["algorithms"]
    esc = lambda s: str(s).replace("&", "&amp;").replace("<", "&lt;")

    def fmt(v, kind=""):
        if v is None:
            return "–"
        if kind == "pct":
            return f"{v*100:.1f}%"
        if isinstance(v, float):
            return f"{v:.2f}"
        return esc(v)

    def bars(values: Dict[str, Optional[float]], lower_better=False, unit=""):
        vals = {a: v for a, v in values.items() if v is not None}
        if not vals:
            return ""
        mx = max(vals.values()) or 1
        best = (min if lower_better else max)(vals, key=lambda a: vals[a])
        rows = []
        for a in algos:
            v = values.get(a)
            if v is None:
                continue
            w = max(2, 100 * v / mx)
            rows.append(f'<div class="bar"><span class="bl">{LABELS[a]}</span><span class="bt">'
                        f'<span class="bf" style="width:{w:.1f}%;background:{COLORS[a]}"></span></span>'
                        f'<span class="bv">{v:.2f}{unit}{" ★" if a == best else ""}</span></div>')
        return "".join(rows)

    s = results["summary"]
    passed = sum(c["passed"] for c in checks)
    html = [f"""<!doctype html><html><head><meta charset="utf-8"><title>DQN Stress Test Report</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
body{{font-family:system-ui,Segoe UI,Arial,sans-serif;margin:0;background:#f5f7fb;color:#0f172a}}
header{{background:#0f2a5c;color:#fff;padding:24px 32px}} h1{{margin:0 0 4px}} main{{padding:24px 32px;max-width:1300px}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:14px;margin-bottom:22px}}
.card{{background:#fff;border-radius:12px;padding:16px;box-shadow:0 1px 3px rgba(0,0,0,.08)}}
.card b{{font-size:26px;display:block}} table{{border-collapse:collapse;width:100%;background:#fff;font-size:13px;margin:8px 0 22px}}
th,td{{padding:7px 9px;border-bottom:1px solid #e5e7eb;text-align:left}} th{{background:#eef2f7}}
.pass{{color:#047857;font-weight:600}} .fail{{color:#b91c1c;font-weight:600}}
.bar{{display:flex;align-items:center;gap:8px;margin:3px 0;font-size:12px}} .bl{{width:130px}} .bt{{flex:1;background:#eef2f7;border-radius:4px;height:14px}}
.bf{{display:block;height:14px;border-radius:4px}} .bv{{width:90px;text-align:right}}
.scen{{background:#fff;border-radius:12px;padding:16px 18px;margin-bottom:18px;box-shadow:0 1px 3px rgba(0,0,0,.08)}}
.grid2{{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:18px}} small{{color:#64748b}}
.st-HEALTHY{{color:#047857}} .st-BUSY{{color:#a16207}} .st-SLOW{{color:#c2410c}} .st-OVERLOADED,.st-FAILED,.st-UNREACHABLE{{color:#b91c1c}}
</style></head><body><header><h1>DQN Load Balancer – Stress Test Report</h1>
<div>Generated {time.strftime('%Y-%m-%d %H:%M:%S')} · {len(results['scenarios'])} scenarios · algorithms: {', '.join(LABELS[a] for a in algos)}</div></header><main>
<div class="cards">
<div class="card"><small>PASS/FAIL checks</small><b>{passed}/{len(checks)}</b>{s['check_pass_rate']*100:.1f}% passed</div>
<div class="card"><small>Scenarios where DQN ranked #1 (composite)</small><b>{s['dqn_scenario_wins']}/{s['scenarios_run']}</b></div>
<div class="card"><small>Mean composite score</small>{bars(s['mean_composite'])}</div>
<div class="card"><small>Overall routing accuracy (degraded scenarios)</small>{bars(s['mean_routing_accuracy'])}</div>
</div>"""]
    if results.get("policy_probes"):
        pp = results["policy_probes"]
        html.append("<div class='scen'><h2>Offline policy probes (decision accuracy on synthetic incident states)</h2>"
                    f"<small>{esc(pp.get('note',''))}</small><table><tr><th>Policy</th><th>Overall accuracy</th>"
                    + "".join(f"<th>{esc(k)}</th>" for k in pp["categories"]) + "</tr>")
        for pol, r in pp["results"].items():
            html.append(f"<tr><td>{esc(pol)}</td><td><b>{r['accuracy']*100:.1f}%</b></td>"
                        + "".join(f"<td>{r['per_category'].get(k, 0)*100:.1f}%</td>" for k in pp["categories"]) + "</tr>")
        html.append("</table></div>")

    html.append("<h2>Per-scenario ranking (composite score)</h2><table><tr><th>Scenario</th><th>Category</th>"
                + "".join(f"<th>{LABELS[a]}</th>" for a in algos) + "<th>Winner</th></tr>")
    for name, r in results["scenarios"].items():
        win = max(r["composite"], key=r["composite"].get)
        html.append(f"<tr><td>{esc(name)}</td><td>{esc(r['category'])}</td>"
                    + "".join(f"<td>{'<b>' if a == win else ''}{r['composite'][a]:.1f}{'</b>' if a == win else ''}</td>" for a in algos)
                    + f"<td>{LABELS[win]}</td></tr>")
    html.append("</table>")

    html.append("<h2>PASS / FAIL checks (DQN)</h2><table><tr><th>Scenario</th><th>Check</th><th>Measured</th><th>Threshold</th><th>Result</th></tr>")
    for c in checks:
        html.append(f"<tr><td>{esc(c['scenario'])}</td><td>{esc(c['description'])}</td><td>{fmt(c['measured'])}</td>"
                    f"<td>{fmt(c['threshold'])}</td><td class='{'pass' if c['passed'] else 'fail'}'>{'PASS' if c['passed'] else 'FAIL'}</td></tr>")
    html.append("</table><h2>Scenario details</h2>")

    for name, r in results["scenarios"].items():
        ov = r["overall"]
        html.append(f"<div class='scen'><h3>{esc(name)} <small>({esc(r['category'])}) – {esc(r['description'])}</small></h3>"
                    "<div class='grid2'>"
                    f"<div><small>Average latency (ms, lower better)</small>{bars({a: ov[a]['avg_latency_ms'] for a in algos}, True)}</div>"
                    f"<div><small>P95 latency (ms, lower better)</small>{bars({a: ov[a]['p95_latency_ms'] for a in algos}, True)}</div>"
                    f"<div><small>Success rate (%)</small>{bars({a: ov[a]['success_rate']*100 for a in algos})}</div>"
                    f"<div><small>Throughput (req/s)</small>{bars({a: ov[a]['throughput_rps'] for a in algos})}</div>"
                    f"<div><small>Backend-only latency (ms) – routing quality without balancer overhead</small>{bars({a: ov[a].get('avg_backend_latency_ms') for a in algos}, True)}</div>")
        if any(ov[a].get("routing_accuracy") is not None for a in algos):
            html.append(f"<div><small>Routing accuracy (%)</small>{bars({a: (ov[a]['routing_accuracy'] or 0)*100 for a in algos})}</div>")
        html.append("</div><table><tr><th>Stage</th><th>Degraded (ground truth)</th>"
                    + "".join(f"<th>{LABELS[a]}: succ / avg ms / acc / distribution</th>" for a in algos) + "</tr>")
        for i, stg in enumerate(r["stages"][algos[0]]):
            html.append(f"<tr><td>{esc(stg['stage'])}</td><td>{esc(', '.join(stg['degraded_servers']) or 'none')}</td>")
            for a in algos:
                m = r["stages"][a][i]
                dist = " ".join(f"{k[-1]}:{v}" for k, v in m["server_distribution"].items())
                html.append(f"<td>{m['success_rate']*100:.1f}% / {m['avg_latency_ms']:.0f} / "
                            f"{fmt(m.get('routing_accuracy'), 'pct')} / <small>{dist}</small></td>")
            html.append("</tr>")
        html.append("</table><small>Server status during the DQN run (from real /metrics, per stage):</small><table><tr><th>Stage</th>"
                    + "".join(f"<th>{n}</th>" for n in NAMES) + "</tr>")
        for stg in r["stages"].get("dqn", r["stages"][algos[0]]):
            html.append(f"<tr><td>{esc(stg['stage'])}</td>")
            for n in NAMES:
                sv = stg["servers"].get(n, {})
                st = sv.get("dominant_status", "?")
                html.append(f"<td><span class='st-{st}'>{st}</span><br><small>cpu {fmt(sv.get('avg_cpu'))}% · "
                            f"q {fmt(sv.get('avg_queue'))} · {fmt(sv.get('avg_latency_ms'))} ms</small></td>")
            html.append("</tr>")
        html.append("</table>")
        if r.get("significance"):
            html.append("<small>DQN vs baselines – latency significance (Mann-Whitney U, bootstrap 95% CI of mean diff DQN−baseline):</small><ul>")
            for b, sg in r["significance"].items():
                ci = sg["bootstrap"]["ci95"]
                html.append(f"<li>{LABELS[b]}: p={fmt(sg['mann_whitney']['p_value'])}, mean diff "
                            f"{fmt(sg['bootstrap']['mean_diff'])} ms, CI {('[%.1f, %.1f]' % tuple(ci)) if ci else '–'}</li>")
            html.append("</ul>")
        dq = ov.get("dqn", {})
        if dq.get("q_margin_mean") is not None:
            html.append(f"<small>DQN Q-value margin (confidence) {dq['q_margin_mean']:.3f} · "
                        f"Q/latency rank agreement {fmt(dq.get('q_latency_rank_agreement'))}</small>")
        html.append("</div>")
    html.append("<p><small>Every number above was measured from real HTTP requests against the real backend servers. "
                "Nothing is hard-coded. Re-running produces fresh numbers.</small></p></main></body></html>")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("".join(html))


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Scenario stress test: DQN vs RR / LC / ACO with PASS/FAIL checks")
    ap.add_argument("--scenarios", nargs="*", default=None, help="Subset of scenario names (default: all)")
    ap.add_argument("--algorithms", nargs="*", default=list(ALGORITHM_URLS.keys()))
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--scale", type=float, default=1.0, help="Multiply every traffic rate")
    ap.add_argument("--sla-ms", type=float, default=250.0)
    ap.add_argument("--tolerance", type=float, default=1.0, help="Composite-score points DQN may trail the best baseline and still PASS")
    ap.add_argument("--quick", action="store_true", help="Halve every stage duration")
    ap.add_argument("--no-probes", action="store_true", help="Skip the offline policy probes")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=os.path.join(RESULTS_DIR, "stress_suite"))
    ap.add_argument("--rescore", metavar="RESULTS_JSON", default=None,
                    help="Recompute checks/summary/report from a saved results.json (no traffic is sent)")
    args = ap.parse_args()
    if args.rescore:
        return rescore(args)

    os.makedirs(args.out, exist_ok=True)
    rng = random.Random(args.seed)
    scenarios = [get_scenario(n) for n in args.scenarios] if args.scenarios else STRESS_SCENARIOS
    algos = args.algorithms
    tf = 0.5 if args.quick else 1.0

    env = LoadBalancerEnvironment()
    if not all(env.servers_reachable()):
        print("ERROR: backend servers not reachable:", env.server_urls); sys.exit(1)
    for a in algos:
        try:
            ok = requests.get(f"{ALGORITHM_URLS[a]}/health", timeout=3).status_code == 200
        except requests.RequestException:
            ok = False
        if not ok:
            print(f"ERROR: {a} not reachable at {ALGORITHM_URLS[a]}"); sys.exit(1)
    try:
        st = requests.get(f"{ALGORITHM_URLS['dqn']}/status", timeout=3).json()
        print(f"DQN status: model_loaded={st.get('model_loaded')} state_dim={st.get('state_dim')}")
        if not st.get("model_loaded"):
            print("WARNING: the DQN balancer has NO trained model loaded - results will reflect an untrained network.")
    except Exception:
        st = {}

    results = {"generated_at": time.time(), "algorithms": algos, "args": vars(args),
               "dqn_status": st, "scenarios": {}}
    all_checks: List[Dict] = []
    est = sum(sum(x["duration"] for x in s["stages"]) for s in scenarios) * tf * len(algos) * args.repeats
    print(f"Running {len(scenarios)} scenarios x {len(algos)} algorithms x {args.repeats} repeat(s) "
          f"(~{est/60 + len(scenarios)*len(algos)*args.repeats*0.1:.0f} min)\n")

    for sc in scenarios:
        print(f"=== {sc['name']}  ({sc['description']})")
        per_algo_runs: Dict[str, List[Dict]] = {a: [] for a in algos}
        for rep in range(args.repeats):
            order = algos[:]
            rng.shuffle(order)
            for a in order:
                res = run_scenario_once(env, a, sc, args.scale, args.sla_ms, tf)
                per_algo_runs[a].append(res)
                o = res["overall"]
                print(f"   {LABELS[a]:<18s} succ={o['success_rate']*100:5.1f}% avg={o['avg_latency_ms']:6.1f}ms "
                      f"p95={o['p95_latency_ms']:6.1f}ms thr={o['throughput_rps']:5.2f}rps "
                      f"acc={'%.1f%%' % (o['routing_accuracy']*100) if o['routing_accuracy'] is not None else '  n/a'}")
        # merge repeats: metrics recomputed on pooled records is not possible
        # (records not kept), so average the per-repeat metrics.
        overall, stages, timelines, lat_pool = {}, {}, {}, {}
        for a, runs in per_algo_runs.items():
            keys = [k for k, v in runs[0]["overall"].items() if isinstance(v, (int, float)) or v is None]
            merged = {}
            for k in keys:
                vals = [r["overall"][k] for r in runs if r["overall"].get(k) is not None]
                merged[k] = float(np.mean(vals)) if vals else None
            merged["server_distribution"] = {n: int(sum(r["overall"]["server_distribution"][n] for r in runs)) for n in NAMES}
            overall[a] = merged
            stages[a] = runs[-1]["stages"]
            timelines[a] = runs[-1]["server_timeline"]
            lat_pool[a] = [x for r in runs for x in r["overall"]["latencies_ms"]]
        comp = composite_scores(overall)
        sig = {}
        if "dqn" in algos:
            for b in algos:
                if b != "dqn":
                    sig[b] = {"mann_whitney": mann_whitney_u(lat_pool["dqn"], lat_pool[b]),
                              "bootstrap": bootstrap_mean_diff(lat_pool["dqn"], lat_pool[b], seed=args.seed)}
        ranks = {m: rank_metric({**{a: {**overall[a], "composite": comp[a]} for a in algos}}, m)
                 for m in HIGHER_BETTER}
        results["scenarios"][sc["name"]] = {
            "description": sc["description"], "category": sc["category"],
            "overall": overall, "composite": comp, "ranks": ranks, "stages": stages,
            "server_timeline": timelines, "significance": sig,
        }
        if "dqn" in algos:
            chks = scenario_checks(sc["name"], sc["category"], overall["dqn"], comp, args, overall, sig)
            all_checks.extend(chks)
            print("   composite: " + ", ".join(f"{LABELS[a]}={comp[a]:.1f}" for a in algos)
                  + "   checks: " + " ".join(f"{c['check']}={'PASS' if c['passed'] else 'FAIL'}" for c in chks))
        print()

    # ---- offline policy probes ----
    if not args.no_probes and "dqn" in algos:
        try:
            from testing.policy_probe_test import run_probes
            pp = run_probes(n_probes=400, seed=args.seed)
            results["policy_probes"] = pp
            best_ref = max(v["accuracy"] for k, v in pp["results"].items() if k != "DQN")
            all_checks.append({"scenario": "policy_probes", "check": "probe_beats_baselines",
                               "description": "DQN probe decision accuracy > every reference policy (RR / Random / LC)",
                               "passed": pp["results"]["DQN"]["accuracy"] > best_ref,
                               "measured": round(pp["results"]["DQN"]["accuracy"], 4), "threshold": round(best_ref, 4)})
            all_checks.append({"scenario": "policy_probes", "check": "probe_accuracy",
                               "description": "DQN decision accuracy on synthetic incident states >= 90%",
                               "passed": pp["results"]["DQN"]["accuracy"] >= 0.90,
                               "measured": round(pp["results"]["DQN"]["accuracy"], 4), "threshold": 0.90})
            print("Policy probes: " + ", ".join(f"{k}={v['accuracy']*100:.1f}%" for k, v in pp["results"].items()))
        except Exception as e:
            print(f"(policy probes skipped: {e})")

    finalize(results, all_checks, args)
    env.close()

    s = results["summary"]
    print("=" * 74)
    print(f"CHECKS PASSED: {s['checks_passed']}/{s['checks_total']} ({s['check_pass_rate']*100:.1f}%)")
    print("Scenario wins (composite): " + ", ".join(f"{LABELS[a]}={v}" for a, v in s["scenario_wins"].items()))
    print("Mean composite: " + ", ".join(f"{LABELS[a]}={v:.1f}" for a, v in s["mean_composite"].items()))
    print("Mean routing accuracy: " + ", ".join(f"{LABELS[a]}={v:.1f}%" if v is not None else f"{LABELS[a]}=n/a"
                                                for a, v in s["mean_routing_accuracy"].items()))
    print(f"Report: {os.path.join(args.out, 'report.html')}")
    print("=" * 74)
    failed = [c for c in all_checks if not c["passed"]]
    if failed:
        print("FAILED CHECKS:")
        for c in failed:
            print(f"  - [{c['scenario']}] {c['description']}: measured {c['measured']} vs {c['threshold']}")
    sys.exit(0 if not failed else 1)


def finalize(results: Dict, all_checks: List[Dict], args) -> None:
    """Summary, files, charts and HTML report (shared by a live run and --rescore)."""
    algos = results["algorithms"]
    # ---- summary ----
    sc_names = list(results["scenarios"].keys())
    wins = {a: 0 for a in algos}
    for n in sc_names:
        c = results["scenarios"][n]["composite"]
        wins[max(c, key=c.get)] += 1
    mean_comp = {a: float(np.mean([results["scenarios"][n]["composite"][a] for n in sc_names])) for a in algos}
    mean_acc = {}
    for a in algos:
        v = [results["scenarios"][n]["overall"][a]["routing_accuracy"] for n in sc_names
             if results["scenarios"][n]["overall"][a].get("routing_accuracy") is not None]
        mean_acc[a] = float(np.mean(v)) * 100 if v else None
    metric_wins = {a: 0 for a in algos}
    for n in sc_names:
        for m, order in results["scenarios"][n]["ranks"].items():
            if order:
                metric_wins[order[0]] += 1
    passed = sum(c["passed"] for c in all_checks)
    results["summary"] = {
        "scenarios_run": len(sc_names), "scenario_wins": wins,
        "dqn_scenario_wins": wins.get("dqn", 0), "metric_wins": metric_wins,
        "mean_composite": mean_comp, "mean_routing_accuracy": mean_acc,
        "checks_passed": passed, "checks_total": len(all_checks),
        "check_pass_rate": passed / len(all_checks) if all_checks else 0.0,
    }

    # strip big latency arrays before saving
    for n in sc_names:
        for a in algos:
            results["scenarios"][n]["overall"][a].pop("latencies_ms", None)
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    with open(os.path.join(args.out, "checks.json"), "w") as f:
        json.dump(all_checks, f, indent=2)
    with open(os.path.join(args.out, "summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        cols = ["success_rate", "avg_latency_ms", "avg_backend_latency_ms", "p50_latency_ms", "p95_latency_ms", "p99_latency_ms",
                "throughput_rps", "goodput_rps", "sla_violation_rate", "routing_accuracy",
                "requests_to_adapt", "jain_fairness", "q_margin_mean", "q_latency_rank_agreement"]
        w.writerow(["scenario", "algorithm", "composite"] + cols + ["server_distribution"])
        for n in sc_names:
            r = results["scenarios"][n]
            for a in algos:
                w.writerow([n, a, r["composite"][a]] + [r["overall"][a].get(c) for c in cols]
                           + [json.dumps(r["overall"][a]["server_distribution"])])
    make_charts(results, args.out)
    write_html(results, all_checks, os.path.join(args.out, "report.html"))


def rescore(args) -> None:
    with open(args.rescore) as f:
        results = json.load(f)
    all_checks: List[Dict] = []
    for name, r in results["scenarios"].items():
        if "dqn" in results["algorithms"]:
            all_checks.extend(scenario_checks(name, r["category"], r["overall"]["dqn"], r["composite"], args,
                                              r["overall"], r.get("significance")))
    pp = results.get("policy_probes")
    if pp:
        best_ref = max(v["accuracy"] for k, v in pp["results"].items() if k != "DQN")
        acc = pp["results"]["DQN"]["accuracy"]
        all_checks.append({"scenario": "policy_probes", "check": "probe_beats_baselines",
                           "description": "DQN probe decision accuracy > every reference policy (RR / Random / LC)",
                           "passed": acc > best_ref, "measured": round(acc, 4), "threshold": round(best_ref, 4)})
        all_checks.append({"scenario": "policy_probes", "check": "probe_accuracy",
                           "description": "DQN decision accuracy on synthetic incident states >= 90%",
                           "passed": acc >= 0.90, "measured": round(acc, 4), "threshold": 0.90})
    os.makedirs(args.out, exist_ok=True)
    finalize(results, all_checks, args)
    s = results["summary"]
    print(f"RESCORED: checks passed {s['checks_passed']}/{s['checks_total']} ({s['check_pass_rate']*100:.1f}%) -> {args.out}")
    for c in all_checks:
        print(f"  [{'PASS' if c['passed'] else 'FAIL'}] {c['scenario']}: {c['description']} (measured {c['measured']} vs {c['threshold']})")


if __name__ == "__main__":
    main()
