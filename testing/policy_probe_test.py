#!/usr/bin/env python3
"""
testing/policy_probe_test.py
=============================
OFFLINE decision-accuracy test of the trained DQN policy ("unit tests for
the Q-network"). No servers or Docker needed - only the model file.

How it works
  1. Generate N synthetic-but-realistic incident snapshots of the 4
     servers. Each server is drawn from one of these conditions, with
     value ranges calibrated against what the real backend servers report
     under the matching workload preset (servers/workload.py):
        healthy, busy-but-healthy        -> ACCEPTABLE targets
        overloaded, slow, queue-buildup,
        cpu-starved, hard-failed         -> WRONG targets
  2. Build the exact same 46-dim state the balancer builds
     (dqn/state.build_state) and ask the network for its Q-values.
  3. A decision is CORRECT if argmax(Q) is an acceptable server.

Probe categories (each needs >= 1 acceptable server):
  one_overloaded, one_slow, one_failed, one_queue_buildup, one_cpu_starved,
  two_degraded, three_degraded (only one good server left!),
  busy_vs_degraded (the only alternatives are busy-but-healthy servers).

Reference policies on the SAME probes:
  Round Robin (state-blind -> expected accuracy = share of good servers),
  Random, Least Connections (min active_requests, can see nothing else).

Usage
  python testing/policy_probe_test.py --n 1000
  python testing/policy_probe_test.py --model models/dqn_model.pt

Writes results/policy_probes.json. Exit code 0 if DQN accuracy >= --threshold.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dqn.config import MODEL_PATH, RESULTS_DIR  # noqa: E402
from dqn.state import build_state  # noqa: E402

N = 4
GOOD = {"healthy", "busy"}

# (low, high) ranges per metric and condition
RANGES = {
    #                cpu        mem(%)     queue     active    lat_ms       err          capacity
    "healthy":     ((2, 30),   (6, 18),   (0, 1),   (0, 3),   (10, 70),    (0.0, 0.04), (12, 20)),
    "busy":        ((25, 55),  (12, 35),  (0, 3),    (1, 6),   (40, 120),   (0.0, 0.05), (8, 12)),
    "overloaded":  ((80, 100), (80, 100), (3, 20),  (3, 8),   (180, 700),  (0.10, 0.40), (4, 4)),
    "slow":        ((10, 35),  (8, 20),   (0, 4),   (1, 6),   (260, 800),  (0.0, 0.05), (10, 10)),
    "queue_buildup": ((20, 45), (10, 20), (4, 20),  (3, 3),   (120, 450),  (0.02, 0.10), (3, 3)),
    "cpu_starved": ((82, 100), (10, 20),  (0, 4),   (1, 5),   (60, 250),   (0.02, 0.08), (10, 10)),
    "failed":      ((1, 20),   (5, 15),   (0, 0),   (0, 1),   (5, 20),     (0.55, 1.0),  (12, 12)),
}

CATEGORIES = {
    "one_overloaded": ["overloaded"],
    "one_slow": ["slow"],
    "one_failed": ["failed"],
    "one_queue_buildup": ["queue_buildup"],
    "one_cpu_starved": ["cpu_starved"],
    "two_degraded": None,     # 2 random bad conditions
    "three_degraded": None,   # 3 random bad, only 1 good
    "busy_vs_degraded": None, # good ones are 'busy', bad ones degraded
}
BAD = ["overloaded", "slow", "queue_buildup", "cpu_starved", "failed"]


def sample_server(cond: str, r: random.Random) -> Dict:
    (c, m, q, a, lat, err, cap) = RANGES[cond]
    u = lambda lo_hi: r.uniform(*lo_hi)
    cpu, mem = u(c), u(m)
    queue, active = int(round(u(q))), int(round(u(a)))
    capacity = int(round(u(cap)))
    rate = r.uniform(1.0, 8.0)
    error = u(err)
    healthy = cond != "failed" and error < 0.5
    qr = min(1.0, queue / max(1, capacity * 2))
    ar = min(1.0, active / max(1, capacity * 2))
    lp = min(1.0, 0.35 * cpu / 100 + 0.20 * mem / 100 + 0.25 * qr + 0.20 * ar)
    return {
        "cpu_usage": cpu, "memory_usage": mem, "queue_length": queue, "active_requests": active,
        "capacity": capacity, "request_rate": rate, "throughput": rate * (1 - error),
        "average_response_time_ms": u(lat), "error_rate": error, "health": healthy,
        "load_pressure": lp, "manually_failed": cond == "failed",
    }


def make_probe(category: str, r: random.Random):
    if category == "two_degraded":
        bad = r.sample(BAD, 2) if r.random() < 0.7 else [r.choice(BAD)] * 2
    elif category == "three_degraded":
        bad = [r.choice(BAD) for _ in range(3)]
    elif category == "busy_vs_degraded":
        bad = [r.choice(BAD) for _ in range(r.choice([1, 2]))]
    else:
        bad = list(CATEGORIES[category])
    good_kind = "busy" if category == "busy_vs_degraded" else None
    conds = bad + [good_kind or r.choice(["healthy", "healthy", "busy"]) for _ in range(N - len(bad))]
    r.shuffle(conds)
    metrics = [sample_server(cd, r) for cd in conds]
    acceptable = [i for i, cd in enumerate(conds) if cd in GOOD]
    return metrics, conds, acceptable


def run_probes(n_probes: int = 800, seed: int = 0, model_path: str = MODEL_PATH) -> Dict:
    from dqn.agent import DQNAgent
    agent = DQNAgent()
    agent.load(model_path)
    r = random.Random(seed)
    cats = list(CATEGORIES)
    pol_rr = {"i": -1}
    rr_rng, rnd = random.Random(seed + 1), random.Random(seed + 2)

    def rr(metrics):
        pol_rr["i"] = (pol_rr["i"] + 1) % N
        return pol_rr["i"]

    def lc(metrics):
        lo = min(m["active_requests"] for m in metrics)
        return rr_rng.choice([i for i, m in enumerate(metrics) if m["active_requests"] == lo])

    def dqn(metrics):
        q = agent.q_values(build_state(metrics))
        return int(np.argmax(q))

    policies = {"DQN": dqn, "Round Robin": rr, "Random": lambda m: rnd.randrange(N), "Least Connections": lc}
    stats = {p: {"ok": 0, "n": 0, "cat": {c: [0, 0] for c in cats}} for p in policies}
    examples = []
    for k in range(n_probes):
        cat = cats[k % len(cats)]
        metrics, conds, acceptable = make_probe(cat, r)
        for p, fn in policies.items():
            a = fn(metrics)
            ok = a in acceptable
            stats[p]["ok"] += ok; stats[p]["n"] += 1
            stats[p]["cat"][cat][0] += ok; stats[p]["cat"][cat][1] += 1
        if len(examples) < 12:
            q = agent.q_values(build_state(metrics))
            examples.append({"category": cat, "conditions": conds,
                             "q_values": [round(float(x), 4) for x in q],
                             "dqn_choice": int(np.argmax(q)), "acceptable": acceptable})
    return {
        "n_probes": n_probes, "categories": cats,
        "results": {p: {"accuracy": s["ok"] / s["n"],
                        "per_category": {c: (v[0] / v[1] if v[1] else 0.0) for c, v in s["cat"].items()}}
                    for p, s in stats.items()},
        "examples": examples,
        "note": "Synthetic snapshots built with value ranges calibrated to the real backend presets; "
                "a decision is correct when the chosen server is healthy/busy-but-healthy, never degraded or failed.",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default=MODEL_PATH)
    ap.add_argument("--threshold", type=float, default=0.90)
    args = ap.parse_args()
    res = run_probes(args.n, args.seed, args.model)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, "policy_probes.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    cats = res["categories"]
    print(f"{'policy':<18s} {'overall':>8s} " + " ".join(f"{c[:14]:>14s}" for c in cats))
    for p, v in res["results"].items():
        print(f"{p:<18s} {v['accuracy']*100:7.1f}% " + " ".join(f"{v['per_category'][c]*100:13.1f}%" for c in cats))
    acc = res["results"]["DQN"]["accuracy"]
    verdict = "PASS" if acc >= args.threshold else "FAIL"
    print(f"\nDQN probe decision accuracy = {acc*100:.1f}%  (threshold {args.threshold*100:.0f}%) -> {verdict}")
    print(f"Saved {out}")
    sys.exit(0 if verdict == "PASS" else 1)


if __name__ == "__main__":
    main()
