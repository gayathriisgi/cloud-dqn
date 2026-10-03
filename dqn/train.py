#!/usr/bin/env python3
"""
dqn/train.py  (v3 - generalisation-aware training)
===================================================
Trains the SAME DQN (dqn/model.py: state_dim -> 256 -> 128 -> 4, unchanged)
against the REAL backend Flask servers, but with a training procedure that
is explicitly designed to neither OVERFIT nor UNDERFIT:

ANTI-OVERFITTING
  * Train / validation split of workload scenarios. Validation episodes use
    a different random seed, the exact (non-randomised) presets, and two
    held-out scenario families the agent never trains on.
  * Early stopping on the greedy validation score with patience, and the
    BEST-validation checkpoint (not the last one) is what gets saved to
    models/dqn_model.pt. The last weights go to models/dqn_model_last.pt.
  * Held-out TD-loss: transitions collected during validation are kept in
    a separate buffer the network never trains on; the gap between the
    training-buffer TD loss and the held-out TD loss is logged every
    evaluation (classic over-fit detector).
  * Domain randomisation of workload presets (+-25% on every parameter).
  * Server-permutation augmentation: every REAL transition is also stored
    with the 4 servers re-ordered (servers are interchangeable), so the
    network cannot memorise "server N is usually good".
  * Standard DQN target with a frozen target network (NOT Double DQN).
  * AdamW decoupled weight decay (L2 regularisation).
  * Gradient clipping + Huber loss (already present in v2).

ANTI-UNDERFITTING
  * Concurrent agent-driven clients (--workers): several threads route
    through the SAME agent at once, so it learns the cost of herding
    concurrent requests onto one server (train/test match: a real
    balancer serves many requests in parallel).
  * More, longer, more varied experience: background concurrent traffic
    (real queueing!), dynamic mid-episode changes (overload shifts, server
    flapping, failure+recovery), every scenario family visited in a
    shuffled curriculum.
  * Linear epsilon schedule over a fixed fraction of training (instead of
    the v2 per-learn-step decay that reached epsilon_min after ~600 steps
    and then stopped exploring).
  * Cosine learning-rate schedule (large steps early, fine steps late).
  * Underfit check: the trained policy must beat simple reference policies
    (round robin, random, least-active) on the SAME validation episodes,
    otherwise the fit report says UNDERFIT.

Outputs
  models/dqn_model.pt          best-validation checkpoint (+ training_report)
  models/dqn_model_last.pt     final weights
  logs/training_history.csv    per-episode training log (v2 columns kept)
  logs/training_history.json
  logs/validation_history.json per-evaluation train/val scores + TD losses
  logs/fit_report.json         final GOOD FIT / OVERFIT / UNDERFIT diagnosis
  results/training_fit_curves.png

Usage
  python dqn/train.py                         # sensible defaults (~15-25 min)
  python dqn/train.py --episodes 120 --steps-per-episode 50
  python dqn/train.py --quick                 # smoke test (~2 min)

Prerequisite: the 4 backend servers must be reachable
  docker compose up -d server1 server2 server3 server4
"""

from __future__ import annotations

import argparse
import copy
import csv
import dataclasses
import json
import math
import os
import random
import sys
import threading
import time
from typing import Callable, Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from common.formulas import jain_fairness_index, summarize_latencies  # noqa: E402
from dqn.agent import DQNAgent  # noqa: E402
from dqn.config import (DQN_HYPERPARAMS, MODEL_PATH, RESULTS_DIR,  # noqa: E402
                        TRAINING_LOG_DIR)
from dqn.environment import (SCENARIOS, BackgroundTraffic,  # noqa: E402
                             LoadBalancerEnvironment)
from dqn.reward import load_variance  # noqa: E402
from dqn.replay_buffer import ReplayBuffer  # noqa: E402

# Presets that count as "degraded" - used ONLY for logging the
# bad-routing rate (fraction of requests sent to a degraded server while a
# non-degraded one existed). It is never fed to the network or the reward.
DEGRADED_PRESETS = {"high", "cpu_pressure", "memory_pressure", "queue_buildup",
                    "high_latency", "overload"}

TRAIN_MODES = [
    "uniform_idle", "uniform_low", "uniform_medium", "uniform_high",
    "single_overload", "double_overload", "cpu_pressure_one",
    "memory_pressure_one", "queue_buildup_one", "high_latency_one",
    "unequal_load", "failure_then_recovery", "mixed_dynamic",
    # v3 dynamic families
    "overload_shift", "flapping", "hard_failure",
]
# Families NEVER seen during training (generalisation check).
HELDOUT_MODES = ["slow_plus_overload", "three_degraded"]
# Backwards-compatible alias used by older scripts/docs.
EPISODE_MODES = TRAIN_MODES


# ----------------------------------------------------------------------
# Episode configuration (fully pre-generated so the SAME episode can be
# replayed for several policies - essential for a fair validation)
# ----------------------------------------------------------------------
def build_episode_config(mode: str, n: int, rng: random.Random, steps: int,
                         bg_max: float) -> Dict:
    servers = list(range(n))
    events: List[Dict] = []   # {"t": step, "phase": {...}} / {"t", "fail": i} / {"t", "recover": i}
    phase: Dict[int, str]

    def victim():
        return rng.choice(servers)

    if mode == "uniform_idle":
        phase = {i: "idle" for i in servers}
    elif mode == "uniform_low":
        phase = {i: "low" for i in servers}
    elif mode == "uniform_medium":
        phase = {i: "medium" for i in servers}
    elif mode == "uniform_high":
        phase = {i: "high" for i in servers}
    elif mode == "single_overload":
        v = victim(); phase = {i: ("overload" if i == v else "low") for i in servers}
    elif mode == "double_overload":
        vs = set(rng.sample(servers, 2)); phase = {i: ("overload" if i in vs else "low") for i in servers}
    elif mode in ("cpu_pressure_one", "memory_pressure_one", "queue_buildup_one", "high_latency_one"):
        preset = mode.replace("_one", "")
        v = victim(); phase = {i: (preset if i == v else "low") for i in servers}
    elif mode == "unequal_load":
        presets = ["idle", "low", "medium", "high"]; rng.shuffle(presets)
        phase = {i: presets[i % 4] for i in servers}
    elif mode == "failure_then_recovery":
        phase = {i: "medium" for i in servers}
        v = victim()
        events = [{"t": 0, "fail": v}, {"t": steps // 2, "recover": v}]
    elif mode == "hard_failure":
        phase = {i: "low" for i in servers}
        events = [{"t": 0, "fail": victim()}]
    elif mode == "mixed_dynamic":
        pool = list(SCENARIOS.keys())
        phase = {i: rng.choice(pool) for i in servers}
        events = [{"t": steps // 2, "phase": {i: rng.choice(pool) for i in servers}}]
    elif mode == "overload_shift":
        a, b = rng.sample(servers, 2)
        phase = {i: ("overload" if i == a else "low") for i in servers}
        events = [{"t": steps // 2, "phase": {i: ("overload" if i == b else "low") for i in servers}}]
    elif mode == "flapping":
        v = victim()
        phase = {i: "low" for i in servers}
        period = max(4, steps // 5)
        for k, t in enumerate(range(period, steps, period)):
            events.append({"t": t, "phase": {v: ("overload" if k % 2 == 0 else "low")}})
    elif mode == "slow_plus_overload":       # held-out
        a, b = rng.sample(servers, 2)
        phase = {i: ("high_latency" if i == a else "overload" if i == b else "low") for i in servers}
    elif mode == "three_degraded":           # held-out
        good = victim()
        bad = ["overload", "high_latency", "queue_buildup"]; rng.shuffle(bad)
        it = iter(bad)
        phase = {i: ("low" if i == good else next(it)) for i in servers}
    else:
        phase = {i: "low" for i in servers}

    return {"mode": mode, "phase": phase, "events": events, "steps": steps,
            "bg_rate": round(rng.uniform(0.0, bg_max), 2)}


def degraded_set(phase: Dict[int, str], failed: set) -> set:
    return {i for i, p in phase.items() if p in DEGRADED_PRESETS} | set(failed)


# ----------------------------------------------------------------------
# Reference (non-learning) policies used for the UNDERFIT check
# ----------------------------------------------------------------------
class RoundRobinPolicy:
    def __init__(self, n): self.n, self.i = n, -1
    def __call__(self, state, metrics):
        self.i = (self.i + 1) % self.n
        return self.i, None


class RandomPolicy:
    def __init__(self, n, seed=0): self.n, self.r = n, random.Random(seed)
    def __call__(self, state, metrics): return self.r.randrange(self.n), None


class LeastActivePolicy:
    """Least (active + queued) requests, read from the same /metrics the
    DQN sees - a strong, simple, *non-learning* reference."""
    def __init__(self, n, seed=0): self.n, self.r = n, random.Random(seed)
    def __call__(self, state, metrics):
        load = []
        for m in metrics:
            m = m or {}
            if not m.get("health", False):
                load.append(1e9)
            else:
                load.append(float(m.get("active_requests", 0)) + float(m.get("queue_length", 0)))
        lo = min(load)
        return self.r.choice([i for i, v in enumerate(load) if v == lo]), None


def dqn_greedy_policy(agent: DQNAgent):
    def _p(state, metrics):
        a, q, _ = agent.select_action(state, epsilon=0.0)
        return a, q
    return _p


# ----------------------------------------------------------------------
# Server-permutation data augmentation
# ----------------------------------------------------------------------
def permute_transition(state, action, next_state, n_servers, r: random.Random):
    """The 4 backend servers are interchangeable: if routing to server A in
    state s produced reward r, then in the state where A's metrics sit in
    slot B, routing to B produces the same r. Re-ordering the per-server
    feature blocks of (s, s') and re-labelling the action yields a VALID
    transition built purely from REAL measurements (no synthetic numbers).

    This multiplies the real experience and forces the network to learn
    "avoid the server whose metrics look bad" instead of positional
    shortcuts like "server3 is usually fine" - a classic overfitting mode
    when one server happened to be healthy more often during training."""
    from dqn.config import FEATURES_PER_SERVER as F
    perm = list(range(n_servers))
    while perm == list(range(n_servers)):
        r.shuffle(perm)
    def p(x):
        x = np.asarray(x, dtype=np.float32)
        blocks = x[: n_servers * F].reshape(n_servers, F)[perm].reshape(-1)
        return np.concatenate([blocks, x[n_servers * F:]])
    # new slot j holds old server perm[j]; old action a is now slot perm.index(a)
    return p(state), perm.index(int(action)), p(next_state)


# ----------------------------------------------------------------------
# Running one episode
# ----------------------------------------------------------------------
def run_episode(env: LoadBalancerEnvironment, cfg: Dict, *,
                policy: Optional[Callable] = None,
                agent: Optional[DQNAgent] = None,
                learn: bool = False,
                bg: Optional[BackgroundTraffic] = None,
                randomize: float = 0.0,
                rng: Optional[random.Random] = None,
                settle_seconds: float = 0.4,
                epsilon_fn: Optional[Callable[[], float]] = None,
                lr_fn: Optional[Callable[[], float]] = None,
                transition_sink: Optional[ReplayBuffer] = None,
                workers: int = 1,
                augment: int = 0,
                agent_policy_lock: Optional[threading.Lock] = None) -> Dict:
    """If ``learn`` is True the agent acts epsilon-greedily and trains;
    otherwise ``policy`` (greedy DQN or a reference policy) acts and
    nothing is learned."""
    env.reset_all()
    phase = dict(cfg["phase"])
    env.apply_phase(phase, randomize=randomize, rng=rng)
    if bg is not None:
        bg.set_rate(cfg["bg_rate"])
    failed: set = set()
    events_by_t: Dict[int, List[Dict]] = {}
    for ev in cfg["events"]:
        events_by_t.setdefault(ev["t"], []).append(ev)

    def apply_events(t):
        for ev in events_by_t.get(t, []):
            if "fail" in ev:
                env.fail_server(ev["fail"]); failed.add(ev["fail"])
            if "recover" in ev:
                env.recover_server(ev["recover"]); failed.discard(ev["recover"])
            if "phase" in ev:
                phase.update(ev["phase"])
                env.apply_phase(ev["phase"], randomize=randomize, rng=rng)

    apply_events(0)
    time.sleep(settle_seconds)
    state, metrics = env.get_state()

    steps = cfg["steps"]
    rewards, response_times, successes, losses, q_all = [], [], [], [], []
    cpu_vals, mem_vals, queue_vals, thr_vals, var_vals = [], [], [], [], []
    action_counts = [0] * env.n_servers
    counters = {"t": 0, "bad": 0, "bad_possible": 0}
    step_lock = threading.Lock()    # shared step counter / stats / events
    agent_lock = threading.Lock()   # network forward pass + replay + gradient step
    perm_rng = random.Random((rng.random() if rng else random.random()))

    def worker(state, metrics):
        """One concurrent "client". With workers > 1 several of these run at
        once against the SAME agent, so the agent experiences the
        consequences of its own concurrent decisions (queue/active build-up
        when it herds requests onto one server) - exactly the situation it
        faces behind a real load balancer."""
        while True:
            with step_lock:
                t = counters["t"]
                if t >= steps:
                    return
                counters["t"] += 1
                if t > 0:
                    apply_events(t)
                degraded = degraded_set(phase, failed)
            if learn:
                with agent_lock:
                    if epsilon_fn:
                        agent.epsilon = epsilon_fn()
                    if lr_fn:
                        agent.set_learning_rate(lr_fn())
                    action, q_values, _ = agent.select_action(state)
            elif agent_policy_lock is not None:
                with agent_policy_lock:
                    action, q_values = policy(state, metrics)
            else:
                action, q_values = policy(state, metrics)

            next_state, breakdown, info = env.step(action)

            if learn:
                with agent_lock:
                    agent.store(state, action, breakdown.total, next_state, False)
                    for _ in range(augment):
                        ps, pa, pns = permute_transition(state, action, next_state, env.n_servers, perm_rng)
                        agent.store(ps, pa, breakdown.total, pns, False)
                    loss = agent.learn(decay_epsilon=False)
                    agent.step_done()
            else:
                loss = None
                if transition_sink is not None:
                    with step_lock:
                        transition_sink.push(state, action, breakdown.total, next_state, False)

            m = info["server_metrics"]
            with step_lock:
                if len(degraded) < env.n_servers:
                    counters["bad_possible"] += 1
                    if action in degraded:
                        counters["bad"] += 1
                if loss is not None:
                    losses.append(loss)
                rewards.append(breakdown.total)
                response_times.append(info["response_time"])
                successes.append(1.0 if info["success"] else 0.0)
                action_counts[action] += 1
                if q_values is not None:
                    q_all.extend(np.asarray(q_values).tolist())
                cpu_vals.append(np.mean([(x or {}).get("cpu_usage", 100.0) for x in m]))
                mem_vals.append(np.mean([(x or {}).get("memory_usage", 100.0) for x in m]))
                queue_vals.append(np.mean([(x or {}).get("queue_length", 0.0) for x in m]))
                thr_vals.append(sum((x or {}).get("throughput", 0.0) for x in m))
                var_vals.append(load_variance(m))
            state, metrics = next_state, m

    if workers <= 1:
        worker(state, metrics)
    else:
        threads = [threading.Thread(target=worker, args=(state, metrics), daemon=True) for _ in range(workers)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
    bad, bad_possible = counters["bad"], counters["bad_possible"]

    for i in list(failed):
        env.recover_server(i)
    if bg is not None:
        bg.set_rate(0.0)

    lat = summarize_latencies(response_times)
    total_reward = float(sum(rewards))
    return {
        "mode": cfg["mode"],
        "total_reward": total_reward,
        "avg_reward": total_reward / max(1, steps),
        "avg_response_ms": lat["avg_ms"],
        "p95_latency_ms": lat["p95_ms"],
        "throughput": float(np.mean(thr_vals)) if thr_vals else 0.0,
        "success_rate": float(np.mean(successes)) if successes else 0.0,
        "cpu": float(np.mean(cpu_vals)) if cpu_vals else 0.0,
        "memory": float(np.mean(mem_vals)) if mem_vals else 0.0,
        "queue": float(np.mean(queue_vals)) if queue_vals else 0.0,
        "load_variance": float(np.mean(var_vals)) if var_vals else 0.0,
        "fairness": jain_fairness_index(action_counts),
        "bad_routing_rate": (bad / bad_possible) if bad_possible else 0.0,
        "epsilon": agent.epsilon if agent is not None else 0.0,
        "loss": float(np.mean(losses)) if losses else 0.0,
        "q_mean": float(np.mean(q_all)) if q_all else 0.0,
        "q_std": float(np.std(q_all)) if q_all else 0.0,
        "q_max": float(np.max(q_all)) if q_all else 0.0,
        "q_min": float(np.min(q_all)) if q_all else 0.0,
        "action_counts": action_counts,
        "bg_rate": cfg["bg_rate"],
    }


# ----------------------------------------------------------------------
# Fit diagnosis
# ----------------------------------------------------------------------
def diagnose_fit(best_eval: Dict, final_eval: Dict, refs: Dict, gap_tol: float) -> Dict:
    """GOOD FIT / OVERFIT / UNDERFIT from measured numbers only."""
    reasons = []
    best_ref_name = max(refs, key=lambda k: refs[k]["score"])
    best_ref = refs[best_ref_name]["score"]
    val = best_eval["val"]["score"]
    train = best_eval["train_probe"]["score"]
    val_indist = best_eval["val"].get("score_in_distribution") or val
    gap = train - val_indist
    td_ratio = None
    if best_eval.get("train_td_loss") and best_eval.get("val_td_loss"):
        td_ratio = best_eval["val_td_loss"] / max(1e-9, best_eval["train_td_loss"])

    underfit = val <= best_ref
    overfit = gap > gap_tol or (td_ratio is not None and td_ratio > 2.0)
    if underfit:
        reasons.append(f"validation score {val:.4f} does not beat best reference policy "
                       f"'{best_ref_name}' ({best_ref:.4f}) -> model has not learned more than a simple rule")
    if gap > gap_tol:
        reasons.append(f"train-probe score exceeds validation score by {gap:.4f} (> tolerance {gap_tol})")
    if td_ratio is not None and td_ratio > 2.0:
        reasons.append(f"held-out TD loss is {td_ratio:.2f}x the training TD loss")
    verdict = "UNDERFIT" if underfit else ("OVERFIT" if overfit else "GOOD FIT")
    if verdict == "GOOD FIT":
        reasons.append(f"validation score {val:.4f} beats best reference '{best_ref_name}' "
                       f"({best_ref:.4f}); train/val gap {gap:.4f} within tolerance {gap_tol}"
                       + (f"; held-out/train TD-loss ratio {td_ratio:.2f}" if td_ratio else ""))
    return {
        "verdict": verdict, "reasons": reasons,
        "best_val_score": val, "val_score_in_distribution": val_indist,
        "val_score_heldout_families": best_eval["val"].get("score_heldout_families"),
        "train_probe_score": train, "generalisation_gap": gap,
        "td_loss_ratio_val_over_train": td_ratio,
        "reference_scores": {k: v["score"] for k, v in refs.items()},
        "best_reference": best_ref_name,
        "final_val_score": final_eval["val"]["score"],
    }


def plot_fit(val_hist: List[Dict], history: List[Dict], refs: Dict, out_path: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    ref_style = {"round_robin": ("#94a3b8", ":"), "random": ("#a855f7", "-."), "least_active": ("#d97706", "--")}
    ep = [v["episode"] for v in val_hist]
    ax = axes[0]
    ax.plot(ep, [v["train_probe"]["score"] for v in val_hist], "o-", label="train-probe (greedy)")
    ax.plot(ep, [v["val"]["score"] for v in val_hist], "s-", label="validation (held-out)")
    for name, r in refs.items():
        c, ls = ref_style.get(name, ("#64748b", "--"))
        ax.axhline(r["score"], color=c, ls=ls, lw=1.4, label=f"ref: {name} ({r['score']:.3f})")
    best = max(val_hist, key=lambda v: v["val"]["score"])
    ax.axvline(best["episode"], color="k", ls=":", lw=1, label="best checkpoint")
    ax.set_title("Avg reward per request: train vs validation"); ax.set_xlabel("episode"); ax.legend(fontsize=7)
    ax = axes[1]
    ax.plot(ep, [v.get("train_td_loss") or np.nan for v in val_hist], "o-", label="TD loss (train buffer)")
    ax.plot(ep, [v.get("val_td_loss") or np.nan for v in val_hist], "s-", label="TD loss (held-out buffer)")
    ax.set_title("Over-fit detector: TD loss"); ax.set_xlabel("episode"); ax.legend(fontsize=8)
    ax = axes[2]
    ax.plot(ep, [v["val"]["bad_routing_rate"] * 100 for v in val_hist], "s-", label="DQN bad-routing % (val)")
    for name, r in refs.items():
        c, ls = ref_style.get(name, ("#64748b", "--"))
        ax.axhline(r["bad_routing_rate"] * 100, color=c, ls=ls, lw=1.4, label=f"ref: {name} ({r['bad_routing_rate']*100:.1f}%)")
    ax.set_title("Requests sent to a degraded server (%)"); ax.set_xlabel("episode"); ax.legend(fontsize=7)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ----------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Train the DQN load balancer (v3, generalisation-aware).")
    p.add_argument("--episodes", type=int, default=96)
    p.add_argument("--steps-per-episode", type=int, default=50)
    p.add_argument("--settle-seconds", type=float, default=0.6)
    p.add_argument("--val-every", type=int, default=8, help="Evaluate on validation set every N episodes.")
    p.add_argument("--patience", type=int, default=5, help="Early-stop after N evaluations without improvement.")
    p.add_argument("--min-episodes", type=int, default=48, help="Never early-stop before this many episodes.")
    p.add_argument("--val-episodes", type=int, default=8, help="Validation episodes per evaluation.")
    p.add_argument("--randomize", type=float, default=0.25, help="Domain randomisation strength (0 disables).")
    p.add_argument("--bg-max-rps", type=float, default=6.0, help="Max background (other clients) traffic during training.")
    p.add_argument("--workers", type=int, default=4, help="Concurrent agent-driven clients per episode (train AND validation).")
    p.add_argument("--augment", type=int, default=3, help="Server-permutation copies stored per real transition (0 disables).")
    p.add_argument("--explore-fraction", type=float, default=0.5, help="Fraction of training over which epsilon decays linearly.")
    p.add_argument("--gamma", type=float, default=None, help="Override discount factor (default from config).")
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--gap-tolerance", type=float, default=0.15, help="Max acceptable train-val score gap.")
    p.add_argument("--save-every", type=int, default=10, help="(kept for compatibility) also save last weights every N episodes")
    p.add_argument("--model-path", type=str, default=MODEL_PATH)
    p.add_argument("--log-dir", type=str, default=TRAINING_LOG_DIR)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--quick", action="store_true", help="Tiny smoke-test run.")
    args = p.parse_args()

    if args.quick:
        args.episodes, args.steps_per_episode, args.val_every = 12, 20, 4
        args.val_episodes, args.min_episodes, args.patience = 3, 8, 2
        args.workers = min(args.workers, 2)

    random.seed(args.seed); np.random.seed(args.seed)
    try:
        import torch
        torch.manual_seed(args.seed)
    except Exception:
        pass
    train_rng = random.Random(args.seed)
    rand_rng = random.Random(args.seed + 1)

    env = LoadBalancerEnvironment()
    reachable = env.servers_reachable()
    if not all(reachable):
        print("ERROR: not all backend servers are reachable:")
        for url, ok in zip(env.server_urls, reachable):
            print(f"  {url} -> {'OK' if ok else 'UNREACHABLE'}")
        print("\nStart them first:  docker compose up -d server1 server2 server3 server4")
        sys.exit(1)

    overrides = {}
    if args.gamma is not None:
        overrides["gamma"] = args.gamma
    if args.lr is not None:
        overrides["learning_rate"] = args.lr
    hp = dataclasses.replace(DQN_HYPERPARAMS, **overrides) if overrides else DQN_HYPERPARAMS
    print(f"DQN hyperparameters: {hp}")
    agent = DQNAgent(hp)
    if args.resume and os.path.exists(args.model_path):
        print(f"Resuming from checkpoint: {args.model_path}")
        agent.load(args.model_path, strict_epsilon=True)

    n, steps = env.n_servers, args.steps_per_episode
    total_steps = args.episodes * steps

    # ---- fixed validation / train-probe sets (same for every evaluation and every policy)
    val_rng = random.Random(args.seed + 1000)
    val_modes = TRAIN_MODES[:]          # distinct families, fresh seed
    val_rng.shuffle(val_modes)
    val_modes = HELDOUT_MODES + val_modes[: max(0, args.val_episodes - len(HELDOUT_MODES))]
    val_configs = [build_episode_config(m, n, val_rng, steps, args.bg_max_rps) for m in val_modes]
    # Train-probe = the SAME scenario families as the in-distribution part
    # of the validation set, but sampled from the training distribution
    # (training RNG stream + domain randomisation). Score(train-probe) -
    # Score(val, in-distribution part) is therefore a like-for-like
    # generalisation gap.
    probe_rng = random.Random(args.seed + 2000)
    indist_modes = [c["mode"] for c in val_configs if c["mode"] not in HELDOUT_MODES]
    probe_configs = [build_episode_config(m, n, probe_rng, steps, args.bg_max_rps) for m in indist_modes]
    print(f"Validation modes: {[c['mode'] for c in val_configs]}  (held-out families: {HELDOUT_MODES})")

    bg = BackgroundTraffic(env.server_urls, timeout=env.request_timeout)

    # NOTE: validation episodes deliberately run with the same per-config
    # background rate as training, so the comparison is like-for-like.
    def eval_with_bg(configs, policy, sink=None, randomize=0.0):
        out = []
        r = random.Random(args.seed + 3000)  # identical randomisation every evaluation
        for c in configs:
            bg.set_rate(c["bg_rate"])
            out.append(run_episode(env, c, policy=policy, settle_seconds=args.settle_seconds,
                                   transition_sink=sink, bg=bg, randomize=randomize, rng=r,
                                   workers=args.workers))
        indist = [e for e in out if e["mode"] not in HELDOUT_MODES]
        held = [e for e in out if e["mode"] in HELDOUT_MODES]
        return {
            "score_in_distribution": float(np.mean([e["avg_reward"] for e in indist])) if indist else None,
            "score_heldout_families": float(np.mean([e["avg_reward"] for e in held])) if held else None,
            "score": float(np.mean([e["avg_reward"] for e in out])),
            "success_rate": float(np.mean([e["success_rate"] for e in out])),
            "avg_response_ms": float(np.mean([e["avg_response_ms"] for e in out])),
            "p95_latency_ms": float(np.mean([e["p95_latency_ms"] for e in out])),
            "bad_routing_rate": float(np.mean([e["bad_routing_rate"] for e in out])),
            "per_mode": {e["mode"]: round(e["avg_reward"], 4) for e in out},
        }

    print("\nEvaluating reference policies on the validation set (for the underfit check)...")
    refs = {
        "round_robin": eval_with_bg(val_configs, RoundRobinPolicy(n)),
        "random": eval_with_bg(val_configs, RandomPolicy(n, args.seed)),
        "least_active": eval_with_bg(val_configs, LeastActivePolicy(n, args.seed)),
    }
    for k, v in refs.items():
        print(f"  ref {k:<13s} score={v['score']:.4f} succ={v['success_rate']*100:5.1f}% "
              f"lat={v['avg_response_ms']:6.1f}ms bad_routing={v['bad_routing_rate']*100:5.1f}%")

    os.makedirs(args.log_dir, exist_ok=True)
    csv_path = os.path.join(args.log_dir, "training_history.csv")
    json_path = os.path.join(args.log_dir, "training_history.json")
    val_path = os.path.join(args.log_dir, "validation_history.json")
    fit_path = os.path.join(args.log_dir, "fit_report.json")
    last_path = os.path.join(os.path.dirname(os.path.abspath(args.model_path)), "dqn_model_last.pt")

    fieldnames = [
        "episode", "mode", "total_reward", "avg_reward", "avg_response_ms",
        "p95_latency_ms", "throughput", "success_rate", "cpu", "memory",
        "queue", "load_variance", "fairness", "epsilon", "loss",
        "q_mean", "q_std", "q_max", "q_min", "action_counts", "elapsed_seconds",
        "bad_routing_rate", "bg_rate", "learning_rate",
    ]

    global_step = {"t": 0}

    def epsilon_fn():
        frac = min(1.0, global_step["t"] / max(1, args.explore_fraction * total_steps))
        return hp.epsilon_start + frac * (hp.epsilon_min - hp.epsilon_start)

    def lr_fn():
        frac = min(1.0, global_step["t"] / max(1, total_steps))
        return hp.lr_min + 0.5 * (hp.learning_rate - hp.lr_min) * (1 + math.cos(math.pi * frac))

    history, val_hist = [], []
    val_buffer = ReplayBuffer(capacity=6000)
    best_score, best_eval, best_state, bad_evals = -1e9, None, None, 0
    start_time = time.time()
    curriculum: List[str] = []
    stopped_early_at = None

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for ep in range(1, args.episodes + 1):
            if not curriculum:
                curriculum = TRAIN_MODES[:]
                train_rng.shuffle(curriculum)
            mode = curriculum.pop()
            cfg = build_episode_config(mode, n, train_rng, steps, args.bg_max_rps)
            ep_start = time.time()

            def eps_tick():
                e = epsilon_fn(); global_step["t"] += 1; return e

            res = run_episode(env, cfg, agent=agent, learn=True, bg=bg,
                              randomize=args.randomize, rng=rand_rng,
                              settle_seconds=args.settle_seconds,
                              epsilon_fn=eps_tick, lr_fn=lr_fn,
                              workers=args.workers, augment=args.augment)
            res["episode"] = ep
            res["elapsed_seconds"] = round(time.time() - ep_start, 2)
            res["learning_rate"] = agent.current_learning_rate()
            history.append(res)
            row = {k: res.get(k) for k in fieldnames}
            row["action_counts"] = json.dumps(res["action_counts"])
            writer.writerow(row); f.flush()
            print(f"[Ep {ep:3d}/{args.episodes}] {mode:<20s} r={res['avg_reward']:6.3f} "
                  f"resp={res['avg_response_ms']:6.1f}ms succ={res['success_rate']*100:5.1f}% "
                  f"bad={res['bad_routing_rate']*100:5.1f}% bg={res['bg_rate']:4.1f}rps "
                  f"eps={res['epsilon']:.3f} lr={res['learning_rate']:.2e} loss={res['loss']:.4f}")

            if ep % args.save_every == 0:
                agent.save(last_path)

            if ep % args.val_every == 0 or ep == args.episodes:
                v = eval_with_bg(val_configs, dqn_greedy_policy(agent), sink=val_buffer)
                tp = eval_with_bg(probe_configs, dqn_greedy_policy(agent), randomize=args.randomize)
                ev = {"episode": ep, "val": v, "train_probe": tp,
                      "train_td_loss": agent.td_loss_on(agent.memory),
                      "val_td_loss": agent.td_loss_on(val_buffer),
                      "epsilon": agent.epsilon, "learning_rate": agent.current_learning_rate()}
                val_hist.append(ev)
                # A checkpoint can only become "best" once gradient updates
                # have actually happened (an untrained net that gets lucky
                # on a few episodes must never be shipped).
                improved = agent.learn_steps > 0 and v["score"] > best_score + 1e-4
                print(f"  >> VALIDATION ep={ep}: val={v['score']:.4f} train_probe={tp['score']:.4f} "
                      f"gap={tp['score']-(v['score_in_distribution'] or v['score']):+.4f} "
                      f"heldout={v['score_heldout_families'] or 0:.4f} succ={v['success_rate']*100:5.1f}% "
                      f"bad={v['bad_routing_rate']*100:5.1f}% td(train/val)="
                      f"{ev['train_td_loss'] or 0:.4f}/{ev['val_td_loss'] or 0:.4f} "
                      f"{'** new best **' if improved else ''}")
                if improved:
                    best_score, best_eval, bad_evals = v["score"], ev, 0
                    best_state = copy.deepcopy(agent.online_net.state_dict())
                    agent.save(args.model_path, extra={"best_validation": ev, "reference_scores": refs})
                    print(f"     saved best checkpoint -> {args.model_path}")
                else:
                    bad_evals += 1
                with open(val_path, "w") as vf:
                    json.dump({"reference_policies": refs, "evaluations": val_hist}, vf, indent=2)
                if bad_evals >= args.patience and ep >= args.min_episodes:
                    stopped_early_at = ep
                    print(f"  >> EARLY STOPPING at episode {ep} (no validation improvement for "
                          f"{args.patience} evaluations). Best = episode {best_eval['episode']}.")
                    break

    agent.save(last_path)
    with open(json_path, "w") as jf:
        json.dump(history, jf, indent=2)

    if best_eval is None:  # e.g. run too short for any gradient update
        best_eval = val_hist[-1]
    final_eval = val_hist[-1]
    fit = diagnose_fit(best_eval, final_eval, refs, args.gap_tolerance)
    fit.update({
        "best_episode": best_eval["episode"], "episodes_run": len(history),
        "stopped_early_at": stopped_early_at, "elapsed_seconds": round(time.time() - start_time, 1),
        "validation_modes": [c["mode"] for c in val_configs], "heldout_modes": HELDOUT_MODES,
        "hyperparams": {k: (v if not isinstance(v, (list, tuple)) else list(v))
                        for k, v in dataclasses.asdict(hp).items()},
        "args": vars(args),
        "best_val_metrics": best_eval["val"],
    })
    # embed the report in the best checkpoint as well
    if best_state is not None:
        agent.online_net.load_state_dict(best_state)
        agent.target_net.load_state_dict(best_state)
        agent.save(args.model_path, extra={"best_validation": best_eval, "reference_scores": refs, "fit_report": fit})
    with open(fit_path, "w") as ff:
        json.dump(fit, ff, indent=2)
    plot_fit(val_hist, history, refs, os.path.join(RESULTS_DIR, "training_fit_curves.png"))

    bg.close(); env.close()
    print("\n" + "=" * 72)
    print(f"FIT DIAGNOSIS: {fit['verdict']}")
    for r in fit["reasons"]:
        print(f"  - {r}")
    print(f"Best checkpoint: episode {fit['best_episode']} -> {args.model_path}")
    print(f"Logs: {csv_path}, {val_path}, {fit_path}")
    print("Next: curl -X POST http://localhost:8003/reload_model   then   python testing/stress_test_suite.py")
    print("=" * 72)


if __name__ == "__main__":
    main()
