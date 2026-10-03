"""
dqn/reward.py
=============
Multi-objective reward function (Section 3 of the spec).

    reward = + w_success   * (+1 success / -1 failure)
             + w_throughput * throughput_norm(system-wide, post-action)
             + w_health     * (+1 healthy chosen server / -1 unhealthy)
             - w_latency    * latency_norm(this request's actual response time)
             - w_cpu        * cpu_norm(chosen server, post-action)
             - w_memory     * memory_norm(chosen server, post-action)
             - w_queue      * queue_norm(chosen server, post-action)
             - w_active     * active_norm(chosen server, post-action)
             - w_error      * error_norm(chosen server, post-action)
             - w_imbalance  * imbalance_norm(load_pressure variance across all servers)

All weights come from ``dqn.config.RewardWeights`` (configurable via
environment variables); every term is a normalized quantity in
[0, 1] (or [-1, 1] for the two "how did the request go" terms), so no
single metric can dominate purely because of scale - a genuinely large
weight is a deliberate configuration choice, not a units accident.

This function is deliberately *only* a function of measured outcomes
(what happened after the action was executed). It never looks at which
action index was chosen, or encodes a rule like "if server == 2 then
...". The DQN has to learn, from reward signal alone, that certain
regions of state space are associated with good outcomes - which is
what makes the resulting policy an emergent, learned behaviour instead
of a hidden heuristic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

from common.settings import get_settings
from dqn.config import REWARD_WEIGHTS, RewardWeights

SETTINGS = get_settings()


@dataclass
class RewardBreakdown:
    total: float
    success_term: float
    throughput_term: float
    health_term: float
    latency_penalty: float
    cpu_penalty: float
    memory_penalty: float
    queue_penalty: float
    active_penalty: float
    error_penalty: float
    imbalance_penalty: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "total": round(self.total, 4),
            "success_term": round(self.success_term, 4),
            "throughput_term": round(self.throughput_term, 4),
            "health_term": round(self.health_term, 4),
            "latency_penalty": round(self.latency_penalty, 4),
            "cpu_penalty": round(self.cpu_penalty, 4),
            "memory_penalty": round(self.memory_penalty, 4),
            "queue_penalty": round(self.queue_penalty, 4),
            "active_penalty": round(self.active_penalty, 4),
            "error_penalty": round(self.error_penalty, 4),
            "imbalance_penalty": round(self.imbalance_penalty, 4),
        }


def _latency_ms(metrics: Dict) -> float:
    lm = metrics.get("average_response_time_ms")
    if lm is None:
        lm = float(metrics.get("average_response_time", 0.0)) * 1000.0
    return float(lm)


def load_variance(all_server_metrics: Sequence[Dict]) -> float:
    pressures = [float((m or {}).get("load_pressure", 1.0)) for m in all_server_metrics]
    if not pressures:
        return 0.0
    mean = sum(pressures) / len(pressures)
    return sum((p - mean) ** 2 for p in pressures) / len(pressures)


def compute_reward(
    success: bool,
    response_time_s: float,
    chosen_server_metrics: Dict,
    all_server_metrics: Sequence[Dict],
    weights: RewardWeights = REWARD_WEIGHTS,
) -> RewardBreakdown:
    s = SETTINGS
    chosen = chosen_server_metrics or {}

    latency_norm = s.clamp01((response_time_s * 1000.0) / s.latency_norm_ref_ms)
    cpu_norm = s.clamp01(float(chosen.get("cpu_usage", 100.0)) / 100.0)
    mem_norm = s.clamp01(float(chosen.get("memory_usage", 100.0)) / 100.0)
    queue_norm = s.clamp01(float(chosen.get("queue_length", s.queue_norm_ref)) / s.queue_norm_ref)
    active_norm = s.clamp01(float(chosen.get("active_requests", s.active_norm_ref)) / s.active_norm_ref)
    error_norm = s.clamp01(float(chosen.get("error_rate", 1.0)))
    healthy = bool(chosen.get("health", False))

    n = max(1, len(all_server_metrics))
    total_throughput = sum(float((m or {}).get("throughput", 0.0)) for m in all_server_metrics)
    throughput_norm = s.clamp01(total_throughput / (n * s.throughput_norm_ref))

    imbalance_norm = s.clamp01(load_variance(all_server_metrics) / s.load_variance_norm_ref)

    success_term = weights.w_success * (1.0 if success else -1.0)
    throughput_term = weights.w_throughput * throughput_norm
    health_term = weights.w_health * (1.0 if healthy else -1.0)

    latency_penalty = weights.w_latency * latency_norm
    cpu_penalty = weights.w_cpu * cpu_norm
    memory_penalty = weights.w_memory * mem_norm
    queue_penalty = weights.w_queue * queue_norm
    active_penalty = weights.w_active * active_norm
    error_penalty = weights.w_error * error_norm
    imbalance_penalty = weights.w_imbalance * imbalance_norm

    total = (
        success_term + throughput_term + health_term
        - latency_penalty - cpu_penalty - memory_penalty
        - queue_penalty - active_penalty - error_penalty - imbalance_penalty
    )
    total = max(-weights.reward_clip, min(weights.reward_clip, total))

    return RewardBreakdown(
        total=total,
        success_term=success_term,
        throughput_term=throughput_term,
        health_term=health_term,
        latency_penalty=latency_penalty,
        cpu_penalty=cpu_penalty,
        memory_penalty=memory_penalty,
        queue_penalty=queue_penalty,
        active_penalty=active_penalty,
        error_penalty=error_penalty,
        imbalance_penalty=imbalance_penalty,
    )
