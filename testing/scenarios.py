"""
testing/scenarios.py
=====================
Shared scenario definitions so that Round Robin, Least Connections,
ACO and DQN are always evaluated under identical traffic and identical
backend workload conditions (Section 7: "same backend servers, same
traffic scenarios, same evaluation duration, same request conditions
for fair comparison").
"""

from __future__ import annotations

from typing import Dict, List

# A single canonical traffic scenario used by every algorithm in
# testing/compare_algorithms.py: a ramp-up, a normal plateau, a burst,
# then a sustained high-load segment. Each segment is
# {pattern, duration (s), rate (req/s)} and is fed straight into
# traffic.traffic_generator.rate_at().
EVALUATION_TRAFFIC_SCENARIO: List[Dict] = [
    {"pattern": "ramp_up", "duration": 10, "rate": 8},
    {"pattern": "normal", "duration": 15, "rate": 10},
    {"pattern": "burst", "duration": 15, "rate": 10},
    {"pattern": "high", "duration": 10, "rate": 12},
]


def evaluation_scenario_duration(scenario: List[Dict] = EVALUATION_TRAFFIC_SCENARIO) -> float:
    return float(sum(seg["duration"] for seg in scenario))


# Backend workload during the whole algorithm comparison: kept
# realistic-but-moderate (not idle, not permanently overloaded) so
# differences between algorithms are actually visible.
COMPARISON_BACKEND_WORKLOAD: Dict[int, str] = {0: "low", 1: "medium", 2: "low", 3: "medium"}


# The five-phase adaptive-behaviour experiment (Section 8). Every
# phase names the workload preset (from dqn.environment.SCENARIOS)
# applied to each backend server index, plus the traffic pattern/rate
# driven at the balancer under test for that phase.
ADAPTATION_PHASES: List[Dict] = [
    {
        "name": "phase1_all_healthy",
        "duration": 20,
        "workload": {0: "low", 1: "low", 2: "low", 3: "low"},
        "traffic_pattern": "normal",
        "traffic_rate": 10,
    },
    {
        "name": "phase2_overload_server1",
        "duration": 25,
        "workload": {0: "overload", 1: "low", 2: "low", 3: "low"},
        "traffic_pattern": "normal",
        "traffic_rate": 10,
    },
    {
        "name": "phase3_overload_server3",
        "duration": 25,
        # server1 stays overloaded so the experiment can show the DQN
        # juggling *two* simultaneously-bad servers before phase 4
        # recovers the first one.
        "workload": {0: "overload", 1: "low", 2: "overload", 3: "low"},
        "traffic_pattern": "normal",
        "traffic_rate": 10,
    },
    {
        "name": "phase4_recover_server1",
        "duration": 25,
        "workload": {0: "low", 1: "low", 2: "overload", 3: "low"},
        "traffic_pattern": "normal",
        "traffic_rate": 10,
    },
    {
        "name": "phase5_burst_traffic",
        "duration": 20,
        "workload": {0: "low", 1: "low", 2: "low", 3: "low"},
        "traffic_pattern": "burst",
        "traffic_rate": 16,
    },
]


def adaptation_total_duration(phases: List[Dict] = ADAPTATION_PHASES) -> float:
    return float(sum(p["duration"] for p in phases))


# Failure/recovery experiment (Section 17): which server index to fail
# and for how long, before recovering it.
FAILURE_TEST_CONFIG = {
    "server_index": 1,  # server2 (0-indexed)
    "pre_failure_duration": 15,
    "failure_duration": 25,
    "post_recovery_duration": 25,
    "traffic_pattern": "normal",
    "traffic_rate": 10,
}
