"""
testing/stress_scenarios.py
============================
Scenario catalogue for testing/stress_test_suite.py.

Every scenario is a list of STAGES. A stage says
  * how long it lasts and what traffic pattern/rate is driven at the
    balancer under test (patterns from traffic/traffic_generator.py),
  * which workload preset (dqn.environment.SCENARIOS) each backend server
    gets at the start of the stage,
  * which servers are hard-failed (POST /fail) or recovered (POST /recover).

Stage state is CUMULATIVE: a server keeps its preset / failed flag until a
later stage changes it - exactly like a real incident timeline.

Because we create the conditions ourselves, we know the GROUND TRUTH of
which servers are degraded in each stage. That is what makes a real
"routing accuracy" measurable: a decision is correct when it avoids every
degraded server while at least one healthy server exists. The balancers
themselves never see this ground truth - they only see /metrics.
"""

from __future__ import annotations

from typing import Dict, List

DEGRADED_PRESETS = {"high", "cpu_pressure", "memory_pressure", "queue_buildup",
                    "high_latency", "overload"}

ALL_LOW = {0: "low", 1: "low", 2: "low", 3: "low"}


def _st(name, duration, pattern, rate, workload=None, fail=None, recover=None) -> Dict:
    return {"name": name, "duration": duration, "pattern": pattern, "rate": rate,
            "workload": workload or {}, "fail": fail or [], "recover": recover or []}


STRESS_SCENARIOS: List[Dict] = [
    # ---------------- traffic-shape scenarios (healthy servers) ----------
    {"name": "low_traffic", "category": "traffic",
     "description": "Very light load (~3 req/s) on four healthy servers.",
     "stages": [_st("light_load", 15, "low", 10, ALL_LOW)]},
    {"name": "normal_traffic", "category": "traffic",
     "description": "Steady 10 req/s, mixed low/medium backends (the original comparison setup).",
     "stages": [_st("steady", 20, "normal", 10, {0: "low", 1: "medium", 2: "low", 3: "medium"})]},
    {"name": "high_traffic", "category": "traffic",
     "description": "Sustained high load (~20 req/s).",
     "stages": [_st("high_load", 20, "high", 10, ALL_LOW)]},
    {"name": "spike_traffic", "category": "traffic",
     "description": "Mostly quiet with sharp random request spikes.",
     "stages": [_st("spiky", 20, "spike", 10, ALL_LOW)]},
    {"name": "burst_traffic", "category": "traffic",
     "description": "Alternating 4x bursts and 0.4x lulls.",
     "stages": [_st("bursts", 20, "burst", 8, {0: "low", 1: "medium", 2: "low", 3: "medium"})]},
    {"name": "ramp_up", "category": "traffic",
     "description": "Request rate climbs linearly from 1 to ~24 req/s.",
     "stages": [_st("ramp", 20, "ramp_up", 12, ALL_LOW)]},

    {"name": "flash_crowd", "category": "traffic",
     "description": "Registration-rush flash crowd: sustained ~36 req/s on healthy servers (herding test).",
     "stages": [_st("flash_crowd", 15, "sustained_high", 12, ALL_LOW)]},

    # ---------------- server-condition scenarios -------------------------
    {"name": "slow_server", "category": "degradation",
     "description": "server1 becomes slow (250 ms base latency).",
     "stages": [_st("server1_slow", 20, "normal", 10, {0: "high_latency", 1: "low", 2: "low", 3: "low"})]},
    {"name": "single_overload", "category": "degradation",
     "description": "server1 overloaded (92% CPU, 420 MB, capacity 4, 15% errors).",
     "stages": [_st("server1_overloaded", 20, "normal", 10, {0: "overload", 1: "low", 2: "low", 3: "low"})]},
    {"name": "double_overload", "category": "degradation",
     "description": "server1 and server3 overloaded at the same time.",
     "stages": [_st("two_overloaded", 20, "normal", 10, {0: "overload", 1: "low", 2: "overload", 3: "low"})]},
    {"name": "cpu_memory_pressure", "category": "degradation",
     "description": "server1 CPU-starved, server2 memory-starved.",
     "stages": [_st("resource_pressure", 20, "normal", 10, {0: "cpu_pressure", 1: "memory_pressure", 2: "low", 3: "low"})]},
    {"name": "queue_buildup_high_traffic", "category": "degradation",
     "description": "server4 has tiny capacity (3) while traffic is high - queues build up.",
     "stages": [_st("queue_buildup", 20, "high", 10, {0: "low", 1: "low", 2: "low", 3: "queue_buildup"})]},
    {"name": "heterogeneous_servers", "category": "degradation",
     "description": "One idle, one low, one medium and one high-load server.",
     "stages": [_st("unequal", 20, "normal", 12, {0: "idle", 1: "low", 2: "medium", 3: "high"})]},
    {"name": "spike_during_overload", "category": "degradation",
     "description": "Request spikes arrive while server2 is overloaded.",
     "stages": [_st("spike_overload", 20, "spike", 10, {0: "low", 1: "overload", 2: "low", 3: "low"})]},

    # ---------------- failure / dynamic scenarios ------------------------
    {"name": "hard_failure", "category": "failure",
     "description": "server2 hard-fails (fast 90% error responses - a 'black hole').",
     "stages": [_st("server2_failed", 20, "normal", 10, ALL_LOW, fail=[1])]},
    {"name": "failure_and_recovery", "category": "failure",
     "description": "Healthy -> server2 fails -> server2 recovers.",
     "stages": [_st("healthy", 10, "normal", 10, ALL_LOW),
                _st("server2_failed", 15, "normal", 10, fail=[1]),
                _st("server2_recovered", 15, "normal", 10, recover=[1])]},
    {"name": "flapping_server", "category": "failure",
     "description": "server1 flips between overloaded and healthy every 6 s.",
     "stages": [_st("flap_overload_1", 6, "normal", 10, {0: "overload", 1: "low", 2: "low", 3: "low"}),
                _st("flap_healthy_1", 6, "normal", 10, {0: "low"}),
                _st("flap_overload_2", 6, "normal", 10, {0: "overload"}),
                _st("flap_healthy_2", 6, "normal", 10, {0: "low"})]},
    {"name": "cascading_degradation", "category": "failure",
     "description": "Servers degrade one after another (slow -> overload -> queue buildup).",
     "stages": [_st("one_degraded", 10, "normal", 10, {0: "high_latency", 1: "low", 2: "low", 3: "low"}),
                _st("two_degraded", 10, "normal", 10, {1: "overload"}),
                _st("three_degraded", 10, "normal", 10, {2: "queue_buildup"})]},
    {"name": "overload_migration", "category": "failure",
     "description": "The hotspot moves: server1 overloaded, then server4 instead.",
     "stages": [_st("hotspot_server1", 12, "normal", 10, {0: "overload", 1: "low", 2: "low", 3: "low"}),
                _st("hotspot_server4", 12, "normal", 10, {0: "low", 3: "overload"})]},
]


def scenario_names() -> List[str]:
    return [s["name"] for s in STRESS_SCENARIOS]


def get_scenario(name: str) -> Dict:
    for s in STRESS_SCENARIOS:
        if s["name"] == name:
            return s
    raise KeyError(name)
