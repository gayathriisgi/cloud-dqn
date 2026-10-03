"""
mycamu/config.py
================
Every service address MyCamu talks to is configurable through environment
variables, so the same code runs:

* inside Docker Compose  -> service DNS names, backends on port 5000
      ROUND_ROBIN_URL=http://round-robin:8000   SERVER1_URL=http://server1:5000 ...
* on a bare workstation  -> localhost + the published host ports
      (defaults below: 8000-8003 balancers, 5001-5004 backends)

Inside the Docker network the backend servers listen on 5000; 5001-5004
are only HOST port mappings and must never be used between containers.
"""

import os


def _url(name: str, default: str) -> str:
    return os.environ.get(name, default).rstrip("/")


BALANCER_URLS = {
    "round_robin": _url("ROUND_ROBIN_URL", "http://localhost:8000"),
    "least_connections": _url("LEAST_CONNECTIONS_URL", "http://localhost:8001"),
    "aco": _url("ACO_URL", "http://localhost:8002"),
    "dqn": _url("DQN_URL", "http://localhost:8003"),
}

SERVER_URLS = {
    "server1": _url("SERVER1_URL", "http://localhost:5001"),
    "server2": _url("SERVER2_URL", "http://localhost:5002"),
    "server3": _url("SERVER3_URL", "http://localhost:5003"),
    "server4": _url("SERVER4_URL", "http://localhost:5004"),
}

ALGORITHM_LABELS = {
    "dqn": "DQN",
    "round_robin": "Round Robin",
    "least_connections": "Least Connections",
    "aco": "ACO",
}

DATA_DIR = os.environ.get("MYCAMU_DATA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
DB_PATH = os.environ.get("MYCAMU_DB_PATH", os.path.join(DATA_DIR, "mycamu.db"))
PORT = int(os.environ.get("MYCAMU_PORT", "8080"))
DEFAULT_ALGORITHM = os.environ.get("MYCAMU_DEFAULT_ALGORITHM", "dqn")

REQUEST_TIMEOUT = float(os.environ.get("MYCAMU_REQUEST_TIMEOUT", "10"))
HEALTH_TIMEOUT = float(os.environ.get("MYCAMU_HEALTH_TIMEOUT", "2"))
MAX_TRAFFIC_REQUESTS = int(os.environ.get("MYCAMU_MAX_TRAFFIC_REQUESTS", "2000"))

# Workload presets - identical numbers to dqn/environment.py SCENARIOS so
# a "slow" server in MyCamu is exactly the "high_latency" condition the
# DQN was trained/evaluated on. "normal" = the backend's own defaults
# (servers/workload.py WorkloadParams).
WORKLOAD_PRESETS = {
    "normal": {"cpu_level": 5, "memory_mb": 16, "base_latency_ms": 15, "jitter_ms": 10, "capacity": 12, "error_rate": 0.01},
    "slow": {"cpu_level": 20, "memory_mb": 40, "base_latency_ms": 250, "jitter_ms": 100, "capacity": 10, "error_rate": 0.02},
    "overload": {"cpu_level": 92, "memory_mb": 420, "base_latency_ms": 180, "jitter_ms": 80, "capacity": 4, "error_rate": 0.15},
    "medium": {"cpu_level": 30, "memory_mb": 70, "base_latency_ms": 35, "jitter_ms": 15, "capacity": 12, "error_rate": 0.02},
}
