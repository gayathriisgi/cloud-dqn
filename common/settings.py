"""
common/settings.py
===================
Single source of truth for infrastructure-level configuration:
backend server addresses, ports, timeouts and the normalization
reference constants used to build the DQN state vector.

Every other module (backend Flask servers, all four load balancers,
the DQN package, the traffic generator, the testing suite) imports
its server list / timeouts / normalization constants from here so
that training, inference and evaluation are always built on exactly
the same numbers.

All values can be overridden with environment variables so the same
code runs unmodified inside Docker (service DNS names such as
``http://server1:5000``) and on a bare workstation
(``http://localhost:5001``) for local development/testing.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def _default_server_urls() -> list:
    """Four backend servers.

    Inside Docker, service names resolve via the compose network
    (``server1`` .. ``server4``). Outside Docker (local development,
    the automated tests, training against a bare workstation) the
    same servers are reached on localhost via the published ports
    5001-5004. ``SERVER_URLS`` always wins when set explicitly.
    """
    raw = os.environ.get("SERVER_URLS")
    if raw:
        return [u.strip().rstrip("/") for u in raw.split(",") if u.strip()]

    if _env_bool("USE_DOCKER_DNS", False):
        return [f"http://server{i}:5000" for i in range(1, 5)]

    return [f"http://localhost:{5000 + i}" for i in range(1, 5)]


@dataclass(frozen=True)
class Settings:
    # ------------------------------------------------------------------
    # Backend servers
    # ------------------------------------------------------------------
    server_urls: list = field(default_factory=_default_server_urls)
    server_names: list = field(
        default_factory=lambda: [f"server{i}" for i in range(1, 5)]
    )
    request_timeout: float = field(
        default_factory=lambda: _env_float("REQUEST_TIMEOUT", 5.0)
    )
    metrics_timeout: float = field(
        default_factory=lambda: _env_float("METRICS_TIMEOUT", 1.5)
    )

    # ------------------------------------------------------------------
    # Load balancer ports (Flask API port / Prometheus exporter port)
    # ------------------------------------------------------------------
    round_robin_port: int = field(default_factory=lambda: _env_int("RR_PORT", 8000))
    round_robin_metrics_port: int = field(
        default_factory=lambda: _env_int("RR_METRICS_PORT", 9000)
    )
    least_conn_port: int = field(default_factory=lambda: _env_int("LC_PORT", 8001))
    least_conn_metrics_port: int = field(
        default_factory=lambda: _env_int("LC_METRICS_PORT", 9001)
    )
    aco_port: int = field(default_factory=lambda: _env_int("ACO_PORT", 8002))
    aco_metrics_port: int = field(
        default_factory=lambda: _env_int("ACO_METRICS_PORT", 9002)
    )
    dqn_port: int = field(default_factory=lambda: _env_int("DQN_PORT", 8003))
    dqn_metrics_port: int = field(
        default_factory=lambda: _env_int("DQN_METRICS_PORT", 9003)
    )

    # ------------------------------------------------------------------
    # Normalization reference constants
    # ------------------------------------------------------------------
    # These are the "100%" points used to squash raw metrics into [0, 1]
    # before they are fed to the neural network. They are deliberately
    # generous relative to the workload simulator's capacity so that a
    # genuinely overloaded server saturates close to 1.0 while a healthy
    # server stays well below it - giving the network a usable gradient
    # across the whole normal-to-overloaded range.
    queue_norm_ref: float = field(default_factory=lambda: _env_float("QUEUE_NORM_REF", 20.0))
    active_norm_ref: float = field(default_factory=lambda: _env_float("ACTIVE_NORM_REF", 30.0))
    latency_norm_ref_ms: float = field(
        default_factory=lambda: _env_float("LATENCY_NORM_REF_MS", 1000.0)
    )
    rate_norm_ref: float = field(default_factory=lambda: _env_float("RATE_NORM_REF", 40.0))
    throughput_norm_ref: float = field(
        default_factory=lambda: _env_float("THROUGHPUT_NORM_REF", 40.0)
    )
    load_variance_norm_ref: float = field(
        default_factory=lambda: _env_float("LOAD_VARIANCE_NORM_REF", 0.12)
    )

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------
    log_level: str = field(default_factory=lambda: os.environ.get("LOG_LEVEL", "INFO"))
    n_servers: int = 4

    def clamp01(self, value: float) -> float:
        return max(0.0, min(1.0, value))


SETTINGS = Settings()


def get_settings() -> Settings:
    """Return the process-wide settings singleton (re-reads nothing;
    environment variables are captured once at import time)."""
    return SETTINGS
