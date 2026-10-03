"""
dqn/config.py
=============
All DQN hyperparameters, reward weights and the state-vector layout in
one place. Every numeric knob mentioned in the README ("reward weights
are configurable", "state dimension must match between training and
inference") lives here, so that ``train.py``, ``inference.py`` and
``load_balancer/dqn_balancer.py`` are structurally unable to disagree
about the shape of the problem.

Everything can be overridden with environment variables so the same
image can be retrained with different hyperparameters purely through
``docker-compose.yml`` / ``.env`` without touching code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List

from common.settings import get_settings

SETTINGS = get_settings()

# ----------------------------------------------------------------------
# STATE VECTOR LAYOUT  (see dqn/state.py for the construction code)
# ----------------------------------------------------------------------
# Per server (10 features), for each of the 4 servers, in server order:
#   [0] cpu_usage_norm          cpu_usage / 100                        clip[0,1]
#   [1] memory_usage_norm       memory_usage / 100                     clip[0,1]
#   [2] queue_length_norm       queue_length / QUEUE_NORM_REF          clip[0,1]
#   [3] active_requests_norm    active_requests / ACTIVE_NORM_REF      clip[0,1]
#   [4] latency_norm            avg_response_time_ms / LATENCY_NORM_REF_MS  clip[0,1]
#   [5] request_rate_norm       request_rate / RATE_NORM_REF           clip[0,1]
#   [6] throughput_norm         throughput / THROUGHPUT_NORM_REF       clip[0,1]
#   [7] error_rate_norm         error_rate (already a 0-1 fraction)    clip[0,1]
#   [8] health                  1.0 healthy / 0.0 unhealthy
#   [9] load_pressure           composite 0-1, computed server-side
#
# Followed by 6 GLOBAL features (computed across all servers):
#   [40] total_request_rate_norm     sum(request_rate)  / (N * RATE_NORM_REF)
#   [41] total_active_norm           sum(active_requests)/ (N * ACTIVE_NORM_REF)
#   [42] overall_latency_norm        mean(avg_response_time_ms) / LATENCY_NORM_REF_MS
#   [43] overall_throughput_norm     sum(throughput) / (N * THROUGHPUT_NORM_REF)
#   [44] global_error_rate           mean(error_rate_norm across servers)
#   [45] traffic_intensity           sum(active_requests) / (N * ACTIVE_NORM_REF)
#                                     (system-wide occupancy of processing
#                                      capacity - distinct from raw request
#                                      rate, captures "how full is the
#                                      system right now")
#
# STATE_DIM = N_SERVERS * 10 + 6 = 4 * 10 + 6 = 46
FEATURES_PER_SERVER = 10
GLOBAL_FEATURES = 6


def compute_state_dim(n_servers: int) -> int:
    return n_servers * FEATURES_PER_SERVER + GLOBAL_FEATURES


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


@dataclass(frozen=True)
class RewardWeights:
    """Multi-objective reward weights (Section 3 of the spec).

    Every weight below is intentionally kept in a similar order of
    magnitude (0.3 - 1.0) so that no single term can dominate the
    learning signal by construction; see README "Multi-metric
    explanation" for the worked-through justification of each number.
    The reward is computed in dqn/reward.py and NEVER encodes a
    hand-written rule like "if cpu > 80 pick server2" - it only scores
    the outcome of whichever action the network already chose, which is
    what allows a genuine value function to be learned instead of a
    heuristic being disguised as one.
    """

    # Positive terms
    w_success: float = field(default_factory=lambda: _env_float("REWARD_W_SUCCESS", 1.0))
    w_throughput: float = field(default_factory=lambda: _env_float("REWARD_W_THROUGHPUT", 0.5))
    w_health: float = field(default_factory=lambda: _env_float("REWARD_W_HEALTH", 0.5))

    # Negative terms (subtracted)
    w_latency: float = field(default_factory=lambda: _env_float("REWARD_W_LATENCY", 1.0))
    w_cpu: float = field(default_factory=lambda: _env_float("REWARD_W_CPU", 0.4))
    w_memory: float = field(default_factory=lambda: _env_float("REWARD_W_MEMORY", 0.3))
    w_queue: float = field(default_factory=lambda: _env_float("REWARD_W_QUEUE", 0.4))
    w_active: float = field(default_factory=lambda: _env_float("REWARD_W_ACTIVE", 0.3))
    w_error: float = field(default_factory=lambda: _env_float("REWARD_W_ERROR", 0.8))
    w_imbalance: float = field(default_factory=lambda: _env_float("REWARD_W_IMBALANCE", 0.5))

    reward_clip: float = field(default_factory=lambda: _env_float("REWARD_CLIP", 5.0))


@dataclass(frozen=True)
class DQNHyperparams:
    n_servers: int = SETTINGS.n_servers
    action_dim: int = SETTINGS.n_servers
    state_dim: int = field(init=False)

    hidden_sizes: List[int] = field(default_factory=lambda: [256, 128])

    learning_rate: float = field(default_factory=lambda: _env_float("DQN_LR", 1e-3))
    # v3: 0.90 (was 0.95). Routing is close to a contextual bandit - the
    # consequence of one decision fades within a few requests - so a
    # shorter horizon gives lower-variance Bellman targets and smaller,
    # better-calibrated Q-values (less over-estimation / over-fitting).
    gamma: float = field(default_factory=lambda: _env_float("DQN_GAMMA", 0.90))

    epsilon_start: float = field(default_factory=lambda: _env_float("DQN_EPSILON_START", 1.0))
    epsilon_min: float = field(default_factory=lambda: _env_float("DQN_EPSILON_MIN", 0.05))
    epsilon_decay: float = field(default_factory=lambda: _env_float("DQN_EPSILON_DECAY", 0.995))

    batch_size: int = field(default_factory=lambda: _env_int("DQN_BATCH_SIZE", 64))
    buffer_capacity: int = field(default_factory=lambda: _env_int("DQN_BUFFER_CAPACITY", 20000))
    target_update_freq: int = field(default_factory=lambda: _env_int("DQN_TARGET_UPDATE_FREQ", 200))
    min_replay_before_learn: int = field(default_factory=lambda: _env_int("DQN_MIN_REPLAY", 256))
    grad_clip_norm: float = field(default_factory=lambda: _env_float("DQN_GRAD_CLIP", 5.0))

    # --- Generalisation / anti-overfitting controls (v3) -----------------
    # None of these change the network ARCHITECTURE (still
    # state_dim -> 256 -> 128 -> action_dim); they only change how the
    # weights are fitted. The algorithm stays
    # STANDARD DQN (Mnih et al. 2015): target = r + gamma * max_a Q_target(s', a).
    #   weight_decay  : decoupled L2 regularisation (AdamW).
    #   lr_min        : floor of the cosine learning-rate schedule driven
    #                   by dqn/train.py (large steps early, fine steps late).
    #   tau           : Polyak soft target update factor applied every learn
    #                   step (0 = keep the original hard copy every
    #                   target_update_freq steps).
    weight_decay: float = field(default_factory=lambda: _env_float("DQN_WEIGHT_DECAY", 1e-5))
    lr_min: float = field(default_factory=lambda: _env_float("DQN_LR_MIN", 1e-4))
    tau: float = field(default_factory=lambda: _env_float("DQN_TAU", 0.0))

    model_version: str = "cloudload-dqn-v3"

    def __post_init__(self):
        object.__setattr__(self, "state_dim", compute_state_dim(self.n_servers))


DQN_HYPERPARAMS = DQNHyperparams()
REWARD_WEIGHTS = RewardWeights()

MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(os.path.dirname(__file__), "..", "models"))
MODEL_PATH = os.environ.get("DQN_MODEL_PATH", os.path.join(MODEL_DIR, "dqn_model.pt"))
TRAINING_LOG_DIR = os.environ.get("TRAINING_LOG_DIR", os.path.join(os.path.dirname(__file__), "..", "logs"))
RESULTS_DIR = os.environ.get("RESULTS_DIR", os.path.join(os.path.dirname(__file__), "..", "results"))

# Inference-time exploration. Defaults to pure exploitation (argmax Q),
# as required by the spec ("normally select argmax(Q)"); can be set >0
# to demonstrate continued exploration during a live demo.
INFERENCE_EPSILON = _env_float("DQN_INFERENCE_EPSILON", 0.0)

# Inference-time TIE TOLERANCE for argmax (v3). Servers whose Q-value is
# within DQN_TIE_MARGIN of the best Q-value are treated as tied and one of
# them is chosen uniformly at random. Measured on the real system: equally
# healthy servers differ by ~0.1-0.2 in Q, while a degraded server sits
# >= 0.32 (slow) to ~3.4 (failed) below the best - so 0.15 only breaks
# ties between genuinely equivalent servers. Without it, a strict argmax
# keeps sending every request to whichever equivalent server happens to be
# 0.01 higher (herding) and re-admits a recovered server very late.
# This does NOT change the network, the Q-values or training - it is the
# standard "argmax with random tie-breaking" rule (the same reason Least
# Connections breaks ties randomly). Set to 0 for a strict argmax.
TIE_MARGIN = _env_float("DQN_TIE_MARGIN", 0.15)

# Optional continual/online learning at the balancer (off by default -
# the balancer normally serves a frozen, reproducible policy trained by
# dqn/train.py; see README "DQN inference" vs "DQN training").
ONLINE_LEARNING_ENABLED = os.environ.get("DQN_ONLINE_LEARNING", "false").strip().lower() in (
    "1", "true", "yes", "on",
)
