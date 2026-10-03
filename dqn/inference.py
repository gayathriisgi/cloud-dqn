"""
dqn/inference.py
=================
Thin, load-balancer-facing wrapper around ``DQNAgent`` used by
``load_balancer/dqn_balancer.py``.

Loading is strict about state-dimension compatibility (see
``dqn.agent.ModelDimensionMismatchError``) - if someone changes the
metric set or server count and forgets to retrain, this raises loudly
at startup instead of quietly serving garbage Q-values.

If no trained checkpoint exists yet, the balancer still starts (so a
fresh clone of the project can be demoed end-to-end before training),
but ``model_loaded`` is reported as ``False`` everywhere it matters
(the ``/status`` endpoint and every routing decision's JSON) so nobody
mistakes an untrained network's Q-values for a trained policy.
"""

from __future__ import annotations

import logging
import os
from typing import Optional, Tuple

import numpy as np

from dqn.agent import DQNAgent, ModelDimensionMismatchError

# Inference-time CPU threading (deployment setting, does not touch the
# network). The balancer already serves many requests on many Flask
# threads; letting every tiny 46->256->128->4 forward pass ALSO spawn
# intra-op worker threads makes them fight for the same cores and adds
# tens of ms of latency under concurrent load. One intra-op thread per
# request is the standard setting for small-model online inference.
try:
    import torch as _torch
    _torch.set_num_threads(max(1, int(os.environ.get("DQN_TORCH_THREADS", "1"))))
except Exception:  # pragma: no cover
    pass
from dqn.config import DQN_HYPERPARAMS, INFERENCE_EPSILON, MODEL_PATH, TIE_MARGIN

logger = logging.getLogger(__name__)


class DQNInferenceService:
    def __init__(self, model_path: str = MODEL_PATH, epsilon: float = INFERENCE_EPSILON,
                 tie_margin: float = TIE_MARGIN):
        self.model_path = model_path
        self.epsilon = epsilon
        self.tie_margin = max(0.0, float(tie_margin))
        self.agent = DQNAgent(DQN_HYPERPARAMS)
        self.model_loaded = False
        self.load_error: Optional[str] = None
        self._try_load()

    def _try_load(self) -> None:
        if not os.path.exists(self.model_path):
            self.load_error = (
                f"No trained model found at '{self.model_path}'. "
                "Serving with an UNTRAINED network (random-ish Q-values). "
                "Run: python dqn/train.py"
            )
            logger.warning(self.load_error)
            return
        try:
            # strict_epsilon=True so the agent's own .epsilon reflects the
            # trained checkpoint's final exploration rate (used only for
            # informational reporting in /status - actual action
            # selection always goes through self.epsilon on THIS class,
            # which defaults to INFERENCE_EPSILON regardless).
            self.agent.load(self.model_path, strict_epsilon=True)
            self.model_loaded = True
            logger.info("Loaded DQN checkpoint from %s (state_dim=%d, action_dim=%d)",
                        self.model_path, DQN_HYPERPARAMS.state_dim, DQN_HYPERPARAMS.action_dim)
        except ModelDimensionMismatchError as e:
            self.load_error = str(e)
            logger.error(self.load_error)
        except Exception as e:  # pragma: no cover - defensive
            self.load_error = f"Failed to load model at '{self.model_path}': {e}"
            logger.error(self.load_error)

    def reload(self) -> None:
        self._try_load()

    def predict(self, state: np.ndarray, epsilon: Optional[float] = None) -> Tuple[int, np.ndarray, bool]:
        action, q, exploring, _ = self.select(state, epsilon)
        return action, q, exploring

    def select(self, state: np.ndarray, epsilon: Optional[float] = None):
        """Returns (action, q_values, was_exploration, tie_candidates).

        epsilon-greedy as before; on a greedy step, every action whose
        Q-value is within ``tie_margin`` of the maximum is a candidate and
        one is drawn uniformly (tie_margin=0 -> strict argmax)."""
        eps = self.epsilon if epsilon is None else epsilon
        action, q, exploring = self.agent.select_action(state, epsilon=eps)
        if exploring or self.tie_margin <= 0:
            return action, q, exploring, [int(action)]
        best = float(np.max(q))
        candidates = [int(i) for i in np.flatnonzero(q >= best - self.tie_margin)]
        return int(np.random.choice(candidates)), q, False, candidates

    def status(self) -> dict:
        return {
            "model_path": self.model_path,
            "model_loaded": self.model_loaded,
            "load_error": self.load_error,
            "state_dim": DQN_HYPERPARAMS.state_dim,
            "action_dim": DQN_HYPERPARAMS.action_dim,
            "epsilon": self.epsilon,
            "tie_margin": self.tie_margin,
            "training_epsilon_at_save": self.agent.epsilon,
            "steps_done": self.agent.steps_done,
        }
