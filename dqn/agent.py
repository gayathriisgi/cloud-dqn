"""
dqn/agent.py
============
DQNAgent: epsilon-greedy action selection, the target-network Bellman
learning step, and strict save/load with embedded state-dimension
metadata (Section 2: "Save trained model metadata/state dimension and
detect incompatible models instead of silently loading them.").

Training and inference both go through this one class - there is no
separate "training agent" vs "inference agent" implementation to keep
in sync.
"""

from __future__ import annotations

import os
import time
from dataclasses import asdict
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from dqn.config import DQN_HYPERPARAMS, DQNHyperparams
from dqn.model import QNetwork
from dqn.replay_buffer import ReplayBuffer


class ModelDimensionMismatchError(Exception):
    """Raised when a saved checkpoint's state/action dimension does not
    match the currently configured DQN. This is deliberately a hard
    failure instead of a silent reshape/truncate, per the spec's
    requirement to "detect incompatible models instead of silently
    loading them"."""


class DQNAgent:
    def __init__(self, hyperparams: DQNHyperparams = DQN_HYPERPARAMS, device: Optional[str] = None):
        self.hp = hyperparams
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.online_net = QNetwork(self.hp.state_dim, self.hp.action_dim, self.hp.hidden_sizes).to(self.device)
        self.target_net = QNetwork(self.hp.state_dim, self.hp.action_dim, self.hp.hidden_sizes).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        wd = float(getattr(self.hp, "weight_decay", 0.0) or 0.0)
        if wd > 0:
            # AdamW = Adam with *decoupled* weight decay (L2 regularisation
            # that does not interfere with the adaptive step size).
            self.optimizer = optim.AdamW(self.online_net.parameters(), lr=self.hp.learning_rate, weight_decay=wd)
        else:
            self.optimizer = optim.Adam(self.online_net.parameters(), lr=self.hp.learning_rate)
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss - standard, robust DQN choice

        self.memory = ReplayBuffer(self.hp.buffer_capacity)

        self.epsilon = self.hp.epsilon_start
        self.steps_done = 0
        self.learn_steps = 0

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------
    def q_values(self, state: np.ndarray) -> np.ndarray:
        self.online_net.eval()
        with torch.no_grad():
            x = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q = self.online_net(x).squeeze(0).cpu().numpy()
        self.online_net.train()
        return q

    def select_action(self, state: np.ndarray, epsilon: Optional[float] = None) -> Tuple[int, np.ndarray, bool]:
        """Returns (action, q_values, was_exploration).

        Q-values are always computed (even on an exploratory step) so
        callers (the DQN balancer's Prometheus exporters and JSON
        response) can always show "what the network currently believes"
        regardless of whether this particular decision was random.
        """
        eps = self.epsilon if epsilon is None else epsilon
        q = self.q_values(state)
        if np.random.rand() < eps:
            action = np.random.randint(self.hp.action_dim)
            return int(action), q, True
        return int(np.argmax(q)), q, False

    # ------------------------------------------------------------------
    # Experience replay + learning
    # ------------------------------------------------------------------
    def store(self, state, action, reward, next_state, done) -> None:
        self.memory.push(state, action, reward, next_state, done)

    def _bellman_target(self, rewards_t, next_states_t, dones_t):
        """Standard DQN Bellman target (Mnih et al., 2015):
        y = r + gamma * max_a Q_target(s', a) * (1 - done)."""
        with torch.no_grad():
            max_next_q = self.target_net(next_states_t).max(dim=1).values
            return rewards_t + self.hp.gamma * max_next_q * (1.0 - dones_t)

    def _to_tensors(self, batch):
        states, actions, rewards, next_states, dones = batch
        return (
            torch.as_tensor(states, dtype=torch.float32, device=self.device),
            torch.as_tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1),
            torch.as_tensor(rewards, dtype=torch.float32, device=self.device),
            torch.as_tensor(next_states, dtype=torch.float32, device=self.device),
            torch.as_tensor(dones, dtype=torch.float32, device=self.device),
        )

    def td_loss_on(self, buffer: ReplayBuffer, max_samples: int = 2048) -> Optional[float]:
        """Huber TD-error on a buffer WITHOUT any gradient step.

        Used by dqn/train.py to compare the loss on transitions the
        network was trained on (replay memory) with the loss on a
        HELD-OUT validation buffer it never trained on. A widening gap
        (train loss falling, validation loss rising) is the textbook
        overfitting signal; both staying high is underfitting.
        """
        if len(buffer) < 32:
            return None
        n = min(max_samples, len(buffer))
        batch = buffer.sample(n)
        s_t, a_t, r_t, ns_t, d_t = self._to_tensors(batch)
        self.online_net.eval()
        with torch.no_grad():
            q = self.online_net(s_t).gather(1, a_t).squeeze(1)
            target = self._bellman_target(r_t, ns_t, d_t)
            loss = self.loss_fn(q, target)
        self.online_net.train()
        return float(loss.item())

    def set_learning_rate(self, lr: float) -> None:
        for g in self.optimizer.param_groups:
            g["lr"] = lr

    def current_learning_rate(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])

    def learn(self, decay_epsilon: bool = True) -> Optional[float]:
        """One gradient step on a replay minibatch.

        ``decay_epsilon=False`` lets dqn/train.py drive epsilon with its
        own step-based schedule instead of the legacy per-learn-step
        multiplicative decay (kept as default for the online-learning
        path in load_balancer/dqn_balancer.py)."""
        if len(self.memory) < max(self.hp.min_replay_before_learn, self.hp.batch_size):
            return None

        states, actions, rewards, next_states, dones = self.memory.sample(self.hp.batch_size)

        states_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        next_states_t = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        dones_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device)

        q_values = self.online_net(states_t).gather(1, actions_t).squeeze(1)

        target_q = self._bellman_target(rewards_t, next_states_t, dones_t)

        loss = self.loss_fn(q_values, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), self.hp.grad_clip_norm)
        self.optimizer.step()

        if decay_epsilon:
            self.epsilon = max(self.hp.epsilon_min, self.epsilon * self.hp.epsilon_decay)

        self.learn_steps += 1
        tau = float(getattr(self.hp, "tau", 0.0) or 0.0)
        if tau > 0:
            with torch.no_grad():
                for tp, op in zip(self.target_net.parameters(), self.online_net.parameters()):
                    tp.mul_(1.0 - tau).add_(op, alpha=tau)
        elif self.learn_steps % self.hp.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return float(loss.item())

    def step_done(self) -> None:
        self.steps_done += 1

    # ------------------------------------------------------------------
    # Save / load with strict dimension metadata
    # ------------------------------------------------------------------
    def save(self, path: str, extra: Optional[dict] = None) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        checkpoint = {
            "model_version": self.hp.model_version,
            "saved_at": time.time(),
            "state_dim": self.hp.state_dim,
            "action_dim": self.hp.action_dim,
            "hidden_sizes": self.hp.hidden_sizes,
            "hyperparams": asdict(self.hp),
            "online_state_dict": self.online_net.state_dict(),
            "target_state_dict": self.target_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "steps_done": self.steps_done,
            "learn_steps": self.learn_steps,
        }
        if extra:
            checkpoint["training_report"] = extra
        # Write to a temp file then atomically replace, so a balancer that
        # hot-reloads (/reload_model) never sees a half-written file.
        tmp = path + ".tmp"
        torch.save(checkpoint, tmp)
        os.replace(tmp, path)

    def load(self, path: str, strict_epsilon: bool = False) -> dict:
        """Load a checkpoint, raising ModelDimensionMismatchError if the
        checkpoint's state/action dimension (or hidden layer shape)
        doesn't match this agent's current configuration - never
        silently loads an incompatible model.
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        ckpt_state_dim = checkpoint.get("state_dim")
        ckpt_action_dim = checkpoint.get("action_dim")
        ckpt_hidden = checkpoint.get("hidden_sizes")

        if ckpt_state_dim != self.hp.state_dim or ckpt_action_dim != self.hp.action_dim:
            raise ModelDimensionMismatchError(
                f"Checkpoint at '{path}' was trained with state_dim={ckpt_state_dim}, "
                f"action_dim={ckpt_action_dim}, but the current configuration expects "
                f"state_dim={self.hp.state_dim}, action_dim={self.hp.action_dim}. "
                "This usually means the backend metric set or server count changed "
                "since the model was trained. Retrain with `python dqn/train.py` "
                "before serving inference with load_balancer/dqn_balancer.py."
            )
        if ckpt_hidden is not None and list(ckpt_hidden) != list(self.hp.hidden_sizes):
            raise ModelDimensionMismatchError(
                f"Checkpoint at '{path}' uses hidden layer sizes {ckpt_hidden}, but the "
                f"current configuration expects {self.hp.hidden_sizes}. Retrain with "
                "`python dqn/train.py` or restore the matching DQN_HIDDEN_SIZES config."
            )

        self.online_net.load_state_dict(checkpoint["online_state_dict"])
        self.target_net.load_state_dict(checkpoint.get("target_state_dict", checkpoint["online_state_dict"]))
        self.online_net.to(self.device)
        self.target_net.to(self.device)

        if "optimizer_state_dict" in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            except Exception:
                pass  # optimizer state is best-effort; never blocks inference

        if strict_epsilon:
            self.epsilon = checkpoint.get("epsilon", self.epsilon)
        self.steps_done = checkpoint.get("steps_done", self.steps_done)
        self.learn_steps = checkpoint.get("learn_steps", self.learn_steps)
        return checkpoint
