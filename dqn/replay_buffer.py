"""
dqn/replay_buffer.py
=====================
Uniform-sampling experience replay buffer. Experience replay is what
turns a stream of highly-correlated, temporally adjacent observations
(request N and request N+1 usually see almost the same server state)
into i.i.d.-ish minibatches, which is what makes the Bellman update in
``dqn/agent.py`` stable enough to converge.
"""

from __future__ import annotations

import random
from collections import deque
from typing import Tuple

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int = 20000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool) -> None:
        self.buffer.append((
            np.asarray(state, dtype=np.float32),
            int(action),
            float(reward),
            np.asarray(next_state, dtype=np.float32),
            float(done),
        ))

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.stack(states).astype(np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.stack(next_states).astype(np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)
