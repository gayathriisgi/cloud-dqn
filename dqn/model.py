"""
dqn/model.py
============
The Q-network used by both training and inference. A plain
fully-connected network is intentionally chosen over anything fancier
(dueling heads, noisy layers, etc.) - the point of this project is to
demonstrate that a *correctly built multi-metric state and reward*
produces genuinely adaptive behaviour, not to chase a fancier
architecture. The same network class is imported by ``dqn/agent.py``
for both the online network and the target network.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """Input(state_dim) -> [Linear -> ReLU] * len(hidden_sizes) -> Linear(action_dim)

    Outputs one Q-value per backend server (action = server index).
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_sizes: List[int] = (256, 128)):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_sizes = list(hidden_sizes)

        layers = []
        in_dim = state_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, action_dim))
        self.net = nn.Sequential(*layers)

        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
