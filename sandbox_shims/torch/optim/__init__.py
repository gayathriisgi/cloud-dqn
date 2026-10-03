"""VALIDATION-ONLY shim - see torch/__init__.py docstring. Real Adam
update rule operating on Tensor.data/.grad, so the DQN's weights
genuinely move in the direction that reduces the Huber loss computed
by dqn/agent.py.
"""
import numpy as np


class Adam:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.t = 0
        self.m = [np.zeros_like(p.data, dtype=np.float64) for p in self.params]
        self.v = [np.zeros_like(p.data, dtype=np.float64) for p in self.params]

    def zero_grad(self):
        for p in self.params:
            p.grad = np.zeros_like(p.data, dtype=np.float64)

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            g = p.grad
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g ** 2)
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            update = self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            p.data = (p.data - update).astype(p.data.dtype)

    def state_dict(self):
        return {"t": self.t, "m": self.m, "v": self.v}

    def load_state_dict(self, state):
        self.t = state.get("t", self.t)
        self.m = state.get("m", self.m)
        self.v = state.get("v", self.v)
