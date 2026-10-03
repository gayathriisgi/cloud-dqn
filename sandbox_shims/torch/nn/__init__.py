"""VALIDATION-ONLY shim - see torch/__init__.py docstring."""
import numpy as np

from .. import Tensor
import torch as _torch


def _grad_enabled():
    return _torch._GRAD_ENABLED


class Module:
    def __init__(self):
        object.__setattr__(self, "_parameters", {})
        object.__setattr__(self, "_modules", {})
        object.__setattr__(self, "training", True)

    def __setattr__(self, name, value):
        if isinstance(value, Tensor):
            self.__dict__.setdefault("_parameters", {})[name] = value
        elif isinstance(value, Module):
            self.__dict__.setdefault("_modules", {})[name] = value
        object.__setattr__(self, name, value)

    def parameters(self):
        for p in self._parameters.values():
            yield p
        for m in self._modules.values():
            yield from m.parameters()

    def _named(self, prefix=""):
        items = []
        for name, p in self._parameters.items():
            items.append((f"{prefix}{name}", p))
        for name, m in self._modules.items():
            items.extend(m._named(f"{prefix}{name}."))
        return items

    def state_dict(self):
        return {name: Tensor(np.array(p.data), requires_grad=False) for name, p in self._named()}

    def load_state_dict(self, state_dict):
        by_name = dict(self._named())
        for name, value in state_dict.items():
            if name in by_name:
                data = value.data if isinstance(value, Tensor) else np.asarray(value)
                by_name[name].data = np.array(data, dtype=by_name[name].data.dtype)
        return self

    def to(self, *a, **k):
        return self

    def train(self, mode=True):
        object.__setattr__(self, "training", mode)
        for m in self._modules.values():
            m.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class Linear(Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        limit = np.sqrt(1.0 / in_features)
        self.weight = Tensor(np.random.uniform(-limit, limit, size=(out_features, in_features)))
        self.bias = Tensor(np.zeros(out_features, dtype=np.float32))

    def forward(self, x: Tensor) -> Tensor:
        out_data = x.data @ self.weight.data.T + self.bias.data
        track = _grad_enabled() and (x.requires_grad or self.weight.requires_grad or self.bias.requires_grad)
        if not track:
            return Tensor(out_data, requires_grad=False)

        def _backward():
            if x.requires_grad:
                x.grad += out.grad @ self.weight.data
            if self.weight.requires_grad:
                self.weight.grad += out.grad.T @ x.data
            if self.bias.requires_grad:
                self.bias.grad += out.grad.sum(axis=0)
        out = Tensor(out_data, children=(x, self.weight, self.bias), backward_fn=_backward, requires_grad=True)
        return out


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        out_data = np.maximum(0, x.data)
        if not (_grad_enabled() and x.requires_grad):
            return Tensor(out_data, requires_grad=False)

        def _backward():
            x.grad += out.grad * (x.data > 0)
        out = Tensor(out_data, children=(x,), backward_fn=_backward, requires_grad=True)
        return out


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self._seq = list(modules)
        for i, m in enumerate(self._seq):
            self._modules[str(i)] = m

    def __iter__(self):
        return iter(self._seq)

    def forward(self, x):
        for m in self._seq:
            x = m(x)
        return x


class SmoothL1Loss(Module):
    """Huber loss, beta=1.0, mean reduction - matches torch's default
    `nn.SmoothL1Loss()`. Implemented as one fused autodiff op (rather
    than composed from generic sub/abs/where ops) purely for shim
    simplicity; the forward value and the analytic gradient below are
    both the exact real Huber loss / its exact derivative.
    """

    def forward(self, pred: Tensor, target) -> Tensor:
        target_data = target.data if isinstance(target, Tensor) else np.asarray(target, dtype=np.float64)
        diff = pred.data - target_data
        abs_diff = np.abs(diff)
        loss_arr = np.where(abs_diff < 1.0, 0.5 * diff ** 2, abs_diff - 0.5)
        out_data = np.mean(loss_arr)

        if not (_grad_enabled() and pred.requires_grad):
            return Tensor(out_data, requires_grad=False)

        def _backward():
            grad_diff = np.where(abs_diff < 1.0, diff, np.sign(diff)) / diff.size
            pred.grad += out.grad * grad_diff
        out = Tensor(out_data, children=(pred,), backward_fn=_backward, requires_grad=True)
        return out


class _Init:
    @staticmethod
    def kaiming_uniform_(tensor_, nonlinearity="relu"):
        fan_in = tensor_.data.shape[1] if tensor_.data.ndim > 1 else tensor_.data.shape[0]
        gain = np.sqrt(2.0)  # relu gain
        bound = gain * np.sqrt(3.0 / fan_in)
        tensor_.data = np.random.uniform(-bound, bound, size=tensor_.data.shape).astype(tensor_.data.dtype)
        return tensor_

    @staticmethod
    def zeros_(tensor_):
        tensor_.data = np.zeros_like(tensor_.data)
        return tensor_


init = _Init()


class _Utils:
    @staticmethod
    def clip_grad_norm_(parameters, max_norm):
        params = list(parameters)
        total_norm_sq = sum(float(np.sum(p.grad ** 2)) for p in params)
        total_norm = float(np.sqrt(total_norm_sq))
        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1.0:
            for p in params:
                p.grad = p.grad * clip_coef
        return total_norm


utils = _Utils()
