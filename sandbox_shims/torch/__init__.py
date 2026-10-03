"""VALIDATION-ONLY minimal-autograd shim for a small subset of PyTorch.

Not shipped with the project. This sandbox has no network access to
install the real ~200MB CPU wheel, so this shim lets the REAL
dqn/agent.py, dqn/model.py and load_balancer/dqn_balancer.py code run
end-to-end with GENUINE gradient descent (a small reverse-mode
autodiff engine over numpy arrays, in the spirit of micrograd), not
just "no exceptions were raised". `.backward()` really does propagate
gradients, `optimizer.step()` really does update weights with Adam,
and `nn.utils.clip_grad_norm_` really does clip. This is enough to
prove the DQN training loop genuinely reduces loss and genuinely
differentiates Q-values across states - i.e. that the *wiring* in
dqn/agent.py is correct - using exactly the same PyTorch call pattern
a real torch install would see.
"""
import pickle

import numpy as np

float32 = np.float32
int64 = np.int64
int32 = np.int32

_GRAD_ENABLED = True


class _NoGrad:
    def __enter__(self):
        global _GRAD_ENABLED
        self._prev = _GRAD_ENABLED
        _GRAD_ENABLED = False
        return self

    def __exit__(self, *exc):
        global _GRAD_ENABLED
        _GRAD_ENABLED = self._prev
        return False


def no_grad():
    return _NoGrad()


def _unbroadcast(grad, shape):
    """Reduce `grad` (from a broadcasted forward op) back down to
    `shape` by summing over the axes that were broadcast."""
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for i, dim in enumerate(shape):
        if dim == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad


class Tensor:
    """A numpy-backed tensor with a minimal reverse-mode autodiff tape.
    Named `Tensor` (not `Var`) purely so it reads naturally as a
    drop-in for `torch.Tensor` from the real project code's point of
    view.
    """

    __slots__ = ("data", "grad", "_children", "_backward_fn", "requires_grad")

    def __init__(self, data, dtype=None, children=(), backward_fn=None, requires_grad=None):
        if dtype is None:
            dtype = data.dtype if isinstance(data, np.ndarray) else np.float32
        self.data = np.asarray(data, dtype=dtype)
        self.grad = np.zeros_like(self.data, dtype=np.float64)
        self._children = children
        self._backward_fn = backward_fn or (lambda: None)
        self.requires_grad = _GRAD_ENABLED if requires_grad is None else requires_grad

    # -- shape/dtype-ish helpers -----------------------------------
    @property
    def shape(self):
        return self.data.shape

    def float(self):
        return Tensor(self.data.astype(np.float32), requires_grad=self.requires_grad)

    def long(self):
        return Tensor(self.data.astype(np.int64), requires_grad=False)

    def to(self, *args, **kwargs):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return np.asarray(self.data)

    def item(self):
        return float(np.asarray(self.data).reshape(-1)[0])

    def tolist(self):
        return np.asarray(self.data).tolist()

    def __repr__(self):
        return f"Tensor({self.data!r})"

    # -- autodiff ops -------------------------------------------------
    def unsqueeze(self, dim):
        out_data = np.expand_dims(self.data, dim)
        if not (_GRAD_ENABLED and self.requires_grad):
            return Tensor(out_data, requires_grad=False)

        def _backward():
            self.grad += np.squeeze(out.grad, axis=dim)
        out = Tensor(out_data, children=(self,), backward_fn=_backward, requires_grad=True)
        return out

    def squeeze(self, dim=None):
        out_data = np.squeeze(self.data, axis=dim) if dim is not None else np.squeeze(self.data)
        if not (_GRAD_ENABLED and self.requires_grad):
            return Tensor(out_data, requires_grad=False)

        def _backward():
            self.grad += np.reshape(out.grad, self.data.shape)
        out = Tensor(out_data, children=(self,), backward_fn=_backward, requires_grad=True)
        return out

    def gather(self, dim, index):
        idx = index.data if isinstance(index, Tensor) else np.asarray(index)
        out_data = np.take_along_axis(self.data, idx, axis=dim)
        if not (_GRAD_ENABLED and self.requires_grad):
            return Tensor(out_data, requires_grad=False)

        def _backward():
            g = np.zeros_like(self.data, dtype=np.float64)
            np.put_along_axis(g, idx, out.grad, axis=dim)
            self.grad += g
        out = Tensor(out_data, children=(self,), backward_fn=_backward, requires_grad=True)
        return out

    def max(self, dim=None):
        class _MaxResult:
            pass
        r = _MaxResult()
        if dim is None:
            r.values = Tensor(np.max(self.data))
            r.indices = Tensor(np.argmax(self.data))
            return r

        argmax = np.argmax(self.data, axis=dim)
        values_data = np.max(self.data, axis=dim)
        if not (_GRAD_ENABLED and self.requires_grad):
            r.values = Tensor(values_data, requires_grad=False)
            r.indices = Tensor(argmax, requires_grad=False)
            return r

        def _backward():
            g = np.zeros_like(self.data, dtype=np.float64)
            np.put_along_axis(g, np.expand_dims(argmax, axis=dim), np.expand_dims(out.grad, axis=dim), axis=dim)
            self.grad += g
        out = Tensor(values_data, children=(self,), backward_fn=_backward, requires_grad=True)
        r.values = out
        r.indices = Tensor(argmax, requires_grad=False)
        return r

    def __add__(self, other):
        other_t = other if isinstance(other, Tensor) else Tensor(np.asarray(other, dtype=np.float64), requires_grad=False)
        out_data = self.data + other_t.data
        track = _GRAD_ENABLED and (self.requires_grad or other_t.requires_grad)
        if not track:
            return Tensor(out_data, requires_grad=False)

        def _backward():
            if self.requires_grad:
                self.grad += _unbroadcast(out.grad, self.data.shape)
            if other_t.requires_grad:
                other_t.grad += _unbroadcast(out.grad, other_t.data.shape)
        out = Tensor(out_data, children=(self, other_t), backward_fn=_backward, requires_grad=True)
        return out

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self.__add__(-other if not isinstance(other, Tensor) else other.__neg__())

    def __rsub__(self, other):
        return self.__neg__().__add__(other)

    def __neg__(self):
        out_data = -self.data
        if not (_GRAD_ENABLED and self.requires_grad):
            return Tensor(out_data, requires_grad=False)

        def _backward():
            self.grad += -out.grad
        out = Tensor(out_data, children=(self,), backward_fn=_backward, requires_grad=True)
        return out

    def __mul__(self, other):
        other_t = other if isinstance(other, Tensor) else Tensor(np.asarray(other, dtype=np.float64), requires_grad=False)
        out_data = self.data * other_t.data
        track = _GRAD_ENABLED and (self.requires_grad or other_t.requires_grad)
        if not track:
            return Tensor(out_data, requires_grad=False)

        def _backward():
            if self.requires_grad:
                self.grad += _unbroadcast(out.grad * other_t.data, self.data.shape)
            if other_t.requires_grad:
                other_t.grad += _unbroadcast(out.grad * self.data, other_t.data.shape)
        out = Tensor(out_data, children=(self, other_t), backward_fn=_backward, requires_grad=True)
        return out

    def __rmul__(self, other):
        return self.__mul__(other)

    def backward(self, grad=None):
        topo, visited = [], set()

        def build(v):
            if id(v) not in visited:
                visited.add(id(v))
                for child in v._children:
                    build(child)
                topo.append(v)
        build(self)
        self.grad = np.ones_like(self.data, dtype=np.float64) if grad is None else np.asarray(grad, dtype=np.float64)
        for v in reversed(topo):
            v._backward_fn()


def as_tensor(data, dtype=None, device=None):
    return Tensor(data, dtype=dtype, requires_grad=False)


def tensor(data, dtype=None, device=None):
    return Tensor(data, dtype=dtype, requires_grad=False)


def device(spec="cpu"):
    return spec


class _Cuda:
    @staticmethod
    def is_available():
        return False


cuda = _Cuda()


def save(obj, path):
    with open(path, "wb") as f:
        pickle.dump(_strip(obj), f)


def _strip(obj):
    """Convert any Tensor objects in a (possibly nested) checkpoint
    dict to plain numpy arrays before pickling, and back to Tensor on
    load - mirrors how real torch.save/load happily round-trips
    state_dict()'s tensors."""
    if isinstance(obj, Tensor):
        return ("__tensor__", obj.data)
    if isinstance(obj, dict):
        return {k: _strip(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_strip(v) for v in obj]
    return obj


def _restore(obj):
    if isinstance(obj, tuple) and len(obj) == 2 and obj[0] == "__tensor__":
        return Tensor(obj[1], requires_grad=False)
    if isinstance(obj, dict):
        return {k: _restore(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_restore(v) for v in obj]
    return obj


def load(path, map_location=None, weights_only=False):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return _restore(obj)


# ---- nn / optim ----
from . import nn  # noqa: E402
from . import optim  # noqa: E402
