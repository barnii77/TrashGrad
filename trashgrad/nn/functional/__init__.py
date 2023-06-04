from trashgrad.autograd import Tensor, tensor
import trashgrad.nn.functional._Operations as ops
from trashgrad.nn import Module
import numpy as np
import cupy as cp


def dropout(t: Tensor, rate: float) -> Tensor:
    assert 0 <= rate <= 1
    return t * tensor(t.lib.random.choice((0., 1.), t.shape, p=(rate, 1 - rate)))


def standardize(t: Tensor, axis=-1) -> Tensor:
    tmp = t - t.mean(axis, keepdims=True)
    return tmp / tmp.std(axis)


def relu(t: Tensor, leak=0.) -> Tensor:
    return ops.ReLU(leak)(t)


def tanh(t: Tensor) -> Tensor:
    return ops.Tanh()(t)


def sigmoid(t: Tensor) -> Tensor:
    return ops.Sigmoid()(t)


def softmax(t: Tensor) -> Tensor:
    exp_t = t.exp()
    return exp_t / exp_t.sum(-1, keepdims=True)
    #return #ops.Softmax()(t)


def silu(t: Tensor):
    return t * sigmoid(t)


def gelu(t: Tensor):
    return t * sigmoid(1.702*t)  # an approximation of x/2 * (1 + erf(x/sqrt(2)))


def one_hot(label, num_labels, mode="cpu") -> Tensor:
    out = np.zeros((num_labels,))
    out[label] = 1.
    out = tensor(np.zeros((num_labels,)), mode=mode)
    return out


def mse(prediction, label) -> Tensor:
    label = label if isinstance(label, Tensor) else Tensor(label)
    return ((prediction - label) ** 2).mean()


def cross_entropy(logits: Tensor, label) -> Tensor:
    return nll(softmax(logits), label)


def nll(prediction, label) -> Tensor:
    if len(label.shape) != 1:
        raise Exception(f"NLL expected label shape (batch_size,), but got {label.shape}. If you have dimensions of size 1 in your label shape, try using label.flatten() instead.")
    return -(prediction[prediction.lib.arange(0, label.size), label] + 1e-3).log().mean()


class LayerNorm(Module):
    def __init__(self, insize: int, mode="cpu"):
        lib = {"cpu": np, "gpu": cp}[mode]
        self.w = lib.ones((insize,))
        self.b = lib.ones((insize,))

    def forward(self, x: Tensor):
        return self.w * standardize(x) + self.b

#__all__ = [mse, cel, nll, relu, tanh, sigmoid, softmax, layernorm, dropout]
