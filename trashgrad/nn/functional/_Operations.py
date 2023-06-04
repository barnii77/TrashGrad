from trashgrad.autograd.tensor import Tensor
from trashgrad.autograd._Function import Function
from trashgrad.nn import Module


class ReLU(Function):
    backward_required_args_kwargs = set()

    def __init__(self, leak=0.):
        self.leak = leak

    def _forward(self, tensor: Tensor):
        out = tensor.data.copy()
        self.neg = out < 0
        out[self.neg] *= self.leak
        return out

    def _backward(self, grad, req: Tensor, tensor: Tensor):
        grad[self.neg] *= self.leak
        return grad


class Tanh(Function):
    backward_required_args_kwargs = set()

    def _forward(self, tensor: Tensor):
        self.out = tensor.lib.tanh(tensor.data)
        return self.out

    def _backward(self, grad, req: Tensor, tensor: Tensor):
        return grad * (1 - self.out ** 2)


class Sigmoid(Function):
    backward_required_args_kwargs = set()

    def _forward(self, tensor: Tensor):
        self.out = 1 / (1 + 1 / tensor.lib.exp(tensor.data))
        return self.out

    def _backward(self, grad, req: Tensor, tensor: Tensor):
        return grad * self.out * (1 - self.out)


class Softmax(Function):
    backward_required_args_kwargs = set()

    def _forward(self, t: Tensor):
        self.out = t.lib.exp(t.data - t.data.max())
        self.out /= self.out.sum(-1, keepdims=True)
        return self.out

    def _backward(self, grad, req: Tensor, t: Tensor):
        idx = tuple(slice(None) for _ in range(len(self.out.shape))) + (t.lib.newaxis,)
        M = t.lib.repeat(self.out[idx], t.shape[-2], axis=-1)
        return (M * (t.lib.identity(M.shape[-1]).reshape((M.shape[-1], M.shape[-1], *(1 for _ in range(len(M.shape) - 2)))) - t.lib.transpose(M, axes=(
            *range(len(M.shape) - 2), -1, -2)))) @ grad
