import math

import numpy as np

import Functions as f
from Optimizer import Default as DefaultOptim


class Tensor:
    def __init__(self, data: np.ndarray, parents=tuple(), requires_grad=False, Optim=None,
                 logging=False, name="Tensor", dynamic=True):
        """Optim is pointer to class Adam etc"""
        self.requires_grad = requires_grad
        self.data = data
        self.downstreams = parents
        self.transforms = {}
        # contains a sequence of transforms like transposes etc to be performed on the gradient + branches of the graph
        # to be backpropagated through with the temporal gradient. .backward will go through the reversed list of ops.
        # finally, the first branch in the list will be the parents of the tensor so you do backprop on them too.
        if Optim is not None and requires_grad:
            self.optim = Optim(self)
        else:
            self.optim = DefaultOptim(self)
        self.logging = logging
        self.name = name
        self.dynamic = dynamic  # if set to True, the tensor will, if possible, overwrite it's own value instead of making a new tensor

    def __mul__(self, other):
        x = Tensor(self.data * other.data, (self, other), logging=self.logging, requires_grad=any(map(lambda p: p.requires_grad, self.downstreams)))

        self._add_transform(lambda grad: grad * other.data, x)
        other._add_transform(lambda grad: grad * self.data, x)

        return x

    def __matmul__(self, other):
        # dotaxes = axes in which dot product is computed
        x = Tensor(self.data @ other.data, (self, other), logging=self.logging, requires_grad=any(map(lambda p: p.requires_grad, self.downstreams)))

        self._add_transform(lambda grad: grad @ np.transpose(other.data, axes=(
                *range(len(other.data.shape) - 2), len(other.data.shape) - 1, len(other.data.shape) - 2)), x)
        other._add_transform(lambda grad: np.transpose(self.data, axes=(
                *range(len(self.data.shape) - 2), len(self.data.shape) - 1, len(self.data.shape) - 2)) @ grad, x)

        return x

    def __add__(self, other):
        if self.dynamic:
            self.data += other.data
            x = self
        else:
            x = Tensor(self.data + other.data, (self, other), logging=self.logging, requires_grad=any(map(lambda p: p.requires_grad, self.downstreams)))

        self._add_transform(lambda grad: grad, x, (other,) if self.dynamic else tuple())
        other._add_transform(lambda grad: grad, x)

        return x

    def __sub__(self, other):
        if self.dynamic:
            self.data -= other.data
            x = self
        else:
            x = Tensor(self.data - other.data, (self, other), logging=self.logging, requires_grad=any(map(lambda p: p.requires_grad, self.downstreams)))

        self._add_transform(lambda grad: grad, x, (other,) if self.dynamic else tuple())
        other._add_transform(lambda grad: -grad, x)

        return x

    def __pow__(self, power, modulo=None):
        x = Tensor(self.data ** power, (self,), logging=self.logging, requires_grad=any(map(lambda p: p.requires_grad, self.downstreams)))

        self._add_transform(lambda grad: grad * power * self.data ** (power - 1), x)

        return x

    def __neg__(self):
        if self.dynamic:
            self.data = -self.data
            x = self
        else:
            x = Tensor(-self.data, (self,), logging=self.logging, requires_grad=any(map(lambda p: p.requires_grad, self.downstreams)))

        self._add_transform(lambda grad: -grad, x)

        return x

    def __repr__(self):
        sep = '      \n'
        return f"Tensor{sep.join(self.data.__repr__()[5:].splitlines())}"

    def mul(self, other):
        return self * other

    def matmul(self, other):
        return self @ other

    def add(self, other):
        return self + other

    def sub(self, other):
        return self - other

    def pow(self, power, modulo=None):
        return self ** power

    def neg(self):
        return -self

    def transpose(self, axes):
        if self.dynamic:
            self.data = self.data.transpose(axes)
            x = self
        else:
            x = Tensor(self.data.transpose(axes), (self,), logging=self.logging, requires_grad=any(map(lambda p: p.requires_grad, self.downstreams)))

        self._add_transform(lambda grad: grad.transpose(axes), x)

        return x

    def relu(self, alpha=.02):
        if self.dynamic:
            self.data[self.data < 0] *= alpha
            x = self

            def _f(grad):
                grad[self.data < 0] *= alpha
                return grad
        else:
            x = Tensor(np.copy(self.data), (self,), logging=self.logging, requires_grad=any(map(lambda p: p.requires_grad, self.downstreams)))
            x.data[x.data < 0] *= alpha

            def _f(grad):
                grad[self.data < 0] *= alpha
                return grad

        self._add_transform(_f, x)

        return x

    def tanh(self):
        if self.dynamic:
            self.data = np.tanh(self.data)
            x = self
        else:
            x = Tensor(np.tanh(self.data), (self,), logging=self.logging, requires_grad=any(map(lambda p: p.requires_grad, self.downstreams)))

        self._add_transform(lambda grad: grad * (1 - x.data ** 2), x)
        # 1 - np.tanh(self.data) ** 2

        return x

    def sigmoid(self):
        if self.dynamic:
            self.data = 1 / (1 + np.exp(-self.data))
            x = self
        else:
            x = Tensor(1 / (1 + np.exp(-self.data)), (self,), logging=self.logging, requires_grad=any(map(lambda p: p.requires_grad, self.downstreams)))

        self._add_transform(lambda grad: grad * x.data * (1 - x.data), x)

        return x

    def softmax(self):
        x = np.exp(self.data)
        if self.dynamic:
            self.data = x
            x = self
        else:
            x = Tensor(x / x.sum(), (self,), logging=self.logging, requires_grad=any(map(lambda p: p.requires_grad, self.downstreams)))

        def _f(grad):
            M = x.data @ np.ones((*x.data.shape[:-2], x.data.shape[-1], x.data.shape[-2]))
            return (M * (np.identity(M.shape[-1]) - np.transpose(M, axes=(
                *range(len(M.shape) - 2), len(M.shape) - 1, len(M.shape) - 2)))) @ grad

        self._add_transform(_f, x)

        return x

    def exp(self):
        if self.dynamic:
            self.data = np.exp(self.data)
            x = self
        else:
            x = Tensor(np.exp(self.data), (self,), logging=self.logging, requires_grad=any(map(lambda p: p.requires_grad, self.downstreams)))

        self._add_transform(lambda grad: x.data * grad, x)

        return x

    def log(self):
        x = Tensor(np.log(self.data), (self,), logging=self.logging, requires_grad=any(map(lambda p: p.requires_grad, self.downstreams)))

        self._add_transform(lambda grad: grad * (1 / self.data), x)

        return x

    def reshape(self, shape):
        if self.dynamic:
            self.data = self.data.reshape(shape)
            x = self
        else:
            x = Tensor(self.data.reshape(shape), (self,), logging=self.logging, requires_grad=any(map(lambda p: p.requires_grad, self.downstreams)))

        self._add_transform(lambda grad: grad.reshape(self.data.shape), x)

        return x

    def flatten(self):
        return self.reshape((self.data.shape[0], math.prod(self.data.shape[1:]), 1))

    def correlate(self, kernels):
        x = f.correlate_kernels(self.data, kernels.data)
        x = Tensor(x, (self, kernels), logging=self.logging, requires_grad=any(map(lambda p: p.requires_grad, self.downstreams)))

        self._add_transform(lambda grad: f.convolve_equal_depth_loop(grad, kernels.data), x)
        kernels._add_transform(lambda grad: f.correlate_batches(self.data, grad), x)

        return x

    def concat(self, *others, axis=0):
        tensors = (self, *others)
        assert (len(tensors[i - 1].shape) == len(tensors[i].shape) for i in range(len(tensors)))
        x = Tensor(np.concatenate([t.data for t in tensors], axis=axis), tensors, logging=self.logging, requires_grad=any(map(lambda p: p.requires_grad, self.downstreams)))

        start_of_self = 0
        for t in tensors:
            t._add_transform(lambda grad: np.split([start_of_self, start_of_self + t.data.shape[axis]], axis=axis)[1], x)
            start_of_self += t.data.shape[axis]

        return x

    def copy(self):
        """Will not copy gradient, optimizer data and operation operations that were performed on the Tensor
        Will copy value, parents, trainable boolean and type of optimizer

        a = Tensor(some data)
        b = Tensor(some data)
        y = a @ b
        z = a.copy()
        -->
        a.gradf = {y: lambda grad: grad @ b.data}
        z.gradf = {}
        (operations performed on this tensor)"""
        return Tensor(self.data.copy(), self.downstreams, self.requires_grad, logging=self.logging)

    def copydata(self):
        return self.data.copy()

    def dropout(self, rate=.05, magnitude_correction=True):
        assert 0 <= rate < 1
        if magnitude_correction:
            dropout_tensor = Tensor(np.random.choice((0, 1), self.data.shape, p=(rate, 1 - rate)) / (1 - rate),
                                    logging=self.logging, requires_grad=any(map(lambda p: p.requires_grad, self.downstreams)))
        else:
            dropout_tensor = Tensor(np.random.choice((0, 1), self.data.shape, p=(rate, 1 - rate)),
                                    logging=self.logging, requires_grad=any(map(lambda p: p.requires_grad, self.downstreams)))
        return self * dropout_tensor

    def pool(self, sizes, criterion, criterion_included, poolfunc=f.pool):
        """size: pool size; criterion: np.max | np.min | np.average ... pointer; criterion included: np.argmax | np.argmin | lambda x: x ... pointer"""
        x, included = poolfunc(self.data, sizes, criterion, criterion_included)
        x = Tensor(x, (self,), logging=self.logging, requires_grad=any(map(lambda p: p.requires_grad, self.downstreams)))

        self._add_transform(lambda grad: (grad.flatten() * included).reshape(self.data.shape), x)

        return x

    def maxpool(self, sizes):
        return self.pool(sizes, np.max, np.argmax)

    def minpool(self, sizes):
        return self.pool(sizes, np.min, np.argmin)

    def averagepool(self, sizes):
        return self.pool(sizes, np.average, lambda x: range(x.size), poolfunc=f.averagepool)

    def batchnorm(self):
        inv_batchsums = np.apply_along_axis(lambda a: 1 / a.sum(), 0, self.data)

        if self.dynamic:
            self.data *= inv_batchsums
            x = self.data
        else:
            x = Tensor(self.data * inv_batchsums, (self,), logging=self.logging)

        self._add_transform(lambda grad: grad * inv_batchsums, x)

        return x

    def euclid_batchnorm(self):
        batchdivs = np.apply_along_axis(lambda a: 1 / np.sqrt((a * a).sum()))

        if self.dynamic:
            self.data *= batchdivs
            x = self
        else:
            x = Tensor(self.data * batchdivs, (self,), logging=self.logging)

        self._add_transform(lambda grad: grad * batchdivs, x)

        return x

    def _add_transform(self, function, x, others=tuple()):
        """add a function of the gradient to the transforms that will be applied to it in the backward pass"""
        # x = upstream
        if not isinstance(others, tuple):
            others = tuple(others)
        self.downstreams += others
        if self.transforms.get(x) is None:
            self.transforms[x] = []
        self.transforms[x].append((function, others))

    def zero_grad(self, x,
                  delete_parents=True):  # set to True to delete connections between tensors in both directions if not stated otherwise
        # x = upstream
        self.transforms[x] = []
        self.optim.zero_grad(x)

        for downstream in self.downstreams:
            downstream.zero_grad(x, delete_parents)

        if delete_parents:
            self.downstreams = tuple()

    def backward(self, gradient, x):
        # x = upstream
        if self.logging:
            print(f"backward @ {self.name}")
        for transform, others in reversed(self.transforms[x]):
            gradient = transform(gradient)
            for other in others:
                other.backward(gradient, self)

        self.optim.update(gradient.sum(axis=0, keepdims=True) / gradient.shape[0], x)
        for downstream in self.downstreams:
            if downstream.requires_grad:
                downstream.backward(gradient, self)

    def optimize(self, x, lr=0.01):
        # x = upstream
        if self.logging:
            print(f"optimize @ {self.name}")
        self.optim.step(lr, x)
        for downstream in self.downstreams:
            if downstream.requires_grad:
                downstream.optimize(self, lr)
