import numpy as np
from Optimizer import Default as DefaultOptim, Adam
import Functions as f
from scipy import signal


# todo: implement a system for a tensor having a list of functions that need to be applied before passing it into the gradient
# todo: update function, so having 2 copies of the data because of a reshape can be avoided
# todo: => reshape would write to this instruction list instead of making a new tensor => reshape(self, shape) -> self


class Tensor:
    def __init__(self, data: np.ndarray, parents=tuple(), trainable=False, Optim=None,
                 logging=False):  # , name="tensor"
        """Optim is pointer to class Adam etc"""
        self.trainable = trainable
        self.data = data
        self.parents = parents
        self.gradf = {}  # functions of gradient incorporating the local derivative into the gradient
        if Optim is not None and trainable:
            self.optim = Optim(self)
        else:
            self.optim = DefaultOptim(self)
        self.logging = logging

    # self.name = name

    def __mul__(self, other):
        if isinstance(other, Tensor):
            x = Tensor(self.data * other.data, (self, other), logging=self.logging)

            self.gradf[x] = lambda grad: grad * other.data
            other.gradf[x] = lambda grad: grad * self.data
        else:
            x = Tensor(self.data * other, (self, other), logging=self.logging)

            self.gradf[x] = lambda grad: grad * other

        return x

    def __matmul__(self, other):
        # dotaxes = axes in which dot product is computed
        if isinstance(other, Tensor):
            x = Tensor(self.data @ other.data, (self, other), logging=self.logging)

            self.gradf[x] = lambda grad: grad @ np.transpose(other.data, axes=(
                *range(len(other.data.shape) - 2), len(other.data.shape) - 1, len(other.data.shape) - 2))
            other.gradf[x] = lambda grad: np.transpose(self.data, axes=(
                *range(len(self.data.shape) - 2), len(self.data.shape) - 1, len(self.data.shape) - 2)) @ grad
        elif isinstance(other, np.ndarray):
            x = Tensor(self.data @ other, (self, other), logging=self.logging)

            self.gradf[x] = lambda grad: grad @ np.transpose(other, axes=(
                *range(len(other.shape) - 2), len(other.shape) - 1, len(other.shape) - 2))
        else:
            raise Exception(f"Matmul not defined for Tensor and {type(other)}")

        return x

    def __add__(self, other):
        if isinstance(other, Tensor):
            x = Tensor(self.data + other.data, (self, other), logging=self.logging)

            self.gradf[x] = lambda grad: grad
            other.gradf[x] = lambda grad: grad
        else:
            x = Tensor(self.data + other, (self, other), logging=self.logging)

            self.gradf[x] = lambda grad: grad

        return x

    def __sub__(self, other):
        if isinstance(other, Tensor):
            x = Tensor(self.data - other.data, (self, other), logging=self.logging)

            self.gradf[x] = lambda grad: grad
            other.gradf[x] = lambda grad: -grad
        else:
            x = Tensor(self.data - other, (self, other), logging=self.logging)

            self.gradf[x] = lambda grad: grad

        return x

    def __pow__(self, power, modulo=None):
        x = Tensor(self.data ** power, (self,), logging=self.logging)

        self.gradf[x] = lambda grad: grad * power * self.data ** (power - 1)

        return x

    def __neg__(self):
        return Tensor(-self.data, (self,), logging=self.logging)

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
        x = Tensor(self.data.transpose(axes), (self,), logging=self.logging)

        self.gradf[x] = lambda grad: grad.transpose(axes)

        return x

    def relu(self, alpha=.02):
        x = Tensor(np.copy(self.data), (self,), logging=self.logging)
        x.data[x.data < 0] *= alpha

        def _f(grad):
            grad[self.data < 0] *= alpha
            return grad

        self.gradf[x] = _f

        return x

    def tanh(self):
        x = Tensor(np.tanh(self.data), (self,), logging=self.logging)

        self.gradf[x] = lambda grad: grad * (1 - x.data ** 2)
        # 1 - np.tanh(self.data) ** 2

        return x

    def sigmoid(self):
        x = Tensor(1 / (1 + np.exp(-self.data)), (self,), logging=self.logging)

        self.gradf[x] = lambda grad: grad * x.data * (1 - x.data)

        return x

    def softmax(self):
        x = np.exp(self.data)
        x = Tensor(x / x.sum(), (self,), logging=self.logging)

        def _f(grad):
            M = x.data @ np.ones((*x.data.shape[:-2], x.data.shape[-1], x.data.shape[-2]))
            return (M * (np.identity(M.shape[-1]) - np.transpose(M, axes=(
                *range(len(M.shape) - 2), len(M.shape) - 1, len(M.shape) - 2)))) @ grad

        self.gradf[x] = _f

        return x

    def exp(self):
        x = Tensor(np.exp(self.data), (self,), logging=self.logging)

        self.gradf[x] = lambda grad: x.data * grad

        return x

    def log(self):
        x = Tensor(np.log(self.data), (self,), logging=self.logging)

        self.gradf[x] = lambda grad: grad * (1 / self.data)

        return x

    def reshape(self, shape):
        x = Tensor(self.data.reshape(shape), (self,), logging=self.logging)

        self.gradf[x] = lambda grad: grad.reshape(self.data.shape)

        return x

    def flatten(self):
        product = 1
        for i in self.data.shape[1:]:
            product *= i
        return self.reshape((self.data.shape[0], product, 1))

    def correlate(self, kernels):  # , few_loops=True
        if isinstance(kernels, Tensor):
            # if few_loops:
            x = f.correlate_kernels(self.data, kernels.data)
            # else:
            #    x = f.correlate_loop(self.data, kernels.data)
            x = Tensor(x, (self, kernels), logging=self.logging)

            self.gradf[x] = lambda grad: f.convolve_equal_depth_loop(grad, kernels.data)
            kernels.gradf[x] = lambda grad: f.correlate_batches(self.data, grad)
        elif isinstance(kernels, np.ndarray):
            # if few_loops:
            x = f.correlate_kernels(self.data, kernels)
            # else:
            #    x = f.correlate_loop(self.data, kernels)
            x = Tensor(x, (self, kernels), logging=self.logging)

            self.gradf[x] = lambda grad: f.convolve_equal_depth_loop(grad, kernels)
        else:
            raise Exception(f"Can only correlate tensor with Tensor and np.ndarray, not {type(kernels)}")

        return x

    def concat(self, *others, axis=0):
        tensors = (self, *others)
        assert (len(tensors[i - 1].shape) == len(tensors[i].shape) for i in range(len(tensors)))
        x = Tensor(np.concatenate([t.data for t in tensors], axis=axis), tensors, logging=self.logging)

        start_of_self = 0
        for t in tensors:
            t.gradf[x] = lambda grad: grad[start_of_self:start_of_self + t.data.shape[axis]]
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
        return Tensor(self.data.copy(), self.parents, self.trainable, logging=self.logging)

    def copydata(self):
        return self.data.copy()

    def dropout(self, rate=.05, magnitude_correction=True):
        assert 0 <= rate < 1
        if magnitude_correction:
            dropout_tensor = Tensor(np.random.choice((0, 1), self.data.shape, p=(rate, 1 - rate)) / (1 - rate), (self,),
                                    logging=self.logging)
        else:
            dropout_tensor = Tensor(np.random.choice((0, 1), self.data.shape, p=(rate, 1 - rate)), (self,),
                                    logging=self.logging)
        return self * dropout_tensor  # a little less efficient than writing out the expression because dropout_tensor
        # is not a parent, so setting the gradf is unnecessary

    def pool(self, sizes, criterion, criterion_included, poolfunc=f.pool):
        """size: pool size; criterion: np.max | np.min | np.average ... pointer; criterion included: np.argmax | np.argmin | lambda x: x ... pointer"""
        x, included = poolfunc(self.data, sizes, criterion, criterion_included)
        x = Tensor(x, (self,), logging=self.logging)

        self.gradf[x] = lambda grad: (grad.flatten() * included).reshape(self.data.shape)

        return x

    def maxpool(self, sizes):
        return self.pool(sizes, np.max, np.argmax)

    def minpool(self, sizes):
        return self.pool(sizes, np.min, np.argmin)

    def averagepool(self, sizes):
        return self.pool(sizes, np.average, lambda x: range(x.size), poolfunc=f.averagepool)

    def batchnorm(self):
        batchsums = np.apply_along_axis(lambda a: 1 / a.sum(), 0, self.data)

        x = Tensor(self.data * batchsums, (self,), logging=self.logging)

        self.gradf[x] = lambda grad: grad * batchsums

        return x

    def euclid_batchnorm(self):
        batchdivs = np.apply_along_axis(lambda a: 1 / np.sqrt((a * a).sum()))

        x = Tensor(self.data * batchdivs, (self,), logging=self.logging)

        self.gradf[x] = lambda grad: grad * batchdivs

        return x

    def zero_grad(self,
                  delete_parents=True):  # set to True to delete connections between tensors in both directions if not stated otherwise
        self.gradf = {}
        if delete_parents:
            self.parents = tuple()
        self.optim.zero_grad()

    def backward(self, gradient, child=None):
        if self.logging:
            print(f"backward")
        if child is not None:
            gradient = self.gradf[child](gradient)
        self.optim.update(gradient.sum(axis=0, keepdims=True) / gradient.shape[0])
        for parent in self.parents:
            parent.backward(gradient, self)

    def optimize(self, lr=.1):
        if self.logging:
            print(f"optimize")
        self.optim.autostep(lr)
