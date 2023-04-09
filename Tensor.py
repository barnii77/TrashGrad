import math
import warnings

import numpy as np
import cupy as cp
import cupyx.scipy.signal as cpxsignal

from scipy import signal

import ConvFunctions as cf
from Optimizer import StdOptim


class Tensor:
    def __init__(self, data, upstreams=tuple(), requires_grad=False, trainable=False, Optim=None,
                 logging=False, name="Tensor", dynamic=False, mode="cpu"):
        """Warning: dynamic can lead to performance increases (mainly memory) but is not threadsafe"""
        if dynamic:
            warnings.warn(f"{name} is dynamic, note that this is not threadsafe")
        '''if mode == "gpu" and cp.get_array_module(data) is np:
            data = cp.asarray(data)
        elif mode == "cpu" and cp.get_array_module(data) is cp:
            data = cp.asnumpy(data)'''

        self.requires_grad = requires_grad or trainable
        self.trainable = trainable
        self.data = data
        self.upstreams = upstreams  # parents
        self.transforms = {}
        # contains a sequence of transforms like transposes etc to be performed on the gradient + branches of the graph
        # to be backpropagated through with the temporal gradient. .backward will go through the reversed list of ops.
        # finally, the first branch in the list will be the parents of the tensor so you do backprop on them too.
        if Optim is not None:
            self.optim = Optim(self)
        else:
            self.optim = StdOptim(self)
        self.logging = logging
        self.name = name
        self.dynamic = dynamic  # more memory efficient, but not threadsafe (-> dont use IF multiple threads operate on the same tensor instance)
        self.next_op_new_tensor_forced = False
        self.mode = mode

        if mode == "gpu":
            self.lib = cp
            self.siglib = cpxsignal
        elif mode == "cpu":
            self.lib = np
            self.siglib = signal
        else:
            raise Exception(f"Mode must be 'cpu' or 'gpu', not {mode}")

    def delete(self, delete_graph=False):
        """delete_graph = True: Will delete everything that references this Tensor object (children, itself and parents)
        delete_graph = False: Will delete all references to this Tensor object so it's picked up by the garbage collector"""
        for upstream in self.upstreams:
            upstream.transforms.pop(self)
            if delete_graph:
                upstream.delete(delete_graph)

        for downstream in self.transforms.keys():
            downstream.upstreams = tuple(filter(lambda x: x != self, downstream.upstreams))
            if delete_graph:
                downstream.delete(delete_graph)

    def cpu(self):
        if self.mode == "gpu":
            x = Tensor(cp.asnumpy(self.data),
                       (self,),
                       logging=self.logging,
                       requires_grad=self.requires_grad,
                       dynamic=self.dynamic,
                       mode="cpu")

            self._add_transform(lambda grad: cp.asarray(grad), x)
        else:
            x = self

        return x

    def gpu(self):
        if self.mode == "gpu":
            x = Tensor(cp.asarray(self.data),
                       (self,),
                       logging=self.logging,
                       requires_grad=self.requires_grad,
                       dynamic=self.dynamic,
                       mode="gpu")

            self._add_transform(lambda grad: cp.asnumpy(grad), x)
        else:
            x = self

        return x

    def __mul__(self, other):
        x = Tensor(self.data * other.data,
                   (self, other),
                   logging=self.logging or other.logging,
                   requires_grad=self.requires_grad or other.requires_grad,
                   dynamic=self.dynamic or other.dynamic,
                   mode=self.mode)

        self._add_transform(lambda grad: grad * other.data, x)
        other._add_transform(lambda grad: grad * self.data, x)

        return x

    def __matmul__(self, other):
        # dotaxes = axes in which dot product is computed
        x = Tensor(self.data @ other.data,
                   (self, other),
                   logging=self.logging or other.logging,
                   requires_grad=self.requires_grad or other.requires_grad,
                   dynamic=self.dynamic or other.dynamic,
                   mode=self.mode)

        self._add_transform(lambda grad: grad @ other.data.transpose((
            *range(len(other.data.shape) - 2), len(other.data.shape) - 1, len(other.data.shape) - 2)), x)
        other._add_transform(lambda grad: self.data.transpose((
            *range(len(self.data.shape) - 2), len(self.data.shape) - 1, len(self.data.shape) - 2)) @ grad, x)

        return x

    def __add__(self, other):
        if self.dynamic and self.upstreams and not self.next_op_new_tensor_forced:
            self.data += other.data
            x = self
        else:
            x = Tensor(self.data + other.data,
                       (self, other),
                       logging=self.logging or other.logging,
                       requires_grad=self.requires_grad or other.requires_grad,
                       dynamic=self.dynamic or other.dynamic,
                       mode=self.mode)

        self._add_transform(lambda grad: grad, x, (other,) if self.dynamic and self.upstreams and not self.next_op_new_tensor_forced else tuple())
        other._add_transform(lambda grad: grad, x)

        return x

    def __sub__(self, other):
        if self.dynamic and self.upstreams and not self.next_op_new_tensor_forced:
            self.data -= other.data
            x = self
        else:
            x = Tensor(self.data - other.data,
                       (self, other),
                       logging=self.logging or other.logging,
                       requires_grad=self.requires_grad or other.requires_grad,
                       dynamic=self.dynamic or other.dynamic,
                       mode=self.mode)

        self._add_transform(lambda grad: grad, x, (other,) if self.dynamic and self.upstreams and not self.next_op_new_tensor_forced else tuple())
        other._add_transform(lambda grad: -grad, x)

        return x

    def __pow__(self, power, modulo=None):
        x = Tensor(self.data ** power,
                   (self,),
                   logging=self.logging,
                   requires_grad=self.requires_grad,
                   dynamic=self.dynamic,
                   mode=self.mode)

        self._add_transform(lambda grad: grad * power * self.data ** (power - 1), x)

        return x

    def __neg__(self):
        if self.dynamic and self.upstreams and not self.next_op_new_tensor_forced:
            self.data = -self.data
            x = self
        else:
            x = Tensor(-self.data,
                       (self,),
                       logging=self.logging,
                       requires_grad=self.requires_grad,
                       dynamic=self.dynamic,
                       mode=self.mode)

        self._add_transform(lambda grad: -grad, x)

        return x

    def __truediv__(self, other):
        inv = 1 / other.data
        x = Tensor(self.data * inv,
                   (self, other),
                   logging=self.logging or other.logging,
                   requires_grad=self.requires_grad or other.requires_grad,
                   dynamic=self.dynamic or other.dynamic,
                   mode=self.mode)

        self._add_transform(lambda grad: grad * inv, x, (other,))
        other._add_transform(lambda grad: -grad * self.data * (inv * inv))

        return x

    def __rtruediv__(self, other):
        inv = 1 / self.data
        x = Tensor(other.data * inv,
                   (self, other),
                   logging=self.logging or other.logging,
                   requires_grad=self.requires_grad or other.requires_grad,
                   dynamic=self.dynamic or other.dynamic,
                   mode=self.mode)

        self._add_transform(lambda grad: -grad * other.data * inv * inv, x, (other,))

        return x

    def __abs__(self):
        abs_prime = self.lib.ones(self.data.shape)
        abs_prime[self.data < 0] = -1.0

        if self.dynamic and self.upstreams and not self.next_op_new_tensor_forced:
            self.data = self.lib.abs(self.data)
            x = self
        else:
            x = Tensor(self.lib.abs(self.data),
                       (self,),
                       logging=self.logging,
                       requires_grad=self.requires_grad,
                       dynamic=self.dynamic,
                       mode=self.mode)

        self._add_transform(lambda grad: grad * abs_prime, x)  # alternatively you could define a different function
        # for the case self.dynamic == False which computes 'abs_prime' during the backward pass instead of storing it
        # in memory until then

        return x

    def __getitem__(self, item):
        return self.data[item]

    def __setitem__(self, key, value):
        self.data[key] = value

    def __repr__(self):
        sep = len(self.name) * ' ' + '\n'
        return f"{self.name}{sep.join(str(self.data).splitlines())}"

    def mul(self, other):
        if isinstance(other, Tensor):
            if other.requires_grad or not self.dynamic:
                return self * other

        if self.dynamic and not self.next_op_new_tensor_forced:
            self.data *= other
            x = self
        else:
            x = Tensor(self.data * other,
                       (self,),
                       logging=self.logging,
                       requires_grad=self.requires_grad,
                       dynamic=self.dynamic,
                       mode=self.mode)

        self._add_transform(lambda grad: grad * other, x)

        return x

    def matmul(self, other):
        return self @ other

    def add(self, other, new_tensor=False):
        if self.dynamic and new_tensor and not self.next_op_new_tensor_forced:
            self.dynamic = False
            x = self + other
            self.dynamic = True
        else:
            x = self + other
        return x

    def sub(self, other, new_tensor=False):
        if self.dynamic and new_tensor and not self.next_op_new_tensor_forced:
            self.dynamic = False
            x = self - other
            self.dynamic = True
        else:
            x = self - other
        return x

    def pow(self, power, modulo=None):
        return self ** power

    def neg(self, new_tensor=False):
        if self.dynamic and new_tensor and not self.next_op_new_tensor_forced:
            self.dynamic = False
            x = -self
            self.dynamic = True
        else:
            x = -self
        return x

    def div(self, other):
        if isinstance(other, Tensor):
            if other.requires_grad or not self.dynamic:
                return self / other

        inv = 1 / other

        if self.dynamic and not self.next_op_new_tensor_forced:
            self.data *= inv
            x = self
        else:
            x = Tensor(self.data * inv,
                       (self,),
                       logging=self.logging,
                       requires_grad=self.requires_grad,
                       dynamic=self.dynamic,
                       mode=self.mode)

        self._add_transform(lambda grad: grad * inv, x)

        return x

    def abs(self):
        return abs(self)

    def addall(self, *others):
        tensors = (self, *others)
        assert all([self.mode == i.mode for i in tensors])
        x = Tensor(sum(map(lambda a: a.data, tensors)),
                   tensors,
                   logging=any(map(lambda a: a.logging, tensors)),
                   requires_grad=any(map(lambda a: a.requires_grad, tensors)),
                   dynamic=any(map(lambda a: a.dynamic, tensors)),
                   mode=self.mode)

        for t in tensors:
            t._add_transform(lambda grad: grad, x)

        return x

    def mulall(self, *others):
        tensors = (self, *others)
        assert all([self.mode == i.mode for i in tensors])
        x = Tensor(self.lib.prod(map(lambda a: a.data, tensors)),
                   tensors,
                   logging=any(map(lambda a: a.logging, tensors)),
                   requires_grad=any(map(lambda a: a.requires_grad, tensors)),
                   dynamic=any(map(lambda a: a.dynamic, tensors)),
                   mode=self.mode)

        for t in tensors:
            t._add_transform(lambda grad: grad * self.lib.prod([tensor.data for tensor in tensors if tensor != t]), x)

        return x

    def transpose(self, axes):
        assert max(axes) == len(axes) - 1
        if self.dynamic and self.upstreams and not self.next_op_new_tensor_forced:
            self.data = self.data.transpose(axes)
            x = self
        else:
            x = Tensor(self.data.transpose(axes),
                       (self,),
                       logging=self.logging,
                       requires_grad=self.requires_grad,
                       dynamic=self.dynamic,
                       mode=self.mode)

        inverse_transpose_axes = [None for _ in range(len(axes))]
        for i, val in enumerate(axes):
            inverse_transpose_axes[val] = i
        inverse_transpose_axes = tuple(inverse_transpose_axes)
        self._add_transform(lambda grad: grad.transpose(inverse_transpose_axes), x)

        return x

    def relu(self, alpha=.02):
        if self.dynamic and self.upstreams and not self.next_op_new_tensor_forced:
            self.data[self.data < 0] *= alpha
            x = self

        else:
            x = Tensor(self.data.copy(),
                       (self,),
                       logging=self.logging,
                       requires_grad=self.requires_grad,
                       dynamic=self.dynamic,
                       mode=self.mode)
            x.data[x.data < 0] *= alpha

        if self.dynamic:
            x.next_op_new_tensor_forced = True

        def _f(grad):
            grad[self.data < 0] *= alpha
            return grad

        self._add_transform(_f, x)

        return x

    def tanh(self):
        if self.dynamic and self.upstreams and not self.next_op_new_tensor_forced:
            self.data = self.lib.tanh(self.data)
            x = self
        else:
            x = Tensor(self.lib.tanh(self.data),
                       (self,),
                       logging=self.logging,
                       requires_grad=self.requires_grad,
                       dynamic=self.dynamic,
                       mode=self.mode)

        x.next_op_new_tensor_forced = True

        self._add_transform(lambda grad: grad * (1 - x.data ** 2), x)
        # 1 - np.tanh(self.data) ** 2

        return x

    def sigmoid(self):
        if self.dynamic and self.upstreams and not self.next_op_new_tensor_forced:
            self.data = 1 / (1 + self.lib.exp(-self.data))
            x = self
        else:
            x = Tensor(1 / (1 + self.lib.exp(-self.data)),
                       (self,),
                       logging=self.logging,
                       requires_grad=self.requires_grad,
                       dynamic=self.dynamic,
                       mode=self.mode)

        x.next_op_new_tensor_forced = True

        self._add_transform(lambda grad: grad * x.data * (1 - x.data), x)

        return x

    def softmax(self):
        x = self.lib.exp(self.data)
        if self.dynamic and self.upstreams and not self.next_op_new_tensor_forced:
            self.data = self.lib.apply_along_axis(lambda a: a / a.sum(), 0, x)
            x = self
        else:
            x = Tensor(self.lib.apply_along_axis(lambda a: a / a.sum(), 0, x),
                       (self,),
                       logging=self.logging,
                       requires_grad=self.requires_grad,
                       dynamic=self.dynamic,
                       mode=self.mode)

        x.next_op_new_tensor_forced = True

        def _f(grad):
            M = x.data @ self.lib.ones((*x.data.shape[:-2], x.data.shape[-1], x.data.shape[-2]))
            return (M * (self.lib.identity(M.shape[-1]) - self.lib.transpose(M, axes=(
                *range(len(M.shape) - 2), len(M.shape) - 1, len(M.shape) - 2)))) @ grad

        self._add_transform(_f, x)

        return x

    def exp(self, new_tensor=False):
        if self.dynamic and self.upstreams and not new_tensor and not self.next_op_new_tensor_forced:
            self.data = self.lib.exp(self.data)
            x = self
        else:
            x = Tensor(self.lib.exp(self.data),
                       (self,),
                       logging=self.logging,
                       requires_grad=self.requires_grad,
                       dynamic=self.dynamic,
                       mode=self.mode)

        x.next_op_new_tensor_forced = True

        self._add_transform(lambda grad: x.data * grad, x)

        return x

    def log(self):
        x = Tensor(self.lib.log(self.data),
                   (self,),
                   logging=self.logging,
                   requires_grad=self.requires_grad,
                   dynamic=self.dynamic,
                   mode=self.mode)

        self._add_transform(lambda grad: grad / self.data, x)

        return x

    def sum(self, axis=None, keepdims=False, new_tensor=False):
        shape = self.data.shape

        if self.dynamic and not new_tensor and not self.next_op_new_tensor_forced:
            self.data = self.lib.array([self.data.sum(axis, keepdims)])
            x = self
        else:
            x = Tensor(self.lib.array([self.data.sum(axis, keepdims)]),
                       (self,),
                       logging=self.logging,
                       requires_grad=self.requires_grad,
                       dynamic=self.dynamic,
                       mode=self.mode)

        if keepdims:
            self._add_transform(lambda grad: self.lib.repeat(grad, shape[axis], axis=axis), x)
        else:
            self._add_transform(
                lambda grad: self.lib.repeat(self.lib.expand_dims(grad, axis=axis), shape[axis], axis=axis), x)

        return x

    def mean(self, new_tensor=False):
        shape = self.data.shape
        size = self.data.size

        if self.dynamic and not new_tensor and not self.next_op_new_tensor_forced:
            self.data = self.lib.array([self.lib.mean(self.data)])
            x = self
        else:
            x = Tensor(self.lib.array([self.lib.mean(self.data)]),
                       (self,),
                       logging=self.logging,
                       requires_grad=self.requires_grad,
                       dynamic=self.dynamic,
                       mode=self.mode)

        self._add_transform(lambda grad: self.lib.full(shape, grad / size), x)

        return x

    def reshape(self, shape):
        if self.dynamic and self.upstreams and not self.next_op_new_tensor_forced:
            self.data = self.data.reshape(shape)
            x = self
        else:
            x = Tensor(self.data.reshape(shape),
                       (self,),
                       logging=self.logging,
                       requires_grad=self.requires_grad,
                       dynamic=self.dynamic,
                       mode=self.mode)

        self._add_transform(lambda grad: grad.reshape(self.data.shape), x)

        return x

    def flatten(self):
        return self.reshape((self.data.shape[0], math.prod(self.data.shape[1:]), 1))

    def correlate(self, kernels):
        x = cf.correlate(self.data, kernels.data)
        assert self.mode == kernels.mode
        x = Tensor(x,
                   (self, kernels),
                   logging=self.logging or kernels.logging,
                   requires_grad=self.requires_grad or kernels.requires_grad,
                   dynamic=self.dynamic or kernels.dynamic,
                   mode=self.mode)

        self._add_transform(lambda grad: cf.dl_dx_correlate(grad, kernels.data), x)
        kernels._add_transform(lambda grad: cf.dl_dk_correlate(self.data, grad), x)

        return x

    def concat(self, *others, axis=0):
        tensors = (self, *others)
        assert (len(tensors[i - 1].shape) == len(tensors[i].shape) for i in range(len(tensors)))
        assert all([self.mode == i.mode for i in tensors])
        x = Tensor(self.lib.concatenate([t.data for t in tensors], axis=axis),
                   tensors,
                   logging=self.logging,
                   requires_grad=any(map(lambda p: p.requires_grad, tensors)),
                   dynamic=any(map(lambda p: p.dynamic, tensors)),
                   mode=self.mode)

        start_of_self = 0
        for t in tensors:
            t._add_transform(
                lambda grad: grad[tuple(
                    [slice(None) if a != axis else slice(start_of_self, start_of_self + x.data.shape[a]) for a in
                     range(len(grad.shape))])],
                x)
            start_of_self += t.data.shape[axis]

        return x

    def split(self, indices: tuple, axis=1):
        tensors = [Tensor(j,
                          (self,),
                          logging=self.logging,
                          requires_grad=self.requires_grad,
                          dynamic=self.dynamic,
                          mode=self.mode) for j in self.lib.split(self.data, indices, axis)]

        start_of_self = 0
        for x in tensors:
            self._add_transform(lambda grad: self.lib.concatenate([
                self.lib.zeros(x.data.shape[:axis] + (start_of_self,) + x.data.shape[axis + 1:]),
                grad[tuple([slice(None) if a != axis else slice(start_of_self, start_of_self + x.data.shape[a]) for a in
                            range(len(grad.shape))])],
                self.lib.zeros(
                    x.data.shape[:axis] + (self.data.shape[axis] - start_of_self - x.data.shape[axis],) + x.data.shape[
                                                                                                          axis + 1:])
            ]), x)
            start_of_self += x.data.shape[axis]

        return tensors

    def squeeze(self, axis=tuple()):
        if axis == tuple():
            ones = []
            for a, s in enumerate(self.data.shape):
                if s == 1:
                    ones.append(a)
            ones = tuple(ones)
        else:
            ones = axis
        if self.dynamic:
            self.data = self.lib.squeeze(self.data, None if axis == tuple() else axis)
            x = self
        else:
            x = Tensor(self.lib.squeeze(self.data, None if axis == tuple() else axis),
                       (self,),
                       logging=self.logging,
                       requires_grad=self.requires_grad,
                       dynamic=self.dynamic,
                       mode=self.mode)

        self._add_transform(lambda grad: self.lib.expand_dims(grad, ones), x)

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
        return Tensor(self.data.copy(),
                      self.upstreams,
                      self.requires_grad,
                      logging=self.logging,
                      trainable=self.trainable,
                      Optim=self.optim.__class__,
                      name=self.name,
                      dynamic=self.dynamic,
                      mode=self.mode)

    def dropout(self, rate=.05, magnitude_correction=True):
        assert 0 <= rate < 1
        if magnitude_correction:
            dropout_tensor = Tensor(self.lib.random.choice((0, 1), self.data.shape, p=(rate, 1 - rate)) / (1 - rate),
                                    logging=self.logging,
                                    dynamic=self.dynamic,
                                    mode=self.mode)
        else:
            dropout_tensor = Tensor(self.lib.random.choice((0, 1), self.data.shape, p=(rate, 1 - rate)),
                                    logging=self.logging,
                                    dynamic=self.dynamic,
                                    mode=self.mode)
        return self * dropout_tensor

    def pool(self, sizes, criterion, criterion_included, poolfunc=None):
        """size: pool size; criterion: np.max | np.min | np.average ... pointer; criterion included: np.argmax | np.argmin | lambda x: x ... pointer"""
        if poolfunc is None:
            poolfunc = cf.pool
        x, included = poolfunc(self.data, sizes, criterion, criterion_included)
        x = Tensor(x,
                   (self,),
                   logging=self.logging,
                   requires_grad=self.requires_grad,
                   dynamic=self.dynamic,
                   mode=self.mode)

        self._add_transform(lambda grad: (grad.flatten() * included).reshape(self.data.shape), x)

        return x

    def maxpool(self, sizes):
        return self.pool(sizes, self.lib.max, lambda a: np.unravel_index(self.lib.argmax(a)))

    def minpool(self, sizes):
        return self.pool(sizes, self.lib.min, lambda a: np.unravel_index(self.lib.argmin(a)))

    def averagepool(self, sizes):
        return self.pool(sizes, self.lib.average, lambda x: range(x.size), poolfunc=cf.averagepool)

    def normalize(self):
        batchdivs = np.apply_along_axis(lambda a: 1 / a.sum(), 0, self.data)

        if self.dynamic:
            self.data *= batchdivs
            x = self
        else:
            x = Tensor(self.data * batchdivs,
                   (self,),
                   logging=self.logging,
                   requires_grad=self.requires_grad,
                   dynamic=self.dynamic,
                   mode=self.mode)

        self._add_transform(lambda grad: grad * batchdivs, x)

        return x

    def l2normalize(self):
        batchdivs = self.lib.apply_along_axis(lambda a: 1 / self.lib.sqrt((a * a).sum()))

        if self.dynamic and self.upstreams and not self.next_op_new_tensor_forced:
            self.data *= batchdivs
            x = self
        else:
            x = Tensor(self.data * batchdivs,
                       (self,),
                       logging=self.logging,
                       requires_grad=self.requires_grad,
                       dynamic=self.dynamic,
                       mode=self.mode)

        self._add_transform(lambda grad: grad * batchdivs, x)

        return x

    def standardize(self, standard_deviation=1):
        # looks very confusing I know
        # derivation in the readme.md
        # hope it's correct lol
        ma_g_x = []  # mean(abs(g(x)))
        g_x = []  # g(x)

        def _f(grad):
            """this cannot be parallelized within a tensor so not the best implementation, probably. might try (again) to
            use numba to jit-compile it, but it didn't work last time."""
            # compute derivative array of abs(x)
            abs_prime = self.lib.empty(grad.shape)
            abs_prime[g_x[0] < 0] = -1 / grad.size
            abs_prime[g_x[0] >= 0] = 1 / grad.size
            # compute updated gradient (derivation in readme.md)
            grad *= standard_deviation * (1 - 1 / grad.size) * (ma_g_x[0] - g_x[0] * abs_prime[0]) / ma_g_x[
                0] ** 2  # note: grad.size = g_x[0].size
            ma_g_x.pop(0)  # -> when _f(grad) is called for the grad of the next
            g_x.pop(0)
            return grad

        def _g(ary):
            # apply g(x); note: ary = x in notes
            ary -= self.lib.mean(ary)
            # save value
            g_x.append(ary)
            # compute mean(abs(x))
            meanabs = self.lib.mean(self.lib.abs(ary))
            # save the mean(abs(x)) since it's expensive to compute but only a float to store
            ma_g_x.append(meanabs)
            return ary * standard_deviation / (meanabs + .01)  # .01 is there to make sure you don't have divide by zero and there's a cap to how big it can get

        x = Tensor(self.lib.apply_along_axis(_g, 0, self.data),
                   (self,),
                   logging=self.logging,
                   requires_grad=self.requires_grad,
                   dynamic=self.dynamic,
                   mode=self.mode)

        self._add_transform(lambda grad: self.lib.apply_along_axis(_f, 0, grad), x)

        return x

    def squish(self, lower, upper):
        """squishes the data into a certain range keeping the PROPORTIONS OF DIFFERENCES the same"""
        minv = self.lib.min(self.data)
        squish_factor = (upper - lower) / (self.lib.max(self.data) - minv + .01)
        if self.dynamic:
            self.data *= squish_factor
            self.data += lower - minv * squish_factor
            x = self
        else:
            x = Tensor((self.data * squish_factor + (lower - minv * squish_factor)),
                       (self,),
                       logging=self.logging,
                       requires_grad=self.requires_grad,
                       dynamic=self.dynamic,
                       mode=self.mode)

        self._add_transform(lambda grad: grad * squish_factor, x)

        return x

    def statify(self):
        self.dynamic = False
        return self

    def dynamify(self):
        self.dynamic = True
        return self

    def apply(self, func, func_prime=None, new_tensor=True):
        if func_prime is None:
            func_prime = lambda grad: 0
            warnings.warn(
                f"{self.name}._apply({func}, {None}, {new_tensor}) called without func_prime given. func_prime's default value of lambda grad: 0 used. This will stop gradient flow.")
        if new_tensor or not self.dynamic:
            x = Tensor(func(self.data),
                       (self,),
                       logging=self.logging,
                       requires_grad=self.requires_grad,
                       dynamic=self.dynamic,
                       mode=self.mode)
        else:
            self.data = func(self.data)
            x = self

        self._add_transform(func_prime, x)

        return x

    def apply_along_axis(self, func, func_prime=None, axis=0, new_tensor=True):
        if func_prime is None:
            func_prime = lambda grad: 0
            warnings.warn(
                f"{self.name}._apply({func}, {None}, {new_tensor}) called without func_prime given. func_prime's default value of lambda grad: 0 used. This will stop gradient flow.")
        if new_tensor or not self.dynamic:
            x = Tensor(np.apply_along_axis(func, axis, self.data),
                       (self,),
                       logging=self.logging,
                       requires_grad=self.requires_grad,
                       dynamic=self.dynamic,
                       mode=self.mode)
        else:
            self.data = func(self.data)
            x = self

        self._add_transform(lambda grad: np.apply_along_axis(func_prime, axis, grad), x)

        return x

    def _add_transform(self, function, x, others=tuple()):
        """add a function of the gradient to the transforms that will be applied to it in the backward pass"""
        # x = upstream
        if not self.requires_grad and not any(map(lambda a: a.requires_grad, others)):
            return
        if self.transforms.get(x) is None:
            self.transforms[x] = []
        self.transforms[x].append((function, others))

    def zero_grad(self, x=None, delete_gradient=True,
                  depth=0, max_depth=math.inf):
        """delete_gradient should be set to False for deep systems like RNNs, but to True for shallow systems (MLPs, ..; basically everything without hidden states)"""
        # x = upstream
        if not self.requires_grad:
            return

        if x is None:
            x = self

        if self.logging:
            print(f"zero grad @ {self.name}")

        self.optim.zero_grad(x)

        if depth > max_depth or delete_gradient:
            self.transforms[x] = []

        transform_keys = set(self.transforms.keys())
        transform_keys.remove(self)

        for upstream in self.upstreams:
            upstream.zero_grad(self, not transform_keys, depth + 1, max_depth)

        if not transform_keys:
            self.upstreams = tuple()
            self.transforms[self] = []  # same as self.transforms = {}

    # todo: implement topo
    def backward(self, gradient=None, x=None, depth=0, max_depth=math.inf):
        # x = upstream
        # note: there are no checks if tensor.requires_grad because if it is false, it isn't connected to the graph
        if depth > max_depth:
            return

        if x is None:
            x = self

        if gradient is None:
            gradient = self.lib.ones(self.data.shape)

        if self.logging:
            print(f"backward @ {self.name}")

        # commented out because .dynamify() could trigger this
        '''if not self.dynamic and self.transforms.get(self):
            warnings.warn(
                f"{self.name}: self.dynamic is False (-> threadsafe), but nevertheless, non-threadsafe methods were found")'''

        if x != self:
            if self.transforms.get(x) is None:
                self.transforms[x] = []

            for transform, others in reversed(self.transforms[x]):  # update gradients
                gradient = transform(gradient)
                for other in others:
                    other.backward(gradient, self)

        if self.transforms.get(self) is None:
            self.transforms[self] = []

        for transform, others in reversed(self.transforms[self]):  # only possible if self.dynamic is True
            gradient = transform(gradient)
            for other in others:
                other.backward(gradient, self)

        if self.trainable:
            self.optim.update(gradient.sum(axis=0, keepdims=True) / gradient.shape[0], x)  # average gradient in batch samples

        for upstream in self.upstreams:  # continue backprop in parents
            upstream.backward(gradient, self, depth + 1, max_depth)

    def optimize(self, x=None, lr=0.01, depth=0, max_depth=math.inf):
        # x = upstream
        if depth > max_depth:
            return

        if x is None:
            x = self

        if self.logging:
            print(f"optimize @ {self.name}")

        if self.trainable:
            self.optim.step(lr, x)

        for upstream in self.upstreams:
            upstream.optimize(self, lr, depth + 1, max_depth)
