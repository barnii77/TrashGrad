import math
import numpy as np
from trashgrad.autograd._Function import Function
from trashgrad.autograd import Tensor
import trashgrad.autograd._ConvFunctions as cf
from numba import jit


class AsType(Function):
    backward_required_args_kwargs = set()

    def _forward(self, tensor: Tensor, dtype):
        self.input_dtype = tensor.dtype
        return tensor.data.astype(dtype)

    def _backward(self, grad, req: Tensor, tensor: Tensor, dtype):
        return grad.astype(self.input_dtype)


class Add(Function):
    backward_required_args_kwargs = set()

    def _forward(self, *args: Tensor):
        return sum(map(lambda x: x.data, args[1:]), start=args[0].data)

    def _backward(self, grad, req: Tensor, *args: Tensor):
        return grad


class Sub(Function):
    backward_required_args_kwargs = set()

    def _forward(self, *args: Tensor):
        return args[0].data - sum(map(lambda x: x.data, args[2:]), args[1].data)

    def _backward(self, grad, req: Tensor, *args: Tensor):
        if req is args[0]:
            return grad
        else:
            return -grad


class Mul(Function):
    backward_required_args_kwargs = True

    def _forward(self, *args: Tensor):
        return args[0].lib.prod(list(map(lambda x: x.data, args)))

    def _backward(self, grad, req: Tensor, *args: Tensor):
        return args[0].lib.prod(list(map(lambda x: 1. if x is req else x.data, args)))


class Pow(Function):
    backward_required_args_kwargs = True

    def _forward(self, tensor: Tensor, power: Tensor):
        self.out = tensor.data ** power.data
        return self.out

    def _backward(self, grad, req: Tensor, tensor: Tensor, power: Tensor):
        if req is tensor:
            return power.data * tensor.data ** (power.data - 1)
        else:
            return self.out * tensor.lib.log(tensor.data)


class Div(Function):
    backward_required_args_kwargs = True

    def _forward(self, a: Tensor, b: Tensor):
        return a.data / b.data

    def _backward(self, grad, req: Tensor, a: Tensor, b: Tensor):
        if req is a:
            return grad / b.data
        else:
            return -grad * a.data / b.data ** 2


class Matmul(Function):
    backward_required_args_kwargs = True

    def _forward(self, a: Tensor, b: Tensor):
        return a.data @ b.data

    def _backward(self, grad, req: Tensor, a: Tensor, b: Tensor):
        if req is a:
            return grad @ b.data.transpose(
                (*range(len(b.data.shape) - 2), len(b.data.shape) - 1, len(b.data.shape) - 2))
        else:
            return a.data.transpose(
                (*range(len(a.data.shape) - 2), len(a.data.shape) - 1, len(a.data.shape) - 2)) @ grad


class Neg(Function):
    backward_required_args_kwargs = set()

    def _forward(self, tensor: Tensor):
        return -tensor.data

    def _backward(self, grad, req: Tensor, tensor: Tensor):
        return -grad


class Abs(Function):
    backward_required_args_kwargs = set()

    def _forward(self, tensor: Tensor):
        self.abs_prime = (tensor.data > 0) * 2. - 1.
        return tensor.lib.abs(tensor.data)

    def _backward(self, grad, req: Tensor, tensor: Tensor):
        return grad * self.abs_prime


class Get(Function):
    backward_required_args_kwargs = set()

    def _forward(self, tensor: Tensor, indices):
        if not isinstance(indices, tuple):
            self.indices = indices.data if isinstance(indices, Tensor) else indices
        else:
            self.indices = tuple(index.data if isinstance(index, Tensor) else index for index in indices)
        return tensor.data[self.indices]

    def _backward(self, grad, req: Tensor, tensor: Tensor, indices):
        out = tensor.lib.zeros(tensor.shape)
        out[self.indices] = grad
        return out


class Set(Function):
    backward_required_args_kwargs = set()

    def _forward(self, outer: Tensor, inner: Tensor, indices):
        if not isinstance(indices, tuple):
            self.indices = indices.data if isinstance(indices, Tensor) else indices
        else:
            self.indices = tuple(index.data if isinstance(index, Tensor) else index for index in indices)
        out = outer.data
        out[self.indices] = inner.data
        return out

    def _backward(self, grad, req: Tensor, outer: Tensor, inner: Tensor, indices):
        if req is outer:
            grad[self.indices] = 0.0
            return grad
        else:
            mask = inner.lib.ones(grad.shape, dtype=bool)
            mask[self.indices] = False
            grad[mask] = 0.0


class Exp(Function):
    backward_required_args_kwargs = set()

    def _forward(self, tensor: Tensor):
        self.out = tensor.lib.exp(tensor.data)
        return self.out

    def _backward(self, grad, req: Tensor, tensor: Tensor):
        return grad * self.out


class Log(Function):
    backward_required_args_kwargs = True

    def _forward(self, tensor: Tensor, base: float):
        if base is None:
            return tensor.lib.log(tensor.data)
        self.inv_log_base = 1 / math.log(base)
        return tensor.lib.log(tensor.data) * self.inv_log_base

    def _backward(self, grad, req: Tensor, tensor: Tensor, base: float):
        if base is None:
            return grad / tensor.data
        return grad / tensor.data * self.inv_log_base


class Sqrt(Function):
    backward_required_args_kwargs = set()

    def _forward(self, tensor: Tensor):
        self.out = tensor.lib.sqrt(tensor.data)
        return self.out

    def _backward(self, grad, req: Tensor, tensor: Tensor):
        return grad / (2 * self.out)


class Sum(Function):
    backward_required_args_kwargs = set()

    def _forward(self, tensor, axis: int = None, keepdims=False):
        return tensor.data.sum(axis=axis, keepdims=keepdims)

    def _backward(self, grad, req: Tensor, tensor: Tensor, axis: int = None, keepdims: bool = False):
        if axis is None:
            return tensor.lib.full(tensor.shape, grad)
        elif keepdims:
            return tensor.lib.repeat(grad, tensor.shape[axis], axis=axis)
        return tensor.lib.repeat(tensor.lib.expand_dims(grad, axis=axis), tensor.shape[axis], axis=axis)


class Mean(Function):
    backward_required_args_kwargs = set()

    def _forward(self, tensor: Tensor, axis: int = None, keepdims: bool = False):
        return tensor.lib.array(tensor.lib.mean(tensor.data, axis=axis, keepdims=keepdims))

    def _backward(self, grad, req: Tensor, tensor: Tensor, axis: int = None, keepdims: bool = False):
        if axis is None:
            return tensor.lib.full(tensor.shape, grad / tensor.size)
        elif keepdims:
            return tensor.lib.repeat(grad / tensor.shape[axis], tensor.shape[axis], axis)
        return tensor.lib.repeat(tensor.lib.expand_dims(grad, axis) / tensor.shape[axis], tensor.shape[axis], axis)


class Reshape(Function):
    backward_required_args_kwargs = set()

    def _forward(self, tensor: Tensor, shape: tuple[int]):
        return tensor.data.reshape(shape)

    def _backward(self, grad, req: Tensor, tensor: Tensor, shape: tuple[int]):
        return grad.reshape(tensor.shape)


class Transpose(Function):
    backward_required_args_kwargs = set()

    def _forward(self, tensor: Tensor, axes: tuple[int]):
        self.inverse_transpose_axes = [0 for _ in range(len(axes))]
        for i, val in enumerate(axes):
            self.inverse_transpose_axes[val] = i
        self.inverse_transpose_axes = tuple(self.inverse_transpose_axes)
        return tensor.data.transpose(axes)

    def _backward(self, grad, req: Tensor, tensor: Tensor, axes: tuple[int]):
        return grad.transpose(self.inverse_transpose_axes)


class Flatten(Function):
    backward_required_args_kwargs = set()

    def _forward(self, tensor: Tensor):
        return tensor.data.flatten()

    def _backward(self, grad, req: Tensor, tensor: Tensor):
        return grad.reshape(tensor.shape)


class CrossCorrelateBatches(Function):
    backward_required_args_kwargs = True

    def _forward(self, volumes: Tensor, kernels: Tensor, reduce_input_depth_dimension: bool = True):
        return cf.correlate(volumes.data, kernels.data, reduce_input_depth_dimension=reduce_input_depth_dimension)

    def _backward(self, grad, req: Tensor, volumes: Tensor, kernels: Tensor, reduce_input_depth_dimension: bool = True):
        if req is volumes:
            return cf.dl_dx_correlate(grad, kernels.data)
        else:
            return cf.dl_dk_correlate(volumes.data, grad)


class Concatenate(Function):
    backward_required_args_kwargs = set()

    def _forward(self, *tensors: Tensor, axis=0):
        return tensors[0].lib.concatenate([t.data for t in tensors], axis=axis)

    def _backward(self, grad, req: Tensor, *tensors: Tensor, axis=0):
        start_of_tensor = 0
        for tensor in tensors:
            if tensor is req:
                break
            start_of_tensor += tensor.shape[axis]
        return grad[tuple(
            [slice(None) for _ in range(axis)] + [
                slice(start_of_tensor, start_of_tensor + req.shape[axis])])]


class Squeeze(Function):
    backward_required_args_kwargs = set()

    def _forward(self, tensor: Tensor, axis=None):
        if axis == tuple():
            ones = []
            for a, s in enumerate(tensor.shape):
                if s == 1:
                    ones.append(a)
            self.ones = tuple(ones)
        else:
            self.ones = axis
        return tensor.data.squeeze(None if axis == tuple() else axis)

    def _backward(self, grad, req: Tensor, tensor: Tensor, axis=None):
        return tensor.lib.expand_dims(grad, self.ones)


class Squish(Function):
    backward_required_args_kwargs = set()

    def _forward(self, tensor, lower, upper):
        assert lower != upper
        tensor_min = tensor.lib.min(tensor.data)
        tensor_max = tensor.lib.max(tensor.data)
        assert tensor_max != tensor_min
        self.squish_factor = (upper - lower) / (tensor_max - tensor_min)  # add some delta to it
        return tensor.data + (lower - tensor_min * self.squish_factor)

    def _backward(self, grad, req: Tensor, tensor, lower, upper):
        return grad * self.squish_factor


class Pool(Function):
    backward_required_args_kwargs = set()

    def _forward(self, tensor: Tensor, sizes: tuple[int], criterion, criterion_included, backend=None):
        out, self.included = cf.pool(tensor.data, sizes, criterion, criterion_included, backend=backend)
        return out

    def _backward(self, grad, req: Tensor, tensor: Tensor, sizes: tuple[int], criterion, criterion_included, backend=None):
        return (grad.flatten() * self.included).reshape(tensor.shape)


def _maxpool_criterion(subarray):
    return subarray.max()


@jit(target_backend="cpu")
def _cpu_maxpool_criterion(subarray):
    return subarray.max()


@jit(target_backend="cuda")
def _cuda_maxpool_criterion(subarray):
    return subarray.max()


def _minpool_criterion(subarray):
    return subarray.min()


@jit(target_backend="cpu")
def _cpu_minpool_criterion(subarray):
    return subarray.min()


@jit(target_backend="cuda")
def _cuda_minpool_criterion(subarray):
    return subarray.min()


def _maxpool_criterion_included(subarray):
    return np.unravel_index(subarray.argmax(), subarray.shape)


@jit(target_backend="cpu")
def _cpu_maxpool_criterion_included(subarray):
    return np.unravel_index(subarray.argmax(), subarray.shape)


@jit(target_backend="cuda")
def _cuda_maxpool_criterion_included(subarray):
    return np.unravel_index(subarray.argmax(), subarray.shape)


def _minpool_criterion_included(subarray):
    return np.unravel_index(subarray.argmin(), subarray.shape)


@jit(target_backend="cpu")
def _cpu_minpool_criterion_included(subarray):
    return np.unravel_index(subarray.argmin(), subarray.shape)


@jit(target_backend="cuda")
def _cuda_minpool_criterion_included(subarray):
    return np.unravel_index(subarray.argmin(), subarray.shape)


class MaxPool(Function):
    backward_required_args_kwargs = set()

    def _forward(self, tensor: Tensor, sizes: tuple[int], backend=None):
        if backend is None:
            criterion, criterion_included = _maxpool_criterion, _maxpool_criterion_included
        elif backend == "cpu":
            criterion, criterion_included = _cpu_maxpool_criterion, _cpu_maxpool_criterion_included
        elif backend == "cuda":
            criterion, criterion_included = _cuda_maxpool_criterion, _cuda_maxpool_criterion_included
        else:
            raise Exception(f"{backend} is not a valid backend (None/cpu/cuda)")
        out, self.included = cf.pool(tensor.data, sizes, criterion, criterion_included, backend=backend)
        return out

    def _backward(self, grad, req: Tensor, tensor: Tensor, sizes: tuple[int], backend=None):
        return (grad.flatten() * self.included).reshape(tensor.shape)


class MinPool(Function):
    backward_required_args_kwargs = set()

    def _forward(self, tensor: Tensor, sizes: tuple[int], backend=None):
        if backend is None:
            criterion, criterion_included = _minpool_criterion, _minpool_criterion_included
        elif backend == "cpu":
            criterion, criterion_included = _cpu_minpool_criterion, _cpu_minpool_criterion_included
        elif backend == "cuda":
            criterion, criterion_included = _cuda_minpool_criterion, _cuda_minpool_criterion_included
        else:
            raise Exception(f"{backend} is not a valid backend (None/cpu/cuda)")
        out, self.included = cf.pool(tensor.data, sizes, criterion, criterion_included, backend=backend)
        return out

    def _backward(self, grad, req: Tensor, tensor: Tensor, sizes: tuple[int], backend=None):
        return (grad.flatten() * self.included).reshape(tensor.shape)


class AveragePool(Function):
    backward_required_args_kwargs = set()

    def _forward(self, tensor: Tensor, sizes: tuple[int], backend=None):
        out, self.included = cf.averagepool(tensor.data, sizes, None, None, backend=backend)
        return out

    def _backward(self, grad, req: Tensor, tensor: Tensor, sizes: tuple[int], backend=None):
        return (grad.flatten() * self.included).reshape(tensor.shape)


class ApplyAlongAxis(Function):
    def __init__(self, backward_requires_input_data: bool):
        self.backward_required_args_kwargs = True if backward_requires_input_data else set()

    def _forward(self, tensor: Tensor, axis: int, forwardf: callable, backwardf: callable):
        out = tensor.lib.apply_along_axis(forwardf, axis, tensor.data)
        if self.backward_required_args_kwargs:
            def _backwardf(arr):
                grad, data = tensor.lib.split(arr, (out.shape[axis],))
                return backwardf(grad, data)
            self.backwardf = _backwardf
        else:
            self.backwardf = backwardf
        return out

    def _backward(self, grad, req: Tensor, tensor: Tensor, axis: int, forwardf: callable, backwardf: callable):
        return tensor.lib.apply_along_axis(self.backwardf, axis, tensor.lib.concatenate(grad, tensor.data, axis=axis))
