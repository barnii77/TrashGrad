import cupy as cp
import numpy as np
from trashgrad.autograd.tensor import Tensor
import trashgrad.autograd._Operations as ops


def tensor(data, requires_grad=False, mode="cpu") -> Tensor:
    if mode == "cpu":
        data = np.array(data)
    elif mode == "gpu":
        data = cp.array(data)
    else:
        raise Exception(f"{mode} is not a valid mode")
    return Tensor(data, requires_grad=requires_grad, mode=mode)


def multiply(a: Tensor, b: Tensor) -> Tensor:
    a = a if isinstance(a, Tensor) else tensor(a)
    b = b if isinstance(b, Tensor) else tensor(b)
    return ops.Mul()(a, b)


def matmul(a: Tensor, b: Tensor) -> Tensor:
    a = a if isinstance(a, Tensor) else tensor(a)
    b = b if isinstance(b, Tensor) else tensor(b)
    return ops.Matmul()(a, b)


def add(a: Tensor, b: Tensor) -> Tensor:
    a = a if isinstance(a, Tensor) else tensor(a)
    b = b if isinstance(b, Tensor) else tensor(b)
    return ops.Add()(a, b)


def sub(a: Tensor, b: Tensor) -> Tensor:
    a = a if isinstance(a, Tensor) else tensor(a)
    b = b if isinstance(b, Tensor) else tensor(b)
    return ops.Sub()(a, b)


def divide(a: Tensor, b: Tensor) -> Tensor:
    a = a if isinstance(a, Tensor) else tensor(a)
    b = b if isinstance(b, Tensor) else tensor(b)
    return ops.Div()(a, b)


def mul(a: Tensor, b: Tensor) -> Tensor:
    return multiply(a, b)


def div(a: Tensor, b: Tensor) -> Tensor:
    return divide(a, b)


def power(a: Tensor, b: Tensor) -> Tensor:
    a = a if isinstance(a, Tensor) else tensor(a)
    b = b if isinstance(b, Tensor) else tensor(b)
    return ops.Pow()(a, b)


def pow(a: Tensor, b: Tensor) -> Tensor:
    return power(a, b)


def neg(t: Tensor) -> Tensor:
    return ops.Neg()(t)


def abs(t: Tensor) -> Tensor:
    return ops.Abs()(t)


def get(t: Tensor, slices: tuple) -> Tensor:
    return ops.Get()(t, slices)


def set(outer: Tensor, inner: Tensor, slices: tuple) -> Tensor:
    return ops.Set()(outer, inner, slices)


def exp(t: Tensor) -> Tensor:
    return ops.Exp()(t)


def log(t: Tensor, base: float = None) -> Tensor:
    return ops.Log()(t, base)


def sqrt(t: Tensor) -> Tensor:
    return ops.Sqrt()(t)


def sum(t: Tensor, axis=None, keepdims=False) -> Tensor:
    return ops.Sum()(t, axis, keepdims)


def mean(t: Tensor, axis=None, keepdims=False) -> Tensor:
    return ops.Mean()(t, axis, keepdims)


def reshape(t: Tensor, shape: tuple[int]) -> Tensor:
    return ops.Reshape()(t, shape)


def transpose(t: Tensor, axes: tuple[int]) -> Tensor:
    return ops.Transpose()(t, axes)


def flatten(t: Tensor) -> Tensor:
    return ops.Flatten()(t)


def correlate(volumes: Tensor, kernels: Tensor, reduce_input_depth_dimension=True) -> Tensor:
    return ops.CrossCorrelateBatches()(volumes, kernels, reduce_input_depth_dimension)


def concatenate(*tensors: Tensor, axis=0) -> Tensor:
    return ops.Concatenate()(*tensors, axis=axis)


def chunk(t: Tensor, chunks: int, axis=-1):
    assert t.shape[axis] % chunks == 0
    sections = t.shape[axis] // chunks
    indices = [tuple([slice(None) for _ in range(axis)] + [slice(i * sections, (i + 1) * sections)]) for i in range(chunks)]
    return [t[i] for i in indices]


def squeeze(t: Tensor, axis=None) -> Tensor:
    return ops.Squeeze()(t, axis)


def squish(t: Tensor, lower: float, upper: float) -> Tensor:
    return ops.Squish()(t, lower, upper)


def pool(volumes: Tensor, sizes: tuple[int], criterion, criterion_included, backend=None) -> Tensor:
    return ops.Pool()(volumes, sizes, criterion, criterion_included, backend)


def maxpool(volumes: Tensor, sizes: tuple[int], backend=None) -> Tensor:
    return ops.MaxPool()(volumes, sizes, backend)


def minpool(volumes: Tensor, sizes: tuple[int], backend=None) -> Tensor:
    return ops.MinPool()(volumes, sizes, backend)


def averagepool(volumes: Tensor, sizes: tuple[int], backend=None) -> Tensor:
    return ops.AveragePool()(volumes, sizes, backend)


def std(t: Tensor, axis=None, keepdims=False) -> Tensor:
    size = t.size if axis is None else t.shape[axis]
    return sqrt(sum((t - mean(t, axis, keepdims)) ** 2, axis, keepdims) / (size - 1))


def variance(t: Tensor, axis=None, keepdims=False) -> Tensor:
    return sqrt(mean((t - mean(t, axis, keepdims)) ** 2, axis, keepdims))


def apply_along_axis(t: Tensor, axis: int, forward: callable, backward: callable, backward_requires_input_data: bool = False) -> Tensor:
    return ops.ApplyAlongAxis(backward_requires_input_data)(t, axis, forward, backward)


def astype(t: Tensor, dtype):
    return ops.AsType()(t, dtype)


#__all__ = [apply_along_axis, add, sub, div, mul, divide, multiply, power, pow, squish, sqrt, squeeze, std, variance, pool, maxpool, minpool, averagepool, flatten, transpose, reshape, concatenate, correlate, mean, sum, exp, log, abs, neg, tensor, matmul]
