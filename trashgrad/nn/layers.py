import math
from trashgrad.autograd import Tensor
from trashgrad.autograd import tensor
import numpy as np


class Layer:
    def __call__(self, x: Tensor):
        raise NotImplementedError()

    def parameters(self) -> list[Tensor]:
        raise NotImplementedError()


class Dense(Layer):
    def __init__(self, insize: int, outsize: int, mode="cpu"):
        self.w = tensor(np.random.randn(insize, outsize) / math.sqrt(insize), requires_grad=True, mode=mode)
        self.b = tensor(np.random.randn(outsize) / math.sqrt(insize), requires_grad=True, mode=mode)

    def __call__(self, x: Tensor) -> Tensor:
        return x @ self.w + self.b

    def __repr__(self):
        return "Dense"  # self.w.__repr__() + "\n\n" + self.b.__repr__()

    def parameters(self) -> list[Tensor]:
        return [self.w, self.b]


class Convolutional2D(Layer):
    def __init__(self, num_kernels: int, num_channels: int, kernel_width: int, kernel_height: int, input_width: int, input_height: int, mode="cpu", reduce_input_depth_dimension=True):
        """Input size needs to be equal to the product of the input shape (without batch size)"""
        self.k = tensor(np.random.randn(num_kernels, num_channels, kernel_width, kernel_height) / math.sqrt(num_channels), requires_grad=True, mode=mode)
        self.b = tensor(np.random.randn(num_kernels, input_width - kernel_width + 1, input_height - kernel_height + 1), requires_grad=True, mode=mode)
        self.reduce_input_depth_dimension = reduce_input_depth_dimension

    def __call__(self, x: Tensor) -> Tensor:
        return x.correlate(self.k, reduce_input_depth_dimension=self.reduce_input_depth_dimension) + self.b

    def __repr__(self):
        return "Convolutional"  # self.k.__repr__() + "\n\n" + self.b.__repr__()

    def parameters(self) -> list[Tensor]:
        return [self.k, self.b]
