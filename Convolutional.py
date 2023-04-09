from Tensor import Tensor
import numpy as np


class Convolutional2D:
    def __init__(self, input_size: int, num_kernels: int, kernel_depth: int, kernel_width: int, kernel_height: int, output_depth: int, output_width: int, output_height: int, model="cpu", Optim=None):
        """Input size needs to be equal to the product of the input shape (without batch size)"""
        k_init = Tensor(np.random.randn(1, num_kernels, kernel_depth, kernel_width, kernel_height)).standardize(np.sqrt(input_size)).data
        self.k = Tensor(k_init, trainable=True, Optim=Optim, dynamic=False)
        self.b = Tensor(np.random.randn(1, output_depth, output_width, output_height), trainable=True, Optim=Optim, dynamic=False)

    def __call__(self, x: Tensor):
        return x.correlate(self.k) + self.b

    def __repr__(self):
        return "Convolutional"  # self.k.__repr__() + "\n\n" + self.b.__repr__()
