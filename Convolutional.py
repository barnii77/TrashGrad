from Tensor import Tensor
import numpy as np
from Optimizer import Default as DefaultOptim


class Convolutional2D:
    def __init__(self, num_kernels: int, kernel_depth: int, kernel_width: int, kernel_height: int, output_depth: int, output_width: int, output_height: int, Optim=DefaultOptim):
        self.k = Tensor(np.random.randn(num_kernels, kernel_depth, kernel_width, kernel_height), requires_grad=True, Optim=Optim, dynamic=False)
        self.b = Tensor(np.random.randn(1, output_depth, output_width, output_height), requires_grad=True, Optim=Optim, dynamic=False)

    def __call__(self, x: Tensor):
        return x.correlate(self.k) + self.b

    def __repr__(self):
        return self.k.__repr__() + "\n\n" + self.b.__repr__()
