import numpy as np
from Tensor import Tensor


class Sequential:
    def __init__(self, layers: list):
        self.layers = layers

    def __call__(self, x: Tensor):
        if isinstance(x, np.ndarray):
            x = Tensor(x)
        for layer in self.layers:
            x = layer(x)
        return x
