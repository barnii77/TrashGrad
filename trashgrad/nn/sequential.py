import numpy as np
from trashgrad.autograd import Tensor
from trashgrad.nn import Layer


class Sequential:
    def __init__(self, layers: list):
        self.layers = layers

    def __call__(self, x: Tensor) -> Tensor:
        if isinstance(x, np.ndarray):
            x = Tensor(x)
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self) -> list[Tensor]:
        return sum((layer.parameters() for layer in self.layers if isinstance(layer, Layer)), start=[])
