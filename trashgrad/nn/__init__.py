from trashgrad.nn.layers import Convolutional2D, Dense, Layer
from trashgrad.nn.optim import GD, MomentumGD, RMSProp, Adam
from trashgrad.nn.sequential import Sequential
from trashgrad.autograd import Tensor


class Module:
    features = []

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs) -> Tensor:
        return self.forward(*args, **kwargs)

    def parameters(self) -> list[Tensor]:
        return sum(([feature] if isinstance(feature, Tensor) else feature.parameters() for feature in self.features), start=[])


#__all__ = [Convolutional2D, Dense, GD, MomentumGD, RMSProp, Adam, Sequential]
