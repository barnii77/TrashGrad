from Tensor import Tensor
import numpy as np
from Optimizer import Default as DefaultOptim


class Dense:
	def __init__(self, insize: int, outsize: int, Optim=DefaultOptim):
		self.w = Tensor(np.random.randn(1, outsize, insize), Optim=Optim, dynamic=False, trainable=True)
		self.b = Tensor(np.random.randn(1, outsize, 1), Optim=Optim, dynamic=False, trainable=True)
	
	def __call__(self, x: Tensor):
		return self.w @ x + self.b

	def __repr__(self):
		return "Dense"  # self.w.__repr__() + "\n\n" + self.b.__repr__()
