import numpy as np
from Tensor import Tensor


class Loss:
	y: Tensor
	prime: float
	error: float

	def backward(self, lr=None, zero_grad=False):
		self.y.backward(self.prime, None)
		if lr is not None:
			self.optimize(lr)
			if zero_grad:
				self.zero_grad()

	def optimize(self, lr=.01):
		self.y.optimize(None, lr)

	def zero_grad(self):
		self.y.zero_grad(None)


class MSE(Loss):
	def __init__(self, y: Tensor, ystar, requires_error=False):  # super.__init__() doesnt do anything
		self.name = "Mean Squared Error"
		self.y = y
		self.error = 0.0
		if isinstance(ystar, Tensor):
			if requires_error:
				self.error = ((y.data - ystar.data) ** 2).sum()
			self.prime = y.data - ystar.data
		elif isinstance(ystar, np.ndarray):
			if requires_error:
				self.error = ((y.data - ystar) ** 2).sum()
			self.prime = y.data - ystar
		else:
			raise Exception(f"Exception @ {self}: ystar needs to be Tensor or ndarray, not {type(ystar)}")


class CrossEntropyLoss(Loss):
	def __init__(self, y: Tensor, ystar, requires_error=False):
		self.name = "Cross Entropy Loss"
		self.y = y
		self.error = 0.0
		if isinstance(ystar, Tensor):
			if requires_error:
				self.error = (-ystar.data * np.log(y.data)).sum()
			self.prime = -ystar.data * (1 / y.data)
		elif isinstance(ystar, np.ndarray):
			if requires_error:
				self.error = (-ystar * np.log(y.data)).sum()
			self.prime = -ystar * (1 / y.data)
		else:
			raise Exception(f"Exception @ {self}: ystar needs to be Tensor or ndarray, not {type(ystar)}")
