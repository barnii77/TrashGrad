import math

from Tensor import Tensor


class Loss:
	y: Tensor
	value: Tensor
	name: str

	def backward(self, max_depth=math.inf):
		self.value.backward(max_depth)


class MSE(Loss):
	def __init__(self, y: Tensor, ystar: Tensor):  # super.__init__() doesnt do anything
		self.name = "Mean Squared Error"
		self.value = ((y - ystar) ** 2).sum()


class CrossEntropyLoss(Loss):
	def __init__(self, y: Tensor, ystar):
		self.name = "Cross Entropy Loss"
		self.value = (-ystar * y.log()).sum()
