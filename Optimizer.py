import numpy as np


class Default:
	def __init__(self, tensor):
		self.gradient = 0
		self.tensor = tensor
	
	def zero_grad(self):
		self.gradient = 0
		for parent in self.tensor.parents:
			parent.zero_grad()
	
	def update(self, gradient):
		if self.tensor.trainable:
			self.gradient += gradient
	
	def step(self, lr):
		if self.tensor.trainable:
			self.tensor.data -= lr * self.gradient
	
	def autostep(self, lr):
		self.step(lr)
		for parent in self.tensor.parents:
			parent.optimize(lr)


class Adam:
	def __init__(self, tensor, beta1=.9, beta2=.9):
		self.tensor = tensor
		self.beta1 = beta1
		self.beta2 = beta2
		self.tempbeta1 = beta1
		self.tempbeta2 = beta2
		self.m = 0
		self.v = 0
	
	def update(self, gradient):
		if self.tensor.trainable:
			self.m = (self.m * self.beta1 + (1 - self.beta1) * gradient) / (1 - self.tempbeta1)
			self.v = (self.v * self.beta2 + (1 - self.beta2) * gradient ** 2) / (1 - self.tempbeta2)

	def step(self, lr):
		self.tempbeta1 *= self.beta1
		self.tempbeta2 *= self.beta2
		self.tensor.data -= lr * self.m / (np.sqrt(self.v) + .01)

	def zero_grad(self):
		self.tempbeta1 = self.beta1
		self.tempbeta2 = self.beta2
		self.m = 0
		self.v = 0
		for parent in self.tensor.parents:
			parent.zero_grad()
		# you shouldn't typically zero_grad using Adam

	def autostep(self, lr):
		self.step(lr)
		for parent in self.tensor.parents:
			parent.optimize(lr)
