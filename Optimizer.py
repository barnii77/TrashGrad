import numpy as np


class Default:
	def __init__(self, tensor):
		self.gradient = {}
		self.tensor = tensor
	
	def zero_grad(self, x=None):
		if x is None:
			self.gradient = {}
		else:
			self.gradient[x] = 0
	
	def update(self, gradient, x):
		if self.tensor.requires_grad:
			if self.gradient.get(x) is None:
				self.gradient[x] = 0
			self.gradient[x] += gradient
	
	def step(self, lr, x):
		if self.tensor.requires_grad:
			self.tensor.data -= lr * self.gradient


class Adam:
	def __init__(self, tensor, beta1=.9, beta2=.9):
		self.tensor = tensor
		self.beta1 = beta1
		self.beta2 = beta2
		self.tempbeta1s = {}
		self.tempbeta2s = {}
		self.M = {}
		self.V = {}
	
	def update(self, gradient, x):
		if self.tensor.requires_grad:
			if self.M.get(x) is None:
				self.M[x] = 0
			if self.V.get(x) is None:
				self.V[x] = 0
			if self.tempbeta1s.get(x) is None:
				self.tempbeta1s[x] = self.beta1
			if self.tempbeta2s.get(x) is None:
				self.tempbeta2s[x] = self.beta2
			self.M[x] = (self.M[x] * self.beta1 + (1 - self.beta1) * gradient) / (1 - self.tempbeta1s[x])
			self.V[x] = (self.V[x] * self.beta2 + (1 - self.beta2) * gradient ** 2) / (1 - self.tempbeta2s[x])
			self.tempbeta1s[x] *= self.beta1
			self.tempbeta2s[x] *= self.beta2

	def step(self, lr, x):
		self.tensor.data -= lr * self.M[x] / (np.sqrt(self.V[x]) + .01)

	def zero_grad(self, x):
		self.tempbeta1s[x] = self.beta1
		self.tempbeta2s[x] = self.beta2
		self.M[x] = 0
		self.V[x] = 0
		# you shouldn't typically zero_grad using Adam
