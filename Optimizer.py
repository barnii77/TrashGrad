class SGD:
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
			self.tensor.data -= lr * self.gradient[x]


class Adam:
	def __init__(self, tensor, beta1=.9, beta2=.9):
		self.tensor = tensor
		self.beta1 = beta1
		self.beta2 = beta2
		self.tempbeta1 = beta1
		self.tempbeta2 = beta2
		self.m = 0
		self.v = 0
	
	def update(self, gradient, x):
		if self.tensor.requires_grad:
			self.m = (self.m * self.beta1 + (1 - self.beta1) * gradient) / (1 - self.tempbeta1)
			self.v = (self.v * self.beta2 + (1 - self.beta2) * gradient ** 2) / (1 - self.tempbeta2)
			self.tempbeta1 *= self.beta1
			self.tempbeta2 *= self.beta2

	def step(self, lr, x):
		self.tensor.data -= lr * self.m / (self.tensor.lib.sqrt(self.v) + .01)

	def zero_grad(self, x):
		pass  # Adam zero_grad doesnt make sense (the function has to be there though)

	def reset(self):
		self.tempbeta1 = self.beta1
		self.tempbeta2 = self.beta2
		self.m = 0
		self.v = 0
