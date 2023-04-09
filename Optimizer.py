import warnings


class GD:
	"""This is simply gradient descend without momentum"""
	def __init__(self, tensors):
		self.tensors = set()
		for tensor in tensors:
			if tensor.trainable:
				self.tensors.add(tensor)

	@staticmethod
	def _step(gradient, tensor, lr):
		tensor.data -= lr * gradient

	def step(self, gradients, lr, average_over_batches=True):
		for tensor in self.tensors:
			if tensor.requires_grad:
				if tensor.data.shape[0] == 1:
					if average_over_batches:
						self._step(gradients[tensor].sum(0, keepdims=True) / gradients[tensor].shape[0], tensor, lr)
					else:
						self._step(gradients[tensor].sum(0, keepdims=True), tensor, lr)
				elif tensor.data.shape[0] == gradients[tensor].shape[0]:
					self._step(gradients[tensor], tensor, lr)
					warnings.warn(f"Optimizing tensor '{tensor}', but it has batch size != 1")
				else:
					raise Exception(
						"Tensor and gradients need to have either same batch size, or Tensor batch size = 1")


class SGD(GD):
	"""Gradient descend with momentum"""
	def __init__(self, tensors, beta=.5):
		super().__init__(tensors)
		self.gradients = {tensor: 0. for tensor in tensors}
		self.beta = beta

	def _step(self, gradient, tensor, lr):
		self.gradients[tensor] = self.gradients[tensor] * self.beta + gradient * (1 - self.beta)
		tensor.data -= lr * self.gradients[tensor]


class Adam(GD):
	def __init__(self, tensors, beta1=.5, beta2=.5):
		super().__init__(tensors)
		self.beta1 = beta1
		self.beta2 = beta2
		self.beta1_pow_t = {tensor: beta1 for tensor in tensors}
		self.beta2_pow_t = {tensor: beta2 for tensor in tensors}
		self.M = {tensor: 0 for tensor in tensors}
		self.V = {tensor: 0 for tensor in tensors}

	def _step(self, gradient, tensor, lr):
		self.M[tensor] = (self.M[tensor] * self.beta1 + (1 - self.beta1) * gradient) / (1 - self.beta1_pow_t[tensor])
		self.V[tensor] = (self.V[tensor] * self.beta2 + (1 - self.beta2) * gradient ** 2) / (1 - self.beta2_pow_t[tensor])
		tensor.data -= lr * self.M[tensor] / (tensor.lib.sqrt(self.V[tensor]) + .01)
		self.beta1_pow_t[tensor] *= self.beta1
		self.beta2_pow_t[tensor] *= self.beta2
