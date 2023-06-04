import joblib
from trashgrad.autograd import Tensor


class GD:
	"""This is simply gradient descend without momentum"""
	def __init__(self, tensors, n_jobs: int = None):
		self.tensors = set()
		self.parallelizer = joblib.Parallel(n_jobs=n_jobs) if isinstance(n_jobs, int) else None
		for tensor in tensors:
			if tensor.requires_grad:
				self.tensors.add(tensor)

	@staticmethod
	def _step(tensor: Tensor, gradients: dict, lr: float) -> None:
		tensor.data -= tensor.lib.nanmean(lr * gradients[tensor], 0)

	def step(self, gradients: dict, lr: float) -> None:
		if self.parallelizer is None:
			for tensor in self.tensors:
				self._step(tensor, gradients, lr)
		else:
			self.parallelizer(joblib.delayed(self._step)(tensor, gradients, lr) for tensor in self.tensors)


class MomentumGD(GD):
	"""Gradient descend with momentum"""
	def __init__(self, tensors: list[Tensor], beta=.5):
		super().__init__(tensors)
		self.gradients = {tensor: 0. for tensor in tensors}
		self.beta = beta

	def _step(self, tensor: Tensor, gradients: dict, lr: float) -> None:
		self.gradients[tensor] = self.gradients[tensor] * self.beta + gradients[tensor] * (1 - self.beta)
		tensor.data -= tensor.lib.nanmean(lr * self.gradients[tensor], 0)


class RMSProp(GD):
	def __init__(self, tensors: list[Tensor], beta=.9, eps=1e-2):
		super().__init__(tensors)
		self.gradients = {tensor: 0. for tensor in tensors}
		self.beta = beta
		self.eps = eps

	def _step(self, tensor: Tensor, gradients: dict, lr: float) -> None:
		self.gradients[tensor] = self.gradients[tensor] * self.beta + gradients[tensor] ** 2 * (1 - self.beta)
		tensor.data -= tensor.lib.nanmean(lr * gradients[tensor] / (tensor.lib.sqrt(self.gradients[tensor]) + self.eps), 0)


class Adam(GD):
	def __init__(self, tensors: list[Tensor], beta1=.8, beta2=.8, eps=1e-2):
		super().__init__(tensors)
		self.beta1 = beta1
		self.beta2 = beta2
		self.beta1_pow_t = {tensor: beta1 for tensor in tensors}
		self.beta2_pow_t = {tensor: beta2 for tensor in tensors}
		self.M = {tensor: 0. for tensor in tensors}
		self.V = {tensor: 0. for tensor in tensors}
		self.eps = eps

	def _step(self, tensor: Tensor, gradients: dict, lr: float) -> None:
		self.M[tensor] = (self.M[tensor] * self.beta1 + (1 - self.beta1) * gradients[tensor]) / (1 - self.beta1_pow_t[tensor])
		self.V[tensor] = (self.V[tensor] * self.beta2 + (1 - self.beta2) * gradients[tensor] ** 2) / (1 - self.beta2_pow_t[tensor])
		tensor.data -= tensor.lib.nanmean(lr * self.M[tensor] / (tensor.lib.sqrt(self.V[tensor]) + self.eps), 0)
		self.beta1_pow_t[tensor] *= self.beta1
		self.beta2_pow_t[tensor] *= self.beta2
