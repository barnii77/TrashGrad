import torch


x = torch.tensor([1., 2., 3., 4., 5., 6., 7., 8., 9.], requires_grad=True).resize(1, 1, 9)
#ystar = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0], requires_grad=False)
kernels = torch.randn(1, 1, 1, requires_grad=True)
bias = torch.randn(1, 1, 9, requires_grad=True)
optim = torch.optim.SGD([kernels, bias], 0.01)

y = torch.conv1d(x, kernels, None) + bias
y.backward(y)
print(kernels.grad)
