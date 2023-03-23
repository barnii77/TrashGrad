import torch


'''x = torch.tensor([1., 2., 3., 4., 5., 6., 7., 8., 9.], requires_grad=True).resize(1, 1, 9)
#ystar = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0], requires_grad=False)
kernels = torch.randn(1, 1, 1, requires_grad=True)
bias = torch.randn(1, 1, 9, requires_grad=True)
optim = torch.optim.SGD([kernels, bias], 0.01)

y = torch.conv1d(x, kernels, None) + bias
y.backward(y)
print(kernels.grad)
'''
x = [torch.tensor([0.0, 0.0]), torch.tensor([0.0, 1.0]), torch.tensor([1.0, 0.0]), torch.tensor([1.0, 1.0])]
y = [torch.tensor([0]), torch.tensor([1]), torch.tensor([1]), torch.tensor([0])]
w1 = torch.randn(3, 2, requires_grad=True)
b1 = torch.randn(3, requires_grad=True)
w2 = torch.randn(1, 3, requires_grad=True)
b2 = torch.randn(1, requires_grad=True)
optim = torch.optim.SGD([w1, b1, w2, b2], 1e-02)

for i in range(10):
    for sample, answer in zip(x, y):
        out = torch.matmul(w1, sample) + b1
        out = torch.tanh(out)
        out = torch.matmul(w2, out) + b2
        out = torch.sigmoid(out)
        loss = out - answer
        loss.backward(1e-02)
        optim.step()

print(out)
