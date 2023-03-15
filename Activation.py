from Tensor import Tensor


def relu(x: Tensor):
    return x.relu()


def tanh(x: Tensor):
    return x.tanh()


def sigmoid(x: Tensor):
    return x.sigmoid()


def batchnorm(x: Tensor):
    return x.batchnorm()


def euclid_batchnorm(x: Tensor):
    return x.euclid_batchnorm()


def dropout(x: Tensor, rate=0.05, magnitude_correction=True):
    return x.dropout(rate, magnitude_correction)


def maxpool(x: Tensor, sizes):
    return x.maxpool(sizes)


def minxpool(x: Tensor, sizes):
    return x.minpool(sizes)


def averagepool(x: Tensor, sizes):
    return x.averagepool(sizes)
