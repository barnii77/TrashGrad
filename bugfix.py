import numpy as np
from scipy.signal import correlate


def vcorrelate(a, b):
    return correlate(a, b, "valid")


def mcorrelate(volumes, kernels):
    x = np.empty_like((kernels.shape[0], volumes.shape[0], volumes.shape[1] - kernels.shape[1] + 1, volumes.shape[2] - kernels.shape[2] + 1))
    for i, k in enumerate(kernels):
        x[i] = vcorrelate(volumes, k.reshape((1, *k.shape)))
    x = x.reshape((x.shape[:2], x.shape[3:]))
    x = np.transpose(x, (1, 0, *range(2, len(volumes.shape))))
    return x


volume = np.random.randn(3, 5, 5)
kernel = np.random.randn(3, 2, 2)
out = mcorrelate(volume, kernel)
print(out.shape)
