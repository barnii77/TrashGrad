import numpy as np
from scipy import signal
from numba import jit, prange
import cupy as cp
import cupyx.scipy.signal as cpxsignal


def _correlate(volumes, kernels, lib, siglib, mode="valid", reduce_input_depth_dimension=True):
    if mode == "valid":
        x = lib.empty((kernels.shape[0], volumes.shape[0],
                       *(volumes.shape[i] - kernels.shape[i] + 1 for i in range(1, len(volumes.shape)))))
    elif mode == "full":
        x = lib.empty((kernels.shape[1], volumes.shape[0],
                       *(volumes.shape[i] + kernels.shape[i] - 1 for i in range(1, len(volumes.shape)))))
    else:
        raise Exception(f"mode {mode} is not a valid mode. valid modes are 'valid' & 'full'")

    for i in range(kernels.shape[0]):
        k = kernels[i]
        x[i] = siglib.correlate(volumes, k.reshape((1, *k.shape)), mode)

    if reduce_input_depth_dimension:
        x = x.reshape((x.shape[0], x.shape[1] * x.shape[2], *x.shape[3:]))
    x = x.transpose((1, 0, *range(2, len(volumes.shape))))
    return x


@jit(target_backend="cuda")
def cuda_correlate(volumes, kernels, lib, siglib, mode="valid", reduce_input_depth_dimension=True):
    if mode == "valid":
        x = lib.empty((kernels.shape[0], volumes.shape[0],
                       *(volumes.shape[i] - kernels.shape[i] + 1 for i in range(1, len(volumes.shape)))))
    elif mode == "full":
        x = lib.empty((kernels.shape[1], volumes.shape[0],
                       *(volumes.shape[i] + kernels.shape[i] - 1 for i in range(1, len(volumes.shape)))))
    else:
        raise Exception(f"mode {mode} is not a valid mode. valid modes are 'valid' & 'full'")

    for i in prange(kernels.shape[
                        0]):  # kernels is going to have shape (1, num_kernels, image_shape); image_shape = (num_channels, width, height)
        k = kernels[i]
        x[i] = siglib.correlate(volumes, k.reshape((1, *k.shape)), mode)

    if reduce_input_depth_dimension:
        x = x.reshape((x.shape[0], x.shape[1] * x.shape[2], *x.shape[3:]))
    x = x.transpose((1, 0, *range(2, len(volumes.shape))))
    return x


@jit(target_backend="cpu")
def cpu_correlate(volumes, kernels, lib, siglib, mode="valid", reduce_input_depth_dimension=True):
    if mode == "valid":
        x = lib.empty((kernels.shape[0], volumes.shape[0],
                       *(volumes.shape[i] - kernels.shape[i] + 1 for i in range(1, len(volumes.shape)))))
    elif mode == "full":
        x = lib.empty((kernels.shape[1], volumes.shape[0],
                       *(volumes.shape[i] + kernels.shape[i] - 1 for i in range(1, len(volumes.shape)))))
    else:
        raise Exception(f"mode {mode} is not a valid mode. valid modes are 'valid' & 'full'")

    for i in prange(kernels.shape[0]):
        k = kernels[i]
        x[i] = siglib.correlate(volumes, k.reshape((1, *k.shape)), mode)

    if reduce_input_depth_dimension:
        x = x.reshape((x.shape[0], x.shape[1] * x.shape[2], *x.shape[3:]))
    x = x.transpose((1, 0, *range(2, len(volumes.shape))))
    return x


def correlate(volumes, kernels, lib=None, siglib=None, backend=None,
              reduce_input_depth_dimension=True):
    """Used to compute the cross-correlation between a set of images and a set of kernels assuming both have a batch size and the kernels are stacked in the 2nd dim."""
    if len(kernels.shape) - 1 == len(volumes.shape) and kernels.shape[0] == 1:
        kernels = kernels[0]
    elif len(kernels.shape) != len(volumes.shape):
        raise Exception(f"Number of kernel dimensions {len(kernels.shape)} does not match number of volume dimensions {len(volumes.shape)}")
    if backend is None:
        if lib is None:
            lib = cp.get_array_module(volumes)
        if siglib is None:
            siglib = {np: signal, cp: cpxsignal}[lib]
        return _correlate(volumes, kernels, lib, siglib, reduce_input_depth_dimension=reduce_input_depth_dimension)
    elif backend == "cpu":
        if lib is None:
            lib = np
        if siglib is None:
            siglib = signal
        return cpu_correlate(volumes, kernels, lib, siglib, "valid", reduce_input_depth_dimension)
    elif backend == "cuda":
        if lib is None:
            lib = cp
        if siglib is None:
            siglib = cpxsignal
        return cuda_correlate(volumes, kernels, lib, siglib, "valid", reduce_input_depth_dimension)
    else:
        raise Exception(f"{backend} is not a valid backend. Valid backends are 'cpu' & 'cuda'")


def _dl_dk_correlate(volumes, gradient, lib, siglib, mode="valid"):
    """Used to compute the derivative of the kernel with respect to the loss: dL/dK"""
    # batch size needs to be the same
    if mode == "valid":
        x = lib.empty((volumes.shape[0], gradient.shape[1], volumes.shape[1],
                       *(volumes.shape[i] - gradient.shape[i] + 1 for i in range(2, len(volumes.shape)))))
    elif mode == "full":
        x = lib.empty((volumes.shape[0], gradient.shape[1], volumes.shape[1],
                       *(volumes.shape[i] + gradient.shape[i] - 1 for i in range(2, len(volumes.shape)))))
    else:
        raise Exception(f"mode {mode} is not a valid mode for cross-correlation (valid = 'valid' & 'full')")

    for j in range(gradient.shape[1]):  # for kernel_gradient in gradients (kernel outputs get stacked)
        for i in range(volumes.shape[0]):  # for sample in batch
            g = gradient[i, j]
            x[i, j] = siglib.correlate(volumes[i], g.reshape((1, *g.shape)), mode)

    return x


@jit(target_backend="cpu")
def cpu_dl_dk_correlate(volumes, gradient, lib, siglib, mode="valid"):
    """Used to compute the derivative of the kernel with respect to the loss: dL/dK"""
    # batch size needs to be the same
    if mode == "valid":
        x = lib.empty((volumes.shape[0], gradient.shape[1], volumes.shape[1],
                       *(volumes.shape[i] - gradient.shape[i] + 1 for i in range(2, len(volumes.shape)))))
    elif mode == "full":
        x = lib.empty((volumes.shape[0], gradient.shape[1], volumes.shape[1],
                       *(volumes.shape[i] + gradient.shape[i] - 1 for i in range(2, len(volumes.shape)))))
    else:
        raise Exception(f"mode {mode} is not a valid mode for cross-correlation (valid = 'valid' & 'full')")

    for j in prange(gradient.shape[1]):  # for kernel_gradient in gradients (kernel outputs get stacked)
        for i in prange(volumes.shape[0]):  # for sample in batch
            g = gradient[i, j]
            x[i, j] = siglib.correlate(volumes[i], g.reshape((1, *g.shape)), mode)

    return x


@jit(target_backend="cuda")
def cuda_dl_dk_correlate(volumes, gradient, lib, siglib, mode="valid"):
    """Used to compute the derivative of the kernel with respect to the loss: dL/dK"""
    # batch size needs to be the same
    if mode == "valid":
        x = lib.empty((volumes.shape[0], gradient.shape[1], volumes.shape[1],
                       *(volumes.shape[i] - gradient.shape[i] + 1 for i in range(2, len(volumes.shape)))))
    elif mode == "full":
        x = lib.empty((volumes.shape[0], gradient.shape[1], volumes.shape[1],
                       *(volumes.shape[i] + gradient.shape[i] - 1 for i in range(2, len(volumes.shape)))))
    else:
        raise Exception(f"mode {mode} is not a valid mode for cross-correlation (valid = 'valid' & 'full')")

    for j in prange(gradient.shape[1]):  # for kernel_gradient in gradients (kernel outputs get stacked)
        for i in prange(volumes.shape[0]):  # for sample in batch
            g = gradient[i, j]
            x[i, j] = siglib.correlate(volumes[i], g.reshape((1, *g.shape)), mode)

    return x


def dl_dk_correlate(volumes, gradient, lib=None, siglib=None, backend=None):
    """Used to compute the derivative of the kernel with respect to the loss: dL/dK"""
    # batch size needs to be the same
    assert len(volumes.shape) == len(gradient.shape) and volumes.shape[0] == gradient.shape[0]
    if backend is None:
        if lib is None:
            lib = cp.get_array_module(volumes)
        if siglib is None:
            siglib = {np: signal, cp: cpxsignal}[lib]
        return _dl_dk_correlate(volumes, gradient, lib, siglib)
    elif backend == "cpu":
        if lib is None:
            lib = np
        if siglib is None:
            siglib = signal
        return cpu_dl_dk_correlate(volumes, gradient, lib, siglib, "valid")
    elif backend == "cuda":
        if lib is None:
            lib = cp
        if siglib is None:
            siglib = cpxsignal
        return cuda_dl_dk_correlate(volumes, gradient, lib, siglib, "valid")
    else:
        raise Exception(f"{backend} is not a valid backend. Valid backends are 'cpu' & 'cuda'")


def _dl_dx_correlate(gradient, kernels, lib, siglib,
                     mode="full"):  # there is no convolve forward since it's not actually required
    """Used to compute the derivative of the input with respect to the loss: dL/dX"""
    if mode == "valid":
        x = lib.empty((gradient.shape[0], gradient.shape[1],
                       *(gradient.shape[i] - kernels.shape[i] + 1 for i in range(2, len(gradient.shape)))))
    elif mode == "full":
        x = lib.empty((gradient.shape[0], kernels.shape[1],
                       *(gradient.shape[i] + kernels.shape[i] - 1 for i in range(2, len(gradient.shape)))))
    else:
        raise Exception(f"mode {mode} is not a valid mode for cross-correlation (valid = 'valid' & 'full')")

    for i in range(gradient.shape[0]):  # for sample in batch
        for j in range(kernels.shape[1]):  # for layer in image_depth
            for k in range(kernels.shape[0]):  # for kernel in kernels (gradient.shape[1] == kernels.shape[0] == depth)
                x[i, k] += siglib.convolve(gradient[i, k], kernels[k, j], mode)

    return x


@jit(target_backend="cpu")
def cpu_dl_dx_correlate(gradient, kernels, lib, siglib,
                        mode="full"):  # there is no convolve forward since it's not actually required
    """Used to compute the derivative of the input with respect to the loss: dL/dX"""
    if mode == "valid":
        x = lib.empty((gradient.shape[0], gradient.shape[1],
                       *(gradient.shape[i] - kernels.shape[i] + 1 for i in range(2, len(gradient.shape)))))
    elif mode == "full":
        x = lib.empty((gradient.shape[0], gradient.shape[1],
                       *(gradient.shape[i] + kernels.shape[i] - 1 for i in range(2, len(gradient.shape)))))
    else:
        raise Exception(f"mode {mode} is not a valid mode for cross-correlation (valid = 'valid' & 'full')")

    for i in prange(gradient.shape[0]):  # for sample in batch
        for j in prange(kernels.shape[1]):  # for layer in image_depth
            for k in prange(kernels.shape[0]):  # for kernel in kernels (gradient.shape[1] == kernels.shape[0] == depth)
                x[i, j] += siglib.convolve(gradient[i, k], kernels[k, j], mode)

    return x


@jit(target_backend="cuda")
def cuda_dl_dx_correlate(gradient, kernels, lib, siglib,
                         mode="full"):  # there is no convolve forward since it's not actually required
    """Used to compute the derivative of the input with respect to the loss: dL/dX"""
    if mode == "valid":
        x = lib.empty((gradient.shape[0], gradient.shape[1],
                       *(gradient.shape[i] - kernels.shape[i] + 1 for i in range(2, len(gradient.shape)))))
    elif mode == "full":
        x = lib.empty((gradient.shape[0], gradient.shape[1],
                       *(gradient.shape[i] + kernels.shape[i] - 1 for i in range(2, len(gradient.shape)))))
    else:
        raise Exception(f"mode {mode} is not a valid mode for cross-correlation (valid = 'valid' & 'full')")

    for i in prange(gradient.shape[0]):  # for sample in batch
        for j in prange(kernels.shape[1]):  # for layer in image_depth
            for k in prange(kernels.shape[0]):  # for kernel in kernels (gradient.shape[1] == kernels.shape[0] == depth)
                x[i, j] += siglib.convolve(gradient[i, k], kernels[k, j], mode)

    return x


# todo: remove mode for dl_d..._correlate functions
def dl_dx_correlate(gradient, kernels, lib=None, siglib=None,
                    backend=None):
    """Used to compute the derivative of the input with respect to the loss: dL/dX"""
    if len(kernels.shape) - 1 == len(gradient.shape) and kernels.shape[0] == 1:
        kernels = kernels[0]
    elif len(kernels.shape) != len(gradient.shape):
        raise Exception(f"Number of kernel dimensions {len(kernels.shape)} does not match number of gradient dimensions {len(gradient.shape)}")
    if backend is None:
        if lib is None:
            lib = cp.get_array_module(gradient)
        if siglib is None:
            siglib = {np: signal, cp: cpxsignal}[lib]
        return _dl_dx_correlate(gradient, kernels, lib, siglib)
    elif backend == "cpu":
        if lib is None:
            lib = np
        if siglib is None:
            siglib = signal
        return cpu_dl_dx_correlate(gradient, kernels, lib, siglib, "full")
    elif backend == "cuda":
        if lib is None:
            lib = cp
        if siglib is None:
            lib = cpxsignal
        return cuda_dl_dx_correlate(gradient, kernels, lib, siglib, "full")
    else:
        raise Exception(f"{backend} is not a valid backend. Valid backends are 'cpu' & 'cuda'")


def _abstract_pool_grad_processing(subarr, criterion_included, lib):
    ret = lib.zeros(subarr.shape)
    crit_included = criterion_included(subarr)
    for i in range(len(crit_included)):  # index contributing to result (thus has nonzero gradient)
        idx, influence = crit_included[i]
        ret[idx] = influence
    return ret


@jit(target_backend="cpu")
def _cpu_abstract_pool_grad_processing(subarr, criterion_included, lib):
    ret = lib.zeros(subarr.shape)
    crit_included = criterion_included(subarr)
    for i in prange(len(crit_included)):  # index contributing to result (thus has nonzero gradient)
        idx, influence = crit_included[i]
        ret[idx] = influence
    return ret


@jit(target_backend="cuda")
def _cuda_abstract_pool_grad_processing(subarr, criterion_included, lib):
    ret = lib.zeros(subarr.shape)
    crit_included = criterion_included(subarr)
    for i in prange(len(crit_included)):  # index contributing to result (thus has nonzero gradient)
        idx, influence = crit_included[i]
        ret[idx] = influence
    return ret


def _pool(data, sizes, criterion, criterion_included, lib):
    out = lib.empty((data.shape[0], *(data.shape[i + 1] // sizes[i] for i in range(len(data.shape) - 1))))
    outgrad = lib.empty(data.shape)
    subarrays = data.reshape((data.shape[0], np.prod(out.shape[1:]), -1))

    for i in range(data.shape[0]):  # for sample in batch
        subarrays_i = subarrays[i]
        res = lib.empty(subarrays_i.shape[0:1])
        resgrad = lib.empty((*res.shape, *subarrays_i[0].shape))
        for j in range(subarrays_i.shape[0]):
            subarray = subarrays_i[j]
            res[j] = criterion(subarray)
            resgrad[j] = _abstract_pool_grad_processing(subarray, criterion_included, lib)

        out[i] = res.reshape(out.shape[1:])
        outgrad[i] = resgrad.reshape(data.shape[1:])

    return out, outgrad


@jit(target_backend="cpu")
def cpu_pool(data, sizes, criterion, criterion_included, lib):
    out = lib.empty((data.shape[0], *(data.shape[i + 1] // sizes[i] for i in range(len(data.shape) - 1))))
    outgrad = lib.empty(data.shape)
    subarrays = data.reshape((data.shape[0], np.prod(out.shape[1:]), -1))

    for i in prange(data.shape[0]):  # for sample in batch
        subarrays_i = subarrays[i]
        res = lib.empty(subarrays_i.shape[0:1])
        resgrad = lib.empty((*res.shape, *subarrays_i[0].shape))
        for j in prange(subarrays_i.shape[0]):
            subarray = subarrays_i[j]
            res[j] = criterion(subarray)
            resgrad[j] = _cpu_abstract_pool_grad_processing(subarray, criterion_included, lib)

        out[i] = res.reshape(out.shape[1:])
        outgrad[i] = resgrad.reshape(data.shape[1:])

    return out, outgrad


@jit(target_backend="cuda")
def cuda_pool(data, sizes, criterion, criterion_included, lib):
    out = lib.empty((data.shape[0], *(data.shape[i + 1] // sizes[i] for i in range(len(data.shape) - 1))))
    outgrad = lib.empty(data.shape)
    subarrays = data.reshape((data.shape[0], np.prod(out.shape[1:]), -1))

    for i in prange(data.shape[0]):  # for sample in batch
        subarrays_i = subarrays[i]
        res = lib.empty(subarrays_i.shape[0:1])
        resgrad = lib.empty((*res.shape, *subarrays_i[0].shape))
        for j in prange(subarrays_i.shape[0]):
            subarray = subarrays_i[j]
            res[j] = criterion(subarray)
            resgrad[j] = _cuda_abstract_pool_grad_processing(subarray, criterion_included, lib)

        out[i] = res.reshape(out.shape[1:])
        outgrad[i] = resgrad.reshape(data.shape[1:])

    return out, outgrad


def pool(data, sizes, criterion, criterion_included, lib=None, backend=None):
    """criterion needs to return a value of datatype data.dtype, criterion_included needs to return an iterable of indices and local gradients of values that influence the result.
    an example criterion_included function might be:

    def max_pooling_criterion_included(pooling_subarray: np.ndarray|cp.ndarray):
        lib = cp.get_array_module(pooling_subarray)  # numpy if pooling_subarray lies on cpu, cupy otherwise

        # since lib.argmax will return an int, np.unravel_index is better suited than cp.unravel_index (which would require a conversion from int to cupy.ndarray)
        idx = np.unravel_index(lib.argmax(pooling_subarray), pooling_subarray.shape)

        return ((idx, 1),)  # make it iterable; Note: you could also return slices and arrays instead
        # 1 because the local gradient inside the pooling function is 1 for max/min pooling (but not for average pooling)
    """
    if backend is None:
        if backend is None:
            if lib is None:
                lib = cp.get_array_module(data)
            return _pool(data, sizes, criterion, criterion_included, lib)
    elif backend == "cpu":
        if lib is None:
            lib = np
        return cpu_pool(data, sizes, criterion, criterion_included, lib)
    elif backend == "cuda":
        if lib is None:
            lib = cp
        return cuda_pool(data, sizes, criterion, criterion_included, lib)
    else:
        raise Exception(f"{backend} is not a valid backend. Valid backends are 'cpu' & 'cuda'")


def _averagepool(data, sizes, _, __, lib):
    subarray_size = np.prod(sizes)
    out = lib.empty((data.shape[0], *(data.shape[i + 1] // sizes[i] for i in range(len(data.shape) - 1))))
    outgrad = lib.empty(data.shape)
    subarrays = data.reshape((data.shape[0], np.prod(out.shape[1:]), -1))

    for i in range(data.shape[0]):
        subarrays_i = subarrays[i]
        res = lib.empty(subarrays_i.shape[0:1])
        resgrad = lib.empty((*res.shape, *subarrays_i[0].shape))
        for j in range(subarrays_i.shape[0]):
            subarray = subarrays_i[j]
            res[j] = lib.mean(subarray)
            resgrad[j] = lib.ones(subarray.shape) / subarray_size

        out[i] = res.reshape(out.shape[1:])
        outgrad[i] = resgrad.reshape(data.shape[1:])
    return out, outgrad


@jit(target_backend="cpu")
def cpu_averagepool(data, sizes, _, __, lib):
    subarray_size = np.prod(sizes)
    out = lib.empty((data.shape[0], *(data.shape[i + 1] // sizes[i] for i in range(len(data.shape) - 1))))
    outgrad = lib.empty(data.shape)
    subarrays = data.reshape((data.shape[0], np.prod(out.shape[1:]), -1))

    for i in prange(data.shape[0]):
        subarrays_i = subarrays[i]
        res = lib.empty(subarrays_i.shape[0:1])
        resgrad = lib.empty((*res.shape, *subarrays_i[0].shape))
        for j in prange(subarrays_i.shape[0]):
            subarray = subarrays_i[j]
            res[j] = lib.mean(subarray)
            resgrad[j] = lib.ones(subarray.shape) / subarray_size

        out[i] = res.reshape(out.shape[1:])
        outgrad[i] = resgrad.reshape(data.shape[1:])
    return out, outgrad


@jit(target_backend="cuda")
def cuda_averagepool(data, sizes, _, __, lib):
    subarray_size = np.prod(sizes)
    out = lib.empty((data.shape[0], *(data.shape[i + 1] // sizes[i] for i in range(len(data.shape) - 1))))
    outgrad = lib.empty(data.shape)
    subarrays = data.reshape((data.shape[0], np.prod(out.shape[1:]), -1))

    for i in prange(data.shape[0]):
        subarrays_i = subarrays[i]
        res = lib.empty(subarrays_i.shape[0:1])
        resgrad = lib.empty((*res.shape, *subarrays_i[0].shape))
        for j in prange(subarrays_i.shape[0]):
            subarray = subarrays_i[j]
            res[j] = lib.mean(subarray)
            resgrad[j] = lib.ones(subarray.shape) / subarray_size

        out[i] = res.reshape(out.shape[1:])
        outgrad[i] = resgrad.reshape(data.shape[1:])
    return out, outgrad


def averagepool(data, sizes, _=None, __=None, lib=None, backend=None):  # todo: implement cpu and cuda jit here
    if backend is None:
        if backend is None:
            if lib is None:
                lib = cp.get_array_module(data)
            return _averagepool(data, sizes, None, None, lib=lib)
    elif backend == "cpu":
        if lib is None:
            lib = np
        return cpu_averagepool(data, sizes, None, None, lib=lib)
    elif backend == "cuda":
        if lib is None:
            lib = cp
        return cuda_averagepool(data, sizes, None, None, lib=lib)
    else:
        raise Exception(f"{backend} is not a valid backend")
