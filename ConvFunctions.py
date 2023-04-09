import numpy as np
from scipy import signal


def correlate(volumes, kernels, lib=np, siglib=signal, mode="valid", remove_input_depth_dimension=True):
    """Used to compute the cross-correlation between a set of images and a set of kernels assuming both have a batch size and the kernels are stacked in the 2nd dim."""
    assert len(volumes.shape) == len(kernels.shape) - 1  # note that the kernel has a batch size of 1 (it's assumed everywhere tensors have batch sizes)
    if mode == "valid":
        x = lib.empty((kernels.shape[1], volumes.shape[0],
                       *(volumes.shape[i] - kernels.shape[i + 1] + 1 for i in range(1, len(volumes.shape)))))
    elif mode == "full":
        x = lib.empty((kernels.shape[1], volumes.shape[0],
                       *(volumes.shape[i] + kernels.shape[i + 1] - 1 for i in range(1, len(volumes.shape)))))
    else:
        raise Exception(f"mode {mode} is not a valid mode. valid modes are 'valid' & 'full'")

    for i, k in enumerate(kernels[0]):
        x[i] = siglib.correlate(volumes, k.reshape((1, *k.shape)), mode)

    if remove_input_depth_dimension:
        x = x.reshape((*x.shape[:2], *x.shape[3:]))
    x = x.transpose((1, 0, *range(2, len(volumes.shape))))
    return x


def dl_dk_correlate(volumes, gradient, lib=np, siglib=signal, mode="valid"):
    """Used to compute the derivative of the kernel with respect to the loss: dL/dK"""
    # batch size needs to be the same
    assert len(volumes.shape) == len(gradient.shape) and volumes.shape[0] == gradient.shape[0]
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


'''def correlate_forward_looped(volumes, kernels, lib=np, siglib=signal, mode="valid",
                           remove_input_depth_dimension=True):  # less efficient (only potentially better with jit-compilation, thus kept)
    """volumes = images"""
    assert len(volumes.shape) == len(kernels.shape)
    if mode == "valid":
        x = lib.empty((volumes.shape[0],
                       *(volumes.shape[i] - kernels.shape[i] + 1 for i in range(1, len(volumes.shape)))))
    elif mode == "full":
        x = lib.empty((volumes.shape[0],
                       *(volumes.shape[i] + kernels.shape[i] - 1 for i in range(1, len(volumes.shape)))))
    else:
        raise Exception(f"mode {mode} is not a valid mode for cross-correlation (valid = 'valid' & 'full')")

    for i in range(volumes.shape[0]):
        for j in range(kernels.shape[0]):
            x[i, j] = siglib.correlate(volumes[i], kernels[j], mode)

    if remove_input_depth_dimension:
        x = x.reshape((*x.shape[:2], *x.shape[3:]))
    return x'''


def dl_dx_correlate(volumes, kernels, lib=np, siglib=signal, mode="full"):  # there is no convolve forward since it's not actually required
    """Used to compute the derivative of the input with respect to the loss: dL/dX"""
    assert len(volumes.shape) == len(kernels.shape) - 1
    if mode == "valid":
        x = lib.empty((volumes.shape[0], volumes.shape[1],
                       *(volumes.shape[i] - kernels.shape[i + 1] + 1 for i in range(2, len(volumes.shape)))))
    elif mode == "full":
        x = lib.empty((volumes.shape[0], volumes.shape[1],
                       *(volumes.shape[i] + kernels.shape[i + 1] - 1 for i in range(2, len(volumes.shape)))))
    else:
        raise Exception(f"mode {mode} is not a valid mode for cross-correlation (valid = 'valid' & 'full')")

    for i in range(volumes.shape[0]):  # for sample in batch
        for j in range(kernels.shape[2]):  # for layer in image_depth
            for k in range(kernels.shape[1]):  # for kernel in kernels (gradient.shape[1] == kernels.shape[0] == depth)
                x[i, j] += siglib.convolve(volumes[i, k], kernels[0, k, j], mode)

    return x


def pool(data, sizes, criterion, criterion_included, lib=np):
    """criterion needs to return a value of datatype data.dtype, criterion_included needs to return an iterable of indices and local gradients of values that influence the result.
    an example criterion_included function might be:

    def max_pooling_criterion_included(pooling_subarray: np.ndarray|cp.ndarray):
        lib = cp.get_array_module(pooling_subarray)  # numpy if pooling_subarray lies on cpu, cupy otherwise

        # since lib.argmax will return an int, np.unravel_index is better suited than cp.unravel_index (which would require a conversion from int to cupy.ndarray)
        idx = np.unravel_index(lib.argmax(pooling_subarray), pooling_subarray.shape)

        return ((idx, 1),)  # make it iterable
        # 1 because the local gradient inside the pooling function is 1 for max/min pooling (but not for average pooling)
    """
    out = lib.empty((data.shape[0], *(dim // sizes[i] for i, dim in enumerate(data.shape[1:]))))
    outgrad = lib.empty(data.shape)

    def _f(subarr):
        ret = lib.zeros(subarr.shape)
        for idx, influence in criterion_included(subarr):  # index contributing to result (thus has nonzero gradient)
            ret[idx] = influence
        return ret

    for i in range(data.shape[0]):  # for sample in batch
        '''subarrays = lib.split(data[i], data.shape[1] // sizes[0], 0)
        for axis in range(1, len(data.shape) - 2):
            new_subarrays = []
            for ary in subarrays:
                new_subarrays.extend(lib.split(ary.data, ary.shape[axis] // sizes[axis], axis))
            subarrays = new_subarrays
        res = lib.array([criterion(subarr) for subarr in subarrays])
        for subarr in subarrays:
            out_grad.append(_f(subarr))  # included.reshape(data.shape[2:])'''
        subarrays = lib.split(data[i].flatten(), np.prod(out.shape[1:]))
        res = lib.empty((len(subarrays),))
        resgrad = lib.empty((*res.shape, *subarrays[0].shape))
        for j, subarray in enumerate(subarrays):
            res[j] = criterion(subarray)
            resgrad[j] = _f(subarray)

        out[i] = res.reshape(out.shape[1:])
        outgrad[i] = resgrad.reshape(data.shape[1:])

    return out, outgrad


def averagepool(data, sizes, _, __, lib=np):
    subarray_size = np.prod(sizes)
    out = lib.empty((data.shape[0], *(dim // sizes[i] for i, dim in enumerate(data.shape[1:]))))
    outgrad = lib.empty(data.shape)

    for i in range(data.shape[0]):
        subarrays = lib.split(data[i].flatten(), np.prod(out.shape[1:]))
        res = lib.empty((len(subarrays),))
        resgrad = lib.empty((*res.shape, *subarrays[0].shape))
        for j, subarray in enumerate(subarrays):
            res[j] = lib.mean(subarray)
            resgrad[j] = lib.ones(subarray.shape) / subarray_size

        out[i] = res.reshape(out.shape[1:])
        outgrad[i] = resgrad.reshape(data.shape[1:])
    return out, outgrad
