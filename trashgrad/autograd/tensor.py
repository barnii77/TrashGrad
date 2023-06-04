import math
import joblib
import numpy as np
import cupy as cp
import cupyx.scipy.signal as cpxsignal
from scipy import signal
import trashgrad.autograd as ag


class Tensor:
    def __init__(self,
                 data,
                 upstreams=tuple(),
                 requires_grad=False,
                 name="Tensor",
                 ops_until_forget=None,
                 base_ops_until_forget=math.inf,
                 mode="cpu",
                 _requires_grad=False,
                 backward_parallel_jobs=None):
        if mode == "gpu" and cp.get_array_module(data) is np:
            data = cp.asarray(data)
        elif mode == "cpu" and cp.get_array_module(data) is cp:
            data = cp.asnumpy(data)

        self.requires_grad: bool = requires_grad
        self._requires_grad: bool = requires_grad or _requires_grad
        self.data = data
        self.upstreams: tuple[Tensor] = upstreams
        self.functions = {}
        if backward_parallel_jobs is not None:
            assert isinstance(backward_parallel_jobs, int)
            self.backward_parallelizer: joblib.Parallel = joblib.Parallel(n_jobs=backward_parallel_jobs)
        else:
            self.backward_parallelizer = None

        self.name = name
        if ops_until_forget is None:
            self.ops_until_forget = base_ops_until_forget
        else:
            self.ops_until_forget = ops_until_forget
        self.base_ops_until_forget = base_ops_until_forget
        self.mode = mode
        self.shape: tuple[int, ...] = self.data.shape  # the data never changes, although it may get deleted for mem efficiency. However, self.data.shape is a tuple, thus immutable, so the deletion doesn't cause issues
        self.size: int = self.data.size
        self.dtype = self.data.dtype
        self.ndim: int = len(self.shape)

        if mode == "gpu":
            self.lib = cp
            self.siglib = cpxsignal
        elif mode == "cpu":
            self.lib = np
            self.siglib = signal
        else:
            raise Exception(f"Mode must be 'cpu' or 'gpu', not {mode}")

    # TODO: investigate correctness
    '''def delete(self, delete_graph=False):
        """delete_graph = True: Will delete everything that references this Tensor object (children, itself and parents)
        delete_graph = False: Will delete all references to this Tensor object so it's picked up by the garbage collector"""
        for upstream in self.upstreams:
            upstream.transforms.pop(self)
            if delete_graph:
                upstream.delete(delete_graph)

        for downstream in self.transforms.keys():
            downstream.upstreams = tuple(filter(lambda x: x != self, downstream.upstreams))
            if delete_graph:
                downstream.delete(delete_graph)'''

    def set_backward_parallel_jobs(self, n_jobs: int):
        self.backward_parallelizer = joblib.Parallel(n_jobs=n_jobs)

    def astype(self, dtype):
        return ag.astype(self, dtype)

    def _forget(self):
        del self.data

    def forget(self):
        if self.ops_until_forget <= 0:
            self._forget()

    def unforgettable(self):
        self.ops_until_forget = math.inf
        return self

    def forgettable(self):
        self.ops_until_forget = self.base_ops_until_forget
        return self

    def cpu(self):
        if self.mode == "gpu":
            x = Tensor(cp.asnumpy(self.data),
                       (self,),
                       _requires_grad=self._requires_grad,
                       ops_until_forget=self.base_ops_until_forget,
                       base_ops_until_forget=self.base_ops_until_forget,
                       mode="cpu")

            self._add_function(lambda grad: cp.asarray(grad), x)
        else:
            x = self

        return x

    def gpu(self, device=None):
        if self.mode == "cpu" or device is not None:
            if device is not None:
                with cp.cuda.Device(device):
                    x = Tensor(cp.asarray(self.data),
                               (self,),
                               _requires_grad=self._requires_grad,
                               ops_until_forget=self.base_ops_until_forget,
                               base_ops_until_forget=self.base_ops_until_forget,
                               mode="gpu")
            else:
                x = Tensor(cp.asarray(self.data),
                           (self,),
                           _requires_grad=self._requires_grad,
                           ops_until_forget=self.base_ops_until_forget,
                           base_ops_until_forget=self.base_ops_until_forget,
                           mode="gpu")

            self._add_function(lambda grad: cp.asnumpy(grad), x)
        else:
            x = self

        return x

    def contiguous(self):
        self.data = self.lib.ascontiguousarray(self.data)
        return self

    def __mul__(self, other):
        return ag.mul(self, other)

    def __matmul__(self, other):
        return ag.matmul(self, other)

    def __add__(self, other):
        return ag.add(self, other)

    def __sub__(self, other):
        return ag.sub(self, other)

    def __pow__(self, power, modulo=None):
        return ag.pow(self, power)

    def __neg__(self):
        return ag.neg(self)

    def __truediv__(self, other):
        return ag.div(self, other)

    def __rtruediv__(self, other):
        return ag.div(other, self)

    def __abs__(self):
        return ag.abs(self)

    def __getitem__(self, item):
        return ag.get(self, item)

    def __setitem__(self, key, value):
        return ag.set(self, value, key)

    def __repr__(self):
        sep = len(self.name) * ' ' + '\n'
        return f"{self.name}({sep.join(str(self.data).splitlines())})"

    def mul(self, other):
        return self * other

    def matmul(self, other):
        return self @ other

    def add(self, other):
        return self + other

    def sub(self, other):
        return self - other

    def pow(self, power, modulo=None):
        return self ** power

    def neg(self):
        return -self

    def div(self, other):
        return self / other

    def abs(self):
        return abs(self)

    def transpose(self, axes: tuple[int]):
        assert max(axes) == len(axes) - 1
        return ag.transpose(self, axes)

    def exp(self):
        return ag.exp(self)

    def log(self, base: float = None):
        return ag.log(self, base)

    def sum(self, axis=None, keepdims=False):
        return ag.sum(self, axis, keepdims)

    def mean(self, axis=None, keepdims=False):
        return ag.mean(self, axis, keepdims)

    def reshape(self, shape: tuple[int, ...]):
        return ag.reshape(self, shape)

    def flatten(self):
        return ag.flatten(self)

    def correlate(self, kernels, reduce_input_depth_dimension=True):
        assert self.mode == kernels.mode
        return ag.correlate(self, kernels, reduce_input_depth_dimension)

    def concatenate(self, *others, axis=0):
        return ag.concatenate(*[self, *others], axis=axis)

    def split(self, indices: tuple, axis=1):
        pass  # todo: implement

    def chunk(self, chunks: int, axis=-1):
        return ag.chunk(self, chunks, axis)

    def squeeze(self, axis=tuple()):
        return ag.squeeze(self, axis)

    def pool(self, sizes, criterion, criterion_included, backend=None):
        """size: pool size; criterion: np.max | np.min | np.average ... pointer; criterion included: np.argmax | np.argmin | lambda x: x ... pointer"""
        return ag.pool(self, sizes, criterion, criterion_included, backend)

    def maxpool(self, sizes, backend=None):
        return ag.maxpool(self, sizes, backend)

    def minpool(self, sizes, backend=None):
        return ag.minpool(self, sizes, backend)

    def averagepool(self, sizes, backend=None):
        return ag.averagepool(self, sizes, backend)

    def squish(self, lower: float, upper: float):
        """squishes the data into a certain range keeping the PROPORTIONS OF DIFFERENCES the same"""
        return ag.squish(self, lower, upper)

    def std(self, axis=None, keepdims=False):
        return ag.std(self, axis, keepdims)

    def variance(self, axis=None, keepdims=False):
        return ag.variance(self, axis, keepdims)

    def item(self):
        assert self.size == 1
        return self.data.flatten()[0]

    def apply_along_axis(self, axis: int, forwardf: callable, backwardf: callable, backward_requires_input_data: bool = False):
        return ag.apply_along_axis(self, axis, forwardf, backwardf, backward_requires_input_data)

    def _add_function(self, function, x):
        """add a function of the gradient to the transforms that will be applied to it in the backward pass"""
        # x = upstream
        self.functions[x] = function

    def _backward(self, gradients: dict):
        """This function requires that the self.upstreams (parents) are in the order they were put into the function"""
        def _edge_backward(upstream: Tensor):
            if not upstream._requires_grad:
                return
            gradient = gradients.get(self)
            if gradient is None:
                gradient = self.lib.ones(self.data.shape)
            function = upstream.functions.get(self)
            if function is not None:
                gradient = function.backward(gradient, upstream)
                if gradients.get(upstream) is None:
                    gradients[upstream] = upstream.lib.zeros(gradient.shape)
                gradients[upstream] += gradient

        if self.backward_parallelizer is not None:
            self.backward_parallelizer(joblib.delayed(_edge_backward)(upstream) for upstream in self.upstreams)
        else:
            for upstream in self.upstreams:
                _edge_backward(upstream)

        if not self.requires_grad and self in gradients.keys():
            del gradients[self]

    def backward(self, max_depth=math.inf) -> dict:
        topo = []
        visited = set()

        def build_topo(node, depth=0):
            if node not in visited and depth < max_depth:
                visited.add(node)
                for upstream in node.upstreams:
                    if upstream._requires_grad:
                        build_topo(upstream, depth + 1)
                topo.append(node)

        build_topo(self)
        gradients = {}
        for node in reversed(topo):
            node._backward(gradients)
        return gradients
