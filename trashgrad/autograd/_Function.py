import math
import warnings
from abc import abstractmethod
from trashgrad.autograd import Tensor


class Function:
    backward_required_args_kwargs: set  # set of indices/names of tensors whose data is required for backward. specification:
    # add both the name of the arg as well as its index in the function args to the set to ensure it will land in either
    # args or kwargs if passed in. eg:
    # def _forward(self, tensor: Tensor, factor: float):
    #     return tensor.data * factor
    # backward_required_args_kwargs = {0, 1, 'tensor', 'factor'}
    _args: tuple
    _kwargs: dict
    _called = False
    _index_map: dict

    @abstractmethod
    def _forward(self, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def _backward(self, grad, req: Tensor, *args, **kwargs):
        raise NotImplementedError()

    @staticmethod
    def reduce(grad, tensor: Tensor):
        augmented = tuple(range(len(grad.shape) - tensor.ndim))
        if augmented:
            grad = grad.sum(augmented)
        expanded = []
        for i, j, k in zip(range(len(grad.shape)), tensor.shape, grad.shape):
            if j == 1 and k != 1:
                expanded.append(i)
        if expanded:
            grad = grad.sum(tuple(expanded), keepdims=True)
        return grad

    def _assertions(self, upstreams, *args, **kwargs):
        #assert len(args + tuple(kwargs.values())) > 0
        assert (upstreams[0].base_ops_until_forget == upstreams[i].base_ops_until_forget for i in range(1, len(upstreams)))
        assert (upstreams[0].mode == upstreams[i].mode for i in range(1, len(upstreams)))

    def _result_def(self, upstreams, *args, **kwargs):
        return Tensor(self._forward(*args, **kwargs),
                      upstreams,
                      _requires_grad=any(map(lambda a: a._requires_grad, upstreams)),
                      ops_until_forget=upstreams[0].base_ops_until_forget,
                      base_ops_until_forget=upstreams[0].base_ops_until_forget,
                      mode=upstreams[0].mode
                      )

    def _post_result_def_ops(self, upstreams, x, *args, **kwargs):
        for upstream in upstreams:
            if not upstream._requires_grad:
                continue
            upstream._add_function(self, x)
            upstream.ops_until_forget -= 1
            upstream.forget()

    def _interpret_backward_req_args_kwargs(self, *args, **kwargs):
        warn_for_type_errors = True
        if self.backward_required_args_kwargs is True:
            self.backward_required_args_kwargs = set(range(len(args)))
            self.backward_required_args_kwargs.update(kwargs.keys())
            warn_for_type_errors = False
        for i in self.backward_required_args_kwargs:
            if isinstance(i, int):
                if isinstance(args[i], Tensor):
                    args[i].ops_until_forget = math.inf
                elif warn_for_type_errors:
                    warnings.warn(f"Index {i} of inputs to {self} is not a tensor and will thus be skipped")
            elif isinstance(i, str):
                if isinstance(kwargs[i], Tensor):
                    kwargs[i].ops_until_forget = math.inf
                elif warn_for_type_errors:
                    warnings.warn(f"Value of keyword {i} of inputs to {self} is not a tensor and will thus be skipped")
            else:
                warnings.warn(f"{self}.backward_required_args_kwargs contains an invalid element. It will be ignored.")

    def forward(self, *args, **kwargs) -> Tensor:
        if self._called:
            raise Exception(f"Function {self} was called multiple times, but Functions can only be called once")
        upstreams = tuple(item for item in args + tuple(kwargs.values()) if isinstance(item, Tensor))
        self._assertions(upstreams, *args, **kwargs)
        self._args = args
        self._kwargs = kwargs
        self._interpret_backward_req_args_kwargs(*args, **kwargs)
        x = self._result_def(upstreams, *args, **kwargs)
        #self._index_map = {args[i]: i for i in range(len(args)) if isinstance(args[i], Tensor)}
        #self._index_map.update({kwargs[name]: name for name in kwargs.keys() if isinstance(kwargs[name], Tensor)})
        self._post_result_def_ops(upstreams, x, *args, **kwargs)
        self._called = True
        return x

    def backward(self, grad, req: Tensor):
        return self.reduce(self._backward(grad, req, *self._args, **self._kwargs), req)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
