# From https://github.com/arogozhnikov/einops/blob/master/einops/layers/oneflow.py

import re
from functools import wraps

import oneflow as flow
from einops import rearrange, reduce, repeat
from einops._backends import AbstractBackend
from einops.layers import RearrangeMixin
from oneflow import nn


class Rearrange(RearrangeMixin, flow.nn.Module):
    def forward(self, input):
        return self._apply_recipe(input)


class OneFlowBackend(AbstractBackend):
    framework_name = "oneflow"

    def __init__(self):
        import oneflow as flow

        self.flow = flow

    def is_appropriate_type(self, tensor):
        return isinstance(tensor, self.flow.Tensor)

    def from_numpy(self, x):
        variable = self.flow.from_numpy(x)
        if self.is_float_type(variable):
            # attach grad only to floating types
            variable.requires_grad = True
        return variable

    def to_numpy(self, x):
        return x.detach().cpu().numpy()

    def arange(self, start, stop):
        return self.flow.arange(start, stop, dtype=self.flow.int64)

    def reduce(self, x, operation, reduced_axes):
        for axis in sorted(reduced_axes, reverse=True):
            if operation == "min":
                x, _ = x.min(dim=axis)
            elif operation == "max":
                x, _ = x.max(dim=axis)
            elif operation in ["sum", "mean", "prod"]:
                x = getattr(x, operation)(dim=axis)
            else:
                raise NotImplementedError("Unknown reduction ", operation)
        return x

    def transpose(self, x, axes):
        return x.permute(axes)

    def stack_on_zeroth_dimension(self, tensors: list):
        return self.flow.stack(tensors)

    def add_axes(self, x, n_axes, pos2len):
        repeats = [-1] * n_axes
        for axis_position, axis_length in pos2len.items():
            x = self.add_axis(x, axis_position)
            repeats[axis_position] = axis_length
        return x.expand(*repeats)

    def tile(self, x, repeats):
        return x.repeat(repeats)

    def add_axis(self, x, new_position):
        return self.flow.unsqueeze(x, new_position)

    def is_float_type(self, x):
        return x.dtype in [self.flow.float16, self.flow.float32, self.flow.float64]

    def einsum(self, pattern, *x):
        return self.flow.einsum(pattern, *x)


# From https://github.com/lucidrains/einops-exts/tree/main/einops_exts


class EinopsToAndFrom(nn.Module):
    def __init__(self, from_einops, to_einops, fn):
        super().__init__()
        self.from_einops = from_einops
        self.to_einops = to_einops
        self.fn = fn

    def forward(self, x, **kwargs):
        shape = x.shape
        reconstitute_kwargs = dict(tuple(zip(self.from_einops.split(" "), shape)))
        x = rearrange(x, f"{self.from_einops} -> {self.to_einops}")
        x = self.fn(x, **kwargs)
        x = rearrange(x, f"{self.to_einops} -> {self.from_einops}", **reconstitute_kwargs)
        return x


# checking shape
# @nils-werner
# https://github.com/arogozhnikov/einops/issues/168#issuecomment-1042933838


def check_shape(tensor, pattern, **kwargs):
    return rearrange(tensor, f"{pattern} -> {pattern}", **kwargs)


# do same einops operations on a list of tensors


def _many(fn):
    @wraps(fn)
    def inner(tensors, pattern, **kwargs):
        return (fn(tensor, pattern, **kwargs) for tensor in tensors)

    return inner


# do einops with unflattening of anonymously named dimensions
# (...flattened) ->  ...flattened


def _with_anon_dims(fn):
    @wraps(fn)
    def inner(tensor, pattern, **kwargs):
        regex = r"(\.\.\.[a-zA-Z]+)"
        matches = re.findall(regex, pattern)

        def get_anon_dim_name(t):
            return t.lstrip("...")

        dim_prefixes = tuple(map(get_anon_dim_name, set(matches)))

        update_kwargs_dict = dict()

        for prefix in dim_prefixes:
            assert prefix in kwargs, f'dimension list "{prefix}" was not passed in'
            dim_list = kwargs[prefix]
            assert isinstance(
                dim_list, (list, tuple)
            ), f'dimension list "{prefix}" needs to be a tuple of list of dimensions'
            dim_names = list(map(lambda ind: f"{prefix}{ind}", range(len(dim_list))))
            update_kwargs_dict[prefix] = dict(zip(dim_names, dim_list))

        def sub_with_anonymous_dims(t):
            dim_name_prefix = get_anon_dim_name(t.groups()[0])
            return " ".join(update_kwargs_dict[dim_name_prefix].keys())

        pattern_new = re.sub(regex, sub_with_anonymous_dims, pattern)

        for prefix, update_dict in update_kwargs_dict.items():
            del kwargs[prefix]
            kwargs.update(update_dict)

        return fn(tensor, pattern_new, **kwargs)

    return inner


# generate all helper functions

rearrange_many = _many(rearrange)
repeat_many = _many(repeat)
reduce_many = _many(reduce)

rearrange_with_anon_dims = _with_anon_dims(rearrange)
repeat_with_anon_dims = _with_anon_dims(repeat)
reduce_with_anon_dims = _with_anon_dims(reduce)
