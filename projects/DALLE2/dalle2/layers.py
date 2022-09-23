# from https://github.com/Oneflow-Inc/oneflow/blob/master/python/oneflow/nn/modules/normalization.py
import math
import os
from typing import Union

import oneflow as flow
from oneflow import nn
from oneflow.nn import init
from oneflow.nn.common_types import _size_1_t, _size_2_t, _size_3_t
from oneflow.nn.modules.utils import _pair, _single, _triple

from libai.utils import distributed as dist


class GroupNorm(nn.Module):
    def __init__(
        self,
        num_groups: int,
        num_channels: int,
        eps: float = 1e-05,
        affine: bool = True,
        bias=True,
        *,
        layer_idx=0,
    ) -> None:
        super().__init__()
        assert num_groups > 0, "The num_groups must larger than zero"
        assert num_channels > 0, "The num_channels must larger than zero"
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.weight = nn.Parameter(
                flow.ones(
                    num_channels,
                    dtype=flow.float32,
                    placement=dist.get_layer_placement(layer_idx),
                    sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
                )
            )
            self.bias = nn.Parameter(
                flow.zeros(
                    num_channels,
                    dtype=flow.float32,
                    placement=dist.get_layer_placement(layer_idx),
                    sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
                ),
                requires_grad=bias,
            )
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, input: flow.Tensor) -> flow.Tensor:
        assert len(input.shape) >= 3, "The dimensions of input tensor must larger than 2"
        assert (
            input.shape[1] == self.num_channels
        ), "The channels of input tensor must equal num_channels"
        origin_shape = input.shape
        reshape_to_1d = flow.reshape(input, shape=[origin_shape[0], self.num_groups, -1])
        # https://github.com/Oneflow-Inc/oneflow/pull/8905
        # mean = flow.mean(reshape_to_1d, dim=2, keepdim=True)
        # variance = flow.var(reshape_to_1d, dim=2, unbiased=False, keepdim=True)
        # normalized = (reshape_to_1d - mean) / flow.sqrt(variance + self.eps)
        normalized = layer_norm(
            reshape_to_1d, normalized_shape=(reshape_to_1d.shape[-1:]), eps=self.eps
        )
        normalized = flow.reshape(normalized, shape=[origin_shape[0], self.num_channels, -1])
        if self.weight is not None:
            normalized = normalized * self.weight.reshape(1, self.num_channels, 1)
        if self.bias is not None:
            normalized = normalized + self.bias.reshape(1, self.num_channels, 1)
        res = flow.reshape(normalized, shape=tuple(input.shape))
        return res

    def extra_repr(self) -> str:
        return "{num_groups}, {num_channels}, eps={eps}, affine={affine}".format(**self.__dict__)


def layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-05):
    assert len(input.shape) > len(
        normalized_shape
    ), "Input tensor dim must greater than normalized dim!"
    begin_norm_axis = len(input.shape) - len(normalized_shape)
    begin_params_axis = len(input.shape) - len(normalized_shape)

    elementwise_affine = True if (weight is not None and bias is not None) else False

    for i in range(0, len(normalized_shape)):
        if input.shape[i + begin_params_axis] != normalized_shape[i]:
            raise RuntimeError(
                f"Given normalized_shape={normalized_shape}, expected input with shape [*, {str(normalized_shape)[1:-1]}], but got input of size {input.shape}"
            )

    if not input.is_cuda:
        reduce_axis = []
        for dim in range(len(input.shape)):
            if dim >= begin_norm_axis:
                reduce_axis.append(dim)
        mean = input.mean(dim=reduce_axis, keepdim=True)
        variance = input.var(dim=reduce_axis, unbiased=False, keepdim=True)
        params_shape = input.shape[begin_params_axis:]
        if len(mean.shape) == 1:
            nd_params_shape = [1] * len(input.shape)
            nd_params_shape[begin_norm_axis] = params_shape[0]
            mean = flow.reshape(mean, shape=nd_params_shape)
            variance = flow.reshape(variance, nd_params_shape)
            if weight is not None and params_shape[0] == weight.nelement():
                weight = flow.reshape(weight, shape=nd_params_shape)
            if bias is not None and params_shape[0] == bias.nelement():
                bias = flow.reshape(bias, shape=nd_params_shape)
        elif len(mean.shape) == len(input.shape):
            pass
        else:
            raise ValueError(
                "shape of mean and variance should be 1D or has number of axes and x's"
            )
        variance += eps
        normalized = (input - mean) * variance.rsqrt()
        if elementwise_affine:
            normalized = normalized * weight + bias
        return normalized
    else:
        if elementwise_affine:
            res = flow._C.layer_norm_affine(
                input,
                weight,
                bias,
                begin_norm_axis=begin_norm_axis,
                begin_params_axis=begin_params_axis,
                epsilon=eps,
            )
        else:
            res = flow._C.layer_norm(
                input,
                begin_norm_axis=begin_norm_axis,
                begin_params_axis=begin_params_axis,
                epsilon=eps,
            )
        return res


# Conv2d from https://github.com/Oneflow-Inc/oneflow/blob/master/python/oneflow/nn/modules/conv.py
def get_padding(padding, kernel_size, dilation, stride):
    valid_padding_strings = {"same", "valid"}
    if isinstance(padding, str):
        if padding not in valid_padding_strings:
            raise ValueError(
                "Invalid padding string {!r}, should be one of {}".format(
                    padding, valid_padding_strings
                )
            )
        if padding == "same" and any(s != 1 for s in list(stride)):
            raise ValueError("padding='same' is not supported for strided convolutions")

    out_padding = [0] * len(kernel_size)
    if padding == "same":
        for d, k, i in zip(dilation, kernel_size, range(len(kernel_size) - 1, -1, -1)):
            total_padding = d * (k - 1)
            left_pad = total_padding // 2
            out_padding[i] = left_pad
    return out_padding


class Conv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        *,
        layer_idx=0,
    ):
        super().__init__()
        sbp = dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast])
        assert padding_mode == "zeros"
        self.padding_mode = padding_mode
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.dilation = _pair(dilation)
        self.padding = (
            get_padding(padding, self.kernel_size, self.dilation, self.stride)
            if isinstance(padding, str)
            else _pair(padding)
        )
        self.groups = groups

        if os.getenv("ONEFLOW_ENABLE_NHWC") == "1":
            self.channel_pos = "channels_last"
        else:
            self.channel_pos = "channels_first"

        assert in_channels % groups == 0
        assert out_channels % groups == 0
        self.in_channels = in_channels
        self.out_channels = out_channels
        if self.channel_pos == "channels_first":
            self.weight = flow.nn.Parameter(
                flow.Tensor(
                    out_channels,
                    in_channels // groups,
                    *self.kernel_size,
                    placement=dist.get_layer_placement(layer_idx=layer_idx),
                    sbp=sbp,
                )
            )
        else:
            self.weight = flow.nn.Parameter(
                flow.Tensor(
                    out_channels,
                    *self.kernel_size,
                    in_channels // groups,
                    placement=dist.get_layer_placement(layer_idx=layer_idx),
                    sbp=sbp,
                )
            )

        self.out_channel_groups = out_channels // groups
        self.bias = None
        if bias:
            self.bias = flow.nn.Parameter(
                flow.Tensor(
                    out_channels, placement=dist.get_layer_placement(layer_idx=layer_idx), sbp=sbp
                )
            )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            (fan_in, _) = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def _conv_forward(self, x, weight, bias):
        if self.channel_pos == "channels_first":
            in_channel_axis = 1
        else:
            in_channel_axis = 3
        if x.shape[in_channel_axis] != self.in_channels:
            raise ValueError(
                f"The input channels {x.shape[in_channel_axis]} should be equal to self.in_channels {self.in_channels}."
            )
        return flow._C.conv2d(
            x,
            weight,
            bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            channel_pos=self.channel_pos,
        )

    def forward(self, x):
        return self._conv_forward(x, self.weight, self.bias)

    def extra_repr(self):
        s = "{in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}"
        if self.padding != (0,) * len(self.padding):
            s += ", padding={padding}"
        if self.dilation != (1,) * len(self.dilation):
            s += ", dilation={dilation}"
        if self.groups != 1:
            s += ", groups={groups}"
        if self.bias is None:
            s += ", bias=False"
        if self.padding_mode != "zeros":
            s += ", padding_mode={padding_mode}"
        return s.format(**self.__dict__)


class ConvTranspose2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: _size_2_t = 0,
        output_padding: _size_2_t = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: int = 1,
        padding_mode: str = "zeros",
        *,
        layer_idx=0,
    ) -> None:
        super().__init__()
        sbp = dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast])
        assert padding_mode == "zeros"
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.output_padding = _pair(output_padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        assert in_channels % groups == 0
        assert out_channels % groups == 0
        self.weight = flow.nn.Parameter(
            flow.Tensor(
                in_channels,
                out_channels // groups,
                *self.kernel_size,
                placement=dist.get_layer_placement(layer_idx=layer_idx),
                sbp=sbp,
            )
        )
        self.in_channel_groups = in_channels // groups
        self.filters = out_channels
        self.bias = None
        self._bias_add_op = None
        if bias:
            self.bias = flow.nn.Parameter(
                flow.Tensor(
                    out_channels, placement=dist.get_layer_placement(layer_idx=layer_idx), sbp=sbp
                )
            )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            (fan_in, _) = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        res = flow._C.deconv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.output_padding,
            self.groups,
            self.dilation,
            "channels_first",
        )
        return res
