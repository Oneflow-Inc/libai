# coding=utf-8
# Copyright 2021 The OneFlow Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import oneflow as flow
from oneflow import nn

from libai.layers.activations import build_activation
from libai.utils import distributed as dist


class Linear1D(nn.Module):
    """Linear layer with 1D parallelism which includes column parallelism and row parallelism.
    The linear layer is defined as :math:`Y = XA + b`.

    In column parallelism, A is parallelized along the second dimension as :math:`A = [A_1, ..., A_p]`.
    
    In row parallelism, A is parallelized along the first dimension and X along its second dimension as:
                | A_1 |
                |  .  |
            A = |  .  |         X = [X_1, ..., X_p]
                |  .  |
                | A_p |

    Arguments:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias. Defaults to ``True``.
        parallel: . Defaults to "data".
        init_method: method to initialize weight. Defaults to nn.init.xavier_normal_.
        activation: name of activation function after linear layer. 
        If set to ``None``, it will do nothing. Defaults to None.
        bias_gelu_fusion: If set to ``True``, it will fuse bias adding and elementwise gelu activation. Defaults to ``False``.
        output_dropout_prob: Output dropout probability. Defaults to 0.0.
        bias_dropout_fusion: If set to ``True``, it will fuse bias adding and dropout. Defaults to ``False``.
        layer_idx: layer_idx sign for placement setting. It will be used in pipeline parallelism. Defaults to 0.
    """
    
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        parallel="data",
        init_method=nn.init.xavier_normal_,
        activation=None,
        bias_gelu_fusion=False,
        output_dropout_prob=0.0,
        bias_dropout_fusion=False,
        *,
        layer_idx=0, # Enforce layer_idx passed with keyword
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.need_act = activation is not None
        self.bias_gelu_fusion = bias_gelu_fusion and (activation == "gelu")
        if self.need_act:
            # Not using gelu fusion
            if not bias_gelu_fusion:
                self.activation_func = build_activation(activation)

        self.output_dropout_prob = output_dropout_prob
        self.bias_dropout_fusion = bias_dropout_fusion
        if not self.bias_dropout_fusion and output_dropout_prob > 0.0:
            self.dropout = nn.Dropout(p=self.output_dropout_prob)

        if parallel == "col":
            # column parallel linear weight sbp: [B, S(1)] and bias sbp: [B, S(0)].
            weight_sbp = dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.split(1)])
            bias_sbp = dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.split(0)])
        elif parallel == "row":
            # row parallel linear weight sbp: [B, S(0)] and bias sbp: [B, B]
            weight_sbp = dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.split(0)])
            bias_sbp = dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast])
        elif parallel == "data":
            weight_sbp = flow.sbp.broadcast
            bias_sbp = flow.sbp.broadcast
        else:
            raise KeyError(
                f"{parallel} is not supported! Only support ('data', 'row' and 'col')"
            )

        self.weight = flow.nn.Parameter(
            flow.empty(
                (in_features, out_features),
                dtype=flow.float32,
                # For pipeline parallelism placement.
                placement=dist.get_layer_placement(layer_idx),
                sbp=weight_sbp,
            )
        )
        init_method(self.weight)

        self.bias = flow.nn.Parameter(
            flow.empty(
                (out_features,),
                dtype=flow.float32,
                placement=dist.get_layer_placement(layer_idx),
                sbp=bias_sbp,
            )
        )
        nn.init.zeros_(self.bias)

    def forward(self, x):
        if dist.same_sbp(
            self.weight.sbp, dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.split(1)])
        ):
            # 设定 x sbp: [S(0), B] 确保 w 一定是 [B, S(1)]
            # x.grad sbp: [S(0), P] -> [S(0), B]
            x = x.to_consistent(
                sbp=dist.get_nd_sbp([flow.sbp.split(0), flow.sbp.broadcast])
            )
            x = x.to_consistent(grad_sbp=x.sbp)
        elif dist.same_sbp(
            self.weight.sbp, dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.split(0)])
        ):
            # 设定 x sbp: [S(0), S(1)] 确保 w 一定是 [B, S(0)]
            x = x.to_consistent(
                sbp=dist.get_nd_sbp([flow.sbp.split(0), flow.sbp.split(x.ndim - 1)])
            )

        # matmul sbp sign: [S(0), B] x [B, S(1)] -> [S(0), S(1)]
        # x.grad sbp sign: [S(0), S(1)] x [B, S(0)] (weight.T) -> [S(0), P]
        x = flow._C.matmul(x, self.weight)

        if dist.same_sbp(
            self.weight.sbp, dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.split(0)])
        ):
            # 设定 x sbp: [S(0), P] -> [S(0), B] 进行后面的运算
            x = x.to_consistent(
                sbp=dist.get_nd_sbp([flow.sbp.split(0), flow.sbp.broadcast])
            )

        if self.need_act:
            if self.bias_gelu_fusion:
                x = flow._C.fused_bias_add_gelu(x, self.bias, axis=x.ndim - 1)
            else:
                x = x + self.bias
                x = self.activation_func(x)

        if self.output_dropout_prob > 0.0:
            if (not self.need_act) and self.bias_dropout_fusion:
                x = flow._C.fused_bias_add_dropout(
                    x, self.bias, p=self.output_dropout_prob, axis=x.ndim - 1
                )
            elif self.need_act:
                x = self.dropout(x)
            else:
                x = x + self.bias
                x = self.dropout(x)

        return x


class ColumnParallelLinear(nn.Module):
    """Linear layer with column parallelism.
    The linear layer is defined as Y = XA + b, where A is parallelized along 
    the second dimension as A = [A_1, ..., A_p].
    Arguments:
        layer_idx: the layer index, which determines the placement.
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.
        init_method: method to initialize weights.
        activation: name of the activation function
        bias_gelu_fusion: whether fuse add bias and gelu.
    """

    def __init__(
        self,
        layer_idx,
        input_size,
        output_size,
        init_method=nn.init.xavier_normal_,
        activation=None,
        bias_gelu_fusion=False,
    ):
        super().__init__()
        self.need_act = activation is not None
        self.bias_gelu_fusion = bias_gelu_fusion and (activation == "gelu")

        if self.need_act:
            # Not using gelu fusion
            if not bias_gelu_fusion:
                self.activation_func = build_activation(activation)

        # column parallel linear weight sbp: [B, S(1)]
        self.weight = flow.nn.Parameter(
            flow.empty(
                (input_size, output_size),
                dtype=flow.float32,
                placement=dist.get_layer_placement(
                    layer_idx
                ),  # 这个是为了使用 pipeline parallel
                sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.split(1)]),
            )
        )
        init_method(self.weight)

        # column parallel linear bias sbp: [B, S(0)]
        self.bias = flow.nn.Parameter(
            flow.empty(
                (output_size,),
                dtype=flow.float32,
                placement=dist.get_layer_placement(layer_idx),
                sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.split(0)]),
            )
        )
        flow.nn.init.zeros_(self.bias)

    def forward(self, x):
        # x sbp: [S(0), B]
        # x.grad sbp: [S(0), P] -> [S(0), B]
        x = x.to_consistent(grad_sbp=x.sbp)
        # matmul sbp sign: [S(0), B] x [B, S(1)] -> [S(0), S(1)]
        # x.grad sbp sign: [S(0), S(1)] x [B, S(0)] (weight.T) -> [S(0), P]
        x = flow.matmul(x, self.weight)
        if self.need_act:
            if self.bias_gelu_fusion:
                x = flow._C.fused_bias_add_gelu(x, self.bias, axis=x.ndim - 1)
            else:
                x = x + self.bias
                x = self.activation_func(x)
        else:
            # broadcast_add shape sign:
            # (input_size, output_size) + (output_size, ) = (input_size, output_size)
            # bias_add sbp sign: [S(0), S(1)] + [B, S(0)] = [S(0), S(1)]
            x = x + self.bias

        return x


class RowParallelLinear(flow.nn.Module):
    """Linear layer with row parallelism.
    The linear layer is defined as Y = XA + b, where A is parallelized along 
    the first dimension and X along its second dimension as:
                | A_1 |
                |  .  |
            A = |  .  |         X = [X_1, ..., X_p]
                |  .  |
                | A_p |
    Arguments:
        layer_idx: the layer index, which determines the placement.
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.
        init_method: method to initialize weights.
        output_dropout_prob: dropout probability of output. (Supporting bias and dropout fusion)
        bias_dropout_fusion: whether fuse add bias and dropout.
    """

    def __init__(
        self,
        layer_idx,
        input_size,
        output_size,
        init_method=nn.init.xavier_normal_,
        output_dropout_prob=0.0,
        bias_dropout_fusion=False,
    ):
        super().__init__()
        self.output_dropout_prob = output_dropout_prob

        self.bias_dropout_fusion = bias_dropout_fusion
        if not self.bias_dropout_fusion:
            self.dropout = nn.Dropout(p=self.output_dropout_prob)

        self.weight = flow.nn.Parameter(
            flow.empty(
                (input_size, output_size),
                dtype=flow.float32,
                placement=dist.get_layer_placement(layer_idx),
                sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.split(0)]),
            )
        )
        init_method(self.weight)

        # row parallel linear bias sbp: [B, B]
        self.bias = flow.nn.Parameter(
            flow.empty(
                (output_size,),
                dtype=flow.float32,
                placement=dist.get_layer_placement(layer_idx),
                sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
            )
        )
        flow.nn.init.zeros_(self.bias)

    def forward(self, x):
        # x.sbp: [S(0), S(1)]
        # matmul sbp sign: [S(0), S(1)] x [B, S(0)] -> [S(0), P]
        # backward x.grad sbp sign: [S(0), B] x [B, S(1)] (weight.T) -> [S(0), S(1)]
        x = flow.matmul(x, self.weight)
        # x.sbp: [S(0), P] -> [S(0), B]
        x = x.to_consistent(sbp=dist.get_hidden_sbp())
        if self.output_dropout_prob > 0.0:
            if self.bias_dropout_fusion:
                x = flow._C.fused_bias_add_dropout(
                    x, self.bias, p=self.output_dropout_prob, axis=x.ndim - 1
                )
            else:
                x = x + self.bias
                x = self.dropout(x)
        else:
            x = x + self.bias

        return x
