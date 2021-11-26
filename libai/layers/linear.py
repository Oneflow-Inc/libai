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

from libai.layers import build_activation
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
        layer_idx: A layer_idx sign which determines the placement. It will be used in pipeline parallelism. Defaults to 0.
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
        layer_idx=0,  # Enforce layer_idx passed with keyword
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.parallel = parallel
        self.activation = activation

        self.need_act = activation is not None
        self.bias_gelu_fusion = bias_gelu_fusion and (activation == "gelu") and bias
        if self.need_act and (not self.bias_gelu_fusion):
            # Not using gelu fusion
            self.activation_func = build_activation(activation)

        self.output_dropout_prob = output_dropout_prob
        # bias_dropout fusion optimization can only be done without activation
        self.bias_dropout_fusion = bias_dropout_fusion and (not self.need_act) and bias
        if not self.bias_dropout_fusion and output_dropout_prob > 0.0:
            self.dropout = nn.Dropout(p=self.output_dropout_prob)

        if parallel == "col":
            # Column parallel weight sbp: [B, S(1)] and bias sbp: [B, S(0)].
            weight_sbp = dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.split(1)])
            bias_sbp = dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.split(0)])
        elif parallel == "row":
            # Row parallel weight sbp: [B, S(0)] and bias sbp: [B, B]
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

        self.bias = (
            flow.nn.Parameter(
                flow.zeros(
                    (out_features,),
                    dtype=flow.float32,
                    placement=dist.get_layer_placement(layer_idx),
                    sbp=bias_sbp,
                )
            )
            if bias
            else None
        )

    def forward(self, x):
        if dist.same_sbp(
            self.weight.sbp, dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.split(1)])
        ):
            # When weight sbp sign is [B, S(1)], change x sbp sign to [S(0), B].
            x = x.to_consistent(
                sbp=dist.get_nd_sbp([flow.sbp.split(0), flow.sbp.broadcast])
            )

            # Matmul sbp sign: [S(0), B] x [B, S(1)] -> [S(0), S(1)]
            # Backward x.grad sbp sign: [S(0), S(1)] x [B, S(0)] (weight.T) -> [S(0), P]
            # which cannot do backward pass when sbp sign is [S(0), P]
            # so change x.grad sbp: [S(0), P] -> [S(0), B]
            x = x.to_consistent(grad_sbp=x.sbp)
            x = flow._C.matmul(x, self.weight)

        elif dist.same_sbp(
            self.weight.sbp, dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.split(0)])
        ):
            # When weight sbp sign is [B, S(0)], change x sbp sign to [S(0), S(1)].
            x = x.to_consistent(
                sbp=dist.get_nd_sbp([flow.sbp.split(0), flow.sbp.split(x.ndim - 1)])
            )
            # Matmul sbp sign: [S(0), S(1)] x [B, S(0)] -> [S(0), P]
            # Backward x.grad sbp sign: [S(0), B] x [B, S(1)] (weight.T) -> [S(0), S(1)]
            x = flow._C.matmul(x, self.weight)
            # Change x sbp: [S(0), P] -> [S(0), B] for followup forward pass.
            x = x.to_consistent(
                sbp=dist.get_nd_sbp([flow.sbp.split(0), flow.sbp.broadcast])
            )
        else:
            raise NotImplementedError(f"Not support weight with sbp: {self.weight.sbp}")

        # Flag for deciding if add bias
        bias_add = False
        if self.need_act:
            if self.bias_gelu_fusion:
                x = flow._C.fused_bias_add_gelu(x, self.bias, axis=x.ndim - 1)
                bias_add = True
            else:
                if self.bias is not None:
                    # bias_add sbp sign
                    # column parallelism: [S(0), S(1)] + [B, S(0)] = [S(0), S(1)]
                    # row parallelism: [S(0), B] + [B, B] = [S(0), B]
                    x = x + self.bias
                    bias_add = True
                x = self.activation_func(x)

        if self.output_dropout_prob > 0.0:
            if self.bias_dropout_fusion:
                x = flow._C.fused_bias_add_dropout(
                    x, self.bias, p=self.output_dropout_prob, axis=x.ndim - 1
                )
                bias_add = True
            else:
                if not bias_add and self.bias is not None:
                    x = x + self.bias
                    bias_add = True
                x = self.dropout(x)

        # If no activation and dropout, then bias_add can been done here.
        if not bias_add and self.bias is not None:
            x = x + self.bias

        return x

    def extra_repr(self) -> str:
        return "in_features={}, out_features={}, bias={}, parallel={}, activation={}, dropout={}".format(
            self.in_features,
            self.out_features,
            self.bias is not None,
            self.parallel,
            self.activation,
            self.output_dropout_prob,
        )
