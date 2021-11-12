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
import oneflow.nn.init as init
from core import distribute as dist


class ColumnParallelLinear(flow.nn.Module):
    """Linear layer with column parallelism.

    The linear layer is defined as Y = XA + b, where A is parallelized along 
    the second dimension as A = [A_1, ..., A_p].

    Arguments:
        layer_idx: the layer index, which determines the placement.
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.
        init_method: method to initialize weights.
        need_gelu: whether to use gelu activation function. (Supporting bias and gelu fusion)
        bias_gelu_fusion: whether fuse add bias and gelu.
    """
    def __init__(self, layer_idx, input_size, output_size, init_method=init.xavier_normal_, 
                 need_gelu=False, bias_gelu_fusion=False):
        super().__init__()
        self.need_gelu = need_gelu
        self.bias_gelu_fusion = bias_gelu_fusion

        # col parallel linear weight sbp: [B, S(1)]
        self.weight = flow.nn.Parameter(
            flow.empty(
                (input_size, output_size),
                dtype=flow.float32,
                placement=dist.get_layer_placement(layer_idx),
                sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.split(1)]),
            )
        )
        init_method(self.weight)
        
        # col parallel linear bias sbp: [B, S(0)]
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
        if self.need_gelu:
            if self.bias_gelu_fusion:
                x = flow._C.fused_bias_add_gelu(x, self.bias, axis=x.ndim - 1)
            else:
                x = x + self.bias
                x = flow.gelu(x)
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
    def __init__(self, layer_idx, input_size, output_size, init_method=init.xavier_normal_, 
                 output_dropout_prob=0., bias_dropout_fusion=False):
        super().__init__()
        self.output_dropout_prob = output_dropout_prob

        self.bias_dropout_fusion = bias_dropout_fusion
        if not self.bias_dropout_fusion > 0.:
            self.dropout = flow.nn.Dropout(p=self.output_dropout_prob)

        # col parallel linear weight sbp: [B, S(0)]
        self.weight = flow.nn.Parameter(
            flow.empty(
                (input_size, output_size),
                dtype=flow.float32,
                placement=dist.get_layer_placement(layer_idx),
                sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.split(0)]),
            )
        )
        init_method(self.weight)

        # col parallel linear bias sbp: [B, B]
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
        if self.output_dropout_prob > 0.:
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

