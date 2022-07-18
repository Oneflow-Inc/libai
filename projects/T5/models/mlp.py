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

from libai.layers import Linear, build_activation


class T5MLP(nn.Module):
    def __init__(
        self,
        hidden_size,
        ffn_hidden_size,
        output_dropout_prob=0.0,
        init_method=nn.init.xavier_normal_,
        output_layer_init_method=None,
        bias_gelu_fusion=False,
        bias_dropout_fusion=False,
        *,
        layer_idx=0,
    ):
        super().__init__()
        self.output_dropout_prob = output_dropout_prob
        self.bias_gelu_fusion = bias_gelu_fusion
        self.bias_dropout_fusion = bias_dropout_fusion

        if output_layer_init_method is None:
            output_layer_init_method = init_method

        self.dense_h_to_4h = Linear(
            hidden_size,
            ffn_hidden_size,
            bias=False,
            parallel="col",
            skip_bias_add=bias_gelu_fusion,
            init_method=init_method,
            layer_idx=layer_idx,
        )

        if not bias_gelu_fusion:
            self.activation_func = build_activation("relu")

        self.dense_4h_to_h = Linear(
            ffn_hidden_size,
            hidden_size,
            bias=False,
            parallel="row",
            skip_bias_add=bias_dropout_fusion,
            init_method=output_layer_init_method,
            layer_idx=layer_idx,
        )

        if not bias_dropout_fusion:
            self.dropout = nn.Dropout(self.output_dropout_prob)

    def forward(self, hidden_states):
        intermediate = self.dense_h_to_4h(hidden_states)
        if self.bias_gelu_fusion:
            intermediate, bias = intermediate
            intermediate = flow._C.fused_bias_add_gelu(
                intermediate, bias, axis=intermediate.ndim - 1
            )
        else:
            intermediate = self.activation_func(intermediate)

        output = self.dense_4h_to_h(intermediate)
        if self.bias_dropout_fusion:
            output, bias = output
            output = flow._C.fused_bias_add_dropout(
                output, bias, p=self.output_dropout_prob, axis=output.ndim - 1
            )
        else:
            output = self.dropout(output)
        return output

    def extra_repr(self) -> str:
        return "bias_gelu_fusion={}, bias_dropout_fusion={}, dropout={}".format(
            self.bias_gelu_fusion, self.bias_dropout_fusion, self.output_dropout_prob
        )


class MT5MLP(nn.Module):
    def __init__(
        self,
        hidden_size,
        ffn_hidden_size,
        output_dropout_prob=0.0,
        init_method=nn.init.xavier_normal_,
        output_layer_init_method=None,
        bias_gelu_fusion=False,
        bias_dropout_fusion=False,
        *,
        layer_idx=0,
    ):
        super().__init__()
        self.output_dropout_prob = output_dropout_prob
        self.bias_gelu_fusion = bias_gelu_fusion
        self.bias_dropout_fusion = bias_dropout_fusion

        if output_layer_init_method is None:
            output_layer_init_method = init_method

        self.wi_0 = Linear(
            hidden_size,
            ffn_hidden_size,
            bias=False,
            parallel="col",
            skip_bias_add=bias_gelu_fusion,
            init_method=init_method,
            layer_idx=layer_idx,
        )

        self.wi_1 = Linear(
            hidden_size,
            ffn_hidden_size,
            bias=False,
            parallel="col",
            skip_bias_add=False,
            init_method=init_method,
            layer_idx=layer_idx,
        )

        if not bias_gelu_fusion:
            self.activation_func = build_activation("gelu")

        self.wo = Linear(
            ffn_hidden_size,
            hidden_size,
            bias=False,
            parallel="row",
            skip_bias_add=bias_dropout_fusion,
            init_method=output_layer_init_method,
            layer_idx=layer_idx,
        )

        self.dropout = nn.Dropout(self.output_dropout_prob)

    def forward(self, hidden_states):
        wi_0_out = self.wi_0(hidden_states)
        if self.bias_gelu_fusion:
            wi_0_out, bias = wi_0_out
            hidden_gelu = flow._C.fused_bias_add_gelu(wi_0_out, bias, axis=wi_0_out.ndim - 1)
        else:
            hidden_gelu = self.activation_func(wi_0_out)

        hidden_linear = self.wi_1(hidden_states)
        hidden_states = hidden_gelu * hidden_linear

        output = self.wo(hidden_states)
        if self.bias_dropout_fusion:
            output, bias = output
            output = flow._C.fused_bias_add_dropout(
                output, bias, p=self.output_dropout_prob, axis=output.ndim - 1
            )
        else:
            output = self.dropout(output)
        return output
