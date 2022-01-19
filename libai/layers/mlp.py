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


class MLP(nn.Module):
    """MLP

    MLP will take the input with h hidden state, project it to intermediate
    hidden dimension, perform gelu transformation, and project the
    state back into h hidden dimension.

    Arguments:
        hidden_size: size of each input and output sample.
        ffn_hidden_size: size of each intermediate sample.
        output_dropout_prob: Output dropout probability. Defaults to 0.0.
        init_method: method to initialize the first linear weight.
        Defaults to nn.init.xavier_normal_.
        output_layer_init_method: method to initialize the second linear weight. If set to None,
        it will use ``init_method`` instead. Defaults to None.
        bias_gelu_fusion: If set to ``True``, it will fuse bias adding and elementwise
        gelu activation. Defaults to ``False``.
        bias_dropout_fusion: If set to ``True``, it will fuse bias adding and dropout.
        Defaults to ``False``.
        layer_idx: A layer_idx sign which determines the placement. It will be used in
        pipeline parallelism. Defaults to 0.
    """

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
            bias=True,
            parallel="col",
            skip_bias_add=bias_gelu_fusion,
            init_method=init_method,
            layer_idx=layer_idx,
        )

        if not bias_gelu_fusion:
            self.activation_func = build_activation("gelu")

        self.dense_4h_to_h = Linear(
            ffn_hidden_size,
            hidden_size,
            bias=True,
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
