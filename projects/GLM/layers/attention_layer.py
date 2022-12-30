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

import math

import oneflow as flow
from oneflow import nn

from libai.layers.linear import Linear


class MultiheadAttention(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        attention_dropout_prob=0.0,
        output_dropout_prob=0.0,
        init_method=nn.init.xavier_normal_,
        output_layer_init_method=None,
        bias_dropout_fusion=False,
        scale_mask_softmax_fusion=False,
        apply_query_key_layer_scaling=False,
        attention_scale=1.0,
        *,
        layer_idx=0
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.attention_scale = attention_scale
        if output_layer_init_method is None:
            output_layer_init_method = init_method

        assert (
            hidden_size % num_attention_heads == 0
        ), "hidden_size must be divisible by num_attention_heads."

        self.num_heads = num_attention_heads
        self.head_size = hidden_size // num_attention_heads

        self.attention_dropout_prob = attention_dropout_prob
        self.dropout = nn.Dropout(p=attention_dropout_prob)
        self.norm_factor = 1.0 / math.sqrt(float(self.head_size))
        self.coeff = None
        if apply_query_key_layer_scaling:
            self.coeff = layer_idx + 1
            self.norm_factor /= self.coeff

        self.scale_mask_softmax_fusion = scale_mask_softmax_fusion
        self.bias_dropout_fusion = bias_dropout_fusion

        if self.bias_dropout_fusion:
            self.output_dropout_prob = output_dropout_prob
        else:
            self.output_dropout = nn.Dropout(p=output_dropout_prob)

        self.query_key_value = Linear(
            self.hidden_size,
            self.hidden_size * 3,
            parallel="col",
            init_method=init_method,
            layer_idx=layer_idx,
        )

        self.dense = Linear(
            self.hidden_size,
            self.hidden_size,
            parallel="row",
            init_method=output_layer_init_method,
            skip_bias_add=self.bias_dropout_fusion,
            layer_idx=layer_idx,
        )

    def forward(
        self,
        hidden_states: flow.Tensor,
        attention_mask: flow.Tensor = None,
        mem=None,
    ):
        attention_mask = (
            attention_mask.to_global(placement=hidden_states.placement)
            if attention_mask is not None
            else None
        )

        bsz, tgt_len = hidden_states.size()[:2]

        if mem is not None:
            hidden_states = flow.cat((mem, hidden_states), dim=1)
        query_key_value = self.query_key_value(hidden_states)
        query_key_value = query_key_value.view(bsz, -1, self.num_heads, 3 * self.head_size)
        query_key_value = query_key_value.permute(0, 2, 1, 3)
        query, key, value = flow.chunk(query_key_value, chunks=3, dim=-1)
        if mem is not None:
            query = query[:, :, -tgt_len:]

        if self.attention_scale > 1.0:
            attention_scores = flow.matmul(
                query / math.sqrt(self.attention_scale),
                key / math.sqrt(self.head_size * self.attention_scale),
                transpose_b=True,
            )
        else:
            attention_scores = flow.matmul(query, key, transpose_b=True, alpha=self.norm_factor)

        if self.scale_mask_softmax_fusion:
            attention_weights = flow._C.fused_scale_mask_softmax_dropout(
                attention_scores,
                attention_mask,
                fill_value=-10000.0,
                scale=self.coeff,
                p=self.attention_dropout_prob,
            )[0]
        else:
            if self.coeff is not None:
                attention_scores *= self.coeff
            attention_scores = flow.mul(attention_scores, attention_mask)
            attention_scores = attention_scores - 10000.0 * (1 - attention_mask)
            attention_weights = flow.softmax(attention_scores, dim=-1)
            attention_weights = self.dropout(attention_weights)

        context = flow.matmul(attention_weights, value)
        context = context.transpose(1, 2)
        output = self.dense(context.flatten(2))

        if self.bias_dropout_fusion:
            output, bias = output
            output = flow._C.fused_bias_add_dropout(
                output, bias, p=self.output_dropout_prob, axis=output.ndim - 1
            )
        else:
            output = self.output_dropout(output)

        return output

    def extra_repr(self) -> str:
        return "hidden_size={}, num_heads={}".format(
            self.hidden_size,
            self.num_heads,
        )
