# coding=utf-8
# Copyright 2021 The OneFlow Authors. All rights reserved.
# Copyright 2022 The HuggingFace Inc. team.
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

import oneflow.nn as nn

from libai.layers.layer_norm import LayerNorm
from libai.layers.mlp import MLP
from libai.utils import distributed as dist
from projects.GLM.layers.attention_layer import MultiheadAttention


class TransformerLayer(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        attention_dropout_prob=0.0,
        output_dropout_prob=0.0,
        layernorm_epsilon=1e-5,
        init_method=nn.init.xavier_normal_,
        output_layer_init_method=None,
        bias_gelu_fusion=False,
        bias_dropout_fusion=False,
        scale_mask_softmax_fusion=False,
        apply_query_key_layer_scaling=False,
        attention_scale=1.0,
        *,
        layer_idx=0
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_dropout_prob = attention_dropout_prob
        self.output_dropout_prob = output_dropout_prob
        self.layernorm_epsilon = layernorm_epsilon
        self.attention_scale = attention_scale

        self.layer_idx = layer_idx

        self.bias_gelu_fusion = bias_gelu_fusion
        self.bias_dropout_fusion = bias_dropout_fusion
        self.scale_mask_softmax_fusion = scale_mask_softmax_fusion
        self.apply_query_key_layer_scaling = apply_query_key_layer_scaling

        self.init_method = init_method
        if output_layer_init_method is None:
            output_layer_init_method = init_method
        self.output_layer_init_method = output_layer_init_method

        self.input_layernorm = LayerNorm(
            self.hidden_size, eps=self.layernorm_epsilon, layer_idx=self.layer_idx
        )

        self.attention = self.build_attention()
        self.post_attention_layernorm = LayerNorm(
            self.hidden_size, eps=self.layernorm_epsilon, layer_idx=self.layer_idx
        )

        self.mlp = MLP(
            self.hidden_size,
            4 * self.hidden_size,
            self.output_dropout_prob,
            self.init_method,
            output_layer_init_method=self.output_layer_init_method,
            bias_gelu_fusion=self.bias_gelu_fusion,
            bias_dropout_fusion=self.bias_dropout_fusion,
            layer_idx=self.layer_idx,
        )

    def forward(
        self,
        hidden_states,
        attention_mask,
        mem=None,
    ):
        hidden_states = hidden_states.to_global(placement=dist.get_layer_placement(self.layer_idx))
        attention_mask = (
            attention_mask.to_global(placement=dist.get_layer_placement(self.layer_idx))
            if attention_mask is not None
            else None
        )
        mem = (
            mem.to_global(placement=dist.get_layer_placement(self.layer_idx))
            if mem is not None
            else None
        )

        layernorm_output = self.input_layernorm(hidden_states)
        mem = self.input_layernorm(mem) if mem is not None else None
        attention_output = self.attention(
            layernorm_output,
            attention_mask=attention_mask,
            mem=mem,
        )

        hidden_states = hidden_states + attention_output

        layernorm_output = self.post_attention_layernorm(hidden_states)

        mlp_output = self.mlp(layernorm_output)

        output = hidden_states + mlp_output

        return output

    def build_attention(self):
        return MultiheadAttention(
            self.hidden_size,
            self.num_attention_heads,
            attention_dropout_prob=self.attention_dropout_prob,
            output_dropout_prob=self.output_dropout_prob,
            init_method=self.init_method,
            output_layer_init_method=self.output_layer_init_method,
            bias_dropout_fusion=self.bias_dropout_fusion,
            scale_mask_softmax_fusion=self.scale_mask_softmax_fusion,
            apply_query_key_layer_scaling=self.apply_query_key_layer_scaling,
            attention_scale=self.attention_scale,
            layer_idx=self.layer_idx,
        )
