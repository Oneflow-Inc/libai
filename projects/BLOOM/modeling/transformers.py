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

from oneflow import nn

from libai.layers import LayerNorm
from libai.utils import distributed as dist
from projects.BLOOM.modeling.attention import BloomAttention
from projects.BLOOM.modeling.mlp import BloomMLP


class BloomBlock(nn.Module):
    def __init__(
        self,
        hidden_size,
        n_head,
        layer_norm_epsilon,
        hidden_dropout,
        attention_dropout,
        pretraining_tp,
        slow_but_exact,
        init_method,
        output_layer_init_method,
        apply_residual_connection_post_layernorm,
        layer_idx=0,
    ):
        super().__init__()
        hidden_size = hidden_size

        self.input_layernorm = LayerNorm(hidden_size, eps=layer_norm_epsilon, layer_idx=layer_idx)
        self.num_heads = n_head
        self.self_attention = BloomAttention(
            hidden_size=hidden_size,
            n_head=n_head,
            hidden_dropout=hidden_dropout,
            attention_dropout=attention_dropout,
            pretraining_tp=pretraining_tp,
            slow_but_exact=slow_but_exact,
            init_method=init_method,
            output_layer_init_method=output_layer_init_method,
            layer_idx=layer_idx,
        )
        self.post_attention_layernorm = LayerNorm(
            hidden_size, eps=layer_norm_epsilon, layer_idx=layer_idx
        )

        self.mlp = BloomMLP(
            hidden_size,
            pretraining_tp,
            slow_but_exact,
            hidden_dropout,
            init_method,
            output_layer_init_method,
            layer_idx,
        )

        self.apply_residual_connection_post_layernorm = apply_residual_connection_post_layernorm
        self.hidden_dropout = hidden_dropout
        self.layer_idx = layer_idx

    def forward(
        self,
        hidden_states,
        alibi,
        attention_mask,
        layer_past=None,
        head_mask=None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ):
        # Change placement for pipeline parallelsim
        hidden_states = hidden_states.to_global(placement=dist.get_layer_placement(self.layer_idx))

        alibi = alibi.to_global(placement=dist.get_layer_placement(self.layer_idx))

        # hidden_states shape: (batch_size, seq_length, hidden_size)
        if attention_mask is not None:
            attention_mask = attention_mask.to_global(
                placement=dist.get_layer_placement(self.layer_idx)
            )

        layernorm_output = self.input_layernorm(hidden_states)

        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = hidden_states

        # Self attention.
        attn_outputs = self.self_attention(
            layernorm_output,
            residual,
            layer_past=layer_past,
            attention_mask=attention_mask,
            alibi=alibi,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )

        attention_output = attn_outputs[0]

        outputs = attn_outputs[1:]

        layernorm_output = self.post_attention_layernorm(attention_output)

        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = attention_output

        # MLP.
        output = self.mlp(layernorm_output, residual)

        if use_cache:
            outputs = (output,) + outputs
        else:
            outputs = (output,) + outputs[1:]

        return outputs  # hidden_states, present, attentions
