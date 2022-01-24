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

import oneflow.nn as nn

from libai.utils import distributed as dist

from .attention import MultiheadAttention
from .layer_norm import LayerNorm
from .mlp import MLP


class TransformerLayer(nn.Module):
    """A single transformer layer.

    Transformer layer takes input with size [bsz, seq_length, hidden size] and returns an
    output of the same size.
    The input and output has same sbp sign, (S(0), B).

    Arguments:
        hidden_size: size of hidden state.
        ffn_hidden_size: size of feed forword neural network.
        num_attention_heads: number of attention heads.
        is_decoder: used to specify whether this is transformer encoder layer or transformer
        decoder layer. Default: ``False``.
        attention_dropout_prob: dropout probability of attention weights.
        output_dropout_prob: dropout probability of output.
        layernorm_epsilon: epsilon used in layernorm layer. Default: `1e-5`.
        init_method: method to initialize the input layer weights.
        output_layer_init_method: method to initialize the output layer weights.
        If None, use `init_method`.
        bias_gelu_fusion: whether fuse add bias and gelu. Default: ``False``.
        bias_dropout_fusion: whether fuse add bias and dropout. Default: ``False``.
        scale_mask_softmax_fusion: whether to fuse scale, mask and softmax. Default: ``False``.
        apply_query_key_layer_scaling: if `true`, scaling the attention score by layer index.
        Default: ``False``.
        layer_idx: the layer index, which determines the placement.
    """

    def __init__(
        self,
        hidden_size,
        ffn_hidden_size,
        num_attention_heads,
        is_decoder=False,
        attention_dropout_prob=0.0,
        output_dropout_prob=0.0,
        layernorm_epsilon=1e-5,
        init_method=nn.init.xavier_normal_,
        output_layer_init_method=None,
        bias_gelu_fusion=False,
        bias_dropout_fusion=False,
        scale_mask_softmax_fusion=False,
        apply_query_key_layer_scaling=False,
        *,
        layer_idx=0
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.ffn_hidden_size = ffn_hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_dropout_prob = attention_dropout_prob
        self.output_dropout_prob = output_dropout_prob
        self.layernorm_epsilon = layernorm_epsilon

        self.layer_idx = layer_idx
        self.is_decoder = is_decoder

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

        self.self_attention = self.build_attention(is_cross_attention=False)
        self.post_attention_layernorm = LayerNorm(
            self.hidden_size, eps=self.layernorm_epsilon, layer_idx=self.layer_idx
        )

        if self.is_decoder:
            self.cross_attention = self.build_attention(is_cross_attention=True)
            self.post_cross_attention_layernorm = LayerNorm(
                self.hidden_size, eps=self.layernorm_epsilon, layer_idx=self.layer_idx
            )

        self.mlp = MLP(
            self.hidden_size,
            self.ffn_hidden_size,
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
        encoder_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        use_cache=False,
    ):
        """
        hidden_states: [bsz, seq_length, hidden_size], (S(0), B),
        attention_mask: [bsz, 1, seq_length, seq_length], (S(0), B), the combination of key
        padding mask and casual mask of hidden states.
        encoder_states: [bsz, seq_length, hidden_size], (S(0), B), encoder output, this will be
        used in cross attention.
        encoder_attention_mask: [bsz, 1, seq_length, seq_length], (S(0), B) key padding mask of
        encoder states.
        past_key_value: tuple of key and value, each shape is [src_len, bsz, num_heads, head_size].
        For decoder layer, the past_key_value contains the states both from self attention
        and cross attention.
        use_cache: it will be set to `True`, when the model is in the inference phase and
        used for incremental decoding.
        """
        # Change placement for pipeline parallelsim
        hidden_states = hidden_states.to_consistent(
            placement=dist.get_layer_placement(self.layer_idx)
        )

        # hidden_states shape: (batch_size, seq_length, hidden_size)
        attention_mask = attention_mask.to_consistent(
            placement=dist.get_layer_placement(self.layer_idx)
        )

        if past_key_value is not None:
            if self.is_decoder:
                assert len(past_key_value) == 4
                self_attn_past_key_value = past_key_value[:2]
                cross_attn_past_key_value = past_key_value[2:]
            else:
                self_attn_past_key_value = past_key_value
                cross_attn_past_key_value = None
        else:
            self_attn_past_key_value, cross_attn_past_key_value = None, None

        layernorm_output = self.input_layernorm(hidden_states)
        # todo: use key-value to pass the arguments
        attention_output = self.self_attention(
            layernorm_output, None, attention_mask, self_attn_past_key_value, use_cache
        )
        #    attention_mask=attention_mask,
        #    past_key_value=self_attn_past_key_value,
        #    use_cache=use_cache)

        if use_cache:
            attention_output, presents = attention_output
        hidden_states = hidden_states + attention_output

        layernorm_output = self.post_attention_layernorm(hidden_states)

        if self.is_decoder:
            # todo: use key-value to pass the arguments
            attention_output = self.cross_attention(
                layernorm_output,
                encoder_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                use_cache,
            )
            # attention_mask=encoder_attention_mask,
            # past_key_value=cross_attn_past_key_value,
            # use_cache=use_cache)

            if use_cache:
                attention_output, decoder_presents = attention_output
                presents += decoder_presents

            hidden_states = hidden_states + attention_output
            layernorm_output = self.post_cross_attention_layernorm(hidden_states)

        mlp_output = self.mlp(layernorm_output)
        output = hidden_states + mlp_output

        if use_cache:
            output = (output, presents)
        return output

    def build_attention(self, is_cross_attention=False):
        return MultiheadAttention(
            self.hidden_size,
            self.num_attention_heads,
            is_cross_attention=is_cross_attention,
            attention_dropout_prob=self.attention_dropout_prob,
            output_dropout_prob=self.output_dropout_prob,
            init_method=self.init_method,
            output_layer_init_method=self.output_layer_init_method,
            bias_dropout_fusion=self.bias_dropout_fusion,
            scale_mask_softmax_fusion=self.scale_mask_softmax_fusion,
            apply_query_key_layer_scaling=self.apply_query_key_layer_scaling,
            layer_idx=self.layer_idx,
        )
