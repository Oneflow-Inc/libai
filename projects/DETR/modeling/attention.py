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
from typing import Tuple

import oneflow as flow
from oneflow import nn

from libai.layers.attention import MultiheadAttention
from libai.layers.linear import Linear


class DetrMultiheadAttention(MultiheadAttention):
    """Multi-head attention layer, support self attention and cross attention.

    Args:
        hidden_size: size of hidden state.
        num_attention_heads: number of attention heads.
        attention_dropout_prob: dropout probability of attention weights.
            Defaults to 0.0.
        output_dropout_prob: dropout probability of output. Defaults to 0.0.
        init_method: method to initialize the input layer weights.
            Defaults to ``init.xavier_normal_``.
        output_layer_init_method: method to initialize the output layer weights.
            If None, use ``init_method``.
        bias_dropout_fusion: whether to fuse add bias and dropout.
            Defaults to False.
        scale_mask_softmax_fusion: whether to fuse scale, mask and softmax.
            Defaults to False.
        apply_query_key_layer_scaling: if `True`, scaling the attention score by layer index.
            Defaults to False.
        layer_idx: A layer_idx sign which determines the placements.
            It will be used in pipeline parallelism. Defaults to 0.
    """

    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        output_dropout_prob=0.0,
        attention_dropout_prob=0.0
    ):
        super().__init__(hidden_size=hidden_size, 
                         num_attention_heads=num_attention_heads, 
                         output_dropout_prob=output_dropout_prob,
                         attention_dropout_prob=attention_dropout_prob)

    def forward(
        self,
        hidden_states: flow.Tensor,
        encoder_states: flow.Tensor = None,
        attention_mask: flow.Tensor = None,
        past_key_value: Tuple[flow.Tensor, flow.Tensor] = None,
        use_cache: bool = False,
        key_padding_mask: flow.Tensor = None
    ):
        """

        Args:
            hidden_states (flow.Tensor): shape is [bsz, tgt_len, hidden_size].
            encoder_states (flow.Tensor, optional): shape is [bsz, src_len, hidden_size].
                Defaults to None.
            attention_mask (flow.Tensor, optional): shape is [bsz, 1, tgt_len, src_len].
                It should be the combination of padding mask and casual mask.
                It is the padding mask of source input when used with self-attention in encoder.
                And it is the combination of padding mask of target input and casual mask when
                used with self-attention in decoder. It is the padding mask of source input when
                used with cross-attention in decoder.
                Defaults to None.
            past_key_value (Tuple[flow.Tensor, flow.Tensor], optional): tuple of key and value,
                each shape is [bsz, num_heads, src_len, head_size]. Defaults to None.
            use_cache (bool, optional): it will be set to True, when the model is in the inference
                phase and used for incremental decoding. Defaults to False.
        """

        # hidden_states, encoder_states: [S(0), B]
        # attention_mask: [S(0), B]

        if encoder_states is not None:
            encoder_states = encoder_states.to_global(placement=hidden_states.placement)

        if attention_mask is not None:
            attention_mask = attention_mask.to_global(placement=hidden_states.placement)
            
        # *NOTE: for detr MultiHeadAttention
        query, key, value = hidden_states
        query, key, value = query.permute(1,0,2), key.permute(1,0,2), value.permute(1,0,2)
        
        bsz, tgt_len = query.size()[:2]

        query_key_value = self.query_key_value(hidden_states)
        query_key_value = query_key_value.view(bsz, -1, self.num_heads, 3 * self.head_size)
        query_key_value = query_key_value.permute(
            0, 2, 1, 3
        )  # [bsz, num_heads, src_len, 3 * head_size]
        query, key, value = flow.chunk(query_key_value, chunks=3, dim=-1)
            
        # [bsz, num_heads, tgt_len, src_len] with [S(0), S(1)]
        attention_scores = flow.matmul(query, key, transpose_b=True, alpha=self.norm_factor)
        # [S(0), S(1)] x [S(0), B] = [S(0), S(1)]
        if attention_mask is not None:
            
            # * detr needs key_padding_mask
            if key_padding_mask is not None:
                attention_mask = attention_mask.masked_fill(key_padding_mask, float("-inf"))
                
            if self.scale_mask_softmax_fusion:
                attention_weights = flow._C.fused_scale_mask_softmax(
                    attention_scores, attention_mask, fill_value=-10000.0
                )
            else:
                if self.coeff is not None:
                    attention_scores *= self.coeff
                attention_scores = flow.mul(attention_scores, attention_mask)
                attention_scores = attention_scores - 10000.0 * (1 - attention_mask)
                # TODO(l1aoxingyu): graph will occur `where_scalar` errors when using `masked_fill`
                # attention_scores = attention_scores.masked_fill(1 - attention_mask, -10000.0)

                attention_weights = flow.softmax(attention_scores, dim=-1)
        else:
       
            attention_weights = flow.softmax(attention_scores, dim=-1)

        # [bsz, num_heads, tgt_len, src_len]
        attention_weights = self.dropout(attention_weights)

        # Context shape: [bsz, num_heads, tgt_len, head_size] with [S(0), S(1)]
        context = flow.matmul(attention_weights, value)
        # Change shape: [bsz, num_heads, tgt_len, head_size] -> [bsz, tgt_len, num_heads, head_size]
        context = context.transpose(1, 2)

        # Concat multi-head results from
        # [bsz, tgt_len, num_heads, head_size] -> [bsz, tgt_len, num_heads * head_size]
        # SBP sign: [S(0), S(2)]
        context = context.view(bsz, tgt_len, self.hidden_size)

        # [S(0), S(2)] x [B, S(0)] = [S(0), P] -> [S(0), B]
        output = self.dense(context)

        if self.bias_dropout_fusion:
            output, bias = output
            output = flow._C.fused_bias_add_dropout(
                output, bias, p=self.output_dropout_prob, axis=output.ndim - 1
            )
        else:
            output = self.output_dropout(output)

        if use_cache:
            output = (output, past_key_value)

        return output

    def extra_repr(self) -> str:
        return "hidden_size={}, num_heads={}, is_cross_attention={}".format(
            self.hidden_size,
            self.num_heads,
            self.is_cross_attention,
        )
