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
import oneflow.nn.init as init

from libai.utils import distributed as dist
from .linear import Linear


class MultiheadAttention(nn.Module):
    """Multihead attention layer, support self attention and cross attention.

    Arguments:
        hidden_size: size of hidden state.
        num_attention_heads: number of attention heads.
        is_cross_attention: used to specify whether it is self attention or cross attention.
        attention_dropout_prob: dropout probability of attention weights.
        output_dropout_prob: dropout probability of output.
        init_method: method to initialize the input layer weights.
        output_layer_init_method: method to initialize the output layer weights. If None, use `init_method`.
        bias_dropout_fusion: whether to fuse add bias and dropout.
        scale_mask_softmax_fusion: whether to fuse scale, mask and softmax.
        layer_idx: A layer_idx sign which determines the placements. It will be used in pipeline parallelism. Defaults to 0.
    """
    def __init__(self, hidden_size, num_attention_heads, is_cross_attention=False,
                 attention_dropout_prob=0., output_dropout_prob=0., 
                 init_method=init.xavier_normal_, output_layer_init_method=None, 
                 bias_dropout_fusion=False, scale_mask_softmax_fusion=False, 
                 *, layer_idx=0):
        super().__init__()
        self.hidden_size = hidden_size
        if output_layer_init_method is None:
            output_layer_init_method = init_method

        assert hidden_size % num_attention_heads == 0, "hidden_size must be the multiply of num_attention_heads"

        self.num_heads = num_attention_heads
        self.head_size = hidden_size // num_attention_heads
        
        self.dropout = flow.nn.Dropout(p=attention_dropout_prob)
        self.norm_factor = 1.0 / math.sqrt(float(self.head_size))
        
        self.is_cross_attention = is_cross_attention
        self.scale_mask_softmax_fusion = scale_mask_softmax_fusion
        self.bias_dropout_fusion = bias_dropout_fusion

        if self.bias_dropout_fusion:
            self.output_dropout_prob = output_dropout_prob
        else:
            self.output_dropout = flow.nn.Dropout(p=output_dropout_prob)

        if self.is_cross_attention:
            self.query = Linear(self.hidden_size, self.hidden_size, parallel='col', 
                                init_method=init_method, 
                                layer_idx=layer_idx)
            self.key_value = Linear(self.hidden_size, self.hidden_size * 2, parallel='col', 
                                    init_method=init_method, 
                                    layer_idx=layer_idx)
        else:
            self.query_key_value = Linear(self.hidden_size, self.hidden_size * 3, parallel='col', 
                                          init_method=init_method, 
                                          layer_idx=layer_idx)

        self.dense = Linear(self.hidden_size, self.hidden_size, parallel='row', 
                            init_method=output_layer_init_method, 
                            skip_bias_add=self.bias_dropout_fusion,
                            layer_idx=layer_idx)

    def forward(self, hidden_states, encoder_states=None, attention_mask=None, past_key_value=None, use_cache=False):
        """ hidden_states: [tgt_len, bsz, hidden_size]. We adopted seq_len first setting for faster operation.
            encoder_states: [src_len, bsz, hidden_size].
            attention_mask: [bsz, 1, tgt_len, src_len], it should be the combination of padding mask and casual mask.
                            In case of self attention in encoder, it is the padding mask of source input.
                            In case of self attention in decoder, it is the combination of padding mask of target input and casual mask.
                            In case of cross attention in decoder, it is the padding mask of source input.
            past_key_value: tuple of key and value, each shape is [src_len, bsz, num_heads, head_size].
            use_cache: it will be set to True, when the model is in the inference phase and used for incremental decoding.
        """

        if encoder_states is not None:
            encoder_states = encoder_states.to_consistent(placement=hidden_states.placement)
        
        if attention_mask is not None:
            attention_mask = attention_mask.to_consistent(placement=hidden_states.placement)

        tgt_len, bsz = hidden_states.size()[:-2]

        if self.is_cross_attention:
            # if it is cross attention, key and value should be calculated only once, and the result can be reused.
            query = self.query(hidden_states)
            if past_key_value is not None:
                key, value = past_key_value
            elif encoder_states is not None:
                key_value = self.key_value(encoder_states)
                key_value = query_key_value.view(-1, bsz, self.num_heads, 2 * self.head_size)
                key, value = flow.split(key_value, 2, dim=-1)            # [src_len, bsz, num_heads, head_size]
            else:
                raise ValueError("past_key_value and encoder_states cannot be None at the same time.")
        else:
            # if it is self attention, query, key, and value are all obtained from hidden_states.
            # when in the inference phase of an incremental decoder, hidden_states is the last-added state, 
            # the full key and value could be obtained by concatenating with past_key_value.
            query_key_value = self.query_key_value(hidden_states)
            query_key_value = query_key_value.view(-1, bsz, self.num_heads, 3 * self.head_size)
            query, key, value = flow.split(query_key_value, 3, dim=-1)   # [tgt_len, bsz, num_heads, head_size]
            if past_key_value is not None:
                past_key, past_value = past_key_value
                key = flow.cat((past_key.type_as(key), key), dim=0)
                value = flow.cat((past_value.type_as(value), value), dim=0)
        
        if use_cache:
            past_key_value = (key, value)

        new_shape = (-1, bsz * self.num_heads, self.head_size)
        query = query.view(*new_shape).transpose(0, 1)  # [bsz * num_heads, tgt_len, head_size]
        key = key.view(*new_shape).transpose(0, 1)      # [bsz * num_heads, src_len, head_size]
        value = value.view(*new_shape).transpose(0, 1)  # [bsz * num_heads, src_len, head_size]

        attention_scores = flow.matmul(query, key, transpose_b=True, alpha=self.norm_factor)    # [bsz * num_heads, tgt_len, src_len]
        attention_scores = attention_scores.view(bsz, self.num_heads, tgt_len, -1)              # [bsz, num_heads, tgt_len, src_len]
        
        if attention_mask is not None:
            if self.scale_mask_softmax_fusion:
                attention_weights = flow._C.fused_scale_mask_softmax(attention_scores, attention_mask, fill_value=-10000.0)
            else:
                attention_scores = flow.mul(attention_scores, attention_mask) - 10000.0 * (1.0 - attention_mask)
                attention_weights = flow.softmax(attention_scores, dim=-1)
        else:
            attention_weights = flow.softmax(attention_scores, dim=-1)

        attention_weights = attention_weights.view(bsz * self.num_heads, tgt_len, -1)   # [bsz * num_heads, tgt_len, src_len]
        attention_weights = self.dropout(attention_weights)

        context = flow.matmul(attention_weights, value)                           # [bsz * num_heads, tgt_len, head_size]
        context = context.transpose(0, 1).view(tgt_len, bsz, self.hidden_size)    # [tgt_len, bsz, hidden_size]
        output = self.dense(context)

        if self.bias_dropout_fusion:
            output, bias = output
            output = flow._C.fused_bias_add_dropout(output, bias, p=self.output_dropout_prob, axis=output.ndim - 1)
        else:
            output = self.output_dropout(output)

        if use_cache:
            output = [output, past_key_value]

        return output

    def extra_repr(self) -> str:
        return "hidden_size={}, num_heads={}, is_cross_attention={}".format(
            self.hidden_size, self.num_heads, self.is_cross_attention,
        )

