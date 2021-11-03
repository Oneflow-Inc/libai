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
from core.utils import init_method_normal, scaled_init_method_normal

from .linear import ColumnParallelLinear, RowParallelLinear


class SelfAttention(flow.nn.Module):
    """Parallel self-attention layer.

    Arguments:
        layer_idx: the layer index, which determines the placement.
        hidden_size: size of hidden state.
        num_attention_heads: number of attention heads.
        attention_dropout_prob: dropout probability of attention weights.
        output_dropout_prob: dropout probability of output.
        init_method: method to initialize the input layer weights.
        output_layer_init_method: method to initialize the output layer weights. If None, use `init_method`.
        apply_query_key_layer_scaling: if `true`, scaling the attention score by layer index.
    """
    def __init__(self, layer_idx, hidden_size, num_attention_heads, 
                 attention_dropout_prob=0., output_dropout_prob=0., 
                 init_method=init.xavier_normal_, output_layer_init_method=None,
                 apply_query_key_layer_scaling=False):
        super().__init__()
        self.hidden_size = hidden_size
        if output_layer_init_method is None:
            output_layer_init_method = init_method

        self.num_heads = num_attention_heads
        self.head_size = hidden_size // num_attention_heads
        self.attention_dropout_prob = attention_dropout_prob
        
        self.dropout = flow.nn.Dropout(p=self.attention_dropout_prob)

        self.norm_factor = 1.0 / math.sqrt(float(self.head_size))
        self.coeff = 1.0
        if apply_query_key_layer_scaling:
            self.coeff = float(layer_idx + 1)
            self.norm_factor /= self.coeff

        self.query_key_value = ColumnParallelLinear(layer_idx, self.hidden_size, self.hidden_size * 3, init_method)
        self.dense = RowParallelLinear(layer_idx, self.hidden_size, self.hidden_size, 
                                       init_method=output_layer_init_method, 
                                       output_dropout_prob=output_dropout_prob, 
                                       bias_dropout_fusion=bias_dropout_fusion)

    def _transpose_for_scores(self, tensor):
        """Convert a 3D tensor [b, s, nh * hs] into a 4D tensor [b, nh, s, hs].
        """
        bsz, seq_len = hidden_states.size()[:2]
        tensor = tensor.view(bsz, seq_len, self.num_heads, self.head_size)
        return tensor.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None, layer_past=None, use_cache=False):
        """ hidden_states: shape: [batch_size, seq_len, hidden_size].
            attention_mask: padding mask or ltor mask, [1, 1, seq_len, seq_len].
        """
        query_key_value = self.query_key_value(hidden_states)
        query, key, value = flow.chunk(query_key_value, chunks=3, dim=2)

        query = self._transpose_for_scores(query)   # [batch_size, num_heads, seq_len, head_size]
        key = self._transpose_for_scores(key)
        value = self._transpose_for_scores(value)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = flow.cat((past_key.type_as(key), key), dim=-2)
            value = flow.cat((past_value.type_as(value), value), dim=-2)
        present = (key, value)

        # [batch_size, num_heads, seq_len, seq_len]
        attention_scores = flow.matmul(query, key, transpose_b=True, alpha=self.norm_factor)

        if use_cache and attention_mask is not None:
            with flow.no_grad():
                source_length = attention_scores.size(3)
                if layer_past is not None:
                    attention_mask = attention_mask[..., source_length - 1, :source_length].unsqueeze(2)
                else:
                    attention_mask = attention_mask[..., :source_length, :source_length]
    
        if attention_mask is not None:
            attention_scores = flow.mul(attention_scores, attention_mask) - 10000.0 * (1.0 - attention_mask)
        
        attention_weights = flow.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # [batch_size, num_heads, seq_len, head_size]
        context = flow.matmul(attention_weights, value)

        # [batch_size, seq_len, num_heads, head_size]
        context = context.permute(0, 2, 1, 3)
        bsz, seq_len = context.size()[:-2]
        context = context.view(bsz, seq_len, -1)
        output = self.dense(context)

        if use_cache:
            output = [output, present]

        return output


class CrossAttention(flow.nn.Module):
    """Parallel cross-attention layer.

    Arguments:
        layer_idx: the layer index, which determines the placement.
        hidden_size: size of hidden state.
        num_attention_heads: number of attention heads.
        attention_dropout_prob: dropout probability of attention weights.
        output_dropout_prob: dropout probability of output.
        init_method: method to initialize the input layer weights.
        output_layer_init_method: method to initialize the output layer weights. If None, use `init_method`.
        apply_query_key_layer_scaling: if `true`, scaling the attention score by layer index.
    """
    def __init__(self, layer_idx, hidden_size, num_attention_heads, 
                 attention_dropout_prob=0., output_dropout_prob=0., 
                 init_method=init.xavier_normal_, output_layer_init_method=None,
                 apply_query_key_layer_scaling=False):
        super().__init__()
        self.hidden_size = hidden_size
        if output_layer_init_method is None:
            output_layer_init_method = init_method

        self.num_heads = num_attention_heads
        self.head_size = hidden_size // num_attention_heads
        self.attention_dropout_prob = attention_dropout_prob
        
        self.dropout = flow.nn.Dropout(p=self.attention_dropout_prob)

        self.norm_factor = 1.0 / math.sqrt(float(self.head_size))
        self.coeff = 1.0
        if apply_query_key_layer_scaling:
            self.coeff = float(layer_idx + 1)
            self.norm_factor /= self.coeff

        self.query = ColumnParallelLinear(layer_idx, self.hidden_size, self.hidden_size, init_method=init_method)
        self.key_value = ColumnParallelLinear(layer_idx, self.hidden_size, self.hidden_size * 2, init_method=init_method)
        self.dense = RowParallelLinear(layer_idx, self.hidden_size, self.hidden_size, 
                                       init_method=output_layer_init_method, 
                                       output_dropout_prob=output_dropout_prob, 
                                       bias_dropout_fusion=bias_dropout_fusion)

    def _transpose_for_scores(self, tensor):
        """Convert a 3D tensor [b, s, nh * hs] into a 4D tensor [b, nh, s, hs].
        """
        bsz, seq_len = hidden_states.size()[:2]
        tensor = tensor.view(bsz, seq_len, self.num_heads, self.head_size)
        return tensor.permute(0, 2, 1, 3)

    def forward(self, hidden_states, encoder_states, attention_mask=None):
        # hidden_states shape: (batch_size, seq_len, hidden_size)
        # sbp: [S(0), B]
        query = self.query(hidden_states)
        key_value = self.key_value(encoder_states)
        key, value = flow.chunk(key_value, chunks=2, dim=2)

        query = self._transpose_for_scores(query)
        key = self._transpose_for_scores(key)
        value = self._transpose_for_scores(value)

        attention_scores = flow.matmul(query, key, transpose_b=True, alpha=self.norm_factor)

        if attention_mask is not None:
            attention_scores = flow.mul(attention_scores, attention_mask) - 10000.0 * (1.0 - attention_mask)
        
        attention_weights = flow.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        context = flow.matmul(attention_weights, value)

        # (batch_size, num_heads, seq_len, head_size) -> (batch_size, seq_len, num_heads, head_size)
        context = context.permute(0, 2, 1, 3)
        bsz, seq_len = context.size()[:-2]
        context = context.view(bsz, seq_len, -1)
        output = self.dense(context)

        return output

