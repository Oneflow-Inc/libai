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

# reference: https://github.com/pytorch/pytorch/blob/master/torch/nn/functional.py

import oneflow as flow
import oneflow.nn.functional as F

from libai.layers.attention import MultiheadAttention


class DetrMultiheadAttention(MultiheadAttention):
    """Multi-head attention layer, support self attention and cross attention.

    Args:
        hidden_size: size of hidden state.
        num_attention_heads: number of attention heads.
        attention_dropout_prob: dropout probability of attention weights.
            Defaults to 0.0.
    """

    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        attention_dropout_prob=0.0
    ):
        super().__init__(hidden_size=hidden_size, 
                         num_attention_heads=num_attention_heads, 
                         attention_dropout_prob=attention_dropout_prob)

        self.num_attention_heads = num_attention_heads
        
    def forward(
        self,
        hidden_states: flow.Tensor,
        attention_mask: flow.Tensor = None,
        key_padding_mask: flow.Tensor = None
    ):
        """

        Args:
            hidden_states (flow.Tensor): shape is [bsz, tgt_len, hidden_size].
            attention_mask (flow.Tensor, optional): shape is [bsz, 1, tgt_len, src_len].
                It should be the combination of padding mask and casual mask.
                It is the padding mask of source input when used with self-attention in encoder.
                And it is the combination of padding mask of target input and casual mask when
                used with self-attention in decoder. It is the padding mask of source input when
                used with cross-attention in decoder.
                Defaults to None.
        """
                
        query, key, value = hidden_states

        # refer to torch.nn.MultiHeadAttention
        tgt_len, bsz = query.shape[:2]
        src_len = key.shape[0]
        
        query_w, key_w, value_w = self.query_key_value.weight.chunk(3, dim=0)
        query_b, key_b, value_b = self.query_key_value.bias.chunk(3, dim=0)
     
        query = self.linear(query, query_w, query_b) 
        key = self.linear(key, key_w, key_b) 
        value = self.linear(value, value_w, value_b)  
        
        # Reshape q, k, v for multihead attention and make them batch first.
        query = query.contiguous().view(tgt_len, bsz * self.num_attention_heads, self.head_size).transpose(0, 1)
        key = key.contiguous().view(key.shape[0], bsz * self.num_attention_heads, self.head_size).transpose(0, 1)
        value = value.contiguous().view(value.shape[0], bsz * self.num_attention_heads, self.head_size).transpose(0, 1)

        # merge key padding and attention masks
        if key_padding_mask is not None:
            assert key_padding_mask.shape == (bsz, src_len), \
                f"expecting key_padding_mask shape of {(bsz, src_len)}, but got {key_padding_mask.shape}"            
            key_padding_mask = key_padding_mask.view(bsz, 1, 1, src_len).expand(-1, self.num_attention_heads, -1, -1).reshape(bsz * self.num_attention_heads, 1, src_len)

            if attention_mask is None:
                attention_mask = key_padding_mask
            elif attention_mask.dtype == flow.bool:
                attention_mask = attention_mask.logical_or(key_padding_mask)
            else:
                attention_mask = attention_mask.masked_fill(key_padding_mask, float("-inf"))
                
                
        attention_scores = flow.bmm(query*self.norm_factor, key.transpose(-2,-1))
                
        # convert mask to float
        if attention_mask is not None:
            if attention_mask.dtype == flow.bool:
                new_attention_mask = flow.zeros_like(attention_mask).to(dtype=query.dtype)
                new_attention_mask = new_attention_mask.masked_fill(attention_mask, float("-inf"))
                attention_mask = new_attention_mask
            attention_scores = attention_scores + attention_mask
            # NOTE: The += op leads error sbp signature
            # attention_scores += attention_mask

        attention_weights = flow.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        context = flow.bmm(attention_weights, value)
        
        # Change shape: [bsz*num_heads, tgt_len, head_size] -> [tgt_len, bsz*num_heads, head_size] -> [tgt_len, bsz, embed_dim]
        context = context.transpose(0,1).contiguous().view(tgt_len, bsz, self.hidden_size)
        output = self.dense(context)
        return output
    
    def linear(self, x, w, b):
        
        return F.linear(x, weight=w, bias=b)
        