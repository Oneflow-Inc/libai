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


from oneflow import nn
import oneflow as flow
from typing import Optional


class ScaledDotProduct(nn.Module):
    def __init__(
        self,
        norm_factor: float,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm_factor = norm_factor
        self.attn_drop = nn.Dropout(dropout)

    def forward(
        self,
        query: flow.Tensor,
        key: flow.Tensor,
        value: flow.Tensor,
        attention_mask: Optional[flow.Tensor] = None,
    ) -> flow.Tensor:
        bsz = query.size(0)
        tgt_len = query.size(1)

        # Raw attention scores with [bsz*num_heads, tgt_len, src_len]
        attention_scores = flow.matmul(query, key, transpose_b=True, alpha=self.norm_factor)
        attention_scores = attention_scores.view(
            bsz, self.num_heads, tgt_len, -1)  # [bsz, num_heads, tgt_len, src_len]

        if attention_mask is not None:
            if self.scale_mask_softmax_fusion:
                attention_weights = flow._C.fused_scale_mask_softmax(
                    attention_scores, attention_mask, fill_value=-10000.0)
            else:
                attention_scores = flow.mul(attention_scores, attention_mask) - 10000.0 * (1.0 - attention_mask)
                attention_weights = flow.softmax(attention_scores, dim=-1)
        else:
            attention_weights = flow.softmax(attention_scores, dim=-1)

        attention_weights = attention_weights.view(
            bsz * self.num_heads, tgt_len, -1)  # [bsz * num_heads, tgt_len, src_len]

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_weights = self.attn_drop(attention_weights)

        # Attention results with [bsz * num_heads, tgt_len, head_size]
        context = flow.matmul(attention_weights, value)

        # Change view from [bsz * num_heads, tgt_len, head_size] -> # [tgt_len, bsz, hidden_size]
        context = context.transpose(0, 1).view(tgt_len, bsz, self.hidden_size)

        return context
