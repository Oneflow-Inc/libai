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

import math

import oneflow as flow
from oneflow import nn
from oneflow.nn import functional as F

from libai.layers import Linear


def dropout_add(x, residual, prob, training):
    """
    Dropout add function

    Args:
        x (`torch.tensor`, *required*):
            input tensor
        residual (`torch.tensor`, *required*):
            esidual tensor
        prob (`float`, *required*):
            dropout probability
        training (`bool`, *required*):
            training mode
    """
    out = F.dropout(x, p=prob, training=training)
    out = residual + out
    return out


class BloomAttention(nn.Module):
    def __init__(
        self,
        hidden_size,
        n_head,
        hidden_dropout,
        attention_dropout,
        pretraining_tp,
        slow_but_exact,
        init_method=nn.init.xavier_normal_,
        output_layer_init_method=None,
        layer_idx=0,
    ):
        super().__init__()
        self.pretraining_tp = pretraining_tp
        self.slow_but_exact = slow_but_exact
        self.hidden_size = hidden_size
        self.num_heads = n_head
        self.head_dim = self.hidden_size // self.num_heads
        self.split_size = self.hidden_size
        self.hidden_dropout = hidden_dropout
        if output_layer_init_method is None:
            output_layer_init_method = init_method

        if self.head_dim * self.num_heads != self.hidden_size:
            raise ValueError(
                f"`hidden_size` must be divisible by num_heads "
                f"(got `hidden_size`: {self.hidden_size} and `num_heads`:"
                f" {self.num_heads})."
            )

        # Layer-wise attention scaling
        self.inv_norm_factor = 1.0 / math.sqrt(self.head_dim)
        self.beta = 1.0

        self.query_key_value = Linear(
            self.hidden_size,
            3 * self.hidden_size,
            bias=True,
            parallel="col",
            init_method=init_method,
            layer_idx=layer_idx,
        )
        self.dense = Linear(
            self.hidden_size,
            self.hidden_size,
            parallel="row",
            init_method=output_layer_init_method,
            layer_idx=layer_idx,
        )
        self.attention_dropout = nn.Dropout(attention_dropout)

    def _split_heads(self, fused_qkv):
        """
        Split the last dimension into (num_heads, head_dim) without making any copies, results share
        same memory storage as `fused_qkv`

        Args:
            fused_qkv (`torch.tensor`, *required*):
                [batch_size, seq_length, num_heads * 3 * head_dim]

        Returns:
            query: [batch_size, seq_length, num_heads, head_dim]
            key: [batch_size, seq_length, num_heads, head_dim]
            value: [batch_size, seq_length, num_heads, head_dim]
        """
        batch_size, seq_length, three_times_hidden_size = fused_qkv.shape
        fused_qkv = fused_qkv.view(batch_size, seq_length, self.num_heads, 3, self.head_dim)
        return fused_qkv[..., 0, :], fused_qkv[..., 1, :], fused_qkv[..., 2, :]

    def _merge_heads(self, x):
        """
        Merge heads together over the last dimenstion

        Args:
            x: (`torch.tensor`, *required*): [batch_size * num_heads, seq_length, head_dim]

        Returns:
            torch.tensor: [batch_size, seq_length, num_heads * head_dim]
        """
        # What we want to achieve is:
        # batch_size * num_heads, seq_len, head_dim -> batch_size, seq_len, num_heads * head_dim
        batch_size_and_num_heads, seq_length, _ = x.shape
        batch_size = batch_size_and_num_heads // self.num_heads

        # First view to decompose the batch size
        # batch_size * num_heads, seq_len, head_dim -> batch_size, num_heads, seq_len, head_dim
        x = x.view(batch_size, self.num_heads, seq_length, self.head_dim)

        # batch_size, num_heads, seq_length, head_dim -> batch_size, seq_length, num_heads, head_dim
        x = x.permute(0, 2, 1, 3)

        # batch_size, seq_len, num_heads, head_dim -> batch_size, seq_len, num_heads * head_dim
        return x.reshape(batch_size, seq_length, self.num_heads * self.head_dim)

    def forward(
        self,
        hidden_states,
        residual,
        alibi,
        attention_mask,
        layer_past=None,
        head_mask=None,
        use_cache=False,
        output_attentions=False,
    ):
        fused_qkv = self.query_key_value(hidden_states)  # [batch_size, seq_length, 3 x hidden_size]

        # 3 x [batch_size, seq_length, num_heads, head_dim]
        (query_layer, key_layer, value_layer) = self._split_heads(fused_qkv)

        batch_size, q_length, _, _ = query_layer.shape

        query_layer = query_layer.transpose(1, 2).reshape(
            batch_size * self.num_heads, q_length, self.head_dim
        )
        key_layer = key_layer.permute(0, 2, 3, 1).reshape(
            batch_size * self.num_heads, self.head_dim, q_length
        )
        value_layer = value_layer.transpose(1, 2).reshape(
            batch_size * self.num_heads, q_length, self.head_dim
        )
        if layer_past is not None:
            past_key, past_value = layer_past
            key_layer = flow.cat((past_key, key_layer), dim=2)
            value_layer = flow.cat((past_value, value_layer), dim=1)

        _, _, kv_length = key_layer.shape

        if use_cache is True:
            present = (key_layer, value_layer)
        else:
            present = None

        matmul_result = flow.baddbmm(
            alibi,
            batch1=query_layer,
            batch2=key_layer,
            beta=self.beta,
            alpha=self.inv_norm_factor,
        )

        attention_scores = matmul_result.view(batch_size, self.num_heads, q_length, kv_length)

        input_dtype = attention_scores.dtype
        attn_weights = flow.masked_fill(
            attention_scores, attention_mask, flow.finfo(attention_scores.dtype).min
        )
        attention_probs = F.softmax(attn_weights, dim=-1).to(input_dtype)

        attention_probs = self.attention_dropout(attention_probs)

        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        attention_probs_reshaped = attention_probs.view(
            batch_size * self.num_heads, q_length, kv_length
        )

        context_layer = flow.bmm(attention_probs_reshaped, value_layer)

        context_layer = self._merge_heads(context_layer)

        if self.pretraining_tp > 1 and self.slow_but_exact:
            slices = self.hidden_size / self.pretraining_tp
            output_tensor = flow.zeros_like(context_layer)
            for i in range(self.pretraining_tp):
                output_tensor = output_tensor + F.linear(
                    context_layer[:, :, int(i * slices) : int((i + 1) * slices)],
                    self.dense.weight[:, int(i * slices) : int((i + 1) * slices)],
                )
        else:
            output_tensor = self.dense(context_layer)

        output_tensor = dropout_add(output_tensor, residual, self.hidden_dropout, self.training)

        outputs = (output_tensor, present)

        if output_attentions:
            outputs += (attention_probs,)

        return outputs
