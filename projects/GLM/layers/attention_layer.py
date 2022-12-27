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

import enum
import math
from typing import Tuple

import oneflow as flow
from oneflow import nn

from libai.layers.linear import Linear
import libai.utils.distributed as dist


class AttnMaskType(enum.Enum):
    padding = 1
    causal = 2


class MultiheadAttention(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        is_cross_attention=False,
        attention_dropout_prob=0.0,
        output_dropout_prob=0.0,
        init_method=nn.init.xavier_normal_,
        output_layer_init_method=None,
        bias_dropout_fusion=False,
        scale_mask_softmax_fusion=False,
        apply_query_key_layer_scaling=False,
        attn_mask_type=AttnMaskType.padding,
        relative_encoding=False,
        performer=False,
        *,
        layer_idx=0
    ):
        super().__init__()
        self.hidden_size = hidden_size
        if output_layer_init_method is None:
            output_layer_init_method = init_method

        assert (
            hidden_size % num_attention_heads == 0
        ), "hidden_size must be divisible by num_attention_heads."

        self.num_heads = num_attention_heads
        self.head_size = hidden_size // num_attention_heads
        self.attn_mask_type = attn_mask_type

        self.attention_dropout_prob = attention_dropout_prob
        self.dropout = nn.Dropout(p=attention_dropout_prob)
        self.norm_factor = 1.0 / math.sqrt(float(self.head_size))
        self.coeff = None
        if apply_query_key_layer_scaling:
            self.coeff = layer_idx + 1
            self.norm_factor /= self.coeff

        self.is_cross_attention = is_cross_attention
        self.scale_mask_softmax_fusion = scale_mask_softmax_fusion
        self.bias_dropout_fusion = bias_dropout_fusion

        if self.bias_dropout_fusion:
            self.output_dropout_prob = output_dropout_prob
        else:
            self.output_dropout = nn.Dropout(p=output_dropout_prob)

        if self.is_cross_attention:
            self.query = Linear(
                self.hidden_size,
                self.hidden_size,
                parallel="col",
                init_method=init_method,
                layer_idx=layer_idx,
            )
            self.key_value = Linear(
                self.hidden_size,
                self.hidden_size * 2,
                parallel="col",
                init_method=init_method,
                layer_idx=layer_idx,
            )
        else:
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
        
        self.relative_encoding = relative_encoding
        if self.relative_encoding:
            self.relative = Linear(
                self.hidden_size,
                self.hidden_size,
                parallel="row",
                init_method=init_method,
                layer_idx=layer_idx,
            )

    def forward(
        self,
        hidden_states: flow.Tensor,
        encoder_states: flow.Tensor = None,
        attention_mask: flow.Tensor = None,
        past_key_value: Tuple[flow.Tensor, flow.Tensor] = None,
        use_cache: bool = False,
        position_embeddings=None,
        r_w_bias=None, 
        r_r_bias=None,
        mem=None,
    ):
        if encoder_states is not None:
            encoder_states = encoder_states.to_global(placement=hidden_states.placement)

        if attention_mask is not None:
            attention_mask = attention_mask.to_global(placement=hidden_states.placement)

        bsz, tgt_len = hidden_states.size()[:2]

        if self.is_cross_attention:
            query = self.query(hidden_states)
            query = query.view(bsz, -1, self.num_heads, self.head_size)
            query = query.permute(0, 2, 1, 3)
            if past_key_value is not None:
                key, value = past_key_value
            elif encoder_states is not None:
                key_value = self.key_value(encoder_states)
                key_value = key_value.view(bsz, -1, self.num_heads, 2 * self.head_size)
                key_value = key_value.permute(0, 2, 1, 3)
                key, value = flow.chunk(key_value, chunks=2, dim=-1)
            else:
                raise ValueError(
                    "past_key_value and encoder_states cannot be None at the same time."
                )
        else:
            if mem is not None:
                hidden_states = flow.cat((mem, hidden_states), dim=1)
            query_key_value = self.query_key_value(hidden_states)
            query_key_value = query_key_value.view(bsz, -1, self.num_heads, 3 * self.head_size)
            query_key_value = query_key_value.permute(
                0, 2, 1, 3
            )
            query, key, value = flow.chunk(query_key_value, chunks=3, dim=-1)
            if mem is not None:
                query = query[:, :, -tgt_len:]
            if past_key_value is not None:
                past_key, past_value = past_key_value
                key = flow.cat((past_key.type_as(key), key), dim=2)
                value = flow.cat((past_value.type_as(value), value), dim=2)

        if use_cache:
            past_key_value = (key, value)
        
        if self.relative_encoding:
            relative_layer_out = self.relative(position_embeddings)
            relative_layer_out = relative_layer_out.view(bsz, -1, self.num_heads, 3 * self.head_size)
            relative_layer_out = relative_layer_out.permute(0, 2, 1, 3)
            rw_head_q = query + r_w_bias.unsqueeze(1)
            ac_score = flow.matmul(rw_head_q, key.transpose(-1, -2))
            rr_head_q = query + r_r_bias.unsqueeze(1)
            bd_score = flow.matmul(rr_head_q, relative_layer_out.transpose(-1, -2))
            bd_score = self._rel_shift(bd_score)
            attention_scores = ac_score + bd_score
            attention_scores = attention_scores / math.sqrt(self.head_size)
        else:
            attention_scores = flow.matmul(query, key, transpose_b=True, alpha=self.norm_factor)

        if attention_mask is not None:
            if self.scale_mask_softmax_fusion:
                if self.attn_mask_type == AttnMaskType.padding:
                    attention_mask = (
                        attention_mask.expand_as(attention_scores) if use_cache else attention_mask
                    )
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
        else:
            if self.scale_mask_softmax_fusion and self.attn_mask_type == AttnMaskType.causal:
                attention_weights = flow._C.fused_scale_tril_softmax_mask_scale(
                    attention_scores,
                    p=self.attention_dropout_prob,
                    diagonal=0,
                    tril_scale_value=self.coeff,
                    tril_fill_value=-10000.0,
                )[0]
            else:
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

        if use_cache:
            output = (output, past_key_value)

        return output

    def extra_repr(self) -> str:
        return "hidden_size={}, num_heads={}, is_cross_attention={}".format(
            self.hidden_size,
            self.num_heads,
            self.is_cross_attention,
        )
        
    @staticmethod
    def _rel_shift(x, zero_triu=False):
        zero_pad = flow.zeros(
            (*x.size()[:-2], x.size(-2), 1),
            dtype=x.dtype,
        )
        zero_pad = zero_pad.to_global(sbp=x.sbp, placement=x.placement)
        x_padded = flow.cat([zero_pad, x], dim=-1)

        x_padded = x_padded.view(*x.size()[:-2], x.size(-1) + 1, x.size(-2))

        x = x_padded[:, :, 1:].view_as(x)

        if zero_triu:
            ones = flow.ones((x.size(0), x.size(1)))
            ones = flow.tril(ones, x.size(1) - x.size(0))[:, :, None, None]
            ones = ones.to_global(
                sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
                placement=x.placement
            )
            x = x * ones

        return x
