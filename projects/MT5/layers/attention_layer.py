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

from libai.layers.linear import Linear
from libai.utils import distributed as dist
from projects.MT5.layers.embed_layer import Embedding


class MultiheadAttention(nn.Module):
    """Multi-head attention layer, support self attention and cross attention.

    Args:
        hidden_size: size of hidden state.
        num_attention_heads: number of attention heads.
        is_cross_attention: used to specify whether it is self attention or cross attention.
            Defaults to False.
        attention_dropout_prob: dropout probability of attention weights.
            Defaults to 0.0.
        output_dropout_prob: dropout probability of output. Defaults to 0.0.
        init_method: method to initialize the input layer weights.
            Defaults to ``init.xavier_normal_``.
        output_layer_init_method: method to initialize the output layer weights.
            If None, use ``init_method``.
        layer_idx: a layer_idx sign which determines the placements.
            It will be used in pipeline parallelism. Defaults to 0.
    """

    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        head_size,
        relative_attention_num_buckets,
        is_cross_attention=False,
        attention_dropout_prob=0.0,
        output_dropout_prob=0.0,
        init_method=nn.init.xavier_normal_,
        output_layer_init_method=None,
        padding_idx=None,
        *,
        layer_idx=0,
        has_relative_attention_bias=False,
        is_decoder=False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.has_relative_attention_bias = has_relative_attention_bias
        self.is_decoder = is_decoder
        self.attention_dropout_prob = attention_dropout_prob

        if output_layer_init_method is None:
            output_layer_init_method = init_method
        self.num_heads = num_attention_heads
        self.head_size = head_size

        self.dropout = nn.Dropout(p=attention_dropout_prob)
        self.norm_factor = 1.0 / math.sqrt(float(self.head_size))

        self.is_cross_attention = is_cross_attention

        self.output_dropout = nn.Dropout(p=output_dropout_prob)

        if self.is_cross_attention:
            self.query = Linear(
                self.hidden_size,
                self.num_heads * self.head_size,
                bias=False,
                parallel="col",
                init_method=init_method,
                layer_idx=layer_idx,
            )
            self.key_value = Linear(
                self.hidden_size,
                self.num_heads * self.head_size * 2,
                bias=False,
                parallel="col",
                init_method=init_method,
                layer_idx=layer_idx,
            )
        else:
            self.query_key_value = Linear(
                self.hidden_size,
                self.num_heads * self.head_size * 3,
                bias=False,
                parallel="col",
                init_method=init_method,
                layer_idx=layer_idx,
            )

        self.dense = Linear(
            self.num_heads * self.head_size,
            self.hidden_size,
            bias=False,
            parallel="row",
            init_method=output_layer_init_method,
            skip_bias_add=False,
            layer_idx=layer_idx,
        )
        if self.has_relative_attention_bias:
            self.relative_attention_bias = Embedding(
                self.relative_attention_num_buckets,
                self.num_heads,
                padding_idx=padding_idx,
                layer_idx=layer_idx,
            )

    def forward(
        self,
        hidden_states: flow.Tensor,
        encoder_states: flow.Tensor = None,
        attention_mask: flow.Tensor = None,
        past_key_value: Tuple[flow.Tensor, flow.Tensor] = None,
        use_cache: bool = False,
        position_bias=None,
        query_length=None,
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

        if encoder_states is not None:
            encoder_states = encoder_states.to_global(placement=hidden_states.placement)

        if attention_mask is not None:
            attention_mask = attention_mask.to_global(placement=hidden_states.placement)

        # hidden_states shape: [seq_len, batch_size, hidden_size]
        real_seq_length, bsz = hidden_states.size()[:2]

        if past_key_value is not None:
            assert (
                len(past_key_value) == 2
            ), "past_key_value should have 2 past states: keys and values."
            f"Got {len(past_key_value)} past states.\n"
            real_seq_length += past_key_value[0].shape[2] if query_length is None else query_length

        key_length = real_seq_length if encoder_states is None else encoder_states.shape[0]

        if self.is_cross_attention:
            query = self.query(hidden_states)
            query = query.view(-1, bsz, self.num_heads, self.head_size)
            query = query.permute(1, 2, 0, 3)  # bsz, num_head, seq_len, head_size

            if past_key_value is not None:
                key, value = past_key_value
            elif encoder_states is not None:
                key_value = self.key_value(encoder_states)
                key_value = key_value.view(-1, bsz, self.num_heads, 2 * self.head_size)
                key_value = key_value.permute(1, 2, 0, 3)
                key, value = flow.chunk(key_value, chunks=2, dim=-1)
            else:
                raise ValueError(
                    "past_key_value and encoder_states cannot be None at the same time."
                )
        else:
            query_key_value = self.query_key_value(hidden_states)
            if use_cache:
                query_key_value = query_key_value.view(bsz, -1, self.num_heads, 3 * self.head_size)
                query_key_value = query_key_value.permute(
                    0, 2, 1, 3
                )  # [bsz, num_heads, src_len, 3 * head_size]
                query, key, value = flow.chunk(query_key_value, chunks=3, dim=-1)
            else:
                attention_scores, value = flow._C.fused_self_attention(
                    query_key_value, head_size=self.head_size, alpha=1
                )
            if past_key_value is not None:
                past_key, past_value = past_key_value
                key = flow.cat((past_key.type_as(key), key), dim=2)
                value = flow.cat((past_value.type_as(value), value), dim=2)

        if use_cache:
            past_key_value = (key, value)

        if self.is_cross_attention or use_cache:
            attention_scores = flow.matmul(query, key, transpose_b=True, alpha=1)

        if position_bias is None:
            if not self.has_relative_attention_bias:
                position_bias = flow.zeros(
                    (1, self.num_heads, real_seq_length, key_length),
                    sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
                    placement=attention_scores.placement,
                )
            else:
                position_bias = self.compute_bias(
                    real_seq_length, key_length, placement=attention_mask.placement
                )

            if past_key_value is not None:
                position_bias = position_bias[:, :, -hidden_states.size(1) :, :]

        if attention_mask is not None:
            if use_cache:
                attention_mask = attention_mask.expand_as(attention_scores)

            attention_weights = flow._C.fused_bias_add_scale_mask_softmax_dropout(
                attention_scores,
                position_bias,
                attention_mask,
                fill_value=-10000.0,
                scale=1,
                p=self.attention_dropout_prob,
            )[0]
        else:
            attention_scores = attention_scores + position_bias
            attention_weights = flow.softmax(attention_scores, dim=-1)
            attention_weights = self.dropout(attention_weights)

        context = flow.matmul(attention_weights, value)

        """ transpose [batch_size, num_head, seq_len, head_size] to
            [seq_len, batch_size, num_head, head_size]
        """
        context = flow._C.transpose(context, perm=(2, 0, 1, 3))

        output = self.dense(context.flatten(2))

        output = self.output_dropout(output)

        if use_cache:
            output = (output, past_key_value)

        output = (output,) + (position_bias,)
        return output

    def extra_repr(self) -> str:
        return "hidden_size={}, num_heads={}, is_cross_attention={}".format(
            self.hidden_size,
            self.num_heads,
            self.is_cross_attention,
        )

    def _relative_position_bucket(
        self, relative_position, bidirectional=True, num_buckets=32, max_distance=128
    ):
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets = (
                relative_buckets + (relative_position > 0).to(flow.long) * num_buckets
            )
            relative_position = flow.abs(relative_position)
        else:
            relative_position = (
                -1
                * flow.min(
                    relative_position,
                    flow.zeros(
                        relative_position.size(),
                        sbp=relative_position.sbp,
                        placement=relative_position.placement,
                    ),
                ).to(flow.long)
            )

        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        relative_postion_if_large = max_exact + (
            flow.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(flow.long)

        relative_postion_if_large = flow.min(
            relative_postion_if_large,
            flow.zeros(
                relative_postion_if_large.size(),
                dtype=relative_postion_if_large.dtype,
                sbp=relative_postion_if_large.sbp,
                placement=relative_postion_if_large.placement,
            ).fill_(num_buckets - 1),
        )

        relative_buckets = relative_buckets + flow.where(
            is_small, relative_position, relative_postion_if_large
        )
        return relative_buckets

    def compute_bias(self, query_length, key_length, placement=None):
        """Compute binned relative position bias"""
        context_position = flow.arange(
            query_length,
            dtype=flow.long,
            sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
            placement=placement,
        )
        memory_position = flow.arange(
            key_length,
            dtype=flow.long,
            sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
            placement=placement,
        )
        relative_position = (
            memory_position[None, :] - context_position[:, None]
        )  # shape (query_length, key_length)

        relative_position_bucket = self._relative_position_bucket(
            relative_position,
            bidirectional=(not self.is_decoder),
            num_buckets=self.relative_attention_num_buckets,
        )  # shape (query_length, key_length)

        values = self.relative_attention_bias(
            relative_position_bucket
        )  # shape (query_length, key_length, num_heads)
        values = values.permute([2, 0, 1]).unsqueeze(
            0
        )  # shape (1, num_heads, query_length, key_length)
        return values
