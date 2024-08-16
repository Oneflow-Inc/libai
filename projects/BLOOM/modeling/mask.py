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

from libai.utils import distributed as dist


def _make_causal_mask(input_ids_shape, past_key_values_length):
    """
    Make causal mask used for self-attention.
    """
    batch_size, target_length = input_ids_shape
    mask = flow.ones(
        (target_length, target_length + past_key_values_length),
        dtype=flow.bool,
        sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
        placement=dist.get_layer_placement(0),
    )
    # ONNX doesn't support `torch.Tensor.triu` properly, thus we use this workaround
    seq_ids = flow.arange(
        target_length,
        dtype=flow.long,
        sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
        placement=dist.get_layer_placement(0),
    )
    mask[:, past_key_values_length:] = seq_ids[:, None] < seq_ids[None, :]

    if past_key_values_length > 0:
        mask[:, :past_key_values_length] = False

    expanded_mask = mask[None, None, :, :].expand(
        batch_size, 1, target_length, target_length + past_key_values_length
    )
    return expanded_mask


def _expand_mask(mask, tgt_length):
    """
    Expands attention_mask from `[batch_size, src_length]` to
    `[batch_size, 1, tgt_length, src_length]`.
    """
    batch_size, src_length = mask.shape
    tgt_length = tgt_length if tgt_length is not None else src_length

    expanded_mask = ~(mask[:, None, None, :].to(flow.bool))
    return expanded_mask.expand(batch_size, 1, tgt_length, src_length)


def build_alibi_tensor(attention_mask, num_heads, dtype):
    batch_size, seq_length = attention_mask.shape
    closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
    base = flow.tensor(
        2 ** (-(2 ** -(math.log2(closest_power_of_2) - 3))),
        sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
        placement=attention_mask.placement,
    )
    powers = flow.arange(
        1,
        1 + closest_power_of_2,
        sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
        placement=attention_mask.placement,
    )
    slopes = flow.pow(base, powers)

    if closest_power_of_2 != num_heads:
        extra_base = flow.tensor(
            2 ** (-(2 ** -(math.log2(2 * closest_power_of_2) - 3))),
            sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
            placement=attention_mask.placement,
        )
        num_remaining_heads = min(closest_power_of_2, num_heads - closest_power_of_2)
        extra_powers = flow.arange(
            1,
            1 + 2 * num_remaining_heads,
            2,
            sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
            placement=attention_mask.placement,
        )
        slopes = flow.cat([slopes, flow.pow(extra_base, extra_powers)], dim=0)

    arange_tensor = ((attention_mask.cumsum(dim=-1) - 1) * attention_mask)[:, None, :]
    alibi = slopes[..., None] * arange_tensor
    return alibi.reshape(batch_size * num_heads, 1, seq_length).to(dtype)
