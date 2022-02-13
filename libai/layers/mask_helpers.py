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
import oneflow.nn as nn

from libai.utils import distributed as dist


class ExtendedMask(nn.Module):
    """Makes the attention mask broadcastable at the head dims."""

    def forward(self, attention_mask):
        if attention_mask.dim() == 4:
            extended_attention_mask = attention_mask
        elif attention_mask.dim() == 3:
            # When we get an attention mask of dimensions [batch_size, tgt_len, src_len]
            # we just need to make it broadcastable to all heads.
            extended_attention_mask = attention_mask.unsqueeze(1)
        elif attention_mask.dim() == 2:
            # We get an attention mask of dimensions [batch_size, src_len]
            # we extend it to [batch_size, 1, 1, src_len]
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        else:
            raise ValueError(f"Wrong shape for attention_mask (shape {attention_mask.shape})")

        extended_attention_mask = extended_attention_mask.to(flow.int8)
        return extended_attention_mask


class CasualMask(nn.Module):
    """Create a casual mask and combine it with the padding mask.
    It will be used in gpt model and T5 decoder.
    """

    def __init__(self, max_positions=1024, *, layer_idx=0):
        super().__init__()
        self.mask = flow.tril(
            flow.ones(
                (max_positions, max_positions),
                dtype=flow.int8,
                placement=dist.get_layer_placement(layer_idx),
                sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
            )
        )

    def forward(self, input_ids, past_length=0, attention_mask=None):
        bsz, tgt_len = input_ids.size()
        casual_mask = self.mask[:tgt_len, :tgt_len]
        if past_length > 0:
            # in case past_key_values are used, we need to add a prefix ones mask to casual mask
            casual_mask = flow.cat(
                [flow.ones(tgt_len, past_length, dtype=flow.int8), casual_mask], dim=-1
            )
        casual_mask = (
            casual_mask.unsqueeze(0).unsqueeze(1).expand(bsz, 1, tgt_len, tgt_len + past_length)
        )
        casual_mask = casual_mask.to_global(sbp=input_ids.sbp)
        if attention_mask is not None:
            assert attention_mask.dim() == 4, "please extend the attention mask first"
            casual_mask = casual_mask * attention_mask
        return casual_mask
