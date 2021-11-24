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
from libai import distribute as dist


class MaskHelper(object):
    def __init__(self):
        super().__init__()
        self.mask = None

    def build_mask_matrix(self, max_positions=1024):
        """Create casual mask matrix for auto-regressive generation."""
        self.mask = flow.tril(
            flow.ones(
                (max_positions, max_positions), 
                dtype=flow.int8, 
                placement=dist.get_layer_placement(0),
                sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast])
            )
        )

    def make_causal_mask(self, input_ids_shape, past_length=0):
        if self.mask is None:
            self.build_mask_matrix()
        
        bsz, tgt_len = input_ids_shape
        mask = self.mask[:tgt_len, :tgt_len]
        if past_length > 0:
            mask = flow.cat([flow.ones(tgt_len, past_length, dtype=flow.int8), mask], dim=-1)
        mask = mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_length)
        return mask
    
    def extend_attention_mask(self, attention_mask, tgt_len=None):
        """Expands attention mask from `[bsz, src_len]` to `[bsz, 1, tgt_len, src_len]`.
        """
        bsz, src_len = attention_mask.size()
        tgt_len = tgt_len if tgt_len is not None else src_len 
        extended_attention_mask = attention_mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(flow.int8)
        return extended_attention_mask
    
    def combine_mask(self, mask1, mask2):
        return mask1 + mask2 > 0