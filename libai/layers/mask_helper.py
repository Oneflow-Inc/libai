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

class MaskHelper(nn.Module):
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
                sbp=[flow.sbp.broadcast, flow.sbp.broadcast]
            )
        )        

    def mask_casual_mask(self, input_ids, past_length=0):
        if self.mask is None:
            self.build_mask_matrix()
        
        bsz, tgt_len = input_ids.size()
        mask = self.mask[:tgt_len, :tgt_len]
        if past_length > 0:
            mask = flow.cat([flow.ones(tgt_len, past_length, dtype=flow.int8), mask], dim=-1)
        mask = mask.unsqueeze(0).unsqueeze(1).expand(bsz, 1, tgt_len, tgt_len + past_length)
        mask = mask.to_consistent(sbp=input_ids.sbp)
        return mask
    
    def get_extend_attention_mask(self, attention_mask, tgt_len=None):
        if attention_mask.dim() == 3:
            # [batch_size, tgt_len, src_len]
            extended_attention_mask = attention_mask.unsqueeze(1)
        elif attention_mask.dim() == 2:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            if tgt_len is not None:
                extended_attention_mask = extended_attention_mask.expand(-1, -1, tgt_len, src_len)

