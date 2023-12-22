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

from libai.utils import distributed as dist


class ExtendedMask(flow.nn.Module):
    def forward(self, x, input_tensor=None, is_decoder=False):
        if x.dim() == 3:
            extended_mask = x[:, None, :, :]
        elif x.dim() == 2:
            if is_decoder:
                extended_mask = self.create_extended_mask_for_decoder(x, input_tensor)
            else:
                extended_mask = x[:, None, None, :]

        return extended_mask

    def create_extended_mask_for_decoder(self, x, input_tensor):
        batch_size, seq_len = input_tensor.size()
        seq_ids = flow.arange(
            seq_len,
            sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
            placement=x.placement,
        )
        causal_mask = (
            seq_ids[None, None, :].repeat(batch_size, seq_len, 1) <= seq_ids[None, :, None]
        )

        causal_mask = causal_mask.to(x.dtype)
        causal_mask = causal_mask.to_global(sbp=x.sbp)
        if causal_mask.shape[1] < x.shape[1]:
            prefix_seq_len = x.shape[1] - causal_mask.shape[1]
            ones = flow.ones(
                (batch_size, seq_len, prefix_seq_len),
                dtype=causal_mask.dtype,
                sbp=causal_mask.sbp,
                placement=causal_mask.placement,
            )
            causal_mask = flow.cat(
                [
                    ones,
                    causal_mask,
                ],
                dim=-1,
            )

        extended_mask = causal_mask[:, None, :, :] * x[:, None, None, :]
        return extended_mask
