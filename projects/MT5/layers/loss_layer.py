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

from libai.layers import ParallelCrossEntropyLoss
from libai.utils import distributed as dist


class MT5Loss(flow.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lm_loss = ParallelCrossEntropyLoss()

    def forward(self, logits, lm_labels, loss_mask):
        lm_labels = lm_labels.to_global(placement=logits.placement)
        lm_loss = self.lm_loss(logits, lm_labels)
        loss_mask = loss_mask.to_global(placement=lm_loss.placement)
        loss_mask = loss_mask.float()
        denominator = loss_mask.sum().to_global(
            sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast])
        )
        lm_loss = flow._C.amp_white_identity(lm_loss)
        lm_loss = flow._C.amp_black_identity(lm_loss)
        masked_lm_loss = flow.sum(lm_loss.view(-1) * loss_mask.view(-1)) / denominator
        masked_lm_loss = masked_lm_loss.to_global(
            sbp=dist.get_nd_sbp([flow.sbp.partial_sum, flow.sbp.broadcast])
        )

        if self.training:
            # token throughput
            done_tokens = (
                flow.zeros(
                    1,
                    sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
                    placement=lm_labels.placement,
                )
                + logits.shape[0] * logits.shape[1]
            )

            # correct token
            correct_tokens = flow.sum(
                (
                    logits.to_global(
                        sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
                        placement=lm_labels.placement,
                    )
                    .argmax(dim=-1)
                    .eq(lm_labels)
                ).float()
            )

            return {
                "mlm_loss": masked_lm_loss,
                "done_tokens": done_tokens,
                "correct_tokens": correct_tokens,
                "denominator": denominator,
            }

        else:
            return {
                "mlm_loss": masked_lm_loss,
            }
