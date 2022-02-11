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
from oneflow import nn

from libai.utils import distributed as dist


class ParallelCrossEntropyLoss(nn.Module):
    def __init__(self, ignore_index=None, reduction="mean"):
        super().__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, logits, target):
        # Change -1 in target to 0 because sparse_softmax_cross_entropy don't accept -1        
        target = target.view(-1)
        target_1 = target * (target >= 0)

        loss = flow._C.sparse_softmax_cross_entropy(
            logits.view(-1, logits.shape[-1]),
            target_1,
        )

        if self.ignore_index is not None:
            non_pad_mask = target.ne(self.ignore_index)
            loss = loss * non_pad_mask.float()
            nsamples = non_pad_mask.sum().to_global(
                sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast])
            )
        else:
            nsamples = target.numel()

        if self.reduction == "none":
            return loss
        elif self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction == "mean":
            loss = loss.sum() / nsamples
        else:
            raise ValueError("Invalid reduction value.")

        loss = loss.to_global(
            sbp=dist.get_nd_sbp([flow.sbp.partial_sum, flow.sbp.broadcast])
        )

        return loss
        