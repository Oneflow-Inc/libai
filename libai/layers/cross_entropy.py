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


class ParallelCrossEntropyLossWithMask(nn.Module):
    def forward(self, vocab_parallel_logits, target, loss_mask=None):
        """Function for the distributed cross entropy
        vocab_parallel_logits with shape (batch_size, seq_length, vocab_size) and sbp sign [S(0), S(2)]
        target with shape (batch_size, seq_length) and sbp sign [S(0), B]
        """
        assert vocab_parallel_logits.ndim == 3
        assert target.ndim == 2
        assert vocab_parallel_logits.shape[0:2] == target.shape

        # Change -1 in target to 0 because sparse_softmax_cross_entropy don't accept -1
        target = target * (target >= 0)

        lm_loss = flow._C.sparse_softmax_cross_entropy(
            vocab_parallel_logits.view(-1, vocab_parallel_logits.shape[-1]),
            target.view(-1),
        )

        if loss_mask is None:
            return lm_loss.mean()

        loss_mask = loss_mask.float()
        # Change loss_mask.sum() sbp sign from [P, B] -> [B, B]
        # because (lm_loss * loss_mask) / loss_mask.sum() cannot accept P / P
        denominator = loss_mask.sum().to_consistent(
            sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast])
        )

        loss = flow.sum(lm_loss.view(-1) * loss_mask.view(-1)) / denominator

        return loss
