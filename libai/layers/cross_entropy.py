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


class ParallelCrossEntropyLoss(nn.Module):
    """This criterion acts like :class:`~flow.nn.CrossEntropyLoss` except it will
    execute distributed cross entropy loss computation cross different GPUs.
    """

    def forward(self, logits: flow.Tensor, target: flow.Tensor):
        """Function for the distributed cross entropy.

        Args:
            logits (flow.Tensor): vocab_parallel_logits with shape
                (batch_size, seq_length, vocab_size) and sbp signature is [S(0), S(2)].
            target (flow.Tensor): target with shape (batch_size, seq_length) and
                sbp signature is [S(0), B].
        """
        assert logits.ndim == 3
        assert target.ndim == 2
        assert logits.shape[0:2] == target.shape

        # target = target.to(flow.int32) # NOTE:npu nll target only support int32 for now
        target = target.to_global(placement=logits.placement)
        lm_loss = flow._C.cross_entropy(
            logits.view(-1, logits.shape[-1]),
            target.view(-1),
            None,
            -100,
            "none",
            0.0
        )
        return lm_loss
