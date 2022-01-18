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
    def forward(self, logits, target):
        """Function for the distributed cross entropy
        vocab_parallel_logits with shape (batch_size, seq_length, vocab_size)
        and sbp sign [S(0), S(2)].
        target with shape (batch_size, seq_length) and sbp sign [S(0), B].
        """
        assert logits.ndim == 3
        assert target.ndim == 2
        assert logits.shape[0:2] == target.shape

        # Change -1 in target to 0 because sparse_softmax_cross_entropy don't accept -1
        target = target * (target >= 0)

        lm_loss = flow._C.sparse_softmax_cross_entropy(
            logits.view(-1, logits.shape[-1]), target.view(-1),
        )
        return lm_loss
