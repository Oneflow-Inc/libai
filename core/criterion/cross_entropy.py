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
from core.criterion import register_criterion

from .base_criterion import BaseLoss
# todo: add other criterion, like kl_div, cross_entropy_with_label_smooth, triple_loss, margin_loss

@register_criterion("cross_entropy")
class ParallelCrossEntropyLoss(BaseLoss):
    def __init__(self, args):
        super().__init__(args)

    def forward(self, logits, labels):
        # logits shape: (batch_size, seq_length, vocab_size)
        # sbp: [S(0), S(2)]
        # labels shape: (batch_size, seq_length)
        # sbp: [S(0), B]
        assert logits.ndim == 3
        assert labels.ndim == 2
        assert logits.shape[0:2] == labels.shape

        if logits.is_consistent and flow.sbp.split(logits.ndim - 1) in logits.sbp:
            loss = flow._C.sparse_softmax_cross_entropy_ms(logits, labels, depth=logits.shape[-1])
        else:
            loss = flow._C.sparse_softmax_cross_entropy(logits, labels, depth=logits.shape[-1])
            loss = flow._C.amp_white_identity(loss)
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
