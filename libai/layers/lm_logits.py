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


class LMLogits(nn.Module):
    def __init__(self, vocab_size, bias=False):
        super().__init__()
        self.bias = (
            nn.Parameter(
                flow.zeros(
                    (vocab_size,),
                    dtype=flow.float32,
                    placement=dist.get_layer_placement(-1),
                    sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.split(0)]),
                )
            )
            if bias
            else None
        )

    def forward(self, input, word_embeddings):
        """LM logits using word embedding weights"""
        # input with sbp sign [S(0), B] and word_embeddings with sbp sign [S(0), B]

        # NOTE(l1aoxingyu): This is for pipeline parallelism
        # change word embedding placement from stage(0) to stage(-1)
        w = word_embeddings.to_consistent(placement=input.placement)

        # NOTE(l1aoxingyu): input x embed^T = logits with sbp sign
        # [S(0), B] x [B, S(1)] --> [S(0), S(1)]
        #     ↑          ↑               ↑
        #   input      embed^T         logits
        # Backward pass input.grad = logits.grad x embed with sbp sign
        # [S(0), S(1)] x [B, S(0)] --> [S(0), P]
        #     ↑             ↑               ↑
        #  logits.grad    embed        input.grad
        # When use input.grad as head node for backward pass, need to convert
        # its sbp sign fromm [S(0), P] --> [S(0), B]
        input = input.to_consistent(grad_sbp=input.sbp)

        logits = flow._C.matmul(input, w, transpose_b=True)
        if self.bias is not None:
            logits = logits + self.bias
        return logits
