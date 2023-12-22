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

from libai.layers import Linear
from libai.utils import distributed as dist


class LMLogits(nn.Module):
    def __init__(self, vocab_size, hidden_size=None, bias=False, model_type="t5", layer_idx=-1):
        super().__init__()
        self.model_type = model_type
        if model_type == "t5":
            self.bias = (
                nn.Parameter(
                    flow.zeros(
                        (vocab_size,),
                        dtype=flow.float32,
                        placement=dist.get_layer_placement(layer_idx),
                        sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.split(0)]),
                    )
                )
                if bias
                else None
            )
        elif model_type == "mt5":
            self.linear = Linear(hidden_size, vocab_size, bias=False, layer_idx=layer_idx)

    def forward(self, input, word_embeddings=None):
        if self.model_type == "t5":
            w = word_embeddings.to_global(placement=input.placement)
            input = input.to_global(grad_sbp=input.sbp)
            logits = flow._C.matmul(input, w, transpose_b=True)
            if self.bias is not None:
                logits = logits + self.bias
        else:
            logits = self.linear(input)
        return logits
