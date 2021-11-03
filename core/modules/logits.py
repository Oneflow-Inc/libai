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
import oneflow.nn.init as init
from core import distribute as dist


class ParallelLogits(flow.nn.Module):
    """LM logits using word embedding weight."""
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states, word_embeddings):
        assert hidden_states.ndim == 3

        w = word_embeddings.to_consistent(placement=hidden_states.placement)
        # h.grad.sbp: [S(0), P] -> [S(0), B]
        h = hidden_states.to_consistent(grad_sbp=hidden_states.sbp)
        # shape sign: (B * S, H) x (H, V) -> (B * S, V)
        # matmul fwd sbp sign: [S(0), B] x [B, S(1)] (wte.T) -> [S(0), S(1)]
        # bwd h.grad sbp sign: [S(0), S(1)] (lgs.grad) x [B, S(0)] (wte) -> [S(0), P] (h.grad)
        logits = flow.matmul(h, w, transpose_b=True)
        return logits
