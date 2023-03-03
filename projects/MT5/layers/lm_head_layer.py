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

from oneflow import nn

from libai.layers import Linear, LMLogits


class LMHead(nn.Module):
    def __init__(self, model_type, hidden_size, vocab_size, hidden_layers):
        super().__init__()
        if model_type == "mt5":
            self.lm_head = Linear(
                hidden_size, vocab_size, bias=False, layer_idx=2 * hidden_layers - 1
            )
        else:
            self.lm_head = LMLogits(vocab_size, bias=True)

    def forward(self, decoder_states, embed_weight=None):
        if isinstance(self.lm_head, Linear):
            logits = self.lm_head(decoder_states)
        else:
            logits = self.lm_head(decoder_states, embed_weight)
        return logits
