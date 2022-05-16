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

from libai.models import BertModel


class BertForSimCSE(BertModel):
    def __init__(self, cfg):
        super().__init__(cfg)

    def forward(self, input_ids, attention_mask, tokentype_ids=None):
        extended_attention_mask = self.extended_attn_mask(attention_mask)
        embedding_output = self.embeddings(input_ids, tokentype_ids)
        total_hidden = []
        hidden_states = embedding_output
        for layer in self.encoders:
            hidden_states = layer(hidden_states, extended_attention_mask)
            total_hidden.append(hidden_states)
        encoder_output = self.final_layernorm(hidden_states)
        pooled_output = self.pooler(encoder_output) if self.pooler is not None else None
        return encoder_output, pooled_output, total_hidden
