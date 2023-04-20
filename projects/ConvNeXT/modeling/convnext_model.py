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

from libai.layers import LayerNorm
from projects.ConvNeXT.modeling.convnext_layers import ConvNextEncoder
from projects.ConvNeXT.modeling.embedding import ConvNextEmbeddings


class ConvNextModel(nn.Module):
    def __init__(
        self,
        num_channels,
        hidden_sizes,
        patch_size,
        depths,
        num_stages,
        drop_path_rate,
        eps,
        layer_norm_eps,
    ):
        super().__init__()

        self.embeddings = ConvNextEmbeddings(
            num_channels, hidden_sizes, patch_size, eps=eps, layer_idx=0
        )
        self.encoder = ConvNextEncoder(hidden_sizes, depths, num_stages, drop_path_rate)
        self.layernorm = LayerNorm(hidden_sizes[-1], eps=layer_norm_eps, layer_idx=-1)

    def forward(self, pixel_values=None):
        embedding_output = self.embeddings(pixel_values)
        encoder_outputs = self.encoder(embedding_output)
        last_hidden_state = encoder_outputs
        pooled_output = self.layernorm(last_hidden_state.mean([-2, -1]))
        return {"last_hidden_state": last_hidden_state, "pooled_output": pooled_output}
