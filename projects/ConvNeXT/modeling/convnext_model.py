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

from libai.config import configurable
from libai.layers import LayerNorm
from projects.ConvNeXT.modeling.convnext_layers import ConvNextEncoder
from projects.ConvNeXT.modeling.embedding import ConvNextEmbeddings


class ConvNextModel(nn.Module):
    @configurable
    def __init__(
        self,
        num_channels,
        patch_size,
        num_stages,
        hidden_sizes,
        depths,
        layer_norm_eps=1e-12,
        drop_path_rate=0.0,
    ):
        super().__init__()

        self.embeddings = ConvNextEmbeddings(num_channels, hidden_sizes, patch_size, layer_idx=0)
        self.encoder = ConvNextEncoder(hidden_sizes, depths, num_stages, drop_path_rate)
        self.layernorm = LayerNorm(hidden_sizes[-1], eps=layer_norm_eps, layer_idx=-1)

        # weight init
        if os.getenv("ONEFLOW_LINEAR_EMBEDDING_SKIP_INIT", "0") != "1":
            self.apply(self._init_weights)

    def forward(self, pixel_values=None):
        embedding_output = self.embeddings(pixel_values)
        encoder_outputs = self.encoder(embedding_output)
        last_hidden_state = encoder_outputs
        pooled_output = self.layernorm(last_hidden_state.mean([-2, -1]))
        return {"last_hidden_state": last_hidden_state, "pooled_output": pooled_output}

    def _init_weight(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    @classmethod
    def from_config(cls, cfg):
        return {
            "num_channels": cfg.num_channels,
            "patch_size": cfg.patch_size,
            "num_stages": cfg.num_stages,
            "hidden_sizes": cfg.hidden_sizes,
            "depths": cfg.depths,
            "layer_norm_eps": cfg.layer_norm_eps,
            "drop_path_rate": cfg.drop_path_rate,
        }
