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
from projects.ConvNeXT.modeling.layer_norm import ConvNextLayerNorm


class ConvNextEmbeddings(nn.Module):
    def __init__(self, num_channels, hidden_sizes, patch_size, layer_idx=0):
        super().__init__()
        self.patch_embeddings = nn.Conv2d(
            num_channels, hidden_sizes[0], kernel_size=patch_size, stride=patch_size
        ).to_global(
            sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
            placement=dist.get_layer_placement(layer_idx),
        )
        self.layernorm = ConvNextLayerNorm(
            hidden_sizes[0], eps=1e-6, data_format="channels_first", layer_idx=layer_idx
        )
        self.num_channels = num_channels
        self.layer_idx = layer_idx

    def forward(self, x):
        num_channels = x.shape[1]
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set "
                "in the configuration."
            )
        embeddings = self.patch_embeddings(x)
        embeddings = self.layernorm(embeddings)
        return embeddings
