# coding=utf-8
# Copyright 2021 The OneFlow Authors. All rights reserved.
# Copyright 2022 The HuggingFace Inc. team.
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

import math

import oneflow as flow
from oneflow import nn

import libai.utils.distributed as dist


class SinePositionalEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings

        position_embedding = flow.zeros(
            num_embeddings,
            embedding_dim,
            dtype=flow.float32,
            placement=dist.get_layer_placement(0),
            sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
        )
        position = flow._C.global_arange(
            start=0,
            end=num_embeddings,
            placement=dist.get_layer_placement(0),
            sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
            dtype=flow.float32,
        ).unsqueeze(1)
        position_range = flow._C.global_arange(
            start=0,
            end=embedding_dim,
            step=2,
            placement=dist.get_layer_placement(0),
            sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
            dtype=flow.float32,
        )
        div_term = flow.exp(position_range * (-math.log(10000.0) / embedding_dim))
        position_embedding[:, : embedding_dim // 2] = flow.sin(position * div_term)
        position_embedding[:, embedding_dim // 2 :] = flow.cos(position * div_term)
        self.register_buffer("position_embedding", position_embedding)

    def forward(self, position_ids):
        position_embeds = flow._C.gather(self.position_embedding, position_ids, axis=0)
        return position_embeds

    def extra_repr(self) -> str:
        s = "num_embeddings={num_embeddings}, embedding_dim={embedding_dim}"
        return s.format(**self.__dict__)
