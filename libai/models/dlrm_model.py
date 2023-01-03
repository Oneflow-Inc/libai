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

from libai.config import configurable
from libai.layers import (
    OneEmbedding,
    MLP,
    FusedMLP,
    Interaction,
)

from libai.utils import distributed as dist

class DLRMModel(nn.Module):
    @configurable
    def __init__(
        self,
        embedding_vec_size=128,
        bottom_mlp=[512, 256, 128],
        top_mlp=[1024, 1024, 512, 256],
        num_dense_fields=13,
        num_sparse_fields=26,
        use_fusedmlp=True,
        persistent_path=None,
        table_size_array=None,
        one_embedding_store_type="cached_host_mem",
        cache_memory_budget_mb=8192,
        interaction_itself=True,
        interaction_padding=True,
    ):
        super(DLRMModel, self).__init__()
        assert (
            embedding_vec_size == bottom_mlp[-1]
        ), "Embedding vector size must equle to bottom MLP output size"
        if use_fusedmlp:
            self.bottom_mlp = MLP(num_dense_fields, bottom_mlp)
        else:
            self.bottom_mlp = FusedMLP(num_dense_fields, bottom_mlp)

        self.embedding = OneEmbedding(
            embedding_vec_size,
            persistent_path,
            table_size_array,
            one_embedding_store_type,
            cache_memory_budget_mb,
        )
        self.interaction = Interaction(
            bottom_mlp[-1],
            num_sparse_fields,
            interaction_itself,
            interaction_padding=interaction_padding,
        )
        if use_fusedmlp:
            self.top_mlp = FusedMLP(
                self.interaction.output_size,
                top_mlp + [1],
                skip_final_activation=True,
            )
        else:
            self.top_mlp = MLP(
                self.interaction.output_size,
                top_mlp + [1],
                skip_final_activation=True,
            )

    def forward(self, dense_fields, sparse_fields) -> flow.Tensor:
        dense_fields = flow.log(dense_fields + 1.0)
        dense_fields = self.bottom_mlp(dense_fields)
        embedding = self.embedding(sparse_fields)
        features = self.interaction(dense_fields, embedding)
        return self.top_mlp(features)

    @classmethod
    def from_config(cls, cfg):
        return {
            "embedding_vec_size": cfg.embedding_vec_size,
            "bottom_mlp": list(cfg.bottom_mlp),
            "top_mlp": list(cfg.top_mlp),
            "num_dense_fields": cfg.num_dense_fields,
            "num_sparse_fields": cfg.num_sparse_fields,
            "use_fusedmlp": cfg.use_fusedmlp,
            "persistent_path": cfg.persistent_path,
            "table_size_array": list(cfg.table_size_array),
            "one_embedding_store_type": cfg.store_type,
            "cache_memory_budget_mb": cfg.cache_memory_budget_mb,
            "interaction_itself": cfg.interaction_itself,
            "interaction_padding": cfg.interaction_padding,
        }
