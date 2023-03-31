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

import oneflow as flow
from oneflow import nn

import libai.utils.distributed as dist
from libai.layers import Embedding, VocabEmbedding
from libai.models.utils import init_method_normal


class GLMEmbedding(nn.Module):
    def __init__(
        self,
        vocab_size,
        hidden_size,
        max_seq_length,
        padding_idx=None,
        init_method=init_method_normal(0.02, 0),
        embedding_dropout_prob=0.0,
        amp_enabled=False,
        block_position_encoding=False,
    ):
        super().__init__()
        self.block_position_encoding = block_position_encoding

        self.word_embeddings = VocabEmbedding(
            vocab_size,
            hidden_size,
            padding_idx=padding_idx,
            init_method=init_method,
            amp_enabled=amp_enabled,
        )

        if block_position_encoding:
            self.position_embeddings = Embedding(
                max_seq_length + 1, hidden_size, init_method=init_method, amp_enabled=amp_enabled
            )
            self.block_position_embeddings = Embedding(
                max_seq_length + 1, hidden_size, init_method=init_method, amp_enabled=amp_enabled
            )
        self.embedding_dropout = nn.Dropout(embedding_dropout_prob)

        self.position_ids = flow.arange(
            max_seq_length,
            dtype=flow.long,
            sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
            placement=dist.get_layer_placement(0),
        ).unsqueeze(0)

        self.block_position_ids = flow.zeros(
            (1, max_seq_length),
            dtype=flow.long,
            sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
            placement=dist.get_layer_placement(0),
        )

    def forward(self, input_ids, position_ids=None):
        bsz, seq_len = input_ids.size()

        if self.block_position_encoding and position_ids is not None:
            position_ids, block_position_ids = position_ids[:, 0], position_ids[:, 1]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_len]
            position_ids = position_ids.expand_as(input_ids).to_global(sbp=input_ids.sbp)
            block_position_ids = self.block_position_ids[:, :seq_len]
            block_position_ids = block_position_ids.expand_as(input_ids).to_global(
                sbp=input_ids.sbp
            )

        word_embeddings = self.word_embeddings(input_ids)

        position_embeddings = self.position_embeddings(position_ids)
        input_embeddings = word_embeddings + position_embeddings

        if self.block_position_encoding:
            block_position_embeddings = self.block_position_embeddings(block_position_ids)
            input_embeddings = input_embeddings + block_position_embeddings

        input_embeddings = self.embedding_dropout(input_embeddings)
        return input_embeddings
