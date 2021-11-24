# coding=utf-8
"""
Copyright 2021 The OneFlow Authors. All rights reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import oneflow as flow
from oneflow import nn

from libai.utils import distributed as dist

__all__ = [
    "VocabEmbedding",
    "PositionalEmbedding",
    "TokenTypeEmbedding",
    "SinePositionalEmbedding",
]


class VocabEmbedding(nn.Module):
    """Construct the word embeddings.
    """

    def __init__(
        self, vocab_size, hidden_size, init_method,
    ):
        super().__init__()
        self.init_method = init_method

        # Word token embedding shape with (vocab_size, hidden_size)
        # sbp: [B, S(0)]
        self.word_embeddings = nn.Parameter(
            flow.empty(
                (vocab_size, hidden_size),
                dtype=flow.float32,
                placement=dist.get_layer_placement(0),
                sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.split(0)]),
            )
        )
        # Initialize the word embedding, waiting for model parallel revision
        self.init_method(self.word_embeddings)

    def forward(self, input_ids):
        # input_ids with shape (batch_size, seq_len), and sbp sign: [S(0), B]

        # Gather forward sbp sign
        # [B, S(0)] x [S(0), B] --> [S(0), P]
        #     ↑           ↑            ↑
        #   embed  input_ids    input_embeds
        input_embeds = flow._C.gather(self.word_embeddings, input_ids, axis=0)
        # Set the embeds sbp from [S(0), P] --> [S(0), B] to get complete embedding results.
        input_embeds = input_embeds.to_consistent(sbp=dist.get_hidden_sbp())

        return input_embeds


class PositionalEmbedding(nn.Module):
    """Construct the trainable positional embeddings.
    """

    def __init__(
        self, max_sequence_length, hidden_size, init_method,
    ):
        super().__init__()
        self.init_method = init_method

        self.position_embeddings = nn.Parameter(
            flow.empty(
                (max_sequence_length, hidden_size),
                dtype=flow.float32,
                placement=dist.get_layer_placement(0),
                sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
            )
        )
        self.init_method(self.position_embeddings)

    def forward(self, position_ids):
        # Position_embeddings with sbp sign: [B, B]
        #   [B, B] x [S(0), B] --> [S(0), B]
        #     ↑         ↑              ↑
        #   embed    pos_ids       pos_embed
        position_embeds = flow._C.gather(self.position_embeddings, position_ids, axis=0)
        return position_embeds


class TokenTypeEmbedding(nn.Module):
    """Construct the token_type embeddings.
    """

    def __init__(
        self, num_tokentypes, hidden_size, init_method,
    ):
        super().__init__()
        self.init_method = init_method

        assert num_tokentypes > 0, ""
        self.tokentype_embeddings = nn.Parameter(
            flow.empty(
                (num_tokentypes, hidden_size),
                dtype=flow.float32,
                placement=dist.get_layer_placement(0),
                sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
            )
        )
        self.init_method(self.tokentype_embeddings)

    def forward(self, tokentype_ids):
        tokentype_embeds = flow._C.gather(
            self.tokentype_embeddings, tokentype_ids, axis=0
        )
        return tokentype_embeds


class SinePositionalEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
