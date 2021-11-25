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

    def __init__(self, num_embeddings, embedding_dim, init_method, padding_idx=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        if padding_idx is not None:
            if padding_idx > 0:
                assert (
                    padding_idx < self.num_embeddings
                ), "Padding_idx must be within num_embeddings"
            elif padding_idx < 0:
                assert (
                    padding_idx >= -self.num_embeddings
                ), "Padding_idx must be within num_embeddings"
                padding_idx = self.num_embeddings + padding_idx
        self.padding_idx = padding_idx
        self.init_method = init_method

        # Word token embedding shape with (vocab_size, hidden_size)
        # sbp: [B, S(0)]
        self.weight = nn.Parameter(
            flow.empty(
                (num_embeddings, embedding_dim),
                dtype=flow.float32,
                placement=dist.get_layer_placement(0),
                sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.split(0)]),
            )
        )
        # Initialize the word embedding
        self.init_method(self.weight)
        # FIXME(Lxy): Fill padding_idx is not supported in nd_sbp right now.
        # self._fill_padding_idx_with_zero()

    def forward(self, input_ids):
        # input_ids with shape (batch_size, seq_len), and sbp sign: [S(0), B]

        # Gather forward sbp sign
        # [B, S(0)] x [S(0), B] --> [S(0), P]
        #     ↑           ↑            ↑
        #   embed  input_ids    input_embeds
        input_embeds = flow._C.gather(self.weight, input_ids, axis=0)
        # Set the embeds sbp from [S(0), P] --> [S(0), B] to get complete embedding results.
        input_embeds = input_embeds.to_consistent(sbp=dist.get_hidden_sbp())

        return input_embeds

    def _fill_padding_idx_with_zero(self) -> None:
        if self.padding_idx is not None:
            with flow.no_grad():
                self.weight[self.padding_idx] = flow.zeros(
                    self.embedding_dim,
                    placement=dist.get_layer_placement(0),
                    sbp=dist.get_nd_sbp(
                        [flow.sbp.broadcast, flow.sbp.broadcast]),
                )

    def extra_repr(self) -> str:
        s = "{num_embeddings}, {embedding_dim}"
        if self.padding_idx is not None:
            s += ". padding_idx={padding_idx}"
        return s.format(**self.__dict__)


class PositionalEmbedding(nn.Module):
    """Construct the trainable positional embeddings.
    """

    def __init__(
        self, num_embeddings, embedding_dim, init_method,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.init_method = init_method

        self.weight = nn.Parameter(
            flow.empty(
                (num_embeddings, embedding_dim),
                dtype=flow.float32,
                placement=dist.get_layer_placement(0),
                sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
            )
        )
        self.init_method(self.weight)

    def forward(self, position_ids):
        # Position_embeddings with sbp sign: [B, B]
        #   [B, B] x [S(0), B] --> [S(0), B]
        #     ↑         ↑              ↑
        #   embed    pos_ids       pos_embed
        position_embeds = flow._C.gather(self.weight, position_ids, axis=0)
        return position_embeds

    def extra_repr(self) -> str:
        s = "{num_embeddings}, {embedding_dim}"
        return s.format(**self.__dict__)


class TokenTypeEmbedding(nn.Module):
    """Construct the token_type embeddings.
    """

    def __init__(
        self, num_embeddings, embedding_dim, init_method,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.init_method = init_method

        assert num_embeddings > 0
        self.weight = nn.Parameter(
            flow.empty(
                (num_embeddings, embedding_dim),
                dtype=flow.float32,
                placement=dist.get_layer_placement(0),
                sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
            )
        )
        self.init_method(self.weight)

    def forward(self, tokentype_ids):
        tokentype_embeds = flow._C.gather(self.weight, tokentype_ids, axis=0)
        return tokentype_embeds

    def extra_repr(self) -> str:
        s = "{num_embeddings}, {embedding_dim}"
        return s.format(**self.__dict__)


class SinePositionalEmbedding(nn.Module):
    """Construct the sin cos positional embeddings.
    """

    def __init__(self, embedding_dim):
        super().__init__()

        self.embedding_dim = embedding_dim

        posit_range = flow._C.consistent_arange(start=0,
                                                end=embedding_dim,
                                                step=2,
                                                placement=dist.get_layer_placement(0),
                                                sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]))
        inv_freq = 1 / (10000 ** (posit_range / embedding_dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq, batch_size=None):
        sinusoid_inp = flow.ger(pos_seq, self.inv_freq)
        pos_emb = flow.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)

        if batch_size is not None:
            return pos_emb[None, :, :].expand(batch_size, -1, -1)
        else:
            return pos_emb[None, :, :]
