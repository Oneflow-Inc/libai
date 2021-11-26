# -*- coding: utf-8 -*-
# Copyright (c) OneFlow, Inc. and its affiliates.

import math
import oneflow as flow
from oneflow import nn
from oneflow.nn import init

from libai.utils import distributed as dist

__all__ = [
    "VocabEmbedding",
    "PositionalEmbedding",
    "TokenTypeEmbedding",
    "SinePositionalEmbedding",
]


class VocabEmbedding(nn.Module):
    """Construct the word embeddings, which may be splited along vocabulary dimension.

    Arguments:
        num_embeddings: size of vocabulary.
        embedding_dim: dimension of embeddings.
        padding_idx: pad index.
        init_method: method to initialize weights.
    """

    def __init__(self, num_embeddings, embedding_dim, padding_idx=None, init_method=init.xavier_normal_):
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
                    sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
                )

    def extra_repr(self) -> str:
        s = "num_embeddings={num_embeddings}, embedding_dim={embedding_dim}"
        if self.padding_idx is not None:
            s += ", padding_idx={padding_idx}"
        return s.format(**self.__dict__)


class PositionalEmbedding(nn.Module):
    """Construct the trainable positional embeddings.

    Arguments:
        num_embeddings: size of vocabulary.
        embedding_dim: dimension of embeddings.
        init_method: method to initialize weights.
    """

    def __init__(
        self, num_embeddings, embedding_dim, init_method=init.xavier_normal_,
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
        s = "num_embeddings={num_embeddings}, embedding_dim={embedding_dim}"
        return s.format(**self.__dict__)


class TokenTypeEmbedding(nn.Module):
    """Construct the token_type embeddings.

    Arguments:
        num_embeddings: size of vocabulary.
        embedding_dim: dimension of embeddings.
        padding_idx: pad index.
        init_method: method to initialize weights.
    """

    def __init__(
        self, num_embeddings, embedding_dim, padding_idx=None, init_method=init.xavier_normal_,
    ):
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
        # FIXME(Lxy): Fill padding_idx is not supported in nd_sbp right now.
        # self._fill_padding_idx_with_zero()

    def forward(self, tokentype_ids):
        tokentype_embeds = flow._C.gather(self.weight, tokentype_ids, axis=0)
        return tokentype_embeds
    
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
        s = "num_embeddings={num_embeddings}, embedding_dim={embedding_dim}"
        if self.padding_idx is not None:
            s += ", padding_idx={padding_idx}"
        return s.format(**self.__dict__)


class SinePositionalEmbedding(nn.Module):
    """Construct the sin cos positional embeddings.

    Arguments:
        num_embeddings: size of vocabulary.
        embedding_dim: dimension of embeddings.
    """

    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings

        position_embedding = flow.zeros(num_embeddings,
                                        embedding_dim,
                                        dtype=flow.float32,
                                        placement=dist.get_layer_placement(0),
                                        sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]))
        position = flow._C.consistent_arange(start=0,
                                             end=num_embeddings,
                                             placement=dist.get_layer_placement(0),
                                             sbp=dist.get_nd_sbp(
                                                 [flow.sbp.broadcast, flow.sbp.broadcast]),
                                             dtype=flow.float32).unsqueeze(1)
        position_range = flow._C.consistent_arange(start=0,
                                                   end=embedding_dim,
                                                   step=2,
                                                   placement=dist.get_layer_placement(0),
                                                   sbp=dist.get_nd_sbp(
                                                       [flow.sbp.broadcast, flow.sbp.broadcast]),
                                                   dtype=flow.float32)
        div_term = flow.exp(
            position_range * (-math.log(10000.0) / embedding_dim))

        position_embedding[:, 0::2] = flow.sin(position * div_term)
        position_embedding[:, 1::2] = flow.cos(position * div_term)
        self.register_buffer('position_embedding', position_embedding)

    def forward(self, position_ids):
        position_embeds = flow._C.gather(self.position_embedding, position_ids, axis=0)
        return position_embeds

    def extra_repr(self) -> str:
        s = "num_embeddings={num_embeddings}, embedding_dim={embedding_dim}"
        return s.format(**self.__dict__)