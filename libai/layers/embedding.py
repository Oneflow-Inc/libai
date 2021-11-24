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
import oneflow.nn.init as init
from libai import distribute as dist


class Embedding(flow.nn.Module):
    """Common embedding module.

    Arguments:
        num_embeddings: size of vocabulary.
        embeddings_dim: dimension of embeddings.
        init_method: method to initialize weights.
        enable_amp: whether apply auto mixed precision (amp).
    """
    def __init__(self, num_embeddings, embeddings_dim, padding_idx=None, init_method=init.xavier_normal_, enable_amp=False):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embeddings_dim = embeddings_dim
        self.padding_idx = padding_idx
        self.enable_amp = enable_amp

        self.weights = flow.nn.Parameter(
            flow.empty(
                (self.num_embeddings, self.embeddings_dim),
                dtype=flow.float32,
                placement=dist.get_layer_placement(0),
                sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
            )
        )

        init_method(self.weights)
        if self.padding_idx is not None:
            with flow.no_grad():
                self.weights[self.padding_idx].fill_(0)
        

    def forward(self, input_ids):
        if self.enable_amp:
            weights = flow._C.amp_white_identity(self.weights)
        embeds = flow._C.gather(self.weights, input_ids, axis=0)
        return embeds


class ParallelEmbedding(flow.nn.Module):
    """Embedding parallelized along vocabulary dimension.

    Arguments:
        num_embeddings: size of vocabulary.
        embeddings_dim: dimension of embeddings.
        padding_idx: pad index.
        init_method: method to initialize weights.
        enable_amp: whether apply auto mixed precision (amp).
    """
    def __init__(self, num_embeddings, embeddings_dim, padding_idx=None, init_method=init.xavier_normal_, enable_amp=False):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embeddings_dim = embeddings_dim
        self.padding_idx = padding_idx
        self.enable_amp = enable_amp

        self.weights = flow.nn.Parameter(
            flow.empty(
                (self.num_embeddings, self.embeddings_dim),
                dtype=flow.float32,
                placement=dist.get_layer_placement(0),
                sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.split(0)]),
            )
        )

        init_method(self.weights)
        if self.padding_idx is not None:
            with flow.no_grad():
                self.weights[self.padding_idx].fill_(0)

    def forward(self, input_ids):
        if self.enable_amp:
            weights = flow._C.amp_white_identity(self.weights)

        embeds = flow._C.gather(weights, input_ids, axis=0)
        return embeds


class PositionalEmbedding(flow.nn.Module):
    """The module learns positional embeddings up to a fixed maximum size.
    This module do not need to parallelized along the vocabulary dim.

    Arguments:
        num_embeddings: size of vocabulary.
        embeddings_dim: dimension of embeddings.
        init_method: method to initialize weights.
        enable_amp: whether apply auto mixed precision (amp).
    """
    def __init__(self, num_embeddings, embeddings_dim, init_method=init.xavier_normal_, enable_amp=False):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embeddings_dim = embeddings_dim
        self.enable_amp = enable_amp

        self.weights = flow.nn.Parameter(
            flow.empty(
                (self.num_embeddings, self.embeddings_dim),
                dtype=flow.float32,
                placement=dist.get_layer_placement(0),
                sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
            )
        )

        init_method(self.weights)
        
        self.position_ids = flow.arange(
            self.num_embeddings, 
            dtype=flow.long, 
            placement=dist.get_layer_placement(0), 
            sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast])
        ).expand((1, -1))

    def forward(self, input_ids_shape, past_length=0):
        seq_length = input_ids_shape[1]
        position_ids = self.position_ids[:, past_length: past_length + seq_length]
        position_ids = position_ids.expand(*input_ids_shape)

        if self.enable_amp:
            weights = flow._C.amp_white_identity(self.weights)

        embeds = flow._C.gather(weights, position_ids, axis=0)
        return embeds
