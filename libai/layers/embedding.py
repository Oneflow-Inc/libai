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

import math

import oneflow as flow
from oneflow import nn
from oneflow.nn import init

from libai.utils import distributed as dist


class Embedding(nn.Module):
    """Construct the trainable embedding module, which does not support parallelization.
    This can be used for positional embedding and token type embedding.

    Arguments:
        num_embeddings: size of vocabulary.
        embedding_dim: dimension of embeddings.
        padding_idx: pad index. Defaults to None.
        init_method: method to initialize weights. Defaults to ``flow.nn.init.xavier_normal_``.
        amp_enabled: fp16 option for embedding weight. Defaults to False.
    """

    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        padding_idx=None,
        init_method=init.xavier_normal_,
        amp_enabled=False,
        layer_idx=0,
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
        self.amp_enabled = amp_enabled

        assert num_embeddings > 0
        self.weight = nn.Parameter(
            flow.empty(
                (num_embeddings, embedding_dim),
                dtype=flow.float32,
                placement=dist.get_layer_placement(layer_idx),
                sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
            )
        )
        self.init_method(self.weight)
        # FIXME(lxy): Fill padding_idx is not supported in nd_sbp right now.
        # self._fill_padding_idx_with_zero()

    def forward(self, input_ids):
        weight = flow._C.amp_white_identity(self.weight) if self.amp_enabled else self.weight
        # embeddings with sbp sign: [B, B]
        #   [B, B] x [S(0), B] --> [S(0), B]
        #     ↑         ↑              ↑
        #   embed    pos_ids       pos_embed
        input_embeds = flow._C.gather(weight, input_ids, axis=0)
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


class VocabEmbedding(nn.Module):
    """Construct the word embeddings, which may be split along vocabulary dimension.

    Arguments:
        num_embeddings: size of vocabulary.
        embedding_dim: dimension of embeddings.
        padding_idx: pad index. Defaults to None.
        init_method: method to initialize weights. Defaults to ``flow.nn.init.xavier_normal_``.
        amp_enabled: fp16 option for embedding weight. Defaults to False.
    """

    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        padding_idx=None,
        init_method=init.xavier_normal_,
        amp_enabled=False,
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
        self.amp_enabled = amp_enabled

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
        weight = flow._C.amp_white_identity(self.weight) if self.amp_enabled else self.weight
        # input_ids with shape (batch_size, seq_len), and sbp sign: [S(0), B]

        # Gather forward sbp sign
        # [B, S(0)] x [S(0), B] --> [S(0), P]
        #     ↑           ↑            ↑
        #   embed  input_ids    input_embeds
        input_embeds = flow._C.gather(weight, input_ids, axis=0)
        # Set the embeds sbp from [S(0), P] --> [S(0), B] to get complete embedding results.
        input_embeds = input_embeds.to_global(sbp=dist.get_hidden_sbp())

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


class SinePositionalEmbedding(nn.Module):
    """Construct the sinusoidal positional embeddings.

    Arguments:
        num_embeddings: size of vocabulary.
        embedding_dim: dimension of embeddings.
    """

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

        position_embedding[:, 0::2] = flow.sin(position * div_term)
        position_embedding[:, 1::2] = flow.cos(position * div_term)
        self.register_buffer("position_embedding", position_embedding)

    def forward(self, position_ids):
        position_embeds = flow._C.gather(self.position_embedding, position_ids, axis=0)
        return position_embeds

    def extra_repr(self) -> str:
        s = "num_embeddings={num_embeddings}, embedding_dim={embedding_dim}"
        return s.format(**self.__dict__)


class PatchEmbedding(nn.Module):
    """2D Image to Patch Embedding

    Arguments:
        img_size: size of input image. Default to 224.
        patch_size: embedded patch size. Default to 16.
        in_chans: input channel's size. Default to 3.
        embed_dim: dimension of embedded patch. Default to 768.
        norm_layer: normalization patch embedding or not. Default to None.
        flatten: flatten patch embedding or keep the 2-D shape. Default to True.
        layer_idx: A layer_idx sign which determines the placement. It will be used in pipeline
        parallelism. Default to 0.
    """

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        norm_layer=None,
        flatten=True,
        *,
        layer_idx=0,
    ):
        super().__init__()
        img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        patch_size = patch_size if isinstance(patch_size, tuple) else (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        ).to_global(
            sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
            placement=dist.get_layer_placement(layer_idx),
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert (
            H == self.img_size[0]
        ), f"Input image height ({H}) doesn't match model ({self.img_size[0]})."
        assert (
            W == self.img_size[1]
        ), f"Input image width ({W}) doesn't match model ({self.img_size[1]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x
