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
import oneflow.nn as nn
from oneflow.nn import init

import libai.utils.distributed as dist
from libai.layers.embedding import VocabEmbedding


class MT5Embedding(flow.nn.Module):
    def __init__(
        self,
        hidden_size,
        vocab_size,
        embedding_dropout_prob,
        pad_token_id=0,
        init_method=flow.nn.init.xavier_normal_,
        amp_enabled=False,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        self.word_embeddings = VocabEmbedding(
            num_embeddings=vocab_size,
            embedding_dim=hidden_size,
            init_method=init_method,
            amp_enabled=amp_enabled,
            padding_idx=pad_token_id,
        )

        self.embedding_dropout = flow.nn.Dropout(embedding_dropout_prob)

    def forward(self, input_ids):
        word_embeddings = self.word_embeddings(input_ids)
        embeddings = self.embedding_dropout(word_embeddings)
        return embeddings


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

    def forward(self, input_ids):
        weight = flow._C.amp_white_identity(self.weight) if self.amp_enabled else self.weight
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
