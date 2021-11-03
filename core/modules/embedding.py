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
from core import distribute as dist


class ParallelEmbedding(flow.nn.Module):
    """Embedding parallelized, including token embedding and position embedding, token type embedding is optional.
    Token embedding is parallelized along vocabulary dimension.
    Position embedding ans type embedding are not parallelized.

    Arguments:
        hidden_size: size of hidden state.
        vocab_size: size of vocabulary.
        max_seq_length: maximum size of sequence, which is used for positional embedding.
        type_vocab_size: size of type vocabulary.
        embedding_dropout_prob: dropout probability of embedding.
        init_method: method to initialize weights.
        enable_norm: whether apply layer normalization to embedding.
        enable_amp: whether apply auto mixed precision (amp).
    """
    def __init__(self, hidden_size, vocab_size, max_seq_length, type_vocab_size=0, embedding_dropout_prob=0., 
                 init_method=init.xavier_normal_, enable_norm=False, enable_amp=False, layernorm_epsilon=1e-5):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        self.type_vocab_size = type_vocab_size
        self.output_dropout_prob = output_dropout_prob

        self.enable_norm = enable_norm
        self.enable_amp = enable_amp

        # token embedding shape (vocab_size, hidden_size)
        # sbp: [B, S(0)]
        self.word_embeddings = flow.nn.Parameter(
            flow.empty(
                (self.vocab_size, self.hidden_size),
                dtype=flow.float32,
                placement=dist.get_layer_placement(0),
                sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.split(0)]),
            )
        )

        # position embedding shape (max_seq_length, hidden_size)
        # sbp: [B, B]
        self.position_embeddings = flow.nn.Parameter(
            flow.empty(
                (self.max_seq_length, self.hidden_size),
                dtype=flow.float32,
                placement=dist.get_layer_placement(0),
                sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
            )
        )

        init_method(self.word_embeddings)
        init_method(self.position_embeddings)
        
        if self.enable_amp:
            self.word_embeddings = flow._C.amp_white_identity(self.word_embeddings)
            self.position_embeddings = flow._C.amp_white_identity(self.position_embeddings)

        self.position_ids = flow.arange(
            self.max_seq_length, dtype=flow.long, placement=dist.get_layer_placement(0), 
            sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast])
        ).expand((1, -1))

        if self.type_vocab_size > 0:
            # token type embedding shape (type_vocab_size, hidden_size)
            # sbp: [B, B]
            self.token_type_embeddings = flow.nn.Parameter(
                flow.empty(
                    (self.type_vocab_size, self.hidden_size),
                    dtype=flow.float32,
                    placement=dist.get_layer_placement(0),
                    sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
                )
            )
            init_method(self.token_type_embeddings)

            if self.enable_amp:
                self.token_type_embeddings = flow._C.amp_white_identity(self.token_type_embeddings)
        
        if self.enable_norm:
            self.LayerNorm = LayerNorm(layer_idx=0, normalized_shape=self.hidden_size, eps=layernorm_epsilon)

        self.dropout = flow.nn.Dropout(p=embedding_dropout_prob)


    def forward(self, token_ids, position_ids=None, type_ids=None):
        # shape: (batch_size, seq_len)      sbp: [S(0), B]
        seq_length = token_ids.size(1)

        word_embeds = flow.gather(self.word_embeddings, token_ids, axis=0)
        word_embeds = word_embeds.to_consistent(sbp=dist.get_hidden_sbp())
        
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]
        position_embeds = flow.gather(self.position_embeddings, position_ids, axis=0)
        embeds = word_embeds + position_embeds

        if self.type_vocab_size > 0:
            assert type_ids is not None, "type id is not specified."
            token_type_embeds = flow.gather(self.token_type_embeddings, token_ids, axis=0)
            embeds = embeds + token_type_embeds            
        
        if self.enable_norm:
            embeds = self.LayerNorm(embeds)
        
        embeds = self.dropout(embeds)
        return embeds

