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

__all__ = [
    "ParallelEmbedding",
    "ColumnParallelLinear",
    "RowParallelLinear",
    "LayerNorm",
    "ParallelMLP",
    "ParallelLogits",
]

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


class ColumnParallelLinear(flow.nn.Module):
    """Linear layer with column parallelism.

    The linear layer is defined as Y = XA + b, where A is parallelized along 
    the second dimension as A = [A_1, ..., A_p].

    Arguments:
        layer_idx: the layer index, which determines the placement.
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.
        init_method: method to initialize weights.
        need_gelu: whether to use gelu activation function. (Supporting bias and gelu fusion)
        bias_gelu_fusion: whether fuse add bias and gelu.
    """
    def __init__(self, layer_idx, input_size, output_size, init_method=init.xavier_normal_, 
                 need_gelu=False, bias_gelu_fusion=False):
        super().__init__()
        self.need_gelu = need_gelu
        self.bias_gelu_fusion = bias_gelu_fusion

        # col parallel linear weight sbp: [B, S(1)]
        self.weight = flow.nn.Parameter(
            flow.empty(
                (input_size, output_size),
                dtype=flow.float32,
                placement=dist.get_layer_placement(layer_idx),
                sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.split(1)]),
            )
        )
        init_method(self.weight)
        
        # col parallel linear bias sbp: [B, S(0)]
        self.bias = flow.nn.Parameter(
            flow.empty(
                (output_size,),
                dtype=flow.float32,
                placement=dist.get_layer_placement(layer_idx),
                sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.split(0)]),
            )
        )
        flow.nn.init.zeros_(self.bias)

    def forward(self, x):
        # x sbp: [S(0), B]
        # x.grad sbp: [S(0), P] -> [S(0), B]
        x = x.to_consistent(grad_sbp=x.sbp)
        # matmul sbp sign: [S(0), B] x [B, S(1)] -> [S(0), S(1)]
        # x.grad sbp sign: [S(0), S(1)] x [B, S(0)] (weight.T) -> [S(0), P]
        x = flow.matmul(x, self.weight)
        if self.need_gelu:
            if self.bias_gelu_fusion:
                x = flow._C.fused_bias_add_gelu(x, self.bias, axis=x.ndim - 1)
            else:
                x = x + self.bias
                x = flow.gelu(x)
        else:
            # broadcast_add shape sign:
            # (input_size, output_size) + (output_size, ) = (input_size, output_size)
            # bias_add sbp sign: [S(0), S(1)] + [B, S(0)] = [S(0), S(1)]
            x = x + self.bias

        return x


class RowParallelLinear(flow.nn.Module):
    """Linear layer with row parallelism.

    The linear layer is defined as Y = XA + b, where A is parallelized along 
    the first dimension and X along its second dimension as:

                | A_1 |
                |  .  |
            A = |  .  |         X = [X_1, ..., X_p]
                |  .  |
                | A_p |

    Arguments:
        layer_idx: the layer index, which determines the placement.
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.
        init_method: method to initialize weights.
        output_dropout_prob: dropout probability of output. (Supporting bias and dropout fusion)
        bias_dropout_fusion: whether fuse add bias and dropout.
    """
    def __init__(self, layer_idx, input_size, output_size, init_method=init.xavier_normal_, 
                 output_dropout_prob=0., bias_dropout_fusion=False):
        super().__init__()
        self.output_dropout_prob = output_dropout_prob

        self.bias_dropout_fusion = bias_dropout_fusion
        if not self.bias_dropout_fusion > 0.:
            self.dropout = flow.nn.Dropout(p=self.output_dropout_prob)

        # col parallel linear weight sbp: [B, S(0)]
        self.weight = flow.nn.Parameter(
            flow.empty(
                (input_size, output_size),
                dtype=flow.float32,
                placement=dist.get_layer_placement(layer_idx),
                sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.split(0)]),
            )
        )
        init_method(self.weight)

        # col parallel linear bias sbp: [B, B]
        self.bias = flow.nn.Parameter(
            flow.empty(
                (output_size,),
                dtype=flow.float32,
                placement=dist.get_layer_placement(layer_idx),
                sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
            )
        )
        flow.nn.init.zeros_(self.bias)

    def forward(self, x):
        # x.sbp: [S(0), S(1)]
        # matmul sbp sign: [S(0), S(1)] x [B, S(0)] -> [S(0), P]
        # backward x.grad sbp sign: [S(0), B] x [B, S(1)] (weight.T) -> [S(0), S(1)]
        x = flow.matmul(x, self.weight)
        # x.sbp: [S(0), P] -> [S(0), B]
        x = x.to_consistent(sbp=dist.get_hidden_sbp())
        if self.dropout > 0.:
            if self.bias_dropout_fusion:
                x = flow._C.fused_bias_add_dropout(
                    x, self.bias, p=self.output_dropout_prob, axis=x.ndim - 1
                )
            else:
                x = x + self.bias
                x = self.dropout(x)
        else:
            x = x + self.bias

        return x



class LayerNorm(flow.nn.Module):
    """Layer normalization. This is same as nn.LayerNorm but add placement and sbp attribution.

    Arguments:
        layer_idx: the layer index, which determines the placement.
        normalized_shape: input shape from an expected input of size.
        eps: a value added to the denominator for numerical stability. Default: 1e-5.
    """
    def __init__(self, layer_idx, normalized_shape, eps=1e-5):
        super().__init__()
        self.normalized_shape = (normalized_shape,)
        self.epsilon = eps

        self.weight = flow.nn.Parameter(
            flow.empty(
                normalized_shape,
                dtype=flow.float32,
                placement=dist.get_layer_placement(layer_idx),
                sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
            )
        )
        flow.nn.init.ones_(self.weight)

        self.bias = flow.nn.Parameter(
            flow.empty(
                normalized_shape,
                dtype=flow.float32,
                placement=dist.get_layer_placement(layer_idx),
                sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
            )
        )
        flow.nn.init.zeros_(self.bias)

    def forward(self, x):
        assert x.shape[-len(self.normalized_shape) :] == self.normalized_shape
        begin_norm_axis = x.ndim - len(self.normalized_shape)
        begin_params_axis = x.ndim - len(self.normalized_shape)
        y = flow._C.layer_norm_affine(
            x,
            self.weight,
            self.bias,
            begin_norm_axis=begin_norm_axis,
            begin_params_axis=begin_params_axis,
            epsilon=self.epsilon,
        )
        return y


class ParallelMLP(flow.nn.Module):
    """
    ParallelMLP will take the input with h hidden state, project it to 4 * h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension.

    Arguments:
        layer_idx: the layer index, which determines the placement.
        hidden_size: size of hidden state.
        output_dropout_prob: dropout probability of output.
        init_method: method to initialize the input layer weights.
        output_layer_init_method: method to initialize the output layer weights. If None, use `init_method`.
        bias_gelu_fusion: whether fuse add bias and gelu.
        bias_dropout_fusion: whether fuse add bias and dropout.
    """
    def __init__(self, layer_idx, hidden_size, output_dropout_prob, init_method, output_layer_init_method=None, bias_gelu_fusion=False, bias_dropout_fusion=False):
        super().__init__()
        if output_layer_init_method is None:
            output_layer_init_method = init_method

        self.c_fc = ColumnParallelLinear(layer_idx, hidden_size, hidden_size * 4, init_method=init_method, 
                                         need_gelu=True, bias_gelu_fusion=bias_gelu_fusion)
        self.c_proj = RowParallelLinear(layer_idx, hidden_size * 4, hidden_size, init_method=output_layer_init_method, 
                                        output_dropout_prob=output_dropout_prob, bias_dropout_fusion=bias_dropout_fusion)

    def forward(self, hidden_states):
        # hidden_states shape: (batch_size, seq_length, hidden_size)
        # sbp: [S(0), B]
        h = self.c_fc(hidden_states)
        h = self.c_proj(h)
        return h


class ParallelLogits(flow.nn.Module):
    """LM logits using word embedding weight."""
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states, word_embeddings):
        assert hidden_states.ndim == 3

        w = word_embeddings.to_consistent(placement=hidden_states.placement)
        # h.grad.sbp: [S(0), P] -> [S(0), B]
        h = hidden_states.to_consistent(grad_sbp=hidden_states.sbp)
        # shape sign: (B * S, H) x (H, V) -> (B * S, V)
        # matmul fwd sbp sign: [S(0), B] x [B, S(1)] (wte.T) -> [S(0), S(1)]
        # bwd h.grad sbp sign: [S(0), S(1)] (lgs.grad) x [B, S(0)] (wte) -> [S(0), P] (h.grad)
        logits = flow.matmul(h, w, transpose_b=True)
        return logits
