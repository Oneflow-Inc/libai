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

import numpy as np
import oneflow as flow
import oneflow.nn.functional as F
from oneflow import einsum, nn

from libai.config import configurable
from libai.layers import LayerNorm, Linear, LMLogits, ParallelCrossEntropyLoss, VocabEmbedding
from libai.models.utils import init_method_normal
from libai.utils import distributed as dist

# rotary positional embedding
# https://arxiv.org/abs/2104.09864


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, *, layer_idx=0):
        super().__init__()
        inv_freq = flow.tensor(
            1.0 / (10000 ** (np.arange(0, dim, 2, dtype=np.float32) / dim)),
            sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
            placement=dist.get_layer_placement(layer_idx),
        )
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, max_seq_len):
        seq = flow.arange(
            max_seq_len,
            dtype=self.inv_freq.dtype,
            sbp=self.inv_freq.sbp,
            placement=self.inv_freq.placement,
        )
        freqs = einsum("i , j -> i j", seq, self.inv_freq)
        return flow.cat((freqs, freqs), dim=-1)


def rotate_half(x):
    # x = rearrange(x, "... (j d) -> ... j d", j=2)
    x = x.reshape(*list(x.shape[:-1]), 2, -1)
    x1 = x[..., 0, :]
    x2 = x[..., 1, :]
    return flow.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(pos, t):
    return (t * pos.cos()) + (rotate_half(t) * pos.sin())


# feedforward
# classic Noam Shazeer paper, except here they use SwiGLU instead of the more popular GEGLU
# https://arxiv.org/abs/2002.05202


class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x


def FeedForward(dim, mult=4, *, layer_idx=0):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        LayerNorm(dim, bias=False, layer_idx=layer_idx),
        Linear(dim, inner_dim * 2, bias=False, parallel="col", layer_idx=layer_idx),
        SwiGLU(),
        Linear(inner_dim, dim, bias=False, parallel="row", layer_idx=layer_idx),
    )


class PalmTransformerLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_head: int = 64,
        num_heads: int = 8,
        ffn_mult: int = 4,
        layernorm_epsilon: float = 1e-5,
        *,
        layer_idx=0
    ):
        """PaLM transformer block with hybrid parallelism"""

        super().__init__()
        self.num_heads = num_heads
        self.dim_head = dim_head
        inner_dim = dim_head * num_heads
        self.attn_inner_dim = num_heads * dim_head
        self.ffn_inner_dim = int(ffn_mult * dim)
        self.ffn_mult = ffn_mult
        self.layer_idx = layer_idx

        # only query has multi head
        # key and value remain as single head
        self.to_q = Linear(dim, inner_dim, bias=False, parallel="col", layer_idx=layer_idx)
        self.to_kv = Linear(dim, dim_head * 2, bias=False, parallel="col", layer_idx=layer_idx)

        self.to_out = Linear(inner_dim, dim, bias=False, parallel="row", layer_idx=layer_idx)

        self.rotary_emb = RotaryEmbedding(self.dim_head, layer_idx=layer_idx)

        self.ffwd = FeedForward(dim, ffn_mult, layer_idx=layer_idx)
        self.norm = LayerNorm(dim, eps=layernorm_epsilon, bias=False, layer_idx=layer_idx)
        self.scale = dim_head ** -0.5

    def get_mask(self, seq):
        if hasattr(self, "mask") and self.mask.shape[-1] >= seq:
            return self.mask[:seq, :seq]

        mask = (
            1
            - flow.ones(
                (seq, seq),
                sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
                placement=dist.get_layer_placement(self.layer_idx),
                dtype=flow.int8,
            ).triu(1)
        )
        self.register_buffer("mask", mask, persistent=False)
        return mask

    def get_rotary_embedding(self, seq):
        if hasattr(self, "pos_emb") and self.pos_emb.shape[-2] >= seq:
            return self.pos_emb[:seq]

        pos_emb = self.rotary_emb(seq)
        self.register_buffer("pos_emb", pos_emb, persistent=False)
        return pos_emb

    def forward(self, x):
        # move x to the stage with right placement
        x = x.to_global(placement=dist.get_layer_placement(self.layer_idx))

        bsz, seq_length = x.size()[0:2]

        # pre-layernorm
        layernorm_output = self.norm(x)

        # fused input linear layer
        query = self.to_q(layernorm_output)
        query = query.view(bsz, -1, self.num_heads, self.dim_head)
        query = query.permute(0, 2, 1, 3)

        key_value = self.to_kv(layernorm_output)
        key, value = flow.chunk(key_value, chunks=2, dim=-1)

        # apply position embedding
        positions = self.get_rotary_embedding(seq_length)
        query, key = map(lambda t: apply_rotary_pos_emb(positions, t), (query, key))

        # apply scale
        query = query * self.scale

        # calculate similarity
        attention_scores = einsum("b h s d, b j d -> b h s j", query, key)

        # apply casual mask
        attention_mask = self.get_mask(seq_length)

        attention_scores = flow.mul(attention_scores, attention_mask)
        attention_scores = attention_scores - 10000.0 * (1 - attention_mask)

        attention_weights = flow.softmax(attention_scores, dim=-1)

        # aggregate values
        attn_out = einsum("b h i j, b j d -> b h i d", attention_weights, value)

        # merge heads
        attn_out = attn_out.transpose(1, 2)
        attn_out = attn_out.view(bsz, seq_length, -1)
        # attn_out = rearrange(attn_out, "b h s d -> b s (h d)")

        attn_out = self.to_out(attn_out)

        # feedforward
        out = self.ffwd(x) + attn_out
        return out + x


class PalmHead(nn.Module):
    def __init__(self, vocab_size, word_embedding_weight):
        super().__init__()
        self.lm_head = LMLogits(vocab_size, bias=False)
        self.loss_func = ParallelCrossEntropyLoss()

        self.word_embedding_weight = word_embedding_weight

    def forward(self, x, lm_labels):
        logits = self.lm_head(x, self.word_embedding_weight)
        if lm_labels is not None:
            lm_loss = self.loss_func(logits, lm_labels)
            lm_loss = lm_loss.mean()
            return {"lm_loss": lm_loss}
        else:
            return {"prediction_scores": logits}


class PaLM(nn.Module):
    @configurable
    def __init__(
        self,
        vocab_size,
        dim,
        depth,
        dim_head=64,
        num_heads=8,
        ffn_mult=4,
        initializer_range=0.02,
        layernorm_eps=1e-12,
        amp_enabled=False,
    ):
        super().__init__()
        init_method = init_method_normal(initializer_range)

        word_embedding = VocabEmbedding(
            vocab_size,
            dim,
            init_method=init_method,
            amp_enabled=amp_enabled,
        )

        self.net = nn.Sequential(
            word_embedding,
            *[
                PalmTransformerLayer(
                    dim,
                    dim_head=dim_head,
                    num_heads=num_heads,
                    ffn_mult=ffn_mult,
                    layernorm_epsilon=layernorm_eps,
                    layer_idx=i,
                )
                for i in range(depth)
            ],
            LayerNorm(dim, bias=False, eps=layernorm_eps, layer_idx=-1),
        )

        self.head = PalmHead(vocab_size, word_embedding.weight)

    def forward(self, input_ids, labels=None):
        output = self.net(input_ids)
        return self.head(output, labels)

    @classmethod
    def from_config(cls, cfg):
        return {
            "vocab_size": cfg.vocab_size,
            "dim": cfg.dim,
            "depth": cfg.depth,
            "dim_head": cfg.dim_head,
            "num_heads": cfg.num_heads,
            "ffn_mult": cfg.ffn_mult,
            "initializer_range": cfg.initializer_range,
            "layernorm_eps": cfg.layernorm_eps,
            "amp_enabled": cfg.amp_enabled,
        }

    @staticmethod
    def set_activation_checkpoint(model):
        for module_block in model.modules():
            if isinstance(module_block.origin, PalmTransformerLayer):
                module_block.config.activation_checkpointing = True

    @staticmethod
    def set_pipeline_stage_id(model: nn.Module):
        dist_utils = dist.get_dist_util()

        for module_block in model.modules():
            if isinstance(module_block.origin, VocabEmbedding):
                module_block.config.set_stage(
                    dist_utils.get_layer_stage_id(0), dist.get_layer_placement(0)
                )
            elif isinstance(module_block.origin, PalmTransformerLayer):
                module_block.config.set_stage(
                    dist_utils.get_layer_stage_id(module_block.layer_idx),
                    dist.get_layer_placement(module_block.layer_idx),
                )
            elif isinstance(module_block.origin, PalmHead):
                module_block.config.set_stage(
                    dist_utils.get_layer_stage_id(-1), dist.get_layer_placement(-1)
                )
        # final layernorm
        model.net[-1].config.set_stage(
            dist_utils.get_layer_stage_id(-1), dist.get_layer_placement(-1)
        )
