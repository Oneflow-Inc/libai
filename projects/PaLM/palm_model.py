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
import oneflow.nn.functional as F
from einops import rearrange
from oneflow import einsum, nn

from libai.config import configurable
from libai.layers import LayerNorm, Linear, LMLogits, ParallelCrossEntropyLoss, VocabEmbedding
from libai.models.utils import init_method_normal, scaled_init_method_normal
from libai.utils import distributed as dist

# rotary positional embedding
# https://arxiv.org/abs/2104.09864


class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = flow.tensor(
            1.0 / (10000 ** (flow.arange(0, dim, 2).float() / dim)),
            sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
            placement=dist.get_layer_placement(0),
        )
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, max_seq_len):
        seq = flow.arange(max_seq_len, dtype=self.inv_freq.dtype)
        freqs = einsum("i , j -> i j", seq, self.inv_freq)
        return flow.cat((freqs, freqs), dim=-1)


def rotate_half(x):
    x = rearrange(x, "... (j d) -> ... j d", j=2)
    x1, x2 = x.unbind(dim=-2)
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


def FeedForward(dim, mult=4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        LayerNorm(dim, bias=False),
        Linear(dim, inner_dim * 2, bias=False, parallel="col"),
        SwiGLU(),
        Linear(inner_dim, dim, bias=False, parallel="row"),
    )


class PalmTransformerLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_head: int = 64,
        num_heads: int = 8,
        ffn_mult: int = 4,
        multi_query: bool = False,
        *,
        layer_idx=0
    ):
        """PaLM transformer block with hybrid parallelism"""

        super().__init__()
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.attn_inner_dim = num_heads * dim_head
        self.ffn_inner_dim = int(ffn_mult * dim)
        self.ffn_mult = ffn_mult
        self.layer_idx = layer_idx

        # build the 2 fused linear layers
        self.multi_query = multi_query

        # calculate the projection size
        if self.multi_query:
            # only query has multi head
            # key and value remain as single head
            input_linear_dim = self.ffn_inner_dim * 2 + dim_head * (num_heads + 2)
        else:
            # conventional multi-head attention
            input_linear_dim = self.ffn_inner_dim * 2 + dim_head * num_heads * 3

        self.fused_input_linear = Linear(dim, input_linear_dim, bias=False, parallel="col")
        self.fused_output_linear = Linear(
            self.ffn_inner_dim + dim_head * num_heads, dim, bias=False, parallel="row"
        )

        self.rotary_emb = RotaryEmbedding(self.dim_head)
        self.swiglu = SwiGLU()
        self.norm = LayerNorm(dim, bias=False)
        self.scale = dim_head ** -0.5

        self.register_buffer("mask", None, persistent=False)
        self.register_buffer("pos_emb", None, persistent=False)

    def get_mask(self, seq):
        if self.mask is not None and self.mask.shape[-1] >= seq:
            return self.mask[:seq, :seq]

        mask = flow.ones(
            (seq, seq),
            sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
            placement=dist.get_layer_placement(self.layer_idx),
            dtype=flow.bool,
        ).triu(1)
        self.register_buffer("mask", mask, persistent=False)
        return mask

    def get_rotary_embedding(self, seq):
        if self.pos_emb is not None and self.pos_emb.shape[-2] >= seq:
            return self.pos_emb[:seq]

        pos_emb = self.rotary_emb(seq)
        self.register_buffer("pos_emb", pos_emb, persistent=False)
        return pos_emb

    def forward(self, x):
        # move x to the stage with right placement
        x = x.to_global(placement=dist.get_layer_placement(self.layer_idx))

        seq_length = x.shape[1]

        # pre-layernorm
        x = self.norm(x)

        # fused input linear layer
        res_pack = self.fused_input_linear(x)

        if self.multi_query:
            q = res_pack.narrow(dim=2, start=0, length=self.attn_inner_dim_per_partition)
            k = res_pack.narrow(
                dim=2, start=self.attn_inner_dim_per_partition, length=self.dim_head_per_partition
            )
            v = res_pack.narrow(
                dim=2,
                start=(self.attn_inner_dim_per_partition + self.dim_head_per_partition),
                length=self.dim_head_per_partition,
            )
            if self.mode_for_gahter is not None:
                k = gather_fwd_reduce_scatter_bwd(
                    k.contiguous(), parallel_mode=self.mode_for_gahter, dim=-1
                )
                v = gather_fwd_reduce_scatter_bwd(
                    v.contiguous(), parallel_mode=self.mode_for_gahter, dim=-1
                )
            ffn_input = res_pack.narrow(
                dim=2,
                start=(self.attn_inner_dim_per_partition + 2 * self.dim_head_per_partition),
                length=self.ffn_inner_dim_per_partition * 2,
            )
        else:
            q = res_pack.narrow(dim=2, start=0, length=self.attn_inner_dim_per_partition)
            k = res_pack.narrow(
                dim=2,
                start=self.attn_inner_dim_per_partition,
                length=self.attn_inner_dim_per_partition,
            )
            v = res_pack.narrow(
                dim=2,
                start=self.attn_inner_dim_per_partition * 2,
                length=self.attn_inner_dim_per_partition,
            )
            ffn_input = res_pack.narrow(
                dim=2,
                start=self.attn_inner_dim_per_partition * 3,
                length=self.ffn_inner_dim_per_partition * 2,
            )

        # arrange the attention embeddings by head
        if not self.multi_query:
            k = rearrange(k, "b s (h d) -> b h s d", h=self.num_heads)
            v = rearrange(v, "b s (h d) -> b h s d", h=self.num_heads)
        q = rearrange(q, "b s (h d) -> b h s d", h=self.num_heads)

        # apply posititon embedding
        positions = self.get_rotary_embedding(seq_length)
        q, k = map(lambda t: apply_rotary_pos_emb(positions, t), (q, k))

        # apply scale
        q = q * self.scale

        # calculate similarity
        if self.multi_query:
            sim = einsum("b h s d, b j d -> b h s j", q, k)
        else:
            # s and n here refer to sequence length
            # n is used only because einsum cannot have 2 same notations
            sim = einsum("b h s d, b h n d -> b h s n", q, k)

        # apply casual mask
        causal_mask = self.get_mask(seq_length)
        sim = sim.masked_fill(causal_mask, -flow.finfo(sim.dtype).max)

        # attention
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        # aggregate values
        if self.multi_query:
            attn_out = einsum("b h i j, b j d -> b h i d", attn, v)
        else:
            attn_out = einsum("b h s n, b h n d -> b h s d", attn, v)

        # merge heads
        attn_out = rearrange(attn_out, "b h s d -> b s (h d)")

        # mlp swiglu
        ffn_input = self.swiglu(ffn_input)

        concat_input = flow.cat([attn_out, ffn_input], dim=-1)
        out = self.fused_output_linear(concat_input)
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
            lm_loss = self.lm_loss(logits, lm_labels)
            lm_loss = lm_loss.mean()
            return {"lm_loss": lm_loss}
        else:
            return {"prediction_scores": logits}


class PaLM(nn.Module):
    @configurable
    def __init__(
        self,
        vocab_size,
        hidden_size,
        hidden_layers,
        num_attention_heads,
        intermediate_size,
        hidden_dropout_prob,
        attention_probs_dropout_prob,
        max_position_embeddings,
        num_tokentypes=2,
        add_pooling_layer=True,
        initializer_range=0.02,
        layernorm_eps=1e-12,
        bias_gelu_fusion=True,
        bias_dropout_fusion=True,
        scale_mask_softmax_fusion=True,
        apply_query_key_layer_scaling=True,
        apply_residual_post_layernorm=False,
        amp_enabled=False,
    ):

        init_method = init_method_normal(initializer_range)
        scaled_init_method = scaled_init_method_normal(initializer_range, hidden_layers)

        word_embedding = VocabEmbedding(
            vocab_size,
            hidden_size,
            init_method=init_method,
            amp_enabled=amp_enabled,
        )

        self.net = nn.Sequential(
            word_embedding,
            *[
                PalmTransformerLayer(
                    hidden_size,
                    intermediate_size,
                    num_attention_heads,
                    attention_dropout_prob=attention_probs_dropout_prob,
                    output_dropout_prob=hidden_dropout_prob,
                    layernorm_epsilon=layernorm_eps,
                    bias_gelu_fusion=bias_gelu_fusion,
                    bias_dropout_fusion=bias_dropout_fusion,
                    scale_mask_softmax_fusion=scale_mask_softmax_fusion,
                    apply_query_key_layer_scaling=apply_query_key_layer_scaling,
                    init_method=init_method,
                    output_layer_init_method=scaled_init_method,
                    apply_residual_post_layernorm=apply_residual_post_layernorm,
                    layer_idx=i,
                )
                for i in range(hidden_layers)
            ],
            LayerNorm(hidden_size, bias=False, eps=layernorm_eps, layer_idx=-1),
        )

        self.head = PalmHead(vocab_size, word_embedding.weight)

    def forward(self, input_ids, labels=None):
        output = self.net(input_ids)
        return self.head(output, labels)

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
                module_block.config.stage_id = dist_utils.get_layer_stage_id(0)
            elif isinstance(module_block.origin, PalmTransformerLayer):
                module_block.config.stage_id = dist_utils.get_layer_stage_id(module_block.layer_idx)
            elif isinstance(module_block.origin, PalmHead):
                module_block.config.stage_id = dist_utils.get_layer_stage_id(-1)

        model.net[-1].config.stage_id = dist_utils.get_layer_stage_id(-1)
