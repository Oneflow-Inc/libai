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
from typing import Tuple

import oneflow as flow
from oneflow import nn

from libai.config import configurable
from libai.inference.generator.generation_utils import Generator
from libai.layers import Linear, RMSLayerNorm, VocabEmbedding
from libai.layers.attention import AttnMaskType
from libai.models.utils import init_method_normal, scaled_init_method_normal
from libai.utils import distributed as dist


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return flow.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    cos = cos[position_ids].unsqueeze(1)  # [seq_len, dim] -> [batch_size, 1, seq_len, head_dim]
    sin = sin[position_ids].unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, dtype=None, layer_idx=0):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        self._set_cos_sin_cache(seq_len=max_position_embeddings, dtype=dtype, layer_idx=layer_idx)

    def _set_cos_sin_cache(self, seq_len, dtype, layer_idx):
        position = flow.arange(
            0,
            self.dim,
            2,
            dtype=dtype,
            sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
            placement=dist.get_layer_placement(layer_idx),
        )
        inv_freq = 1.0 / (self.base ** (position / self.dim))

        self.max_seq_len_cached = seq_len
        t = flow.arange(
            self.max_seq_len_cached,
            dtype=inv_freq.dtype,
            sbp=inv_freq.sbp,
            placement=inv_freq.placement,
        )

        freqs = flow.einsum("i,j->ij", t, inv_freq)
        emb = flow.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype))
        self.register_buffer("sin_cached", emb.sin().to(dtype))

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to_global(placement=x.placement),
            self.sin_cached[:seq_len].to_global(placement=x.placement),
        )


class MLP(nn.Module):
    def __init__(
        self,
        hidden_size,
        intermediate_size,
        init_method=nn.init.xavier_normal_,
        output_layer_init_method=None,
        *,
        layer_idx=0,
    ):
        super().__init__()

        if output_layer_init_method is None:
            output_layer_init_method = init_method

        self.gate_proj = Linear(
            hidden_size,
            intermediate_size,
            bias=False,
            parallel="col",
            init_method=init_method,
            layer_idx=layer_idx,
        )

        self.up_proj = Linear(
            hidden_size,
            intermediate_size,
            bias=False,
            parallel="col",
            init_method=init_method,
            layer_idx=layer_idx,
        )

        self.down_proj = Linear(
            intermediate_size,
            hidden_size,
            bias=False,
            parallel="row",
            init_method=output_layer_init_method,
            layer_idx=layer_idx,
        )

        self.activation_func = nn.SiLU()

    def forward(self, hidden_states):
        gate_out = self.activation_func(self.gate_proj(hidden_states))
        up_out = self.up_proj(hidden_states)
        output = self.down_proj(gate_out * up_out)
        return output


class MultiheadAttention(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        init_method=nn.init.xavier_normal_,
        output_layer_init_method=None,
        scale_mask_softmax_fusion=False,
        attn_mask_type=AttnMaskType.padding,
        *,
        layer_idx=0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        if output_layer_init_method is None:
            output_layer_init_method = init_method

        self.num_heads = num_attention_heads
        self.head_size = hidden_size // num_attention_heads
        self.attn_mask_type = attn_mask_type

        self.norm_factor = 1.0 / math.sqrt(float(self.head_size))

        self.scale_mask_softmax_fusion = scale_mask_softmax_fusion

        self.query_key_value = Linear(
            self.hidden_size,
            self.hidden_size * 3,
            bias=False,
            parallel="col",
            init_method=init_method,
            layer_idx=layer_idx,
        )

        self.o_proj = Linear(
            self.hidden_size,
            self.hidden_size,
            bias=False,
            parallel="row",
            init_method=output_layer_init_method,
            layer_idx=layer_idx,
        )

        self.coeff = None

    def forward(
        self,
        hidden_states: flow.Tensor,
        encoder_states: flow.Tensor = None,
        attention_mask: flow.Tensor = None,
        rotary_emb=None,
        position_ids=None,
        past_key_value: Tuple[flow.Tensor, flow.Tensor] = None,
        use_cache: bool = False,
    ):
        if encoder_states is not None:
            encoder_states = encoder_states.to_global(placement=hidden_states.placement)

        if attention_mask is not None:
            attention_mask = attention_mask.to_global(placement=hidden_states.placement)

        bsz, tgt_len = hidden_states.size()[:2]

        query_key_value = self.query_key_value(hidden_states)
        query_key_value = query_key_value.view(bsz, -1, self.num_heads, 3 * self.head_size)
        query_key_value = query_key_value.permute(
            0, 2, 1, 3
        )  # [bsz, num_heads, src_len, 3 * head_size]
        query, key, value = flow.chunk(query_key_value, chunks=3, dim=-1)

        kv_seq_len = key.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = rotary_emb(value, seq_len=kv_seq_len)
        query, key = apply_rotary_pos_emb(query, key, cos, sin, position_ids)

        if past_key_value is not None:
            past_key, past_value = past_key_value
            key = flow.cat((past_key.type_as(key), key), dim=2)
            value = flow.cat((past_value.type_as(value), value), dim=2)

        # query, key, value: [S(0), S(1)], shape: [bsz, num_heads, seq_length, head_size]
        if use_cache:
            past_key_value = (key, value)

        # [bsz, num_heads, tgt_len, src_len] with [S(0), S(1)]
        attention_scores = flow.matmul(query, key, transpose_b=True, alpha=self.norm_factor)
        attention_weights = attention_scores + attention_mask

        attention_weights = flow.softmax(attention_weights, dim=-1)
        # Context shape: [bsz, num_heads, tgt_len, head_size] with [S(0), S(1)]
        context = flow.matmul(attention_weights, value)

        # Change shape: [bsz, num_heads, tgt_len, head_size] -> [bsz, tgt_len, num_heads, head_size]
        context = context.transpose(1, 2)
        output = self.o_proj(context.flatten(2))

        if use_cache:
            output = (output, past_key_value)

        return output


class CasualMask(nn.Module):
    def __init__(self, max_positions=1024, *, layer_idx=0):
        super().__init__()

        self.mask = flow.full(
            (max_positions, max_positions),
            flow.finfo(flow.float32).min,
            placement=dist.get_layer_placement(layer_idx),
            sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
        )
        mask_cond = flow.arange(
            self.mask.size(-1),
            placement=dist.get_layer_placement(layer_idx),
            sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
        )
        self.mask.masked_fill_(mask_cond < (mask_cond + 1).view(self.mask.size(-1), 1), 0)

    def forward(self, input_ids, past_length=0, attention_mask=None):
        bsz, tgt_len = input_ids.size()
        casual_mask = self.mask[:tgt_len, :tgt_len]
        if past_length > 0:
            # in case past_key_values are used, we need to add a prefix ones mask to casual mask
            casual_mask = flow.cat(
                [flow.ones(tgt_len, past_length, dtype=flow.int8), casual_mask], dim=-1
            )
        casual_mask = (
            casual_mask.unsqueeze(0).unsqueeze(1).expand(bsz, 1, tgt_len, tgt_len + past_length)
        )
        casual_mask = casual_mask.to_global(sbp=input_ids.sbp)
        if attention_mask is not None:
            assert attention_mask.dim() == 4, "please extend the attention mask first"
            casual_mask = casual_mask * attention_mask
        return casual_mask


class LlamaDecoderLayer(nn.Module):
    def __init__(
        self,
        hidden_size,
        intermediate_size,
        num_attention_heads,
        is_decoder=False,
        rms_norm_eps=1e-5,
        init_method=nn.init.xavier_normal_,
        output_layer_init_method=None,
        scale_mask_softmax_fusion=False,
        attn_mask_type=AttnMaskType.padding,
        *,
        layer_idx=0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.rms_norm_eps = rms_norm_eps
        self.attn_mask_type = attn_mask_type

        self.layer_idx = layer_idx
        self.is_decoder = is_decoder

        self.scale_mask_softmax_fusion = scale_mask_softmax_fusion

        self.init_method = init_method
        if output_layer_init_method is None:
            output_layer_init_method = init_method
        self.output_layer_init_method = output_layer_init_method

        self.input_layernorm = RMSLayerNorm(
            self.hidden_size, eps=self.rms_norm_eps, layer_idx=self.layer_idx
        )

        self.self_attn = self.build_attention()
        self.post_attention_layernorm = RMSLayerNorm(
            self.hidden_size, eps=self.rms_norm_eps, layer_idx=self.layer_idx
        )

        self.mlp = MLP(
            self.hidden_size,
            self.intermediate_size,
            self.init_method,
            output_layer_init_method=self.output_layer_init_method,
            layer_idx=self.layer_idx,
        )

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        past_key_value=None,
        rotary_emb=None,
        use_cache=False,
    ):
        hidden_states = hidden_states.to_global(placement=dist.get_layer_placement(self.layer_idx))

        # hidden_states shape: (batch_size, seq_length, hidden_size)
        if attention_mask is not None:
            attention_mask = attention_mask.to_global(
                placement=dist.get_layer_placement(self.layer_idx)
            )

        if past_key_value is not None:
            if self.is_decoder:
                assert len(past_key_value) == 4
                self_attn_past_key_value = past_key_value[:2]
            else:
                self_attn_past_key_value = past_key_value
        else:
            self_attn_past_key_value = None

        layernorm_output = self.input_layernorm(hidden_states)
        attention_output = self.self_attn(
            layernorm_output,
            attention_mask=attention_mask,
            past_key_value=self_attn_past_key_value,
            rotary_emb=rotary_emb,
            use_cache=use_cache,
        )

        if use_cache:
            attention_output, presents = attention_output

        hidden_states = hidden_states + attention_output

        layernorm_output = self.post_attention_layernorm(hidden_states)

        mlp_output = self.mlp(layernorm_output)

        output = hidden_states + mlp_output

        if use_cache:
            output = (output, presents)
        return output

    def build_attention(self):
        return MultiheadAttention(
            self.hidden_size,
            self.num_attention_heads,
            init_method=self.init_method,
            output_layer_init_method=self.output_layer_init_method,
            scale_mask_softmax_fusion=self.scale_mask_softmax_fusion,
            attn_mask_type=self.attn_mask_type,
            layer_idx=self.layer_idx,
        )


class LlamaModel(nn.Module):
    def __init__(
        self,
        hidden_layers,
        vocab_size,
        hidden_size,
        intermediate_size,
        num_attention_heads,
        num_key_value_heads,
        max_position_embeddings=1024,
        rms_norm_eps=1e-5,
        initializer_range=0.02,
        use_scaled_init_for_output_weights=True,
        scale_mask_softmax_fusion=False,
        amp_enabled=False,
        dtype=None,
    ):
        super().__init__()
        init_method = init_method_normal(sigma=initializer_range)
        if use_scaled_init_for_output_weights:
            output_layer_init_method = scaled_init_method_normal(initializer_range, hidden_layers)
        else:
            output_layer_init_method = init_method
        if amp_enabled:
            flow.float16
        else:
            flow.float32
        self.embed_tokens = VocabEmbedding(
            vocab_size, hidden_size, init_method=init_method, amp_enabled=amp_enabled
        )
        self.layers = nn.ModuleList(
            [
                LlamaDecoderLayer(
                    hidden_size,
                    intermediate_size,
                    num_attention_heads,
                    rms_norm_eps=rms_norm_eps,
                    init_method=init_method,
                    output_layer_init_method=output_layer_init_method,
                    scale_mask_softmax_fusion=scale_mask_softmax_fusion,
                    attn_mask_type=AttnMaskType.causal,
                    layer_idx=i,
                )
                for i in range(hidden_layers)
            ]
        )
        self.norm = RMSLayerNorm(hidden_size, eps=rms_norm_eps, layer_idx=-1)

        rotary_dim = self.layers[0].self_attn.head_size
        self.rotary_embed = RotaryEmbedding(
            dim=rotary_dim,
            max_position_embeddings=max_position_embeddings,
            dtype=flow.float32,
            layer_idx=0,
        )

    def forward(
        self,
        input_ids,
        attention_mask=None,
        past_key_values=None,
        use_cache=False,
        set_cache=None,
    ):
        if use_cache:
            presents = []
        input_ids = input_ids.to_global(placement=dist.get_layer_placement(0))
        hidden_states = self.embed_tokens(input_ids)

        for layer, past_key_value in zip(self.layers, past_key_values):
            hidden_states = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                past_key_value=past_key_value,
                rotary_emb=self.rotary_embed,
                use_cache=False,
            )
            if use_cache:
                hidden_states, present = hidden_states
                presents.append(present)

        hidden_states = self.norm(hidden_states)

        if use_cache:
            set_cache(presents)

        return hidden_states


class LlamaForCausalLM(nn.Module, Generator):
    @configurable
    def __init__(
        self,
        hidden_layers,
        vocab_size,
        hidden_size,
        intermediate_size,
        num_attention_heads,
        num_key_value_heads,
        max_position_embeddings=1024,
        rms_norm_eps=1e-5,
        initializer_range=0.02,
        use_scaled_init_for_output_weights=True,
        scale_mask_softmax_fusion=False,
        amp_enabled=False,
        dtype=None,
        cfg=None,
    ):
        super().__init__()
        self.cfg = cfg
        self.model = LlamaModel(
            hidden_layers=hidden_layers,
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            max_position_embeddings=max_position_embeddings,
            rms_norm_eps=rms_norm_eps,
            initializer_range=initializer_range,
            use_scaled_init_for_output_weights=use_scaled_init_for_output_weights,
            scale_mask_softmax_fusion=scale_mask_softmax_fusion,
            amp_enabled=amp_enabled,
            dtype=dtype,
        )
        self.casual_mask = CasualMask(max_position_embeddings, layer_idx=0)
        self.lm_head = Linear(hidden_size, vocab_size, bias=False, layer_idx=-1)

        self.past_key_values = [None] * hidden_layers
        self.past_length = 0

    def forward(self, input_ids, use_cache=False):
        input_ids = input_ids.to_global(placement=dist.get_layer_placement(0))

        if use_cache and self.past_key_values[0] is not None:
            self.past_length = self.past_key_values[0][0].size(-2)
        else:
            self.past_length = 0

        casual_mask = self.casual_mask(
            input_ids,
            past_length=self.past_length,
        )

        output = self.model(
            input_ids,
            attention_mask=casual_mask,
            past_key_values=self.past_key_values,
            use_cache=use_cache,
            set_cache=self.set_cache,
        )

        logits = self.lm_head(output)

        return {"logits": logits}

    def set_cache(self, past_key_values):
        self.past_length = 0 if past_key_values is None else past_key_values[0][0].shape[2]

        if past_key_values is None:
            past_key_values = [None] * self.cfg.hidden_layers

        assert len(past_key_values) == self.cfg.hidden_layers, (
            f"past_key_values's length {len(past_key_values)} doesn't match "
            f"num_layers:' {self.cfg.hidden_layers}"
        )

    @classmethod
    def from_config(cls, cfg):
        return {
            "hidden_layers": cfg.hidden_layers,
            "vocab_size": cfg.vocab_size,
            "hidden_size": cfg.hidden_size,
            "intermediate_size": cfg.intermediate_size,
            "num_attention_heads": cfg.num_attention_heads,
            "num_key_value_heads": cfg.num_key_value_heads,
            "max_position_embeddings": cfg.max_position_embeddings,
            "rms_norm_eps": cfg.rms_norm_eps,
            "initializer_range": cfg.initializer_range,
            "use_scaled_init_for_output_weights": cfg.use_scaled_init_for_output_weights,
            "scale_mask_softmax_fusion": cfg.scale_mask_softmax_fusion,
            "amp_enabled": cfg.amp_enabled,
            "dtype": cfg.dtype,
            "cfg": cfg,
        }
