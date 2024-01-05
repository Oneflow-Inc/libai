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
import oneflow.nn.functional as F
from oneflow import nn

from libai.config import configurable
from libai.inference.generator.generation_utils import Generator
from libai.layers import Embedding, Linear, RMSLayerNorm, VocabEmbedding
from libai.layers.attention import AttnMaskType
from libai.models.utils import init_method_normal, scaled_init_method_normal
from libai.utils import distributed as dist


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return flow.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    cos = cos[position_ids].unsqueeze(1)
    sin = sin[position_ids].unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings

    def forward(self, x, seq_len=None, cos_cached=None, sin_cached=None):
        if seq_len > self.max_position_embeddings:
            raise ValueError(
                f"The maximum supported length is {self.max_position_embeddings}, "
                f"and the current length is{seq_len}."
            )

        return (
            cos_cached[:seq_len].to_global(placement=x.placement),
            sin_cached[:seq_len].to_global(placement=x.placement),
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
        max_position_embeddings,
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

        rotary_dim = self.head_size
        self.rotary_embed = RotaryEmbedding(
            dim=rotary_dim,
            max_position_embeddings=max_position_embeddings,
        )

        self.gate = flow.nn.Parameter(
            flow.zeros(
                1,
                self.num_heads,
                1,
                1,
                placement=dist.get_layer_placement(layer_idx),
                sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
            )
        )

    def forward(
        self,
        hidden_states: flow.Tensor,
        encoder_states: flow.Tensor = None,
        attention_mask: flow.Tensor = None,
        position_ids=None,
        past_key_value: Tuple[flow.Tensor, flow.Tensor] = None,
        cos_cached: flow.Tensor = None,
        sin_cached: flow.Tensor = None,
        use_cache: bool = False,
        adapter=None,
    ):
        if encoder_states is not None:
            encoder_states = encoder_states.to_global(placement=hidden_states.placement)

        if attention_mask is not None:
            attention_mask = attention_mask.to_global(placement=hidden_states.placement)

        if adapter is not None:
            adapter = adapter.to_global(placement=hidden_states.placement)

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
        cos, sin = self.rotary_embed(
            value, seq_len=kv_seq_len, cos_cached=cos_cached, sin_cached=sin_cached
        )
        query, key = apply_rotary_pos_emb(query, key, cos, sin, position_ids)

        # [1, adapter_len, 4096]
        if adapter is not None:
            adapter_len = adapter.shape[1]
            adapter_qkv = self.query_key_value(adapter)
            adapter_qkv = adapter_qkv.view(1, -1, self.num_heads, 3 * self.head_size)
            adapter_qkv = adapter_qkv.permute(0, 2, 1, 3)  # [1, num_heads, src_len, 3 * head_size]
            _, adapter_key, adapter_value = flow.chunk(adapter_qkv, chunks=3, dim=-1)
            adapter_key = adapter_key.repeat(bsz, 1, 1, 1)
            adapter_value = adapter_value.repeat(bsz, 1, 1, 1)
            key = flow.cat([adapter_key, key], dim=2)
            value = flow.cat([adapter_value, value], dim=2)
            extra_mask = flow.zeros(
                bsz,
                1,
                tgt_len,
                adapter_len,
                sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
                placement=attention_mask.placement,
            )
            attention_mask = flow.cat([extra_mask, attention_mask], dim=-1)

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

        if adapter is not None:
            attention_weights = flow.cat(
                [
                    self.gate.tanh().half()
                    * F.softmax(attention_weights[:, :, :, :adapter_len].float(), dim=-1).to(
                        query.dtype
                    ),
                    F.softmax(attention_weights[:, :, :, adapter_len:].float(), dim=-1).to(
                        query.dtype
                    ),
                ],
                dim=-1,
            )
        else:
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
    def __init__(self, max_positions=1024, dtype=flow.float16, *, layer_idx=0):
        super().__init__()
        self.dtype = dtype
        self.mask = flow.full(
            (max_positions, max_positions),
            flow.finfo(dtype).min,
            placement=dist.get_layer_placement(layer_idx),
            sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
        )
        mask_cond = flow.arange(
            self.mask.size(-1),
            placement=dist.get_layer_placement(layer_idx),
            sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
        )
        self.mask.masked_fill_(mask_cond < (mask_cond + 1).view(self.mask.size(-1), 1), 0)
        self.mask = self.mask.to(dtype)

    def forward(self, input_ids, past_length=0, attention_mask=None, input_dtype=None):
        bsz, tgt_len = input_ids.size()
        casual_mask = self.mask[:tgt_len, :tgt_len]
        if past_length > 0:
            # in case past_key_values are used, we need to add a prefix ones mask to casual mask
            casual_mask = flow.cat(
                [flow.ones(tgt_len, past_length, dtype=self.dtype), casual_mask], dim=-1
            )
        casual_mask = (
            casual_mask.unsqueeze(0).unsqueeze(1).expand(bsz, 1, tgt_len, tgt_len + past_length)
        )
        casual_mask = casual_mask.to_global(sbp=input_ids.sbp)
        if attention_mask is not None:
            bsz, src_len = attention_mask.size()
            attention_mask = (
                attention_mask[:, None, None, :]
                .expand(bsz, 1, tgt_len, src_len)
                .to(casual_mask.dtype)
            )
            attention_mask = attention_mask.to_global(placement=casual_mask.placement)
            casual_mask = casual_mask + attention_mask
        if input_dtype is not None:
            casual_mask = casual_mask.to(input_dtype)
        return casual_mask


class LlamaDecoderLayer(nn.Module):
    def __init__(
        self,
        hidden_size,
        intermediate_size,
        num_attention_heads,
        is_decoder=False,
        rms_norm_eps=1e-5,
        max_position_embeddings=None,
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
        self.max_position_embeddings = max_position_embeddings
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
        cos_cached=None,
        sin_cached=None,
        use_cache=False,
        adapter=None,
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
            cos_cached=cos_cached,
            sin_cached=sin_cached,
            use_cache=use_cache,
            adapter=adapter,
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
            self.max_position_embeddings,
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
        max_position_embeddings=1024,
        rms_norm_eps=1e-5,
        initializer_range=0.02,
        use_scaled_init_for_output_weights=True,
        scale_mask_softmax_fusion=False,
        amp_enabled=False,
        cfg=None,
    ):
        super().__init__()
        self.cfg = cfg
        init_method = init_method_normal(sigma=initializer_range)
        if use_scaled_init_for_output_weights:
            output_layer_init_method = scaled_init_method_normal(initializer_range, hidden_layers)
        else:
            output_layer_init_method = init_method

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
                    max_position_embeddings=max_position_embeddings,
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

        self.adapter_query = Embedding(
            cfg.adapter_len * cfg.adapter_layer, hidden_size, amp_enabled=amp_enabled
        )

        self._set_cos_sin_cache(
            rotary_dim=hidden_size // num_attention_heads,
            seq_len=max_position_embeddings,
            dtype=flow.float32,
            layer_idx=0,
        )

    def _set_cos_sin_cache(self, rotary_dim, seq_len, base=10000, dtype=None, layer_idx=0):
        position = flow.arange(
            0,
            rotary_dim,
            2,
            dtype=dtype,
            sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
            placement=dist.get_layer_placement(layer_idx),
        )
        inv_freq = 1.0 / (base ** (position / rotary_dim))

        t = flow.arange(
            seq_len,
            dtype=inv_freq.dtype,
            sbp=inv_freq.sbp,
            placement=inv_freq.placement,
        )

        freqs = flow.einsum("i,j->ij", t, inv_freq)
        emb = flow.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype))
        self.register_buffer("sin_cached", emb.sin().to(dtype))

    def forward(
        self,
        input_ids,
        attention_mask=None,
        past_key_values=None,
        use_cache=False,
        set_cache=None,
    ):
        with flow.no_grad():
            if use_cache:
                presents = []
            input_ids = input_ids.to_global(placement=dist.get_layer_placement(0))
            hidden_states = self.embed_tokens(input_ids)

            for layer, past_key_value in zip(
                self.layers[: -self.cfg.adapter_layer], past_key_values[: -self.cfg.adapter_layer]
            ):
                hidden_states = layer(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    past_key_value=past_key_value,
                    cos_cached=self.cos_cached,
                    sin_cached=self.sin_cached,
                    use_cache=False,
                    adapter=None,
                )
                if use_cache:
                    hidden_states, present = hidden_states
                    presents.append(present)

        adapter_index = 0
        # [num_adapter_layer, 1, adapter_len, 4096]
        adapter = self.adapter_query.weight.reshape(-1, self.cfg.adapter_len, 4096).unsqueeze(1)
        for layer, past_key_value in zip(
            self.layers[-self.cfg.adapter_layer :], past_key_values[-self.cfg.adapter_layer :]
        ):
            hidden_states = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                past_key_value=past_key_value,
                cos_cached=self.cos_cached,
                sin_cached=self.sin_cached,
                use_cache=False,
                adapter=adapter[adapter_index],  # [1, adapter_len, 4096]
            )
            adapter_index += 1
            if use_cache:
                hidden_states, present = hidden_states
                presents.append(present)

        hidden_states = self.norm(hidden_states)

        if use_cache:
            set_cache(presents)

        return hidden_states


class CrossEntropyLoss(nn.Module):
    def forward(self, logits: flow.Tensor, target: flow.Tensor):
        assert logits.ndim == 3
        assert target.ndim == 2
        assert logits.shape[0:2] == target.shape

        target = target.to_global(placement=logits.placement)
        target = target * (target >= 0)

        lm_loss = flow._C.cross_entropy(
            logits.view(-1, logits.shape[-1]), target.view(-1), ignore_index=0
        )
        return lm_loss


class SFTLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lm_loss = CrossEntropyLoss()

    def forward(self, logits, lm_labels):
        lm_loss = self.lm_loss(logits, lm_labels)
        lm_loss = lm_loss.mean()
        return {"lm_loss": lm_loss}


class LlamaForCausalLM(nn.Module, Generator):
    @configurable
    def __init__(
        self,
        hidden_layers,
        vocab_size,
        hidden_size,
        intermediate_size,
        num_attention_heads,
        max_position_embeddings=1024,
        rms_norm_eps=1e-5,
        initializer_range=0.02,
        use_scaled_init_for_output_weights=True,
        scale_mask_softmax_fusion=False,
        amp_enabled=False,
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
            max_position_embeddings=max_position_embeddings,
            rms_norm_eps=rms_norm_eps,
            initializer_range=initializer_range,
            use_scaled_init_for_output_weights=use_scaled_init_for_output_weights,
            scale_mask_softmax_fusion=scale_mask_softmax_fusion,
            amp_enabled=amp_enabled,
            cfg=cfg,
        )
        self.casual_mask = CasualMask(max_position_embeddings, layer_idx=0)
        self.lm_head = Linear(hidden_size, vocab_size, bias=False, layer_idx=-1)
        self.loss_func = SFTLoss()

        self.past_key_values = [None] * hidden_layers
        self.past_length = 0

    def forward(self, input_ids, attention_mask=None, labels=None, use_cache=False):
        input_ids = input_ids.to_global(placement=dist.get_layer_placement(0))
        attention_mask = (
            attention_mask.to_global(placement=dist.get_layer_placement(0))
            if attention_mask is not None
            else attention_mask
        )
        labels = (
            labels.to_global(placement=dist.get_layer_placement(0))
            if labels is not None
            else labels
        )

        if use_cache and self.past_key_values[0] is not None:
            self.past_length = self.past_key_values[0][0].size(-2)
        else:
            self.past_length = 0

        mask = self.casual_mask(
            input_ids,
            past_length=self.past_length,
            attention_mask=attention_mask,
            input_dtype=self.lm_head.weight.dtype,
        )

        output = self.model(
            input_ids,
            attention_mask=mask,
            past_key_values=self.past_key_values,
            use_cache=use_cache,
            set_cache=self.set_cache,
        )

        logits = self.lm_head(output)

        if labels is not None:
            lm_loss = self.loss_func(logits, labels)
            return lm_loss
        else:
            return {"logits": logits}

    def set_cache(self, past_key_values):
        self.past_length = 0 if past_key_values is None else past_key_values[0][0].shape[2]

        if past_key_values is None:
            past_key_values = [None] * self.cfg.hidden_layers

        assert len(past_key_values) == self.cfg.hidden_layers, (
            f"past_key_values's length {len(past_key_values)} doesn't match "
            f"num_layers:' {self.cfg.hidden_layers}"
        )

    def prepare_inputs_for_generation(self, input_ids: flow.Tensor, **kwargs):
        if "attention_mask" in kwargs:
            attention_mask = kwargs.pop("attention_mask").float()
            attention_mask = attention_mask - 1
            attention_mask.masked_fill_(attention_mask == -1, flow.finfo(flow.float32).min)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    @classmethod
    def from_config(cls, cfg):
        return {
            "hidden_layers": cfg.hidden_layers,
            "vocab_size": cfg.vocab_size,
            "hidden_size": cfg.hidden_size,
            "intermediate_size": cfg.intermediate_size,
            "num_attention_heads": cfg.num_attention_heads,
            "max_position_embeddings": cfg.max_position_embeddings,
            "rms_norm_eps": cfg.rms_norm_eps,
            "initializer_range": cfg.initializer_range,
            "use_scaled_init_for_output_weights": cfg.use_scaled_init_for_output_weights,
            "scale_mask_softmax_fusion": cfg.scale_mask_softmax_fusion,
            "amp_enabled": cfg.amp_enabled,
            "cfg": cfg,
        }

    @staticmethod
    def set_activation_checkpoint(model):
        for module_block in model.modules():
            # Old API in OneFlow 0.8
            if hasattr(module_block, "origin"):
                if isinstance(module_block.origin, LlamaDecoderLayer):
                    module_block.config.activation_checkpointing = True
            else:
                if isinstance(module_block.to(nn.Module), LlamaDecoderLayer):
                    module_block.to(nn.graph.GraphModule).activation_checkpointing = True
