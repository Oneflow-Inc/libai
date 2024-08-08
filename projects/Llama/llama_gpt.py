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
from oneflow.nn import init

from libai.config import configurable
from libai.inference.generator.generation_utils import Generator
from libai.layers import DropPath, LayerNorm, Linear, RMSLayerNorm, VocabEmbedding

from libai.layers import build_activation
from libai.layers.activation import Activation

# from libai.layers import MLP
from libai.layers.attention import AttnMaskType
from libai.models.utils import init_method_normal, scaled_init_method_normal
from libai.utils import distributed as dist

LayerNorm_Fn = {"default": LayerNorm, "RMSLayerNorm": RMSLayerNorm}


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

        self.activation_func = build_activation(Activation.GeLU)

        # self.activation_func = nn.SiLU()

    def forward(self, hidden_states):
        gate_out = self.activation_func(self.gate_proj(hidden_states))
        up_out = self.up_proj(hidden_states)
        output = self.down_proj(gate_out * up_out)
        return output


class MultiheadAttention(nn.Module):
    """Multi-head attention layer, support self attention and cross attention.

    Args:
        hidden_size: size of hidden state.
        num_attention_heads: number of attention heads.
        is_cross_attention: used to specify whether it is self attention or cross attention.
            Defaults to False.
        attention_dropout_prob: dropout probability of attention weights.
            Defaults to 0.0.
        output_dropout_prob: dropout probability of output. Defaults to 0.0.
        init_method: method to initialize the input layer weights.
            Defaults to ``init.xavier_normal_``.
        output_layer_init_method: method to initialize the output layer weights.
            If None, use ``init_method``.
        bias_dropout_fusion: whether to fuse add bias and dropout.
            Defaults to False.
        scale_mask_softmax_fusion: whether to fuse scale, mask and softmax.
            Defaults to False.
        apply_query_key_layer_scaling: if `True`, scaling the attention score by layer index.
            Defaults to False.
        layer_idx: a layer_idx sign which determines the placements.
            It will be used in pipeline parallelism. Defaults to 0.
    """

    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        is_cross_attention=False,
        attention_dropout_prob=0.0,
        output_dropout_prob=0.0,
        init_method=nn.init.xavier_normal_,
        output_layer_init_method=None,
        bias_dropout_fusion=False,
        scale_mask_softmax_fusion=False,
        apply_query_key_layer_scaling=False,
        attn_mask_type=AttnMaskType.padding,
        use_rotary_position_embeddings=False,
        rotary_dim=None,
        max_position_embeddings=None,
        *,
        layer_idx=0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        if output_layer_init_method is None:
            output_layer_init_method = init_method

        assert (
            hidden_size % num_attention_heads == 0
        ), "hidden_size must be divisible by num_attention_heads."

        self.num_heads = num_attention_heads
        self.head_size = hidden_size // num_attention_heads
        self.attn_mask_type = attn_mask_type

        self.attention_dropout_prob = attention_dropout_prob
        self.dropout = nn.Dropout(p=attention_dropout_prob)
        self.norm_factor = 1.0 / math.sqrt(float(self.head_size))
        self.coeff = 1.0
        if apply_query_key_layer_scaling:
            self.coeff = layer_idx + 1
            self.norm_factor /= self.coeff

        self.is_cross_attention = is_cross_attention
        self.scale_mask_softmax_fusion = scale_mask_softmax_fusion
        self.bias_dropout_fusion = bias_dropout_fusion

        self.use_rotary_position_embeddings = use_rotary_position_embeddings

        if self.bias_dropout_fusion:
            self.output_dropout_prob = output_dropout_prob
        else:
            self.output_dropout = nn.Dropout(p=output_dropout_prob)

        if self.is_cross_attention:
            self.query = Linear(
                self.hidden_size,
                self.hidden_size,
                parallel="col",
                init_method=init_method,
                layer_idx=layer_idx,
            )
            self.key_value = Linear(
                self.hidden_size,
                self.hidden_size * 2,
                parallel="col",
                init_method=init_method,
                layer_idx=layer_idx,
            )
        else:
            self.query_key_value = Linear(
                self.hidden_size,
                self.hidden_size * 3,
                bias=False,
                parallel="col",
                init_method=init_method,
                layer_idx=layer_idx,
                skip_bias_add=False,
            )

        if self.use_rotary_position_embeddings:
            self.rotary_embed = RotaryEmbedding(
                dim=rotary_dim,
                max_position_embeddings=max_position_embeddings,
            )

        self.dense = Linear(
            self.hidden_size,
            self.hidden_size,
            bias=False,
            parallel="row",
            init_method=output_layer_init_method,
            skip_bias_add=self.bias_dropout_fusion,
            layer_idx=layer_idx,
        )

    def forward(
        self,
        hidden_states: flow.Tensor,
        encoder_states: flow.Tensor = None,
        attention_mask: flow.Tensor = None,
        past_key_value: Tuple[flow.Tensor, flow.Tensor] = None,
        use_cache: bool = False,
        position_ids: flow.Tensor = None,
        cos_cached: flow.Tensor = None,
        sin_cached: flow.Tensor = None,
    ):
        """

        Args:
            hidden_states (flow.Tensor): shape is [bsz, tgt_len, hidden_size].
            encoder_states (flow.Tensor, optional): shape is [bsz, src_len, hidden_size].
                Defaults to None.
            attention_mask (flow.Tensor, optional): shape is [bsz, 1, tgt_len, src_len].
                It should be the combination of padding mask and casual mask.
                It is the padding mask of source input when used with self-attention in encoder.
                And it is the combination of padding mask of target input and casual mask when
                used with self-attention in decoder. It is the padding mask of source input when
                used with cross-attention in decoder.
                Defaults to None.
            past_key_value (Tuple[flow.Tensor, flow.Tensor], optional): tuple of key and value,
                each shape is [bsz, num_heads, src_len, head_size]. Defaults to None.
            use_cache (bool, optional): it will be set to True, when the model is in the inference
                phase and used for incremental decoding. Defaults to False.
        """

        # hidden_states, encoder_states: [S(0), B]
        # attention_mask: [S(0), B]

        if encoder_states is not None:
            encoder_states = encoder_states.to_global(placement=hidden_states.placement)

        if attention_mask is not None:
            attention_mask = attention_mask.to_global(placement=hidden_states.placement)

        bsz, tgt_len = hidden_states.size()[:2]

        if self.is_cross_attention:
            # if it is cross attention, key and value should be calculated only once, and the
            # result can be reused.
            query = self.query(hidden_states)
            query = query.view(bsz, -1, self.num_heads, self.head_size)
            query = query.permute(0, 2, 1, 3)
            if past_key_value is not None:
                key, value = past_key_value
            elif encoder_states is not None:
                key_value = self.key_value(encoder_states)
                key_value = key_value.view(bsz, -1, self.num_heads, 2 * self.head_size)
                key_value = key_value.permute(0, 2, 1, 3)
                key, value = flow.chunk(key_value, chunks=2, dim=-1)
            else:
                raise ValueError(
                    "past_key_value and encoder_states cannot be None at the same time."
                )
        else:
            # if it is self attention, query, key, and value are all obtained from hidden_states.
            # when in the inference phase of an incremental decoder,
            # hidden_states is the last-added state,
            # the full key and value could be obtained by concatenating with past_key_value.
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
            if past_key_value is not None:
                past_key, past_value = past_key_value
                key = flow.cat((past_key.type_as(key), key), dim=2)
                value = flow.cat((past_value.type_as(value), value), dim=2)

        # query, key, value: [S(0), S(1)], shape: [bsz, num_heads, seq_length, head_size]
        if use_cache:
            past_key_value = (key, value)

        # [bsz, num_heads, tgt_len, src_len] with [S(0), S(1)]
        attention_scores = flow.matmul(query, key, transpose_b=True, alpha=self.norm_factor)

        # [S(0), S(1)] x [S(0), B] = [S(0), S(1)]
        if attention_mask is not None:
            if self.scale_mask_softmax_fusion:
                if self.attn_mask_type == AttnMaskType.padding:
                    attention_mask = (
                        attention_mask.expand_as(attention_scores) if use_cache else attention_mask
                    )
                    attention_weights = flow._C.fused_scale_mask_softmax_dropout(
                        attention_scores,
                        attention_mask,
                        fill_value=flow.finfo(attention_scores.dtype).min,
                        scale=self.coeff,
                        p=self.attention_dropout_prob,
                    )[0]
            else:
                if self.coeff is not None:
                    attention_scores *= self.coeff
                attention_scores = flow.mul(attention_scores, attention_mask)
                attention_scores = attention_scores - 10000.0 * (1 - attention_mask)
                # TODO(xingyu.liao): graph will occur `where_scalar` errors
                # when using `masked_fill`
                # attention_scores = attention_scores.masked_fill(1 - attention_mask, -10000.0)
                attention_weights = flow.softmax(attention_scores, dim=-1)
                # [bsz, num_heads, tgt_len, src_len]
                attention_weights = self.dropout(attention_weights)
        else:
            if self.scale_mask_softmax_fusion and self.attn_mask_type == AttnMaskType.causal:
                attention_weights = flow._C.fused_scale_tril_softmax_mask_scale(
                    attention_scores,
                    p=self.attention_dropout_prob,
                    diagonal=0,
                    tril_scale_value=self.coeff,
                    tril_fill_value=flow.finfo(attention_scores.dtype).min,
                )[0]
            else:
                attention_weights = flow.softmax(attention_scores, dim=-1)
                # [bsz, num_heads, tgt_len, src_len]
                attention_weights = self.dropout(attention_weights)

        # Context shape: [bsz, num_heads, tgt_len, head_size] with [S(0), S(1)]
        context = flow.matmul(attention_weights, value)
        # Change shape: [bsz, num_heads, tgt_len, head_size] -> [bsz, tgt_len, num_heads, head_size]
        context = context.transpose(1, 2)

        # Concat multi-head results from
        # [bsz, tgt_len, num_heads, head_size] -> [bsz, tgt_len, num_heads * head_size]
        # SBP sign: [S(0), S(2)]
        # [S(0), S(2)] x [B, S(0)] = [S(0), P] -> [S(0), B]
        output = self.dense(context.flatten(2))

        if self.bias_dropout_fusion:
            output, bias = output
            output = flow._C.fused_bias_add_dropout(
                output, bias, p=self.output_dropout_prob, axis=output.ndim - 1
            )
        else:
            output = self.output_dropout(output)

        if use_cache:
            output = (output, past_key_value)

        return output

    def extra_repr(self) -> str:
        return "hidden_size={}, num_heads={}, is_cross_attention={}".format(
            self.hidden_size,
            self.num_heads,
            self.is_cross_attention,
        )


class TransformerLayer(nn.Module):
    """A single transformer layer.

    Transformer layer takes input with size [bsz, seq_length, hidden size] and returns an
    output of the same size.
    The input and output has same sbp sign, (S(0), B).

    Arguments:
        hidden_size: size of hidden state.
        ffn_hidden_size: size of feed forword neural network.
        num_attention_heads: number of attention heads.
        is_decoder: used to specify whether this is transformer encoder layer or transformer
            decoder layer. Default: ``False``.
        attention_dropout_prob: dropout probability of attention weights.
        output_dropout_prob: dropout probability of output.
        layernorm_epsilon: epsilon used in layernorm layer. Default: `1e-5`.
        init_method: method to initialize the input layer weights.
        output_layer_init_method: method to initialize the output layer weights.
            If None, use `init_method`.
        bias_gelu_fusion: whether fuse add bias and gelu. Default: ``False``.
        bias_dropout_fusion: whether fuse add bias and dropout. Default: ``False``.
        scale_mask_softmax_fusion: whether to fuse scale, mask and softmax. Default: ``False``.
        apply_query_key_layer_scaling: if `true`, scaling the attention score by layer index.
            Default: ``False``.
        apply_residual_post_layernorm: if ``true``, use original BERT residual
            connection ordering. Otherwise, use Megatron BERT residual connection which
            is more stable when scaling model size introduced in
            https://arxiv.org/pdf/1909.08053.pdf.
            Default: ``False``.
        layer_idx: the layer index, which determines the placement.
    """

    def __init__(
        self,
        hidden_size,
        ffn_hidden_size,
        num_attention_heads,
        is_decoder=False,
        attention_dropout_prob=0.0,
        output_dropout_prob=0.0,
        drop_path_prob=0.0,
        layernorm_epsilon=1e-5,
        layernorm_class="defaule",
        init_method=nn.init.xavier_normal_,
        output_layer_init_method=None,
        bias_gelu_fusion=False,
        bias_dropout_fusion=False,
        scale_mask_softmax_fusion=False,
        apply_query_key_layer_scaling=False,
        apply_residual_post_layernorm=False,
        attn_mask_type=AttnMaskType.padding,
        use_rotary_position_embeddings=False,
        max_position_embeddings=1024,
        *,
        layer_idx=0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.ffn_hidden_size = ffn_hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_dropout_prob = attention_dropout_prob
        self.output_dropout_prob = output_dropout_prob
        self.layernorm_epsilon = layernorm_epsilon
        self.attn_mask_type = attn_mask_type

        self.layer_idx = layer_idx
        self.is_decoder = is_decoder

        self.bias_gelu_fusion = bias_gelu_fusion
        self.bias_dropout_fusion = bias_dropout_fusion
        self.scale_mask_softmax_fusion = scale_mask_softmax_fusion
        self.apply_query_key_layer_scaling = apply_query_key_layer_scaling
        self.apply_residual_post_layernorm = apply_residual_post_layernorm
        self.use_rotary_position_embeddings = use_rotary_position_embeddings
        self.max_position_embeddings = max_position_embeddings

        self.init_method = init_method
        if output_layer_init_method is None:
            output_layer_init_method = init_method
        self.output_layer_init_method = output_layer_init_method

        self.drop_path = DropPath(drop_path_prob) if drop_path_prob > 0.0 else nn.Identity()

        LayerNorm = LayerNorm_Fn[layernorm_class]
        self.input_layernorm = LayerNorm(
            self.hidden_size, eps=self.layernorm_epsilon, layer_idx=self.layer_idx
        )

        self.self_attention = self.build_attention(is_cross_attention=False)
        self.post_attention_layernorm = LayerNorm(
            self.hidden_size, eps=self.layernorm_epsilon, layer_idx=self.layer_idx
        )

        if self.is_decoder:
            self.cross_attention = self.build_attention(is_cross_attention=True)
            self.post_cross_attention_layernorm = LayerNorm(
                self.hidden_size, eps=self.layernorm_epsilon, layer_idx=self.layer_idx
            )

        self.mlp = MLP(
            self.hidden_size,
            self.ffn_hidden_size,
            self.init_method,
            output_layer_init_method=self.output_layer_init_method,
            layer_idx=self.layer_idx,
        )

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        encoder_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        use_cache=False,
        sin_cached=None,
        cos_cached=None,
    ):
        """
        Args:
            hidden_states: shape is (batch_size, seq_length, hidden_size),
                sbp signature is (S(0), B).
            attention_mask: the combination of key padding mask and casual mask of hidden states
                with shape (batch_size, 1, seq_length, seq_length) and the sbp
                signature is (S(0), B),
            encoder_states: encoder output with shape (batch_size, seq_length, hidden_size)
                and the sbp signature is (S(0), B), which will be used in cross attention.
            encoder_attention_mask: key padding mask of encoder states with shape
                (batch_size, 1, seq_length, seq_length) and the sbp signature is (S(0), B).
            past_key_value: tuple of key and value, each shape is
                (seq_length, bsz, num_heads, head_size), For decoder layer,
                the past_key_value contains the states both from self attention
                and cross attention.
            use_cache: it will be set to `True` when the model is in the inference phase and
                used for incremental decoding.
        """
        # Change placement for pipeline parallelsim
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
                cross_attn_past_key_value = past_key_value[2:]
            else:
                self_attn_past_key_value = past_key_value
                cross_attn_past_key_value = None
        else:
            self_attn_past_key_value, cross_attn_past_key_value = None, None

        layernorm_output = self.input_layernorm(hidden_states)
        attention_output = self.self_attention(
            layernorm_output,
            attention_mask=attention_mask,
            past_key_value=self_attn_past_key_value,
            use_cache=use_cache,
            sin_cached=sin_cached,
            cos_cached=cos_cached,
        )
        attention_output = self.drop_path(attention_output)

        if use_cache:
            attention_output, presents = attention_output

        if self.apply_residual_post_layernorm:
            residual = layernorm_output
        else:
            residual = hidden_states

        hidden_states = residual + attention_output

        layernorm_output = self.post_attention_layernorm(hidden_states)

        if self.is_decoder:
            attention_output = self.cross_attention(
                layernorm_output,
                encoder_states,
                attention_mask=encoder_attention_mask,
                past_key_value=cross_attn_past_key_value,
                use_cache=use_cache,
            )

            if use_cache:
                attention_output, decoder_presents = attention_output
                presents += decoder_presents

            attention_output = self.drop_path(attention_output)
            if self.apply_residual_post_layernorm:
                residual = layernorm_output
            else:
                residual = hidden_states

            hidden_states = residual + attention_output
            layernorm_output = self.post_cross_attention_layernorm(hidden_states)

        mlp_output = self.mlp(layernorm_output)
        mlp_output = self.drop_path(mlp_output)

        if self.apply_residual_post_layernorm:
            residual = layernorm_output
        else:
            residual = hidden_states

        output = residual + mlp_output

        if use_cache:
            output = (output, presents)
        return output

    def build_attention(self, is_cross_attention=False):
        return MultiheadAttention(
            self.hidden_size,
            self.num_attention_heads,
            is_cross_attention=is_cross_attention,
            attention_dropout_prob=self.attention_dropout_prob,
            output_dropout_prob=self.output_dropout_prob,
            init_method=self.init_method,
            output_layer_init_method=self.output_layer_init_method,
            bias_dropout_fusion=self.bias_dropout_fusion,
            scale_mask_softmax_fusion=self.scale_mask_softmax_fusion,
            apply_query_key_layer_scaling=self.apply_query_key_layer_scaling,
            attn_mask_type=self.attn_mask_type,
            use_rotary_position_embeddings=self.use_rotary_position_embeddings,
            max_position_embeddings=self.max_position_embeddings,
            layer_idx=self.layer_idx,
        )


class Transformer(nn.Module):
    def __init__(
        self,
        hidden_layers,
        hidden_size,
        ffn_hidden_size,
        num_attention_heads,
        attention_dropout_prob=0.0,
        output_dropout_prob=0.0,
        layernorm_epsilon=1e-5,
        layernorm_class="default",
        init_method=init.xavier_normal_,
        output_layer_init_method=None,
        bias_gelu_fusion=False,
        bias_dropout_fusion=False,
        scale_mask_softmax_fusion=False,
        apply_query_key_layer_scaling=False,
        apply_residual_post_layernorm=False,
        use_rotary_position_embeddings=False,
        max_position_embeddings=1024,
    ):
        super().__init__()
        self.hidden_layers = hidden_layers

        def build_layer(layer_number):
            return TransformerLayer(
                hidden_size,
                ffn_hidden_size,
                num_attention_heads,
                attention_dropout_prob=attention_dropout_prob,
                output_dropout_prob=output_dropout_prob,
                layernorm_epsilon=layernorm_epsilon,
                layernorm_class=layernorm_class,
                init_method=init_method,
                output_layer_init_method=output_layer_init_method,
                bias_gelu_fusion=bias_gelu_fusion,
                bias_dropout_fusion=bias_dropout_fusion,
                scale_mask_softmax_fusion=scale_mask_softmax_fusion,
                apply_query_key_layer_scaling=apply_query_key_layer_scaling,
                apply_residual_post_layernorm=apply_residual_post_layernorm,
                attn_mask_type=AttnMaskType.causal,
                use_rotary_position_embeddings=use_rotary_position_embeddings,
                max_position_embeddings=max_position_embeddings,
                layer_idx=layer_number,
            )

        self.layers = nn.ModuleList([build_layer(i) for i in range(self.hidden_layers)])
        LayerNorm = LayerNorm_Fn[layernorm_class]
        self.layernorm_f = LayerNorm(hidden_size, eps=layernorm_epsilon, layer_idx=-1)

    def forward(self, hidden_states, attention_mask, sin_cached, cos_cached):
        # hidden_states shape: (batch_size, seq_length, hidden_size)
        # sbp: [S(0), B]
        for i, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states, attention_mask, sin_cached=sin_cached, cos_cached=cos_cached
            )

        output = self.layernorm_f(hidden_states)

        return output


class GPTEmbedding(nn.Module):
    def __init__(
        self,
        vocab_size,
        hidden_size,
        max_seq_length,
        init_method=init.xavier_normal_,
        embedding_dropout_prob=0.0,
        amp_enabled=False,
    ):
        super().__init__()
        self.token_embeddings = VocabEmbedding(
            vocab_size, hidden_size, init_method=init_method, amp_enabled=amp_enabled
        )
        self.dropout = nn.Dropout(embedding_dropout_prob)

    def forward(self, input_ids):
        token_embeds = self.token_embeddings(input_ids)
        input_embeds = self.dropout(token_embeds)
        return input_embeds


class GPTModel(nn.Module):
    """GPT-2 language model. The output of the forward method is logits.

    Args:
        hidden_layers (int): The number of ``TransformerLayer`` in the gpt model.
        vocab_size (int): The size of vocabulary file.
        hidden_size (int): The size of hidden states.
        ffn_hidden_size (int):
            The size of intermediate layer in feed-forward network for each ``TransformerLayer``.
        num_attention_heads (int):
            The number of attention heads for each attention layer of ``TransformerLayer``.
        max_seq_length (int, optional):
            Max sequence length of input, defines the shape of Position Embeddings in GPTEmebedding.
            Defaults to 1024.
        embedding_dropout_prob (float, optional):
            The dropout ratio for the output of GPTEmbedding Layer. Defaults to 0.0.
        attention_dropout_prob (float, optional):
            The dropout ratio for the output of each attention layer in ``TransformerLayer``.
            Defaults to 0.0.
        output_dropout_prob (float, optional):
            The dropout ratio for the output for each TransformerLayer. Defaults to 0.0.
        layernorm_epsilon (float, optional):
            The epsilon of LayerNorm layer. Defaults to 1e-5.
        initializer_range (float, optional):
            Sigma of the normal distribution in the initialization method. Defaults to 0.02.
        use_scaled_init_for_output_weights (bool, optional): Defaults to ``True``.
        bias_gelu_fusion (bool, optional):
            Whether or not to fuse the computing of bias and gelu. Defaults to ``False``.
        bias_dropout_fusion (bool, optional):
            Whether or not to fuse the computing of dropout and bias. Defaults to ``False``.
        scale_mask_softmax_fusion (bool, optional):
            Whether to fuse the computing of mask and softmax in attention layers.
            Defaults to ``False``.
        apply_query_key_layer_scaling (bool, optional):
            Whether or not to use layer index related scaling in computing attention scores.
            If ``True``, the scaling factor equals to sqrt(d) * (layer_index + 1).
            Defaults to ``False``.
        apply_residual_post_layernorm (bool, optional):
            If set ``True``, use original BERT residual connection ordering otherwise use Megatron
            BERT residual connection which is more stable when scaling model size introduced in
            https://arxiv.org/pdf/1909.08053.pdf.
            Default: ``False``.
        amp_enabled (bool, optional):
            Whether or not to set fp16 for embedding weight in T5 model. Defaults to ``False``.
    """

    def __init__(
        self,
        hidden_layers,
        vocab_size,
        hidden_size,
        ffn_hidden_size,
        num_attention_heads,
        max_seq_length=1024,
        embedding_dropout_prob=0.0,
        attention_dropout_prob=0.0,
        output_dropout_prob=0.0,
        layernorm_epsilon=1e-5,
        layernorm_class="default",
        initializer_range=0.02,
        use_scaled_init_for_output_weights=True,
        bias_gelu_fusion=False,
        bias_dropout_fusion=False,
        scale_mask_softmax_fusion=False,
        apply_query_key_layer_scaling=False,
        apply_residual_post_layernorm=False,
        amp_enabled=False,
        use_rotary_position_embeddings=False,
    ):
        super().__init__()
        init_method = init_method_normal(sigma=initializer_range)
        if use_scaled_init_for_output_weights:
            output_layer_init_method = scaled_init_method_normal(initializer_range, hidden_layers)
        else:
            output_layer_init_method = init_method

        self.embeddings = GPTEmbedding(
            vocab_size,
            hidden_size,
            max_seq_length,
            init_method=init_method,
            embedding_dropout_prob=embedding_dropout_prob,
            amp_enabled=amp_enabled,
        )

        self.transformer = Transformer(
            hidden_layers,
            hidden_size,
            ffn_hidden_size,
            num_attention_heads,
            attention_dropout_prob=attention_dropout_prob,
            output_dropout_prob=output_dropout_prob,
            layernorm_epsilon=layernorm_epsilon,
            layernorm_class=layernorm_class,
            init_method=init_method,
            output_layer_init_method=output_layer_init_method,
            bias_gelu_fusion=bias_gelu_fusion,
            bias_dropout_fusion=bias_dropout_fusion,
            scale_mask_softmax_fusion=scale_mask_softmax_fusion,
            apply_query_key_layer_scaling=apply_query_key_layer_scaling,
            apply_residual_post_layernorm=apply_residual_post_layernorm,
            use_rotary_position_embeddings=use_rotary_position_embeddings,
            max_position_embeddings=max_seq_length,
        )

        self.lm_head = Linear(hidden_size, vocab_size, bias=False, layer_idx=-1)

        self._set_cos_sin_cache(
            rotary_dim=hidden_size // num_attention_heads,
            seq_len=max_seq_length,
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

    @classmethod
    def from_config(cls, cfg):
        return {
            "hidden_layers": cfg.hidden_layers,
            "vocab_size": cfg.vocab_size,
            "hidden_size": cfg.hidden_size,
            "ffn_hidden_size": cfg.ffn_hidden_size,
            "num_attention_heads": cfg.num_attention_heads,
            "max_seq_length": cfg.max_seq_length,
            "embedding_dropout_prob": cfg.embedding_dropout_prob,
            "attention_dropout_prob": cfg.attention_dropout_prob,
            "output_dropout_prob": cfg.output_dropout_prob,
            "layernorm_epsilon": cfg.layernorm_epsilon,
            "initializer_range": cfg.initializer_range,
            "use_scaled_init_for_output_weights": cfg.use_scaled_init_for_output_weights,
            "bias_gelu_fusion": cfg.bias_gelu_fusion,
            "bias_dropout_fusion": cfg.bias_dropout_fusion,
            "scale_mask_softmax_fusion": cfg.scale_mask_softmax_fusion,
            "apply_query_key_layer_scaling": cfg.apply_query_key_layer_scaling,
            "apply_residual_post_layernorm": cfg.apply_residual_post_layernorm,
            "amp_enabled": cfg.amp_enabled,
        }

    def forward(self, input_ids):
        """

        Args:
            input_ids (flow.LongTensor): Indices of input sequence tokens in vocabulary.

        Returns:
            flow.Tensor: logits
        """

        input_ids = input_ids.to_global(placement=dist.get_layer_placement(0))
        input_embeds = self.embeddings(input_ids)

        transformer_output = self.transformer(
            input_embeds,
            attention_mask=None,
            cos_cached=self.cos_cached,
            sin_cached=self.sin_cached,
        )

        output = self.lm_head(transformer_output)

        return output


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
        scale_mask_softmax_fusion=True,
        amp_enabled=False,
        cfg=None,
    ):
        super().__init__()
        self.cfg = cfg
        self.model = GPTModel(
            hidden_layers=hidden_layers,
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            ffn_hidden_size=intermediate_size,
            num_attention_heads=num_attention_heads,
            max_seq_length=max_position_embeddings,
            layernorm_epsilon=rms_norm_eps,
            layernorm_class="RMSLayerNorm",
            initializer_range=initializer_range,
            use_scaled_init_for_output_weights=use_scaled_init_for_output_weights,
            scale_mask_softmax_fusion=scale_mask_softmax_fusion,
            amp_enabled=amp_enabled,
            use_rotary_position_embeddings=True,
        )
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

        logits = self.model(
            input_ids,
        )

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
                if isinstance(module_block.origin, TransformerLayer):
                    module_block.config.activation_checkpointing = True
            else:
                if isinstance(module_block.to(nn.Module), TransformerLayer):
                    module_block.to(nn.graph.GraphModule).activation_checkpointing = True


if __name__ == "__main__":
    from libai.data.structures import DistTensorData
    import torch
    model = LlamaForCausalLM(2, 20000, 768, int(768 * 4), 12)
    input_ids = DistTensorData(flow.LongTensor([[1, 2, 3, 4], [2, 3, 4, 5]]))
    input_ids.to_global()
    ans = model(input_ids.tensor)
    for key, value in model.named_parameters():
        print(key)

    dic = torch.load("meta-llama/Llama-2-7b-hf/pytorch_model-00001-of-00002.bin")
    dic2 = torch.load("meta-llama/Llama-2-7b-hf/pytorch_model-00002-of-00002.bin")
    # print(model.parameters().keys())
    print()
