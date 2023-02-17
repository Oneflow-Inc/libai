# coding=utf-8
# Copyright 2021 The OneFlow Authors. All rights reserved.
# Copyright 2022 The HuggingFace Inc. team.
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

from oneflow import nn
import oneflow as flow
from libai.layers import Embedding, LayerNorm
from projects.BLOOM.modeling.transformers import BloomBlock
from projects.BLOOM.modeling.mask import _make_causal_mask, _expand_mask, build_alibi_tensor
from libai.config import configurable
from libai.utils import distributed as dist
from libai.models.utils import init_method_normal, scaled_init_method_normal


class BloomModel(nn.Module):
    @configurable
    def __init__(
        self,
        vocab_size,
        hidden_size,
        hidden_layers,
        n_head,
        padding_idx,
        pretraining_tp = 1,
        slow_but_exact = False,
        initializer_range = 0.02,
        apply_residual_connection_post_layernorm=False,
        hidden_dropout=0,
        attention_dropout=0,
        amp_enabled=False,
        layer_norm_epsilon=1e-12,
        cfg=None,
    ):
        super().__init__()
        self.cfg = cfg
        self.embed_dim = hidden_size
        self.num_heads = n_head
        self.hidden_layers = hidden_layers
        init_method = init_method_normal(initializer_range)
        scaled_init_method = scaled_init_method_normal(initializer_range, hidden_layers)

        self.word_embeddings = Embedding(
            vocab_size, 
            self.embed_dim, 
            padding_idx=padding_idx,
            init_method=init_method,
            amp_enabled=amp_enabled,
            layer_idx=0
        )

        self.word_embeddings_layernorm = LayerNorm(
            self.embed_dim, 
            eps=layer_norm_epsilon,
            elementwise_affine=False,
            layer_idx=0
        )
        
        self.h = flow.nn.ModuleList(
            [
                BloomBlock(
                    hidden_size=hidden_size,
                    n_head=n_head,
                    layer_norm_epsilon=layer_norm_epsilon,
                    hidden_dropout=hidden_dropout,
                    attention_dropout=attention_dropout,
                    pretraining_tp=pretraining_tp,
                    slow_but_exact=slow_but_exact,
                    init_method=init_method,
                    output_layer_init_method=scaled_init_method,
                    apply_residual_connection_post_layernorm=apply_residual_connection_post_layernorm,
                    layer_idx=i,
                )
                for i in range(hidden_layers)
            ]
        )
        
        # Final Layer Norm
        self.ln_f = LayerNorm(
            self.embed_dim, 
            eps=layer_norm_epsilon,
            elementwise_affine=False,
            layer_idx=hidden_layers-1
        )

    @classmethod
    def from_config(cls, cfg):
        return {
            "vocab_size": cfg.vocab_size,
            "hidden_size": cfg.hidden_size,
            "hidden_layers": cfg.hidden_layers,
            "n_head": cfg.n_head,
            "padding_idx": cfg.padding_idx,
            "pretraining_tp": cfg.pretraining_tp,
            "slow_but_exact": cfg.slow_but_exact,
            "apply_residual_connection_post_layernorm": cfg.apply_residual_connection_post_layernorm,
            "hidden_dropout": cfg.hidden_dropout,
            "attention_dropout": cfg.attention_dropout,
            "amp_enabled": cfg.amp_enabled,
            "layer_norm_epsilon": cfg.layer_norm_epsilon,
            "cfg": cfg,
        }

    def _prepare_attn_mask(
        self,
        attention_mask,
        input_shape,
        past_key_values_length,
    ):
        combined_attention_mask = None
        _, src_length = input_shape
        
        if src_length > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape, past_key_values_length=past_key_values_length
            )
        
        # [batch_size, seq_length] -> [batch_size, 1, tgt_length, src_length]
        expanded_attn_mask = _expand_mask(attention_mask, tgt_length=src_length)
        combined_attention_mask = (
            expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask | combined_attention_mask
        )

        return combined_attention_mask

    def _convert_head_mask_to_5d(self, head_mask, num_hidden_layers):
        """-> [num_hidden_layers x batch x num_heads x seq_length x seq_length]"""
        if head_mask.dim() == 1:
            head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
        elif head_mask.dim() == 2:
            head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
        assert head_mask.dim() == 5, f"head_mask.dim != 5, instead {head_mask.dim()}"
        head_mask = head_mask.to(dtype=self.dtype)  # switch to float if need + fp16 compatibility
        return head_mask

    def get_head_mask(
        self, head_mask, num_hidden_layers, is_attention_chunked = False
    ):
        """
        Prepare the head mask if needed.

        Args:
            head_mask (`torch.Tensor` with shape `[num_heads]` or `[num_hidden_layers x num_heads]`, *optional*):
                The mask indicating if we should keep the heads or not (1.0 for keep, 0.0 for discard).
            num_hidden_layers (`int`):
                The number of hidden layers in the model.
            is_attention_chunked: (`bool`, *optional*, defaults to `False`):
                Whether or not the attentions scores are computed by chunks or not.

        Returns:
            `torch.Tensor` with shape `[num_hidden_layers x batch x num_heads x seq_length x seq_length]` or list with
            `[None]` for each layer.
        """
        if head_mask is not None:
            head_mask = self._convert_head_mask_to_5d(head_mask, num_hidden_layers)
            if is_attention_chunked is True:
                head_mask = head_mask.unsqueeze(-1)
        else:
            head_mask = [None] * num_hidden_layers

        return head_mask

    def forward(
        self,
        input_ids = None,
        attention_mask = None,
        past_key_values = None,
        head_mask = None,
        inputs_embeds =  None,
        use_cache = None,
        output_attentions = None,
    ):
        input_ids = (
            input_ids.to_global(placement=dist.get_layer_placement(0))
            if input_ids is not None
            else input_ids
        )
        attention_mask = (
            attention_mask.to_global(placement=dist.get_layer_placement(0))
            if attention_mask is not None
            else attention_mask
        )
        head_mask = (
            head_mask.to_global(placement=dist.get_layer_placement(0))
            if head_mask is not None
            else head_mask
        )
        inputs_embeds = (
            inputs_embeds.to_global(placement=dist.get_layer_placement(0))
            if inputs_embeds is not None
            else inputs_embeds
        )

        if input_ids is not None:
            batch_size, seq_length = input_ids.shape

        if past_key_values is None:
            past_key_values = tuple([None] * len(self.h))
        
        head_mask = self.get_head_mask(head_mask, self.hidden_layers)
        
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        hidden_states = self.word_embeddings_layernorm(inputs_embeds)
        presents = () if use_cache else None
        
        seq_length_with_past = seq_length
        past_key_values_length = 0
        if past_key_values[0] is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length
        if attention_mask is None:
            attention_mask = flow.ones(
                (batch_size, seq_length_with_past), 
                sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
                placement=dist.get_layer_placement(0),
            )

        alibi = build_alibi_tensor(attention_mask, self.num_heads, hidden_states.dtype)

        causal_mask = self._prepare_attn_mask(
            attention_mask,
            input_shape=(batch_size, seq_length),
            past_key_values_length=past_key_values_length,
        )

        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            outputs = block(
                hidden_states,
                layer_past=layer_past,
                attention_mask=causal_mask,
                head_mask=head_mask[i],
                use_cache=use_cache,
                output_attentions=output_attentions,
                alibi=alibi,
            )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)
                
        hidden_states = self.ln_f(hidden_states)

        return {"last_hidden_state": hidden_states, "past_key_values":presents}
