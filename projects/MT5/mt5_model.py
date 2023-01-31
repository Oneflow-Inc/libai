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

from libai.config import configurable
from libai.inference.generator.generation_utils import Generator
from libai.layers import Linear, LMLogits, RMSLayerNorm
from libai.models.utils import init_method_normal, scaled_init_method_normal
from libai.utils import distributed as dist
from projects.MT5.layers.embed_layer import MT5Embedding
from projects.MT5.layers.loss_layer import MT5Loss
from projects.MT5.layers.mask_layer import ExtendedMask
from projects.MT5.layers.transformer_layer import TransformerLayer
from projects.MT5.utils.mt5_loader import T5LoaderHuggerFace


class MT5Model(flow.nn.Module, Generator):
    @configurable
    def __init__(
        self,
        vocab_size,
        hidden_size,
        hidden_layers,
        num_attention_heads,
        head_size,
        intermediate_size,
        embedding_dropout_prob,
        hidden_dropout_prob,
        attention_probs_dropout_prob,
        relative_attention_num_buckets,
        padding_idx=None,
        initializer_range=0.02,
        layernorm_eps=1e-12,
        amp_enabled=False,
        model_type="mt5",
        cfg=None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.model_type = model_type
        init_method = init_method_normal(initializer_range)
        scaled_init_method = scaled_init_method_normal(initializer_range, hidden_layers)
        self.embedding = MT5Embedding(
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            embedding_dropout_prob=embedding_dropout_prob,
            init_method=init_method,
            amp_enabled=amp_enabled,
        )
        self.extended_attn_mask = ExtendedMask()

        encoder_layers = flow.nn.ModuleList(
            [
                TransformerLayer(
                    hidden_size=hidden_size,
                    ffn_hidden_size=intermediate_size,
                    num_attention_heads=num_attention_heads,
                    head_size=head_size,
                    relative_attention_num_buckets=relative_attention_num_buckets,
                    is_decoder=False,
                    attention_dropout_prob=attention_probs_dropout_prob,
                    output_dropout_prob=hidden_dropout_prob,
                    layernorm_epsilon=layernorm_eps,
                    init_method=init_method,
                    output_layer_init_method=scaled_init_method,
                    padding_idx=padding_idx,
                    layer_idx=i,
                    model_type=model_type,
                    has_relative_attention_bias=bool(i == 0),
                )
                for i in range(hidden_layers)
            ]
        )

        encoder_final_layernorm = RMSLayerNorm(
            (hidden_size,),
            eps=layernorm_eps,
            layer_idx=hidden_layers - 1,
        )

        self.encoder = flow.nn.Sequential()
        self.encoder.add_module("layers", encoder_layers)
        self.encoder.add_module("final_layernorm", encoder_final_layernorm)

        decoder_layers = flow.nn.ModuleList(
            [
                TransformerLayer(
                    hidden_size=hidden_size,
                    ffn_hidden_size=intermediate_size,
                    num_attention_heads=num_attention_heads,
                    head_size=head_size,
                    relative_attention_num_buckets=relative_attention_num_buckets,
                    is_decoder=True,
                    attention_dropout_prob=attention_probs_dropout_prob,
                    output_dropout_prob=hidden_dropout_prob,
                    layernorm_epsilon=layernorm_eps,
                    init_method=init_method,
                    output_layer_init_method=scaled_init_method,
                    padding_idx=padding_idx,
                    layer_idx=i,
                    model_type=model_type,
                    has_relative_attention_bias=bool(i - hidden_layers == 0),
                )
                for i in range(hidden_layers, 2 * hidden_layers)
            ]
        )

        decoder_final_layernorm = RMSLayerNorm(
            (hidden_size,),
            eps=layernorm_eps,
            layer_idx=2 * hidden_layers - 1,
        )

        self.decoder = flow.nn.Sequential()
        self.decoder.add_module("layers", decoder_layers)
        self.decoder.add_module("final_layernorm", decoder_final_layernorm)
        self.past_key_values = [None] * len(self.decoder.layers)
        self.encoder_states = None
        self.past_length = 0

        if model_type == "mt5":
            self.lm_head = Linear(
                hidden_size, vocab_size, bias=False, layer_idx=2 * hidden_layers - 1
            )
        else:
            self.lm_head = LMLogits(vocab_size, bias=False)

    @classmethod
    def from_config(cls, cfg):
        return {
            "vocab_size": cfg.vocab_size,
            "hidden_size": cfg.hidden_size,
            "hidden_layers": cfg.hidden_layers,
            "num_attention_heads": cfg.num_attention_heads,
            "head_size": cfg.head_size,
            "intermediate_size": cfg.intermediate_size,
            "embedding_dropout_prob": cfg.embedding_dropout_prob,
            "hidden_dropout_prob": cfg.hidden_dropout_prob,
            "attention_probs_dropout_prob": cfg.attention_probs_dropout_prob,
            "relative_attention_num_buckets": cfg.relative_attention_num_buckets,
            "padding_idx": cfg.padding_idx,
            "initializer_range": cfg.initializer_range,
            "layernorm_eps": cfg.layernorm_eps,
            "amp_enabled": cfg.amp_enabled,
            "model_type": cfg.model_type,
            "cfg": cfg,
        }

    def forward(
        self,
        encoder_input_ids=None,
        decoder_input_ids=None,
        encoder_attn_mask=None,
        decoder_attn_mask=None,
        encoder_decoder_attn_mask=None,
        use_cache=False,
        only_encoder=False,
    ):

        encoder_input_ids = (
            encoder_input_ids.to_global(placement=dist.get_layer_placement(0))
            if encoder_input_ids is not None
            else encoder_input_ids
        )
        decoder_input_ids = (
            decoder_input_ids.to_global(placement=dist.get_layer_placement(0))
            if decoder_input_ids is not None
            else decoder_input_ids
        )
        encoder_attn_mask = (
            encoder_attn_mask.to_global(placement=dist.get_layer_placement(0))
            if encoder_attn_mask is not None
            else encoder_attn_mask
        )
        decoder_attn_mask = (
            decoder_attn_mask.to_global(placement=dist.get_layer_placement(0))
            if decoder_attn_mask is not None
            else decoder_attn_mask
        )
        encoder_decoder_attn_mask = (
            encoder_decoder_attn_mask.to_global(placement=dist.get_layer_placement(0))
            if encoder_decoder_attn_mask is not None
            else encoder_decoder_attn_mask
        )

        if use_cache and self.encoder_states is not None:
            encoder_states = self.encoder_states
        else:
            position_bias = None
            encoder_decoder_position_bias = None
            self.set_cache(encoder_states=None, past_key_values=None)
            encoder_attn_mask = self.extended_attn_mask(encoder_attn_mask)
            enc_embedding_output = self.embedding(encoder_input_ids)
            # transpose [batch_size, seq_len, embed_size] to [seq_len, batch_size, embed_size]
            enc_hidden_states = enc_embedding_output.transpose(0, 1)

            for layer in self.encoder.layers:
                enc_hidden_states, position_bias = layer(
                    enc_hidden_states,
                    encoder_attn_mask,
                    position_bias=position_bias,
                )
            encoder_states = self.encoder.final_layernorm(enc_hidden_states)

        if only_encoder:
            return encoder_states

        decoder_attn_mask = self.extended_attn_mask(
            decoder_attn_mask, decoder_input_ids, is_decoder=True
        )
        encoder_decoder_attn_mask = self.extended_attn_mask(encoder_decoder_attn_mask)

        dec_embedding_output = self.embedding(decoder_input_ids)
        # transpose [batch_size, seq_len, embed_size] to [seq_len, batch_size, embed_size]
        dec_hidden_states = dec_embedding_output.transpose(0, 1)
        if use_cache:
            presents = []

        position_bias = None
        encoder_decoder_position_bias = None
        for layer, past_key_value in zip(self.decoder.layers, self.past_key_values):
            dec_hidden_states, position_bias, encoder_decoder_position_bias = layer(
                dec_hidden_states,
                decoder_attn_mask,
                encoder_states,
                encoder_decoder_attn_mask,
                past_key_value=past_key_value,
                position_bias=position_bias,
                encoder_decoder_position_bias=encoder_decoder_position_bias,
                use_cache=use_cache,
            )
            if use_cache:
                dec_hidden_states, present = dec_hidden_states
                presents.append(present)
        if use_cache:
            self.set_cache(encoder_states, past_key_values=presents)

        decoder_states = self.decoder.final_layernorm(dec_hidden_states)

        if self.cfg.tie_word_embeddings:
            decoder_states = decoder_states * (self.cfg.hidden_size ** -0.5)

        if self.model_type == "mt5":
            logits = self.lm_head(decoder_states)
        else:
            logits = self.lm_head(decoder_states, self.embedding.word_embeddings.weight)

        return {"logits": logits}

    def set_cache(self, encoder_states, past_key_values):
        self.encoder_states = encoder_states
        self.past_length = 0 if past_key_values is None else past_key_values[0][0].shape[2]

        if past_key_values is None:
            past_key_values = [None] * len(self.decoder.layers)
        assert len(past_key_values) == len(self.decoder.layers), (
            f"past_key_values's length {len(past_key_values)} doesn't match "
            f"decoder num_layers' length {self.decoder.layers}"
        )
        self.past_key_values = past_key_values

    def _reorder_cache(self, beam_idx):
        past_key_values = self.past_key_values
        reordered_decoder_past = ()
        for layer_past_states in past_key_values:
            # get the correct batch idx from layer past batch dim
            # batch dim of `past` is at 2nd position
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # need to set correct `past` for each of the four key / value states
                beam_idx = beam_idx.to_global(placement=layer_past_state.placement)
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(0, beam_idx),
                )

            assert reordered_layer_past_states[0].shape == layer_past_states[0].shape
            assert len(reordered_layer_past_states) == len(layer_past_states)

            reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)
        return reordered_decoder_past

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past=None,
        encoder_attn_mask=None,
        encoder_decoder_attn_mask=None,
        use_cache=None,
        encoder_outputs=None,
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            input_ids = input_ids[:, -1:]
            self.past_key_values = past

        self.encoder_states = encoder_outputs
        decoder_attn_maks = flow.ones(
            input_ids.size(),
            dtype=flow.bool,
            sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
            placement=flow.placement("cuda", list(range(dist.get_world_size()))),
        )
        return {
            "decoder_input_ids": input_ids,
            "decoder_attn_mask": decoder_attn_maks,
            "encoder_attn_mask": encoder_attn_mask,
            "encoder_decoder_attn_mask": encoder_decoder_attn_mask,
            "use_cache": use_cache,
        }


class MT5ForPreTraining(flow.nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        if cfg.pretrained_model_path is not None:
            loader = T5LoaderHuggerFace(MT5Model, cfg, cfg.pretrained_model_path)
            self.mt5_model = loader.load()
        else:
            self.mt5_model = MT5Model(cfg)
        self.loss_func = MT5Loss()

    def set_cache(self, encoder_states, past_key_values):
        self.mt5_model.set_cache(encoder_states, past_key_values)

    def forward(
        self,
        encoder_input_ids,
        decoder_input_ids,
        encoder_attn_mask,
        decoder_attn_mask,
        encoder_decoder_attn_mask,
        lm_labels=None,
        loss_mask=None,
        use_cache=False,
    ):
        logits = self.mt5_model(
            encoder_input_ids,
            decoder_input_ids,
            encoder_attn_mask,
            decoder_attn_mask,
            encoder_decoder_attn_mask,
            use_cache=use_cache,
        )["logits"]
        # transpose [seq_len, batch_size, vocab_size] to [batch_size, seq_len, vocab_size]
        logits = logits.transpose(0, 1)
        if lm_labels is not None:
            lm_loss = self.loss_func(logits, lm_labels, loss_mask)
            return lm_loss
        else:
            return {
                "prediction_scores": logits,
            }

    @staticmethod
    def set_pipeline_stage_id(model):
        dist_utils = dist.get_dist_util()

        # Set pipeline parallelism stage_id
        if hasattr(model.mt5_model.encoder.final_layernorm, "config"):
            # Old API in OneFlow 0.8
            for module_block in model.modules():
                if isinstance(module_block.origin, MT5Embedding):
                    module_block.config.set_stage(
                        dist_utils.get_layer_stage_id(0), dist.get_layer_placement(0)
                    )
                elif isinstance(module_block.origin, ExtendedMask):
                    module_block.config.set_stage(
                        dist_utils.get_layer_stage_id(0), dist.get_layer_placement(0)
                    )
                elif isinstance(module_block.origin, TransformerLayer):
                    module_block.config.set_stage(
                        dist_utils.get_layer_stage_id(module_block.layer_idx),
                        dist.get_layer_placement(module_block.layer_idx),
                    )
                elif isinstance(module_block.origin, MT5Loss):
                    module_block.config.set_stage(
                        dist_utils.get_layer_stage_id(-1), dist.get_layer_placement(-1)
                    )

            model.mt5_model.encoder.final_layernorm.config.set_stage(
                dist_utils.get_layer_stage_id(model.mt5_model.encoder.final_layernorm.layer_idx),
                dist.get_layer_placement(model.mt5_model.encoder.final_layernorm.layer_idx),
            )
            model.mt5_model.decoder.final_layernorm.config.set_stage(
                dist_utils.get_layer_stage_id(model.mt5_model.decoder.final_layernorm.layer_idx),
                dist.get_layer_placement(model.mt5_model.decoder.final_layernorm.layer_idx),
            )
        else:
            for module_block in model.modules():
                if isinstance(module_block.to(nn.Module), MT5Embedding):
                    module_block.to(flow.nn.graph.GraphModule).set_stage(
                        dist_utils.get_layer_stage_id(0), dist.get_layer_placement(0)
                    )
                elif isinstance(module_block.to(nn.Module), ExtendedMask):
                    module_block.to(flow.nn.graph.GraphModule).set_stage(
                        dist_utils.get_layer_stage_id(0), dist.get_layer_placement(0)
                    )
                elif isinstance(module_block.to(nn.Module), TransformerLayer):
                    module_block.to(flow.nn.graph.GraphModule).set_stage(
                        dist_utils.get_layer_stage_id(module_block.layer_idx),
                        dist.get_layer_placement(module_block.layer_idx),
                    )
                elif isinstance(module_block.to(nn.Module), MT5Loss):
                    module_block.to(flow.nn.graph.GraphModule).set_stage(
                        dist_utils.get_layer_stage_id(-1), dist.get_layer_placement(-1)
                    )

            model.mt5_model.encoder.final_layernorm.to(flow.nn.graph.GraphModule).set_stage(
                dist_utils.get_layer_stage_id(model.mt5_model.encoder.final_layernorm.layer_idx),
                dist.get_layer_placement(model.mt5_model.encoder.final_layernorm.layer_idx),
            )
            model.mt5_model.decoder.final_layernorm.to(flow.nn.graph.GraphModule).set_stage(
                dist_utils.get_layer_stage_id(model.mt5_model.decoder.final_layernorm.layer_idx),
                dist.get_layer_placement(model.mt5_model.decoder.final_layernorm.layer_idx),
            )

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
