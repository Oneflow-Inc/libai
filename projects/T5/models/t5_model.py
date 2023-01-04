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
from libai.layers import Linear, LMLogits
from libai.models.t5_model import T5Loss
from libai.models.utils import init_method_normal, scaled_init_method_normal
from libai.utils import distributed as dist
from projects.MT5.utils.mt5_loader import T5LoaderHuggerFace
from projects.T5.models.embedding import T5Embedding
from projects.T5.models.layer_norm import LayerNorm
from projects.T5.models.transformer_layer import TransformerLayer
from projects.T5.utils.mask import ExtendedMask


class T5Model(flow.nn.Module):
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
        initializer_range=0.02,
        layernorm_eps=1e-12,
        amp_enabled=False,
        model_type="t5",
    ) -> None:
        super().__init__()
        self.model_type = model_type
        init_method = init_method_normal(initializer_range)
        scaled_init_method = scaled_init_method_normal(initializer_range, hidden_layers)
        self.embedding = T5Embedding(
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
                    layer_idx=i,
                    model_type=model_type,
                    has_relative_attention_bias=bool(i == 0),
                )
                for i in range(hidden_layers)
            ]
        )

        encoder_final_layernorm = LayerNorm(
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
                    layer_idx=i,
                    model_type=model_type,
                    has_relative_attention_bias=bool(i - hidden_layers == 0),
                )
                for i in range(hidden_layers, 2 * hidden_layers)
            ]
        )

        decoder_final_layernorm = LayerNorm(
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
            "initializer_range": cfg.initializer_range,
            "layernorm_eps": cfg.layernorm_eps,
            "amp_enabled": cfg.amp_enabled,
            "model_type": cfg.model_type,
        }

    def forward(
        self,
        encoder_input_ids,
        decoder_input_ids,
        encoder_attn_mask,
        decoder_attn_mask,
        encoder_decoder_attn_mask,
        use_cache=False,
    ):
        encoder_input_ids = encoder_input_ids.to_global(placement=dist.get_layer_placement(0))
        decoder_input_ids = decoder_input_ids.to_global(placement=dist.get_layer_placement(0))
        encoder_attn_mask = encoder_attn_mask.to_global(placement=dist.get_layer_placement(0))
        decoder_attn_mask = decoder_attn_mask.to_global(placement=dist.get_layer_placement(0))
        encoder_decoder_attn_mask = encoder_decoder_attn_mask.to_global(
            placement=dist.get_layer_placement(0)
        )

        if use_cache and self.encoder_states is not None:
            encoder_states = self.encoder_states
        else:
            position_bias = None
            encoder_decoder_position_bias = None
            self.set_cache(encoder_states=None, past_key_values=None)
            encoder_attn_mask = self.extended_attn_mask(encoder_attn_mask)
            enc_embedding_output = self.embedding(encoder_input_ids)
            enc_hidden_states = enc_embedding_output

            for layer in self.encoder.layers:
                enc_hidden_states, position_bias = layer(
                    enc_hidden_states,
                    encoder_attn_mask,
                    position_bias=position_bias,
                )
            encoder_states = self.encoder.final_layernorm(enc_hidden_states)

        decoder_attn_mask = self.extended_attn_mask(
            decoder_attn_mask, decoder_input_ids, is_decoder=True
        )
        encoder_decoder_attn_mask = self.extended_attn_mask(encoder_decoder_attn_mask)

        dec_embedding_output = self.embedding(decoder_input_ids)
        dec_hidden_states = dec_embedding_output
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

        if self.model_type == "mt5":
            logits = self.lm_head(decoder_states)
        else:
            logits = self.lm_head(decoder_states, self.embedding.word_embeddings.weight)

        return logits

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


class T5ForPreTraining(flow.nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        if cfg.pretrained_model_path is not None:
            loader = T5LoaderHuggerFace(T5Model, cfg, cfg.pretrained_model_path)
            self.t5_model = loader.load()
        else:
            self.t5_model = T5Model(cfg)
        self.loss_func = T5Loss()

    def set_cache(self, encoder_states, past_key_values):
        self.t5_model.set_cache(encoder_states, past_key_values)

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
        logits = self.t5_model(
            encoder_input_ids,
            decoder_input_ids,
            encoder_attn_mask,
            decoder_attn_mask,
            encoder_decoder_attn_mask,
            use_cache=use_cache,
        )

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
        if hasattr(model.t5_model.encoder.final_layernorm, "config"):
            # Old API in OneFlow 0.8
            for module_block in model.modules():
                if isinstance(module_block.origin, T5Embedding):
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
                elif isinstance(module_block.origin, T5Loss):
                    module_block.config.set_stage(
                        dist_utils.get_layer_stage_id(-1), dist.get_layer_placement(-1)
                    )

            model.t5_model.encoder.final_layernorm.config.set_stage(
                dist_utils.get_layer_stage_id(model.t5_model.encoder.final_layernorm.layer_idx),
                dist.get_layer_placement(model.t5_model.encoder.final_layernorm.layer_idx),
            )
            model.t5_model.decoder.final_layernorm.config.set_stage(
                dist_utils.get_layer_stage_id(model.t5_model.decoder.final_layernorm.layer_idx),
                dist.get_layer_placement(model.t5_model.decoder.final_layernorm.layer_idx),
            )
        else:
            for module_block in model.modules():
                if isinstance(module_block.to(nn.Module), T5Embedding):
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
                elif isinstance(module_block.to(nn.Module), T5Loss):
                    module_block.to(flow.nn.graph.GraphModule).set_stage(
                        dist_utils.get_layer_stage_id(-1), dist.get_layer_placement(-1)
                    )

            model.t5_model.encoder.final_layernorm.to(flow.nn.graph.GraphModule).set_stage(
                dist_utils.get_layer_stage_id(model.t5_model.encoder.final_layernorm.layer_idx),
                dist.get_layer_placement(model.t5_model.encoder.final_layernorm.layer_idx),
            )
            model.t5_model.decoder.final_layernorm.to(flow.nn.graph.GraphModule).set_stage(
                dist_utils.get_layer_stage_id(model.t5_model.decoder.final_layernorm.layer_idx),
                dist.get_layer_placement(model.t5_model.decoder.final_layernorm.layer_idx),
            )
