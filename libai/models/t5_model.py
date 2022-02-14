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
from oneflow import nn

from libai.config import configurable
from libai.layers import (
    CasualMask,
    Embedding,
    ExtendedMask,
    LayerNorm,
    LMLogits,
    ParallelCrossEntropyLoss,
    TransformerLayer,
    VocabEmbedding,
)
from libai.utils import distributed as dist

from .build import MODEL_ARCH_REGISTRY
from .utils import ModelType, init_method_normal, scaled_init_method_normal


class T5Embedding(nn.Module):
    def __init__(
        self,
        vocab_size,
        hidden_size,
        max_seq_length,
        embedding_dropout_prob=0.0,
        init_method=nn.init.xavier_normal_,
        layer_idx=0,
    ):
        super().__init__()
        self.token_embeddings = VocabEmbedding(
            vocab_size, hidden_size, init_method=init_method, layer_idx=layer_idx
        )
        self.position_embeddings = Embedding(
            max_seq_length, hidden_size, init_method=init_method, layer_idx=layer_idx
        )
        self.dropout = nn.Dropout(embedding_dropout_prob)

        self.position_ids = flow.arange(
            max_seq_length,
            dtype=flow.long,
            sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
            placement=dist.get_layer_placement(layer_idx),
        ).unsqueeze(0)

    def forward(self, input_ids, past_length=0):
        bsz, seq_length = input_ids.size()

        position_ids = self.position_ids[:, past_length : past_length + seq_length]
        position_ids = position_ids.expand_as(input_ids).to_global(sbp=input_ids.sbp)

        token_embeds = self.token_embeddings(input_ids)
        position_embeds = self.position_embeddings(position_ids)
        input_embeds = token_embeds + position_embeds
        input_embeds = self.dropout(input_embeds)
        return input_embeds

    def word_embeddings(self):
        return self.token_embeddings.weight


class T5Encoder(nn.Module):
    def __init__(
        self,
        embeddings,
        num_layers,
        hidden_size,
        intermediate_size,
        num_attention_heads,
        attention_dropout_prob=0.0,
        output_dropout_prob=0.0,
        layernorm_epsilon=1e-5,
        initializer_range=0.02,
        bias_gelu_fusion=False,
        bias_dropout_fusion=False,
        scale_mask_softmax_fusion=False,
        apply_query_key_layer_scaling=False,
    ):
        super().__init__()
        self.num_layers = num_layers

        init_method = init_method_normal(initializer_range)
        output_layer_init_method = scaled_init_method_normal(initializer_range, num_layers)

        def build_layer(layer_number):
            return TransformerLayer(
                hidden_size,
                intermediate_size,
                num_attention_heads,
                attention_dropout_prob=attention_dropout_prob,
                output_dropout_prob=output_dropout_prob,
                layernorm_epsilon=layernorm_epsilon,
                init_method=init_method,
                output_layer_init_method=output_layer_init_method,
                bias_gelu_fusion=bias_gelu_fusion,
                bias_dropout_fusion=bias_dropout_fusion,
                scale_mask_softmax_fusion=scale_mask_softmax_fusion,
                apply_query_key_layer_scaling=apply_query_key_layer_scaling,
                layer_idx=layer_number,
            )

        self.embeddings = embeddings
        self.extend_mask = ExtendedMask()

        self.layers = nn.ModuleList([build_layer(i) for i in range(self.num_layers)])
        self.layernorm_f = LayerNorm(
            hidden_size, eps=layernorm_epsilon, layer_idx=self.num_layers - 1
        )

    def forward(self, input_ids, attention_mask):
        hidden_states = self.embeddings(input_ids)
        extended_attention_mask = self.extend_mask(attention_mask)

        for i, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states, attention_mask=extended_attention_mask)

        output = self.layernorm_f(hidden_states)
        return output


class T5Decoder(nn.Module):
    def __init__(
        self,
        embeddings,
        num_layers,
        hidden_size,
        intermediate_size,
        num_attention_heads,
        num_encoder_layers,
        attention_dropout_prob=0.0,
        output_dropout_prob=0.0,
        layernorm_epsilon=1e-5,
        initializer_range=0.02,
        bias_gelu_fusion=False,
        bias_dropout_fusion=False,
        scale_mask_softmax_fusion=False,
        apply_query_key_layer_scaling=False,
    ):
        super().__init__()
        self.num_layers = num_layers

        init_method = init_method_normal(initializer_range)
        output_layer_init_method = scaled_init_method_normal(initializer_range, num_layers)

        def build_layer(layer_number):
            return TransformerLayer(
                hidden_size,
                intermediate_size,
                num_attention_heads,
                is_decoder=True,
                attention_dropout_prob=attention_dropout_prob,
                output_dropout_prob=output_dropout_prob,
                layernorm_epsilon=layernorm_epsilon,
                init_method=init_method,
                output_layer_init_method=output_layer_init_method,
                bias_gelu_fusion=bias_gelu_fusion,
                bias_dropout_fusion=bias_dropout_fusion,
                scale_mask_softmax_fusion=scale_mask_softmax_fusion,
                apply_query_key_layer_scaling=apply_query_key_layer_scaling,
                layer_idx=layer_number,
            )

        self.embeddings = embeddings
        self.extend_mask = ExtendedMask()
        self.casual_mask = CasualMask()

        self.layers = nn.ModuleList(
            [build_layer(i + num_encoder_layers) for i in range(self.num_layers)]
        )
        self.layernorm_f = LayerNorm(hidden_size, eps=layernorm_epsilon, layer_idx=-1)

    def forward(
        self,
        input_ids,
        encoder_states,
        attention_mask,
        encoder_attention_mask,
        past_key_values=None,
        use_cache=False,
    ):
        hidden_states = self.embeddings(input_ids)

        presents = []
        if past_key_values is None:
            past_key_values = tuple([None] * self.num_layers)
            past_length = 0
        else:
            past_length = past_key_values[0][0].size[2]

        extended_attention_mask = self.extend_mask(attention_mask)
        extended_attention_mask = self.casual_mask(
            input_ids, past_length=past_length, attention_mask=extended_attention_mask
        )

        extended_encoder_attention_mask = self.extend_mask(encoder_attention_mask)

        for i, (layer, past) in enumerate(zip(self.layers, past_key_values)):
            if self.training:
                hidden_states = layer(
                    hidden_states,
                    attention_mask=extended_attention_mask,
                    encoder_states=encoder_states,
                    encoder_attention_mask=extended_encoder_attention_mask,
                )
            else:
                hidden_states = layer(
                    hidden_states,
                    attention_mask=extended_attention_mask,
                    encoder_states=encoder_states,
                    encoder_attention_mask=extended_encoder_attention_mask,
                    past_key_value=past,
                    use_cache=use_cache,
                )
                if use_cache:
                    hidden_states, present = hidden_states
                    presents.append(present)

        output = self.layernorm_f(hidden_states)

        if use_cache:
            output = output + (presents,)  # todo: unify return format
        return output


class T5Model(nn.Module):
    @configurable
    def __init__(
        self,
        num_encoder_layers,
        num_decoder_layers,
        vocab_size,
        hidden_size,
        intermediate_size,
        num_attention_heads,
        max_seq_length=1024,
        embedding_dropout_prob=0.0,
        attention_dropout_prob=0.0,
        output_dropout_prob=0.0,
        layernorm_epsilon=1e-6,
        initializer_range=0.02,
        bias_gelu_fusion=False,
        bias_dropout_fusion=False,
        scale_mask_softmax_fusion=False,
        apply_query_key_layer_scaling=False,
    ):
        super().__init__()
        self.is_encoder_decoder = True

        self.embeddings = T5Embedding(
            vocab_size,
            hidden_size,
            max_seq_length,
            embedding_dropout_prob=embedding_dropout_prob,
            init_method=init_method_normal(initializer_range),
        )

        self.encoder = T5Encoder(
            self.embeddings,
            num_encoder_layers,
            hidden_size,
            intermediate_size,
            num_attention_heads,
            attention_dropout_prob=attention_dropout_prob,
            output_dropout_prob=output_dropout_prob,
            layernorm_epsilon=layernorm_epsilon,
            initializer_range=initializer_range,
            bias_gelu_fusion=bias_gelu_fusion,
            bias_dropout_fusion=bias_dropout_fusion,
            scale_mask_softmax_fusion=scale_mask_softmax_fusion,
            apply_query_key_layer_scaling=apply_query_key_layer_scaling,
        )

        self.decoder = T5Decoder(
            self.embeddings,
            num_decoder_layers,
            hidden_size,
            intermediate_size,
            num_attention_heads,
            num_encoder_layers=num_encoder_layers,
            attention_dropout_prob=attention_dropout_prob,
            output_dropout_prob=output_dropout_prob,
            layernorm_epsilon=layernorm_epsilon,
            initializer_range=initializer_range,
            bias_gelu_fusion=bias_gelu_fusion,
            bias_dropout_fusion=bias_dropout_fusion,
            scale_mask_softmax_fusion=scale_mask_softmax_fusion,
            apply_query_key_layer_scaling=apply_query_key_layer_scaling,
        )
        self.lm_head = LMLogits(vocab_size, bias=True)

    @classmethod
    def from_config(cls, cfg):
        return {
            "num_encoder_layers": cfg.num_encoder_layers,
            "num_decoder_layers": cfg.num_decoder_layers,
            "vocab_size": cfg.vocab_size,
            "hidden_size": cfg.hidden_size,
            "intermediate_size": cfg.intermediate_size,
            "num_attention_heads": cfg.num_attention_heads,
            "max_seq_length": cfg.max_position_embeddings,
            "embedding_dropout_prob": cfg.embedding_dropout_prob,
            "attention_dropout_prob": cfg.attention_dropout_prob,
            "output_dropout_prob": cfg.hidden_dropout_prob,
            "layernorm_epsilon": cfg.layernorm_epsilon,
            "initializer_range": cfg.initializer_range,
            "bias_gelu_fusion": cfg.bias_gelu_fusion,
            "bias_dropout_fusion": cfg.bias_dropout_fusion,
            "scale_mask_softmax_fusion": cfg.scale_mask_softmax_fusion,
            "apply_query_key_layer_scaling": cfg.apply_query_key_layer_scaling,
        }

    @property
    def model_type(self):
        return ModelType.encoder_decoder

    def forward(
        self,
        input_ids,
        attention_mask,
        decoder_input_ids,
        decoder_attention_mask,
        past_key_values=None,
        use_cache=False,
    ):
        encoder_states = self.encoder(input_ids, attention_mask)
        output = self.decoder(
            input_ids,
            encoder_states,
            decoder_attention_mask,
            attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )

        if not isinstance(output, (tuple, list)):
            output = (output,)
        logits = self.lm_head(output[0], self.embeddings.token_embeddings.weight)
        output = (logits,) + output[1:]
        return output

    def forward_encoder(self, input_ids, attention_mask):
        encoder_states = self.encoder(input_ids, attention_mask)
        return encoder_states

    def forward_decoder(
        self,
        input_ids,
        encoder_states,
        attention_mask,
        encoder_attention_mask,
        past_key_values=None,
        use_cache=False,
    ):
        output = self.decoder(
            input_ids,
            encoder_states,
            attention_mask,
            encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )

        if not isinstance(output, (tuple, list)):
            output = (output,)
        logits = self.lm_head(output[0], self.embeddings.token_embeddings.weight)
        output = (logits,) + output[1:]
        return output

    def reorder_cache(self, past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2])
                + layer_past[2:],
            )
        return reordered_past


@MODEL_ARCH_REGISTRY.register()
class T5ForPretraining(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.t5_model = T5Model(cfg)
        self.loss_func = ParallelCrossEntropyLoss(ignore_index=0)

    def forward(
        self,
        input_ids,
        attention_mask,
        decoder_input_ids,
        decoder_attention_mask,
        labels=None,
        past_key_values=None,
        use_cache=False,
    ):
        outputs = self.t5_model(
            input_ids,
            attention_mask,
            decoder_input_ids,
            decoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )

        if not isinstance(outputs, (tuple, list)):
            outputs = (outputs,)

        if labels is not None:
            logits = outputs[0]
            loss = self.loss_func(logits, labels)
            return {"loss": loss}

        ret_dict = {"logits": outputs[0]}
        if len(outputs) > 1:
            ret_dict["presents"] = outputs[1]

        return ret_dict

    @staticmethod
    def set_pipeline_stage_id(model):
        dist_utils = dist.get_dist_util()

        # Set pipeline parallelism stage_id
        for module_block in model.modules():
            # module.origin can get the original module
            if isinstance(module_block.origin, T5Embedding):
                module_block.config.stage_id = dist_utils.get_layer_stage_id(0)
            elif isinstance(module_block.origin, (ExtendedMask, CasualMask)):
                module_block.config.stage_id = dist_utils.get_layer_stage_id(0)
            elif isinstance(module_block.origin, TransformerLayer):
                module_block.config.stage_id = dist_utils.get_layer_stage_id(module_block.layer_idx)
            elif isinstance(module_block.origin, LMLogits):
                module_block.config.stage_id = dist_utils.get_layer_stage_id(-1)
            elif isinstance(module_block.origin, ParallelCrossEntropyLoss):
                module_block.config.stage_id = dist_utils.get_layer_stage_id(-1)

        model.t5_model.encoder.layernorm_f.config.stage_id = dist_utils.get_layer_stage_id(
            model.t5_model.encoder.num_layers - 1
        )
        model.t5_model.decoder.layernorm_f.config.stage_id = dist_utils.get_layer_stage_id(-1)
