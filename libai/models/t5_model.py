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
from libai.layers import (
    VocabEmbedding,
    Embedding,
    LayerNorm,
    Linear,
    TransformerLayer,
    ParallelCrossEntropyLoss,
    LMLogits,
    ExtendedMask,
    CasualMask,
)
from libai.utils import distributed as dist
from libai.config import configurable

from .utils import init_method_normal, scaled_init_method_normal


class T5Model(nn.Module):
    def __init__(
        self,
        num_encoder_layers, 
        num_decoder_layers,
        vocab_size,
        hidden_size, 
        ffn_hidden_size, 
        num_attention_heads, 
        max_seq_length=1024,
        embedding_dropout_prob=0.,
        attention_dropout_prob=0., 
        output_dropout_prob=0., 
        layernorm_epsilon=1e-5, 
        initializer_range=0.02,
        use_scaled_init_for_output_weights=True,
        bias_gelu_fusion=False,
        bias_dropout_fusion=False,
        scale_mask_softmax_fusion=False,
        apply_query_key_layer_scaling=False
    ):
        super().__init__()
        self.is_encoder_decoder = True

        self.encoder = T5Encoder(
            num_encoder_layers, 
            hidden_size, 
            ffn_hidden_size,
            num_attention_heads, 
            embedding_dropout_prob=embedding_dropout_prob,
            attention_dropout_prob=attention_dropout_prob,
            output_dropout_prob=output_dropout_prob,
            layernorm_epsilon=layernorm_epsilon,
            initializer_range=initializer_range,
            use_scaled_init_for_output_weights=use_scaled_init_for_output_weights,
            bias_gelu_fusion=bias_gelu_fusion,
            bias_dropout_fusion=bias_dropout_fusion,
            scale_mask_softmax_fusion=scale_mask_softmax_fusion,
            apply_query_key_layer_scaling=apply_query_key_layer_scaling,
        )
        self.decoder = T5Decoder(
            num_decoder_layers, 
            hidden_size, 
            ffn_hidden_size,
            num_attention_heads, 
            num_encoder_layers=num_encoder_layers,
            attention_dropout_prob=attention_dropout_prob,
            output_dropout_prob=output_dropout_prob,
            layernorm_epsilon=layernorm_epsilon,
            initializer_range=initializer_range,
            use_scaled_init_for_output_weights=use_scaled_init_for_output_weights,
            bias_gelu_fusion=bias_gelu_fusion,
            bias_dropout_fusion=bias_dropout_fusion,
            scale_mask_softmax_fusion=scale_mask_softmax_fusion,
            apply_query_key_layer_scaling=apply_query_key_layer_scaling,
        )
    
    def forward(self, input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, past_key_values, use_cache):
        encoder_states = self.encoder(input_ids, attention_mask)
        output = self.decoder(
            input_ids, 
            encoder_states, 
            attention_mask, 
            encoder_attention_mask, 
            past_key_values=past_key_values, 
            use_cache=use_cache
        )
        return output
    
    def forward_encoder(self, input_ids, attention_mask):
        encoder_states = self.encoder(input_ids, attention_mask)
        return encoder_states
    
    def forward_decoder(self, input_ids, encoder_states, attention_mask, encoder_attention_mask, past_key_values, use_cache):
        output = self.decoder(
            input_ids, 
            encoder_states, 
            attention_mask, 
            encoder_attention_mask, 
            past_key_values=past_key_values, 
            use_cache=use_cache
        )
        return output


class T5Embedding(nn.Module):
    def __init__(
        self, 
        vocab_size, 
        hidden_size, 
        max_seq_length, 
        init_method=init.xavier_normal_, 
        embedding_dropout_prob=0.,
        layer_idx=0):
        super().__init__()
        self.token_embeddings = VocabEmbedding(vocab_size, hidden_size, init_method=init_method, layer_idx=layer_idx)
        self.position_embeddings = Embedding(max_seq_length, hidden_size, init_method=init_method, layer_idx=layer_idx)
        self.dropout = flow.nn.Dropout(embedding_dropout_prob)

        self.position_ids = flow.arange(
            max_seq_length,
            dtype=flow.long,
            sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
            placement=dist.get_layer_placement(layer_idx),
        ).unsqueeze(0)
    
    def forward(self, input_ids, past_length=0):
        bsz, seq_length = input_ids.size()

        position_ids = self.position_ids[:, past_length: past_length + seq_length]
        position_ids = position_ids.expand_as(input_ids).to_consistent(sbp=input_ids.sbp)

        token_embeds = self.token_embeddings(input_ids)
        position_embeds = self.position_embeddings(position_ids)
        input_embeds = token_embeds + position_embeds
        input_embeds = self.dropout(input_embeds)
        return input_embeds


class T5Encoder(nn.Module):
    def __init__(
        self,
        num_layers, 
        vocab_size,
        hidden_size, 
        ffn_hidden_size, 
        num_attention_heads, 
        max_seq_length=1024,
        embedding_dropout_prob=0.,
        attention_dropout_prob=0., 
        output_dropout_prob=0., 
        layernorm_epsilon=1e-5, 
        initializer_range=0.02,
        use_scaled_init_for_output_weights=True,
        bias_gelu_fusion=False,
        bias_dropout_fusion=False,
        scale_mask_softmax_fusion=False,
        apply_query_key_layer_scaling=False
    ):
        super().__init__()
        self.num_layers = num_layers

        init_method = init_method_normal(std=initializer_range)
        if use_scaled_init_for_output_weights:
            output_layer_init_method = scaled_init_method_normal(initializer_range, num_layers)
        else:
            output_layer_init_method = init_method

        def build_layer(layer_number):
            return TransformerLayer(
                hidden_size,
                ffn_hidden_size,
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

        self.embeddings = T5Embedding(
            vocab_size, 
            hidden_size, 
            max_seq_length, 
            init_method=init_method, 
            embedding_dropout_prob=embedding_dropout_prob
        )
        
        self.extend_mask = ExtendedMask()

        self.layers = nn.ModuleList(
            [build_layer(i) for i in range(self.num_layers)]
        )
        self.layernorm_f = LayerNorm(hidden_size, eps=layernorm_epsilon, layer_idx=self.num_layers - 1)

    def forward(self, input_ids, attention_mask):
        hidden_states = self.embeddings(input_ids)
        extended_attention_mask = self.extend_mask(attention_mask)
        
        for i, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states, extended_attention_mask)

        output = self.layernorm_f(hidden_states)
        return output


class T5Decoder(nn.Module):
    def __init__(
        self,
        num_layers,
        vocab_size,
        hidden_size, 
        ffn_hidden_size, 
        num_attention_heads, 
        num_encoder_layers,
        max_seq_length=1024,
        embedding_dropout_prob=0.,
        attention_dropout_prob=0., 
        output_dropout_prob=0., 
        layernorm_epsilon=1e-5, 
        initializer_range=0.02,
        use_scaled_init_for_output_weights=True,
        bias_gelu_fusion=False,
        bias_dropout_fusion=False,
        scale_mask_softmax_fusion=False,
        apply_query_key_layer_scaling=False
    ):
        super().__init__()
        self.num_layers = num_layers

        init_method = init_method_normal(std=initializer_range)
        if use_scaled_init_for_output_weights:
            output_layer_init_method = scaled_init_method_normal(initializer_range, num_layers)
        else:
            output_layer_init_method = init_method

        def build_layer(layer_number):
            return TransformerLayer(
                hidden_size,
                ffn_hidden_size,
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

        self.embeddings = T5Embedding(
            vocab_size, 
            hidden_size, 
            max_seq_length, 
            init_method=init_method, 
            embedding_dropout_prob=embedding_dropout_prob,
            layer_idx=num_encoder_layers,
        )
        
        self.extend_mask = ExtendedMask()
        self.casual_mask = CasualMask(layer_idx=num_encoder_layers)

        self.layers = nn.ModuleList(
            [build_layer(i + num_encoder_layers) for i in range(self.num_layers)]
        )
        self.layernorm_f = LayerNorm(hidden_size, eps=layernorm_epsilon, layer_idx=-1)

    def forward(
        self, 
        input_ids, 
        encoder_states, 
        attention_mask, 
        encoder_attention_mask=None, 
        past_key_values=None, 
        use_cache=False
    ):
        encoder_attention_mask = encoder_attention_mask.to_consistent(placement=input_ids.placement)
        hidden_states = self.embeddings(input_ids)

        extended_attention_mask = self.extend_mask(attention_mask)
        extended_attention_mask = self.casual_mask(extended_attention_mask)

        extended_encoder_attention_mask = self.extend_mask(encoder_attention_mask)
        
        presents = []
        if past_key_values is None:
            past_key_values = tuple([None] * len(self.num_layers))

        for i, (layer, past) in enumerate(zip(self.layers, past_key_values)):
            if self.training:
                hidden_states = layer(
                    hidden_states, 
                    extended_attention_mask, 
                    encoder_states, 
                    extended_encoder_attention_mask
                )
            else:
                hidden_states = layer(
                    hidden_states, 
                    extended_attention_mask, 
                    encoder_states, 
                    extended_encoder_attention_mask, 
                    past_key_value=past, 
                    use_cache=use_cache
                )
                if use_cache:
                    hidden_states, present = hidden_states
                    presents.append(present)
        
        output = self.layernorm_f(hidden_states)
        if use_cache:
            output = (output, presents)
        
        return output
