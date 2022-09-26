import math

import oneflow as flow
from oneflow import nn

from libai.config import configurable
from libai.layers import (
    LayerNorm,
    Linear,
    SinePositionalEmbedding,
    TransformerLayer,
    VocabEmbedding,
)
from libai.models.utils import init_method_normal, scaled_init_method_normal
from libai.utils import distributed as dist


class ExtendedMask(flow.nn.Module):
    def forward(self, x):
        return x.unsqueeze(1)


class TransformerEmbedding(nn.Module):
    def __init__(
        self,
        vocab_size,
        hidden_size,
        max_sequence_length,
        embedding_dropout_prob,
        init_method=nn.init.xavier_normal_,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        self.word_embedding = VocabEmbedding(vocab_size, hidden_size, init_method=init_method)
        self.positional_encoding = SinePositionalEmbedding(max_sequence_length, hidden_size)
        self.position_ids = flow.arange(
            max_sequence_length,
            dtype=flow.long,
            sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
            placement=dist.get_layer_placement(0),
        ).unsqueeze(0)
        self.embedding_dropout = nn.Dropout(embedding_dropout_prob)

    def forward(self, input_ids):
        seq_length = input_ids.size()[1]

        word_embeddings = self.word_embedding(input_ids)
        position_ids = (
            self.position_ids[:, :seq_length].expand_as(input_ids).to_global(sbp=input_ids.sbp)
        )
        positional_encodings = self.positional_encoding(position_ids)
        embeddings = word_embeddings * math.sqrt(self.hidden_size) + positional_encodings
        embeddings = self.embedding_dropout(embeddings)
        return embeddings


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        hidden_size=512,
        ffn_hidden_size=512,
        hidden_layers=6,
        num_attention_heads=8,
        is_decoder=False,
        initializer_range=0.02,
        attention_dropout_prob=0.1,
        output_dropout_prob=0.1,
        layernorm_epsilon=1e-5,
        bias_gelu_fusion=False,
        bias_dropout_fusion=False,
        scale_mask_softmax_fusion=False,
        apply_query_key_layer_scaling=True,
    ):
        super().__init__()
        self.encoder_layers = nn.ModuleList(
            [
                TransformerLayer(
                    hidden_size=hidden_size,
                    ffn_hidden_size=ffn_hidden_size,
                    num_attention_heads=num_attention_heads,
                    is_decoder=is_decoder,
                    attention_dropout_prob=attention_dropout_prob,
                    output_dropout_prob=output_dropout_prob,
                    layernorm_epsilon=layernorm_epsilon,
                    bias_gelu_fusion=bias_gelu_fusion,
                    bias_dropout_fusion=bias_dropout_fusion,
                    scale_mask_softmax_fusion=scale_mask_softmax_fusion,
                    apply_query_key_layer_scaling=apply_query_key_layer_scaling,
                    init_method=init_method_normal(initializer_range),
                    output_layer_init_method=scaled_init_method_normal(
                        initializer_range, hidden_layers
                    ),
                    layer_idx=i,
                )
                for i in range(hidden_layers)
            ]
        )
        self.encoder_final_layernorm = LayerNorm(
            (hidden_size,), eps=layernorm_epsilon, layer_idx=hidden_layers - 1
        )

    def forward(self, encoder_input_embeddings, encoder_extended_attn_mask):
        enc_hidden_states = encoder_input_embeddings
        for layer in self.encoder_layers:
            enc_hidden_states = layer(enc_hidden_states, encoder_extended_attn_mask)
        encoder_states = self.encoder_final_layernorm(enc_hidden_states)
        return encoder_states


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        hidden_size=512,
        ffn_hidden_size=512,
        hidden_layers=6,
        num_attention_heads=8,
        is_decoder=True,
        initializer_range=0.02,
        attention_dropout_prob=0.1,
        output_dropout_prob=0.1,
        layernorm_epsilon=1e-5,
        bias_gelu_fusion=False,
        bias_dropout_fusion=False,
        scale_mask_softmax_fusion=False,
        apply_query_key_layer_scaling=True,
    ):
        super().__init__()
        self.decoder_layers = nn.ModuleList(
            [
                TransformerLayer(
                    hidden_size=hidden_size,
                    ffn_hidden_size=ffn_hidden_size,
                    num_attention_heads=num_attention_heads,
                    is_decoder=is_decoder,
                    attention_dropout_prob=attention_dropout_prob,
                    output_dropout_prob=output_dropout_prob,
                    layernorm_epsilon=layernorm_epsilon,
                    bias_gelu_fusion=bias_gelu_fusion,
                    bias_dropout_fusion=bias_dropout_fusion,
                    scale_mask_softmax_fusion=scale_mask_softmax_fusion,
                    apply_query_key_layer_scaling=apply_query_key_layer_scaling,
                    init_method=init_method_normal(initializer_range),
                    output_layer_init_method=scaled_init_method_normal(
                        initializer_range, hidden_layers
                    ),
                    layer_idx=i,
                )
                for i in range(hidden_layers, 2 * hidden_layers)
            ]
        )
        self.decoder_final_layernorm = LayerNorm(
            (hidden_size,), eps=layernorm_epsilon, layer_idx=2 * hidden_layers - 1
        )

    def forward(
        self,
        decoder_input_embeddings,
        decoder_extended_attn_mask,
        encoder_states,
        encoder_decoder_extended_attn_mask,
    ):
        dec_hidden_states = decoder_input_embeddings
        for layer in self.decoder_layers:
            dec_hidden_states = layer(
                dec_hidden_states,
                decoder_extended_attn_mask,
                encoder_states,
                encoder_decoder_extended_attn_mask,
            )
        decoder_states = self.decoder_final_layernorm(dec_hidden_states)
        return decoder_states


class TransformerModel(nn.Module):
    @configurable
    def __init__(
        self,
        vocab_size,
        max_position_embeddings,
        hidden_size=512,
        intermediate_size=512,
        hidden_layers=6,
        num_attention_heads=8,
        embedding_dropout_prob=0.1,
        hidden_dropout_prob=0.1,
        attention_dropout_prob=0.1,
        initializer_range=0.02,
        layernorm_epsilon=1e-5,
        bias_gelu_fusion=False,
        bias_dropout_fusion=False,
        scale_mask_softmax_fusion=False,
        apply_query_key_layer_scaling=True,
    ):
        super().__init__()

        self.embedding = TransformerEmbedding(
            vocab_size,
            hidden_size,
            max_position_embeddings,
            embedding_dropout_prob,
            init_method=init_method_normal(initializer_range),
        )

        self.extended_attn_mask = ExtendedMask()

        self.encoder = TransformerEncoder(
            hidden_size=hidden_size,
            ffn_hidden_size=intermediate_size,
            hidden_layers=hidden_layers,
            num_attention_heads=num_attention_heads,
            is_decoder=False,
            initializer_range=0.02,
            attention_dropout_prob=attention_dropout_prob,
            output_dropout_prob=hidden_dropout_prob,
            layernorm_epsilon=layernorm_epsilon,
            bias_gelu_fusion=bias_gelu_fusion,
            bias_dropout_fusion=bias_dropout_fusion,
            scale_mask_softmax_fusion=scale_mask_softmax_fusion,
            apply_query_key_layer_scaling=apply_query_key_layer_scaling,
        )

        self.decoder = TransformerDecoder(
            hidden_size=hidden_size,
            ffn_hidden_size=intermediate_size,
            hidden_layers=hidden_layers,
            num_attention_heads=num_attention_heads,
            is_decoder=True,
            initializer_range=0.02,
            attention_dropout_prob=attention_dropout_prob,
            output_dropout_prob=hidden_dropout_prob,
            layernorm_epsilon=layernorm_epsilon,
            bias_gelu_fusion=bias_gelu_fusion,
            bias_dropout_fusion=bias_dropout_fusion,
            scale_mask_softmax_fusion=scale_mask_softmax_fusion,
            apply_query_key_layer_scaling=apply_query_key_layer_scaling,
        )

        self.lm_head = Linear(
            hidden_size,
            vocab_size,
            layer_idx=-1,
        )

    @classmethod
    def from_config(cls, cfg):
        return {
            "vocab_size": cfg.vocab_size,
            "max_position_embeddings": cfg.max_position_embeddings,
            "hidden_size": cfg.hidden_size,
            "intermediate_size": cfg.intermediate_size,
            "hidden_layers": cfg.hidden_layers,
            "num_attention_heads": cfg.num_attention_heads,
            "embedding_dropout_prob": cfg.embedding_dropout_prob,
            "hidden_dropout_prob": cfg.hidden_dropout_prob,
            "attention_dropout_prob": cfg.attention_dropout_prob,
            "initializer_range": cfg.initializer_range,
            "layernorm_epsilon": cfg.layernorm_epsilon,
            "bias_gelu_fusion": cfg.bias_gelu_fusion,
            "bias_dropout_fusion": cfg.bias_dropout_fusion,
            "scale_mask_softmax_fusion": cfg.scale_mask_softmax_fusion,
            "apply_query_key_layer_scaling": cfg.apply_query_key_layer_scaling,
        }

    def forward(
        self,
        encoder_input_ids,
        decoder_input_ids,
        encoder_attn_mask,
        decoder_attn_mask,
        encoder_decoder_attn_mask,
    ):
        encoder_input_embeddings = self.embedding(encoder_input_ids)
        decoder_input_embeddings = self.embedding(decoder_input_ids)
        encoder_extended_attn_mask = self.extended_attn_mask(encoder_attn_mask)
        decoder_extended_attn_mask = self.extended_attn_mask(decoder_attn_mask)
        encoder_decoder_extended_attn_mask = self.extended_attn_mask(encoder_decoder_attn_mask)

        encoder_states = self.encoder(encoder_input_embeddings, encoder_extended_attn_mask)
        decoder_states = self.decoder(
            decoder_input_embeddings,
            decoder_extended_attn_mask,
            encoder_states,
            encoder_decoder_extended_attn_mask,
        )
        logits = self.lm_head(decoder_states)
        return logits
