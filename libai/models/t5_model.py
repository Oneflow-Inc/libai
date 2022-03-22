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

from libai.config import configurable
from libai.layers import (
    Embedding,
    LayerNorm,
    LMLogits,
    ParallelCrossEntropyLoss,
    TransformerLayer,
    VocabEmbedding,
)
from libai.models.build import MODEL_ARCH_REGISTRY
from libai.models.utils import init_method_normal, scaled_init_method_normal
from libai.utils import distributed as dist


class ExtendedMask(flow.nn.Module):
    def forward(self, x):
        return x.unsqueeze(1)


class T5Embedding(flow.nn.Module):
    def __init__(
        self,
        hidden_size,
        vocab_size,
        max_sequence_length,
        embedding_dropout_prob,
        init_method=flow.nn.init.xavier_normal_,
        amp_enabled=False,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        self.word_embeddings = VocabEmbedding(
            num_embeddings=vocab_size,
            embedding_dim=hidden_size,
            init_method=init_method,
            amp_enabled=amp_enabled,
        )
        self.position_embeddings = Embedding(
            num_embeddings=max_sequence_length,
            embedding_dim=hidden_size,
            init_method=init_method,
            amp_enabled=amp_enabled,
        )
        self.position_ids = flow.arange(
            max_sequence_length,
            dtype=flow.long,
            sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
            placement=dist.get_layer_placement(0),
        ).unsqueeze(0)

        self.embedding_dropout = flow.nn.Dropout(embedding_dropout_prob)

    def forward(self, input_ids, position_ids=None):
        seq_length = input_ids.size()[1]
        word_embeddings = self.word_embeddings(input_ids)

        if position_ids is None:
            # Change position_ids sbp sign: [B, B] -> [S(0), B]
            position_ids = (
                self.position_ids[:, :seq_length].expand_as(input_ids).to_global(sbp=input_ids.sbp)
            )
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = word_embeddings + position_embeddings
        embeddings = self.embedding_dropout(embeddings)
        return embeddings


class T5Model(flow.nn.Module):
    """T5 Model that outputs logits.

    Args:
        vocab_size (int): The size of vocabulary file.
        hidden_size (int): The size of hidden states.
        hidden_layers (int): The number of TransformerLayer in the encoder and decoder.
        num_attention_heads (int):
            The number of attention heads for each attention layer of TransformerLayer.
        intermediate_size (int):
            The size of intermediate layer in feed-forward network for each TransformerLayer.
        embedding_dropout_prob (float): The dropout ratio for the output of T5Embedding Layer.
        hidden_dropout_prob (float): The dropout ratio for the output for each TransformerLayer.
        attention_probs_dropout_prob (float):
            The dropout ratio for the output of each attention layer in TransformerLayer.
        max_position_embeddings (int):
            Max sequence length of input, defines the shape of Position Embeddings in T5Emebedding.
        initializer_range (float, optional):
            Sigma of the normal distribution in the initialization method. Defaults to 0.02.
        layernorm_eps (float, optional): The epsilon of LayerNorm layer. Defaults to 1e-12.
        bias_gelu_fusion (bool, optional):
            Whether or not to fuse the computing of bias and gelu. Defaults to False.
        bias_dropout_fusion (bool, optional):
            Whether or not to fuse the computing of dropout and bias. Defaults to False.
        scale_mask_softmax_fusion (bool, optional):
            Whether to fuse the computing of mask and softmax in attention layers.
            Defaults to False.
        apply_query_key_layer_scaling (bool, optional):
            Whether or not to use layer index related scaling in computing attention scores.
            If True, the scaling factor equals to sqrt(d) * (layer_index + 1). Defaults to True.
        amp_enabled (bool, optional):
            Whether or not to set fp16 for embedding weight in T5 model. Defaults to False.
    """

    @configurable
    def __init__(
        self,
        vocab_size,
        hidden_size,
        hidden_layers,
        num_attention_heads,
        intermediate_size,
        embedding_dropout_prob,
        hidden_dropout_prob,
        attention_probs_dropout_prob,
        max_position_embeddings,
        initializer_range=0.02,
        layernorm_eps=1e-12,
        bias_gelu_fusion=False,
        bias_dropout_fusion=False,
        scale_mask_softmax_fusion=False,
        apply_query_key_layer_scaling=True,
        amp_enabled=False,
    ) -> None:
        super().__init__()
        init_method = init_method_normal(initializer_range)
        scaled_init_method = scaled_init_method_normal(initializer_range, hidden_layers)
        self.embedding = T5Embedding(
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            max_sequence_length=max_position_embeddings,
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
                    is_decoder=False,
                    attention_dropout_prob=attention_probs_dropout_prob,
                    output_dropout_prob=hidden_dropout_prob,
                    layernorm_epsilon=layernorm_eps,
                    init_method=init_method,
                    output_layer_init_method=scaled_init_method,
                    bias_gelu_fusion=bias_gelu_fusion,
                    bias_dropout_fusion=bias_dropout_fusion,
                    scale_mask_softmax_fusion=scale_mask_softmax_fusion,
                    apply_query_key_layer_scaling=apply_query_key_layer_scaling,
                    layer_idx=i,
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
                    is_decoder=True,
                    attention_dropout_prob=attention_probs_dropout_prob,
                    output_dropout_prob=hidden_dropout_prob,
                    layernorm_epsilon=layernorm_eps,
                    init_method=init_method,
                    output_layer_init_method=scaled_init_method,
                    bias_gelu_fusion=bias_gelu_fusion,
                    bias_dropout_fusion=bias_dropout_fusion,
                    scale_mask_softmax_fusion=scale_mask_softmax_fusion,
                    apply_query_key_layer_scaling=apply_query_key_layer_scaling,
                    layer_idx=i,
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

        self.lm_head = LMLogits(vocab_size, bias=True)

    @classmethod
    def from_config(cls, cfg):
        return {
            "vocab_size": cfg.vocab_size,
            "hidden_size": cfg.hidden_size,
            "hidden_layers": cfg.hidden_layers,
            "num_attention_heads": cfg.num_attention_heads,
            "intermediate_size": cfg.intermediate_size,
            "embedding_dropout_prob": cfg.embedding_dropout_prob,
            "hidden_dropout_prob": cfg.hidden_dropout_prob,
            "attention_probs_dropout_prob": cfg.attention_probs_dropout_prob,
            "max_position_embeddings": cfg.max_position_embeddings,
            "initializer_range": cfg.initializer_range,
            "layernorm_eps": cfg.layernorm_eps,
            "bias_gelu_fusion": cfg.bias_gelu_fusion,
            "bias_dropout_fusion": cfg.bias_dropout_fusion,
            "scale_mask_softmax_fusion": cfg.scale_mask_softmax_fusion,
            "apply_query_key_layer_scaling": cfg.apply_query_key_layer_scaling,
            "amp_enabled": cfg.amp_enabled,
        }

    def forward(
        self,
        encoder_input_ids,
        decoder_input_ids,
        encoder_attn_mask,
        decoder_attn_mask,
        encoder_decoder_attn_mask,
    ):
        """

        Args:
            encoder_input_ids (flow.LongTensor):
                Indices of input sequence tokens in vocabulary for encoder.
            decoder_input_ids (flow.LongTensor):
                Indices of input sequence tokens in vocabulary for decoder.
            encoder_attn_mask (flow.LongTensor):
                Mask for encoder to avoid performing attention on
                padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

            decoder_attn_mask (flow.LongTensor):
                Mask for decoder to avoid performing attention on subsequent token indices.
                Mask values have the same meaning as encoder_attn_mask.
            encoder_decoder_attn_mask (flow.LongTensor):
                Mask for decoder to avoid performing attention on encoder padded token indices.
                Mask values have the same meaning as encoder_attn_mask.

        Returns:
            flow.Tensor: logits
        """
        encoder_attn_mask = self.extended_attn_mask(encoder_attn_mask)
        decoder_attn_mask = self.extended_attn_mask(decoder_attn_mask)
        encoder_decoder_attn_mask = self.extended_attn_mask(encoder_decoder_attn_mask)
        enc_embedding_output = self.embedding(encoder_input_ids)
        enc_hidden_states = enc_embedding_output
        for layer in self.encoder.layers:
            enc_hidden_states = layer(enc_hidden_states, encoder_attn_mask)
        encoder_states = self.encoder.final_layernorm(enc_hidden_states)

        dec_embedding_output = self.embedding(decoder_input_ids)
        dec_hidden_states = dec_embedding_output
        for layer in self.decoder.layers:
            dec_hidden_states = layer(
                dec_hidden_states,
                decoder_attn_mask,
                encoder_states,
                encoder_decoder_attn_mask,
            )
        decoder_states = self.decoder.final_layernorm(dec_hidden_states)
        logits = self.lm_head(decoder_states, self.embedding.word_embeddings.weight)
        return logits


class T5Loss(flow.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lm_loss = ParallelCrossEntropyLoss()

    def forward(self, logits, lm_labels, loss_mask):
        lm_loss = self.lm_loss(logits, lm_labels)
        loss_mask = loss_mask.float()
        denominator = loss_mask.sum().to_global(
            sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast])
        )
        masked_lm_loss = flow.sum(lm_loss.view(-1) * loss_mask.view(-1)) / denominator
        masked_lm_loss = masked_lm_loss.to_global(
            sbp=dist.get_nd_sbp([flow.sbp.partial_sum, flow.sbp.broadcast])
        )
        return {"masked_lm_loss": masked_lm_loss}


@MODEL_ARCH_REGISTRY.register()
class T5ForPreTraining(flow.nn.Module):
    """
    T5 Model with classification head on top.
    """

    def __init__(self, cfg) -> None:
        super().__init__()
        self.t5_model = T5Model(cfg)
        self.loss_func = T5Loss()

    def forward(
        self,
        encoder_input_ids,
        decoder_input_ids,
        encoder_attn_mask,
        decoder_attn_mask,
        encoder_decoder_attn_mask,
        lm_labels=None,
        loss_mask=None,
    ):
        """

        Args:
            encoder_input_ids (flow.LongTensor):
                Indices of input sequence tokens in vocabulary for encoder.
            decoder_input_ids (flow.LongTensor):
                Indices of input sequence tokens in vocabulary for decoder.
            encoder_attn_mask (flow.LongTensor):
                Mask for encoder to avoid performing attention on
                padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

            decoder_attn_mask (flow.LongTensor):
                Mask for decoder to avoid performing attention on subsequent token indices.
                Mask values have the same meaning as encoder_attn_mask.
            encoder_decoder_attn_mask (flow.LongTensor):
                Mask for decoder to avoid performing attention on encoder padded token indices.
                Mask values have the same meaning as encoder_attn_mask.
            lm_labels (flow.LongTensor, optional): Labels for computing the masked
                language modeling loss. Indices should be in `[-1, 0, ..., config.vocab_size]`.
                None for evaluating.
            loss_mask (flow.Tensor, optional):
                Mask to avoid performing loss computing on ignored tokens.
                Tokens with indices set to `-1` are ignored (masked), the loss is only computed
                for the tokens with labels in `[0, ..., config.vocab_size]`.
                None for evaluating.

        Returns:
            dict:
                A dict containing :code:`loss_value` or :code:`logits`
                depending on training or evaluation mode.
                :code:`{"masked_lm_loss": loss_value}` when training,
                :code:`{"prediction_scores": logits}` when evaluating.
        """
        logits = self.t5_model(
            encoder_input_ids,
            decoder_input_ids,
            encoder_attn_mask,
            decoder_attn_mask,
            encoder_decoder_attn_mask,
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
        for module_block in model.modules():
            if isinstance(module_block.origin, T5Embedding):
                module_block.config.stage_id = dist_utils.get_layer_stage_id(0)
            elif isinstance(module_block.origin, ExtendedMask):
                module_block.config.stage_id = dist_utils.get_layer_stage_id(0)
            elif isinstance(module_block.origin, TransformerLayer):
                module_block.config.stage_id = dist_utils.get_layer_stage_id(module_block.layer_idx)
            elif isinstance(module_block.origin, LMLogits):
                module_block.config.stage_id = dist_utils.get_layer_stage_id(-1)
            elif isinstance(module_block.origin, T5Loss):
                module_block.config.stage_id = dist_utils.get_layer_stage_id(-1)

        model.t5_model.encoder.final_layernorm.config.stage_id = dist_utils.get_layer_stage_id(
            model.t5_model.encoder.final_layernorm.layer_idx
        )
        model.t5_model.decoder.final_layernorm.config.stage_id = dist_utils.get_layer_stage_id(
            model.t5_model.decoder.final_layernorm.layer_idx
        )
