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
from oneflow.nn import init

from libai.config import configurable
from libai.layers import (
    Embedding,
    LayerNorm,
    LMLogits,
    ParallelCrossEntropyLoss,
    TransformerLayer,
    VocabEmbedding,
)
from libai.layers.attention import AttnMaskType
from libai.utils import distributed as dist

from .utils import init_method_normal, scaled_init_method_normal


class CasualMask(nn.Module):
    """
    Create a casual mask and combine it with the padding mask.
    It will be used in gpt model and T5 decoder.
    When in T5 decoder, the argument `layer_idx` should be set to first decoder layer index.
    """

    def __init__(self, max_positions=1024, *, layer_idx=0):
        super().__init__()
        self.mask = flow.tril(
            flow.ones(
                (max_positions, max_positions),
                dtype=flow.int8,
                placement=dist.get_layer_placement(layer_idx),
                sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
            )
        )

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


class GPTModel(nn.Module):
    """GPT-2 language model. The output of the forward method is logits.

    Args:
        num_layers (int): The number of ``TransformerLayer`` in the gpt model.
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

    @configurable
    def __init__(
        self,
        num_layers,
        vocab_size,
        hidden_size,
        ffn_hidden_size,
        num_attention_heads,
        max_seq_length=1024,
        embedding_dropout_prob=0.0,
        attention_dropout_prob=0.0,
        output_dropout_prob=0.0,
        layernorm_epsilon=1e-5,
        initializer_range=0.02,
        use_scaled_init_for_output_weights=True,
        bias_gelu_fusion=False,
        bias_dropout_fusion=False,
        scale_mask_softmax_fusion=False,
        apply_query_key_layer_scaling=False,
        apply_residual_post_layernorm=False,
        amp_enabled=False,
    ):
        super().__init__()
        init_method = init_method_normal(sigma=initializer_range)
        if use_scaled_init_for_output_weights:
            output_layer_init_method = scaled_init_method_normal(initializer_range, num_layers)
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
            num_layers,
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
            apply_residual_post_layernorm=apply_residual_post_layernorm,
        )

        self.lm_head = LMLogits(vocab_size, bias=False)

    @classmethod
    def from_config(cls, cfg):
        return {
            "num_layers": cfg.num_layers,
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
        input_embeds = self.embeddings(input_ids, 0)

        transformer_output = self.transformer(input_embeds, attention_mask=None)

        output = self.lm_head(transformer_output, self.embeddings.token_embeddings.weight)

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
        self.position_embeddings = Embedding(
            max_seq_length, hidden_size, init_method=init_method, amp_enabled=amp_enabled
        )
        self.dropout = nn.Dropout(embedding_dropout_prob)

        self.position_ids = flow.arange(
            max_seq_length,
            dtype=flow.long,
            sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
            placement=dist.get_layer_placement(0),
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


class Transformer(nn.Module):
    def __init__(
        self,
        num_layers,
        hidden_size,
        ffn_hidden_size,
        num_attention_heads,
        attention_dropout_prob=0.0,
        output_dropout_prob=0.0,
        layernorm_epsilon=1e-5,
        init_method=init.xavier_normal_,
        output_layer_init_method=None,
        bias_gelu_fusion=False,
        bias_dropout_fusion=False,
        scale_mask_softmax_fusion=False,
        apply_query_key_layer_scaling=False,
        apply_residual_post_layernorm=False,
    ):
        super().__init__()
        self.num_layers = num_layers

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
                apply_residual_post_layernorm=apply_residual_post_layernorm,
                attn_mask_type=AttnMaskType.causal,
                layer_idx=layer_number,
            )

        self.layers = nn.ModuleList([build_layer(i) for i in range(self.num_layers)])
        self.layernorm_f = LayerNorm(hidden_size, eps=layernorm_epsilon, layer_idx=-1)

    def forward(self, hidden_states, attention_mask):
        # hidden_states shape: (batch_size, seq_length, hidden_size)
        # sbp: [S(0), B]
        for i, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states, attention_mask)

        output = self.layernorm_f(hidden_states)

        return output


class GPTLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lm_loss = ParallelCrossEntropyLoss()

    def forward(self, logits, lm_labels):
        lm_loss = self.lm_loss(logits, lm_labels)
        lm_loss = lm_loss.mean()
        return {"lm_loss": lm_loss}


class GPTForPreTraining(nn.Module):
    """
    GPT Model with classification head on top.
    """

    def __init__(self, cfg) -> None:
        super().__init__()
        self.GPT_model = GPTModel(cfg)
        self.loss_func = GPTLoss()

    def forward(
        self,
        input_ids,
        labels=None,
    ):
        """

        Args:
            input_ids (flow.LongTensor): Indices of input sequence tokens in vocabulary.
            labels (flow.LongTensor, optional): Labels for computing language modeling loss.
                None for evaluating. Defaults to None.

        Returns:
            dict:
                A dict containing :code:`loss_value` or :code:`logits`
                depending on training or evaluation.
                :code:`{"masked_lm_loss": loss_value}` when training,
                :code:`{"prediction_scores": logits}` when evaluating.
        """
        logits = self.GPT_model(input_ids)
        if labels is not None:
            lm_loss = self.loss_func(logits, labels)
            return lm_loss
        else:
            return {"prediction_scores": logits}

    @staticmethod
    def set_pipeline_stage_id(model: nn.Module):
        dist_utils = dist.get_dist_util()

        for module_block in model.modules():
            if isinstance(module_block.origin, (GPTEmbedding, CasualMask)):
                module_block.config.set_stage(
                    dist_utils.get_layer_stage_id(0), dist.get_layer_placement(0)
                )
            elif isinstance(module_block.origin, TransformerLayer):
                module_block.config.set_stage(
                    dist_utils.get_layer_stage_id(module_block.layer_idx),
                    dist.get_layer_placement(module_block.layer_idx),
                )
            elif isinstance(module_block.origin, (LMLogits, GPTLoss)):
                module_block.config.set_stage(
                    dist_utils.get_layer_stage_id(-1), dist.get_layer_placement(-1)
                )

        model.GPT_model.transformer.layernorm_f.config.set_stage(
            dist_utils.get_layer_stage_id(-1), dist.get_layer_placement(-1)
        )
