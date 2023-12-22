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
from libai.inference.generator.generation_utils import Generator
from libai.layers import Embedding, LayerNorm, LMLogits, VocabEmbedding
from libai.layers.attention import AttnMaskType
from libai.models.gpt_model import GPTLoss
from libai.models.utils import GPT2LoaderHuggerFace, init_method_normal, scaled_init_method_normal
from libai.utils import distributed as dist
from projects.MagicPrompt.layers.transformer_layer import TransformerLayer


class GPTModel(nn.Module, Generator):
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

    @configurable
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
        initializer_range=0.02,
        use_scaled_init_for_output_weights=True,
        bias_gelu_fusion=False,
        bias_dropout_fusion=False,
        scale_mask_softmax_fusion=False,
        apply_query_key_layer_scaling=False,
        apply_residual_post_layernorm=False,
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
            init_method=init_method,
            output_layer_init_method=output_layer_init_method,
            bias_gelu_fusion=bias_gelu_fusion,
            bias_dropout_fusion=bias_dropout_fusion,
            scale_mask_softmax_fusion=scale_mask_softmax_fusion,
            apply_query_key_layer_scaling=apply_query_key_layer_scaling,
            apply_residual_post_layernorm=apply_residual_post_layernorm,
            set_cache=self.set_cache,
        )

        self.past_key_values = [None] * hidden_layers
        self.past_length = 0

        self.lm_head = LMLogits(vocab_size, bias=False)

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
            "cfg": cfg,
        }

    def forward(self, input_ids, use_cache=False):
        """

        Args:
            input_ids (flow.LongTensor): Indices of input sequence tokens in vocabulary.

        Returns:
            flow.Tensor: logits
        """

        input_ids = input_ids.to_global(placement=dist.get_layer_placement(0))

        if use_cache and self.past_key_values[0] is not None:
            self.past_length = self.past_key_values[0][0].size(-2)
        else:
            self.past_length = 0

        input_embeds = self.embeddings(input_ids, self.past_length)

        transformer_output = self.transformer(
            input_embeds,
            attention_mask=None,
            past_key_values=self.past_key_values,
            use_cache=use_cache,
        )

        logits = self.lm_head(transformer_output, self.embeddings.token_embeddings.weight)

        return {"logits": logits}

    def set_cache(self, past_key_values):
        self.past_length = 0 if past_key_values is None else past_key_values[0][0].shape[2]

        if past_key_values is None:
            past_key_values = [None] * self.cfg.hidden_layers

        assert len(past_key_values) == self.cfg.hidden_layers, (
            f"past_key_values's length {len(past_key_values)} doesn't match "
            f"num_layers:' {self.cfg.hidden_layers}"
        )

        self.past_key_values = past_key_values

    def _reorder_cache(self, beam_idx):
        past_key_values = self.past_key_values
        return tuple(
            tuple(
                past_state.index_select(0, beam_idx.to(past_state.device))
                for past_state in layer_past
            )
            for layer_past in past_key_values
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past=None,
        use_cache=None,
    ):
        if past is not None:
            input_ids = input_ids[:, -1:]
            self.past_key_values = past

        return {"input_ids": input_ids, "use_cache": use_cache}


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
        hidden_layers,
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
        set_cache=None,
    ):
        super().__init__()
        self.hidden_layers = hidden_layers
        self.set_cache = set_cache

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

        self.layers = nn.ModuleList([build_layer(i) for i in range(self.hidden_layers)])
        self.layernorm_f = LayerNorm(hidden_size, eps=layernorm_epsilon, layer_idx=-1)

    def forward(self, hidden_states, attention_mask, past_key_values=None, use_cache=False):
        if use_cache:
            presents = []

        for layer, past_key_value in zip(self.layers, past_key_values):
            hidden_states = layer(
                hidden_states,
                attention_mask,
                past_key_value=past_key_value,
                use_cache=use_cache,
            )
            if use_cache:
                hidden_states, present = hidden_states
                presents.append(present)

        output = self.layernorm_f(hidden_states)

        if use_cache:
            self.set_cache(presents)

        return output


class GPTForPreTraining(nn.Module):
    """
    GPT Model with classification head on top.
    """

    def __init__(self, cfg) -> None:
        super().__init__()
        if cfg.pretrained_model_path is not None:
            loader = GPT2LoaderHuggerFace(GPTModel, cfg, cfg.pretrained_model_path)
            self.GPT_model = loader.load()
        else:
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
        logits = self.GPT_model(input_ids)["logits"]
        if labels is not None:
            lm_loss = self.loss_func(logits, labels)
            return lm_loss
        else:
            return {"prediction_scores": logits}

    @staticmethod
    def set_pipeline_stage_id(model: nn.Module):
        dist_utils = dist.get_dist_util()

        if hasattr(model.GPT_model.transformer.layernorm_f, "config"):
            # Old API in OneFlow 0.8
            for module_block in model.modules():
                if isinstance(module_block.origin, (GPTEmbedding)):
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
        else:
            for module_block in model.modules():
                if isinstance(module_block.to(nn.Module), (GPTEmbedding)):
                    module_block.to(nn.graph.GraphModule).set_stage(
                        dist_utils.get_layer_stage_id(0), dist.get_layer_placement(0)
                    )
                elif isinstance(module_block.to(nn.Module), TransformerLayer):
                    module_block.to(nn.graph.GraphModule).set_stage(
                        dist_utils.get_layer_stage_id(module_block.layer_idx),
                        dist.get_layer_placement(module_block.layer_idx),
                    )
                elif isinstance(module_block.to(nn.Module), (LMLogits, GPTLoss)):
                    module_block.to(nn.graph.GraphModule).set_stage(
                        dist_utils.get_layer_stage_id(-1), dist.get_layer_placement(-1)
                    )

            model.GPT_model.transformer.layernorm_f.to(nn.graph.GraphModule).set_stage(
                dist_utils.get_layer_stage_id(-1), dist.get_layer_placement(-1)
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
