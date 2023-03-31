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

import oneflow as flow
import oneflow.nn.functional as F
from oneflow import nn

import libai.utils.distributed as dist
from libai.config import configurable
from libai.inference.generator.generation_utils import Generator
from libai.layers import LayerNorm, LMLogits, ParallelCrossEntropyLoss
from libai.models.utils import init_method_normal, scaled_init_method_normal
from projects.GLM.layers.embedding_layer import GLMEmbedding
from projects.GLM.layers.transformer_layer import TransformerLayer


class Transformer(nn.Module):
    def __init__(
        self,
        num_layers,
        hidden_size,
        num_attention_heads,
        attention_dropout_prob=0.0,
        output_dropout_prob=0.0,
        layernorm_epsilon=1.0e-5,
        init_method=nn.init.xavier_normal_,
        output_layer_init_method=None,
        bias_gelu_fusion=False,
        bias_dropout_fusion=False,
        scale_mask_softmax_fusion=False,
        apply_query_key_layer_scaling=False,
        attention_scale=1.0,
    ):
        super().__init__()
        self.num_layers = num_layers

        def build_layer(layer_number):
            return TransformerLayer(
                hidden_size,
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
                attention_scale=attention_scale,
                layer_idx=layer_number,
            )

        self.layers = nn.ModuleList([build_layer(i) for i in range(self.num_layers)])
        self.final_layernorm = LayerNorm(hidden_size, eps=layernorm_epsilon, layer_idx=-1)

    def forward(self, hidden_states, attention_mask, memory_states=None):
        mem_layers = [hidden_states.detach()]

        for i, layer in enumerate(self.layers):
            mem_i = memory_states[i] if memory_states is not None else None
            hidden_states = layer(hidden_states, attention_mask, mem=mem_i)
            mem_layers.append(hidden_states.detach())

        output = self.final_layernorm(hidden_states)

        return output, mem_layers


class GLMModel(nn.Module):
    @configurable
    def __init__(
        self,
        num_layers,
        vocab_size,
        hidden_size,
        num_attention_heads,
        max_sequence_length=1024,
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
        amp_enabled=False,
        block_position_encoding=False,
        attention_scale=1.0,
        padding_idx=None,
    ):
        super().__init__()
        init_method = init_method_normal(sigma=initializer_range, mean=0)
        if use_scaled_init_for_output_weights:
            output_layer_init_method = scaled_init_method_normal(initializer_range, num_layers)
        else:
            output_layer_init_method = init_method

        self.embeddings = GLMEmbedding(
            vocab_size,
            hidden_size,
            max_sequence_length,
            padding_idx=padding_idx,
            init_method=init_method,
            embedding_dropout_prob=embedding_dropout_prob,
            amp_enabled=amp_enabled,
            block_position_encoding=block_position_encoding,
        )

        self.transformer = Transformer(
            num_layers,
            hidden_size,
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
            attention_scale=attention_scale,
        )

        self.lm_head = LMLogits(vocab_size, bias=False)

    @classmethod
    def from_config(cls, cfg):
        return {
            "num_layers": cfg.num_layers,
            "vocab_size": cfg.vocab_size,
            "hidden_size": cfg.hidden_size,
            "num_attention_heads": cfg.num_attention_heads,
            "max_sequence_length": cfg.max_sequence_length,
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
            "amp_enabled": cfg.amp_enabled,
            "block_position_encoding": cfg.block_position_encoding,
            "attention_scale": cfg.attention_scale,
            "padding_idx": cfg.padding_idx,
        }

    def forward(
        self,
        input_ids,
        position_ids=None,
        attention_mask=None,
        memory_states=None,
        output_predict=True,
    ):
        input_ids = input_ids.to_global(placement=dist.get_layer_placement(0))
        position_ids = (
            position_ids.to_global(placement=dist.get_layer_placement(0))
            if position_ids is not None
            else None
        )
        attention_mask = (
            attention_mask.to_global(placement=dist.get_layer_placement(0))
            if attention_mask is not None
            else None
        )

        batch_size, query_length = input_ids.size()
        memory_length = memory_states[0].size(1) if memory_states is not None else 0
        is_scalar = flow.numel(attention_mask) == 1
        is_sep = is_scalar or flow.numel(attention_mask) == batch_size

        if is_sep:
            sep = attention_mask.item() if is_scalar else attention_mask
            attention_mask = self.build_mask_matrix(
                batch_size, query_length, sep, memory_length=memory_length, is_scalar=is_scalar
            )
        else:
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
            attention_mask = attention_mask[:, :, :, -query_length - memory_length :]

        input_embeds = self.embeddings(input_ids, position_ids)

        logits, mem_layers = self.transformer(
            input_embeds, attention_mask=attention_mask, memory_states=memory_states
        )
        mem_layers = self.update_mems(mem_layers, memory_states)

        if output_predict:
            logits = self.lm_head(logits, self.embeddings.word_embeddings.weight)

        return (logits, mem_layers)

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

    def build_mask_matrix(self, batch_size, seq_length, sep, memory_length=0, is_scalar=False):
        m = flow.tril(
            flow.ones((1, seq_length, seq_length)),
        )
        if is_scalar:
            m[0, :, : int(sep)] = 1
        else:
            m = m.expand(batch_size, -1, -1)
            ids = flow.arange(seq_length, device=sep.device, dtype=sep.dtype).view(1, -1)
            mask = ids < sep.view(-1, 1)
            m = m.masked_fill(mask.unsqueeze(1).expand_as(m), 1)
        if memory_length > 0:
            m = m.expand(batch_size, -1, -1)
            m = flow.cat((flow.ones((batch_size, seq_length, memory_length)), m), dim=2)
        m = m.unsqueeze(1)
        m = m.to_global(
            sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
            placement=dist.get_layer_placement(0),
        )
        return m

    def update_mems(self, hiddens, mems):
        memory_length = mems[0].size(1) if mems is not None else 0
        query_length = hiddens[0].size(1)
        new_memory_length = memory_length + query_length

        new_mems = []
        for i in range(len(hiddens)):
            if new_memory_length <= query_length:
                new_mems.append(hiddens[i][:, -new_memory_length:])
            else:
                new_mems.append(
                    flow.cat((mems[i][:, -new_memory_length + query_length :], hiddens[i]), dim=1)
                )
        return new_mems


class GLMLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_func = ParallelCrossEntropyLoss()

    def forward(self, logits, labels):
        lm_loss = self.loss_func(logits, labels)
        lm_loss = lm_loss.mean()
        return {"lm_loss": lm_loss}


class GLMForMultipleChoice(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.glm = GLMModel(cfg)
        self.loss_func = GLMLoss()

    def forward(
        self,
        input_ids=None,
        position_ids=None,
        attention_mask=None,
        choice_ids=None,
        choice_indices=None,
        labels=None,
        mems=None,
        **kwargs,
    ):
        lm_logits, mem_layers = self.glm(
            input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            memory_states=mems,
            **kwargs,
        )
        outputs = F.log_softmax(lm_logits, dim=-1)
        log_probs = []
        for output, choices, choice_index in zip(outputs, choice_ids, choice_indices):
            log_probs_single = []
            for choice, choice_target_id in zip(choices, choice_index):
                tmp = output[choice_target_id, choice]
                log_probs_single.append(tmp.sum())
            log_probs.append(flow.stack(log_probs_single))
        log_probs = flow.stack(log_probs)
        loss = None
        if labels is not None:
            loss = self.loss_func(log_probs, labels)
        return {"loss": loss, "logits": log_probs, "lm_logits": lm_logits, "mems": mem_layers}


class GLMForConditionalGeneration(nn.Module, Generator):
    @configurable
    def __init__(
        self,
        num_layers,
        vocab_size,
        hidden_size,
        num_attention_heads,
        max_sequence_length=1024,
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
        amp_enabled=False,
        block_position_encoding=False,
        attention_scale=1.0,
        padding_idx=None,
        cfg=None,
    ):
        super().__init__()
        self.cfg = cfg
        self.glm = GLMModel(
            num_layers=num_layers,
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            max_sequence_length=max_sequence_length,
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
            amp_enabled=amp_enabled,
            block_position_encoding=block_position_encoding,
            attention_scale=attention_scale,
            padding_idx=padding_idx,
            cfg=cfg,
        )
        self.loss_func = GLMLoss()

    @classmethod
    def from_config(cls, cfg):
        return {
            "num_layers": cfg.num_layers,
            "vocab_size": cfg.vocab_size,
            "hidden_size": cfg.hidden_size,
            "num_attention_heads": cfg.num_attention_heads,
            "max_sequence_length": cfg.max_sequence_length,
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
            "amp_enabled": cfg.amp_enabled,
            "block_position_encoding": cfg.block_position_encoding,
            "attention_scale": cfg.attention_scale,
            "padding_idx": cfg.padding_idx,
            "cfg": cfg,
        }

    def forward(
        self,
        input_ids=None,
        position_ids=None,
        attention_mask=None,
        labels=None,
        memory_states=None,
        **kwargs,
    ):
        lm_logits, mems = self.glm(
            input_ids, position_ids, attention_mask, memory_states=memory_states, **kwargs
        )
        loss = None
        if labels is not None:
            loss = self.loss_func(lm_logits, labels)
        return {"loss": loss, "logits": lm_logits, "mems": mems}

    def _reorder_cache(self, past, beam_idx):
        if past is None:
            return past
        reordered_decoder_past = ()
        for layer_past_states in past:
            beam_idx = beam_idx.to_global(placement=layer_past_states.placement)
            reordered_decoder_past = reordered_decoder_past + (
                layer_past_states.index_select(0, beam_idx),
            )
        return reordered_decoder_past

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past=None,
        position_ids=None,
        generation_attention_mask=None,
        **kwargs,
    ):
        attention_mask = generation_attention_mask
        # only last token for inputs_ids if past is defined in kwargs
        seq_length = input_ids.shape[1]
        if past:
            if position_ids is not None:
                position_ids = position_ids[:, :, seq_length - 1].unsqueeze(-1)
            if attention_mask is not None:
                attention_mask = attention_mask[:, :, seq_length - 1, :seq_length].unsqueeze(-2)
            input_ids = input_ids[:, -1].unsqueeze(-1)
        else:
            if position_ids is not None:
                position_ids = position_ids[:, :, :seq_length]
            if attention_mask is not None:
                attention_mask = attention_mask[:, :, :seq_length, :seq_length]
        return {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "memory_states": past,
        }

    @staticmethod
    def set_pipeline_stage_id(model: nn.Module):
        dist_utils = dist.get_dist_util()

        if hasattr(model.glm.transformer.final_layernorm, "config"):
            # Old API in OneFlow 0.8
            for module_block in model.modules():
                if isinstance(module_block.origin, GLMEmbedding):
                    module_block.config.set_stage(
                        dist_utils.get_layer_stage_id(0), dist.get_layer_placement(0)
                    )
                elif isinstance(module_block.origin, TransformerLayer):
                    module_block.config.set_stage(
                        dist_utils.get_layer_stage_id(module_block.layer_idx),
                        dist.get_layer_placement(module_block.layer_idx),
                    )
                elif isinstance(module_block.origin, (LMLogits, GLMLoss)):
                    module_block.config.set_stage(
                        dist_utils.get_layer_stage_id(-1), dist.get_layer_placement(-1)
                    )

            model.glm.transformer.final_layernorm.config.set_stage(
                dist_utils.get_layer_stage_id(-1), dist.get_layer_placement(-1)
            )
        else:
            for module_block in model.modules():
                if isinstance(module_block.to(nn.Module), GLMEmbedding):
                    module_block.to(nn.graph.GraphModule).set_stage(
                        dist_utils.get_layer_stage_id(0), dist.get_layer_placement(0)
                    )
                elif isinstance(module_block.to(nn.Module), TransformerLayer):
                    module_block.to(nn.graph.GraphModule).set_stage(
                        dist_utils.get_layer_stage_id(module_block.layer_idx),
                        dist.get_layer_placement(module_block.layer_idx),
                    )
                elif isinstance(module_block.to(nn.Module), (LMLogits, GLMLoss)):
                    module_block.to(nn.graph.GraphModule).set_stage(
                        dist_utils.get_layer_stage_id(-1), dist.get_layer_placement(-1)
                    )

            model.glm.transformer.final_layernorm.to(nn.graph.GraphModule).set_stage(
                dist_utils.get_layer_stage_id(-1), dist.get_layer_placement(-1)
            )
