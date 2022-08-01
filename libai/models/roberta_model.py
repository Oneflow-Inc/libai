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
    Embedding,
    LayerNorm,
    Linear,
    LMLogits,
    ParallelCrossEntropyLoss,
    TransformerLayer,
    VocabEmbedding,
    build_activation,
)
from libai.utils import distributed as dist

from .bert_model import BertEmbeddings, BertExtendedAttnMask, BertModel, BertPooler
from .utils import init_method_normal


class RobertaExtendedAttnMask(BertExtendedAttnMask):
    """
    Same as BertExtendedAttnMask.
    """


class RobertaEmbeddings(BertEmbeddings):
    """
    Same as BertEmbeddings with a tiny tweak for vocab_embeddings and position_embeddings.
    """

    def __init__(
        self,
        vocab_size,
        hidden_size,
        max_sequence_length,
        embedding_dropout_prob,
        num_tokentypes=0,
        pad_token_id=1,
        init_method=nn.init.xavier_normal_,
        amp_enabled=False,
    ):
        super().__init__(
            vocab_size,
            hidden_size,
            max_sequence_length,
            embedding_dropout_prob,
            num_tokentypes=num_tokentypes,
            init_method=init_method,
            amp_enabled=amp_enabled,
        )
        self.pad_token_id = pad_token_id
        self.vocab_embeddings = VocabEmbedding(
            vocab_size,
            hidden_size,
            init_method=init_method,
            amp_enabled=amp_enabled,
            padding_idx=pad_token_id,
        )
        self.position_embeddings = Embedding(
            max_sequence_length,
            hidden_size,
            init_method=init_method,
            amp_enabled=amp_enabled,
            padding_idx=pad_token_id,
        )

        if num_tokentypes > 0:
            self.tokentype_embeddings = Embedding(
                num_tokentypes, hidden_size, init_method=init_method, amp_enabled=amp_enabled
            )
            self.tokentype_ids = flow.zeros(
                1,
                max_sequence_length,
                dtype=flow.long,
                sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
                placement=dist.get_layer_placement(0),
            )
        else:
            self.tokentype_embeddings = None

    def forward(self, input_ids, tokentype_ids=None, position_ids=None):
        seq_length = input_ids.size()[1]

        word_embeddings = self.vocab_embeddings(input_ids)

        if position_ids is None:
            position_ids = self.create_position_ids_from_input_ids(input_ids, self.pad_token_id)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = word_embeddings + position_embeddings

        if self.tokentype_embeddings is not None:
            if tokentype_ids is None:
                tokentype_ids = (
                    self.tokentype_ids[:, :seq_length]
                    .expand_as(input_ids)
                    .to_global(sbp=input_ids.sbp)
                )
            embeddings = embeddings + self.tokentype_embeddings(tokentype_ids)
        embeddings = self.embedding_dropout(embeddings)
        return embeddings

    def create_position_ids_from_input_ids(self, input_ids, pad_token_id):
        mask = input_ids.ne(pad_token_id).int()
        position_ids = (flow.cumsum(mask, dim=1).type_as(mask)) * mask + pad_token_id
        position_ids = position_ids.to_global(sbp=input_ids.sbp, placement=input_ids.placement)
        return position_ids


class RobertaPooler(BertPooler):
    """
    Same as BertPooler.
    """


class RobertaLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.lm_loss = ParallelCrossEntropyLoss()

    def forward(self, lm_output, lm_labels, loss_mask):
        lm_labels = lm_labels.to_global(placement=lm_output.placement)
        loss_mask = loss_mask.to_global(placement=lm_output.placement)
        lm_loss = self.lm_loss(lm_output, lm_labels)
        loss_mask = loss_mask.float()
        # Change loss_mask.sum() sbp sign from [P, B] -> [B, B]
        # because (lm_loss * loss_mask) / loss_mask.sum() cannot accept P / P
        denominator = loss_mask.sum().to_global(
            sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast])
        )
        masked_lm_loss = flow.sum(lm_loss.view(-1) * loss_mask.view(-1)) / denominator
        masked_lm_loss = masked_lm_loss.to_global(
            sbp=dist.get_nd_sbp([flow.sbp.partial_sum, flow.sbp.broadcast])
        )
        loss_dict = {"lm_loss": masked_lm_loss}
        return loss_dict


class RobertaModel(BertModel):
    """The bare Roberta Model transformer outputting raw hidden-states without
    any specific head on top.

        Args:
            vocab_size (int):
                The size of vocabulary file.
            hidden_size (int):
                The size of hidden states.
            hidden_layers (int):
                The number of ``TransformerLayer`` in encoder.
            num_attention_heads (int):
                The number of attention heads for each attention layer of ``TransformerLayer``.
            intermediate_size (int):
                The size of intermediate layer in feed-forward network for each
                ``TransformerLayer``.
            hidden_dropout_prob (float, optional):
                The dropout ratio for the output for each TransformerLayer. Defaults to 0.0.
            attention_probs_dropout_prob (float, optional):
                The dropout ratio for the output of each attention layer in ``TransformerLayer``.
                Defaults to 0.0.
            max_position_embeddings (int):
                Max sequence length of input, defines the shape of Position Embeddings
                in ``RobertaEmbeddings``.
            type_vocab_size (int, optional):
                Number of segment token indices. Defaults to 2.
            add_pooling_layer (bool, optional):
                Whether or not averaging or pooling the sequence of hidden-states for the
                whole input sequence. Defaults to ``True``.
            initializer_range (float, optional):
                Sigma of the normal distribution in the initialization method. Defaults to 0.02.
            layer_norm_eps (float, optional):
                The epsilon of LayerNorm layer. Defaults to 1e-5.
            pad_token_id (int, optional):
                The token id used for padding. Defaults to 1.
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
                Defaults to ``True``.
            apply_residual_post_layernorm (bool, optional):
                If set ``True``, use original BERT(Roberta) residual connection ordering
                otherwise use Megatron BERT residual connection which is more stable
                when scaling model size introduced in https://arxiv.org/pdf/1909.08053.pdf.
                Default: ``False``.
            amp_enabled (bool, optional):
                Whether or not to set fp16 for embedding weight in T5 model. Defaults to ``False``.
    """

    @configurable
    def __init__(
        self,
        vocab_size,
        hidden_size,
        hidden_layers,
        num_attention_heads,
        intermediate_size,
        hidden_dropout_prob,
        attention_probs_dropout_prob,
        max_position_embeddings,
        num_tokentypes=2,
        add_pooling_layer=True,
        initializer_range=0.02,
        layernorm_eps=1e-12,
        pad_token_id=1,
        bias_gelu_fusion=True,
        bias_dropout_fusion=True,
        scale_mask_softmax_fusion=True,
        apply_query_key_layer_scaling=True,
        apply_residual_post_layernorm=False,
        amp_enabled=False,
    ):
        super().__init__(
            vocab_size,
            hidden_size,
            hidden_layers,
            num_attention_heads,
            intermediate_size,
            hidden_dropout_prob,
            attention_probs_dropout_prob,
            max_position_embeddings,
            num_tokentypes=num_tokentypes,
            add_pooling_layer=add_pooling_layer,
            initializer_range=initializer_range,
            layernorm_eps=layernorm_eps,
            bias_gelu_fusion=bias_gelu_fusion,
            bias_dropout_fusion=bias_dropout_fusion,
            scale_mask_softmax_fusion=scale_mask_softmax_fusion,
            apply_query_key_layer_scaling=apply_query_key_layer_scaling,
            apply_residual_post_layernorm=apply_residual_post_layernorm,
            amp_enabled=amp_enabled,
        )

        init_method = init_method_normal(initializer_range)

        # Embeddings
        self.embeddings = RobertaEmbeddings(
            vocab_size,
            hidden_size,
            max_position_embeddings,
            hidden_dropout_prob,
            num_tokentypes,
            pad_token_id,
            init_method,
            amp_enabled,
        )

        # Mask generation
        self.extended_attn_mask = RobertaExtendedAttnMask()
        self.pooler = RobertaPooler(hidden_size, init_method) if add_pooling_layer else None

    @classmethod
    def from_config(cls, cfg):
        return {
            "vocab_size": cfg.vocab_size,
            "hidden_size": cfg.hidden_size,
            "hidden_layers": cfg.hidden_layers,
            "num_attention_heads": cfg.num_attention_heads,
            "intermediate_size": cfg.intermediate_size,
            "hidden_dropout_prob": cfg.hidden_dropout_prob,
            "attention_probs_dropout_prob": cfg.attention_probs_dropout_prob,
            "max_position_embeddings": cfg.max_position_embeddings,
            "num_tokentypes": cfg.num_tokentypes,
            "add_pooling_layer": cfg.add_pooling_layer,
            "initializer_range": cfg.initializer_range,
            "layernorm_eps": cfg.layernorm_eps,
            "pad_token_id": cfg.pad_token_id,
            "bias_gelu_fusion": cfg.bias_gelu_fusion,
            "bias_dropout_fusion": cfg.bias_dropout_fusion,
            "scale_mask_softmax_fusion": cfg.scale_mask_softmax_fusion,
            "apply_query_key_layer_scaling": cfg.apply_query_key_layer_scaling,
            "apply_residual_post_layernorm": cfg.apply_residual_post_layernorm,
            "amp_enabled": cfg.amp_enabled,
        }


class RobertaLMHead(nn.Module):
    def __init__(self, vocab_size, hidden_size, init_method, layer_norm_eps):
        super().__init__()
        self.dense = Linear(
            hidden_size,
            hidden_size,
            bias=True,
            parallel="data",
            init_method=init_method,
            layer_idx=-1,
        )
        self.activation_func = build_activation("gelu")
        self.layernorm = LayerNorm((hidden_size,), eps=layer_norm_eps, layer_idx=-1)

        # NOTE(xzp): LMLogits as a decoder:nn.Linear(hidden_size, vocab_size),
        # it shares the roberta.word_embeddings.weight
        self.lm_logits = LMLogits(vocab_size, bias=True)

    def forward(self, hidden_states, word_embeddings_weight):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.activation_func(hidden_states)
        hidden_states = hidden_states.to_global(
            sbp=dist.get_nd_sbp([flow.sbp.split(0), flow.sbp.broadcast])
        )
        hidden_states = self.layernorm(hidden_states)
        hidden_states = self.lm_logits(hidden_states, word_embeddings_weight)
        return hidden_states


class RobertaPreTrainedModel(nn.Module):
    @staticmethod
    def set_pipeline_stage_id(model):
        dist_utils = dist.get_dist_util()

        # Set pipeline parallelism stage_id
        for module_block in model.modules():
            # module.origin can get the original module
            if isinstance(module_block.origin, RobertaEmbeddings):
                module_block.config.set_stage(
                    dist_utils.get_layer_stage_id(0), dist.get_layer_placement(0)
                )
            elif isinstance(module_block.origin, RobertaExtendedAttnMask):
                module_block.config.set_stage(
                    dist_utils.get_layer_stage_id(0), dist.get_layer_placement(0)
                )
            elif isinstance(module_block.origin, TransformerLayer):
                module_block.config.set_stage(
                    dist_utils.get_layer_stage_id(module_block.layer_idx),
                    dist.get_layer_placement(module_block.layer_idx),
                )
            # `add_pooling_layer` in RobertaForMaskedLM and RobertaForCausalLM.
            # default to False.
            elif isinstance(module_block.origin, RobertaPooler):
                module_block.config.set_stage(
                    dist_utils.get_layer_stage_id(-1), dist.get_layer_placement(-1)
                )
            elif isinstance(module_block.origin, RobertaLMHead):
                module_block.config.set_stage(
                    dist_utils.get_layer_stage_id(-1), dist.get_layer_placement(-1)
                )

        # Set the last layernorm stage id
        model.roberta.final_layernorm.config.set_stage(
            dist_utils.get_layer_stage_id(-1), dist.get_layer_placement(-1)
        )


class RobertaForPreTraining(RobertaPreTrainedModel):
    def __init__(self, cfg):
        super().__init__()

        cfg.add_pooling_layer = False
        self.roberta = RobertaModel(cfg)
        self.lm_head = RobertaLMHead(
            cfg.vocab_size,
            cfg.hidden_size,
            init_method_normal(cfg.initializer_range),
            cfg.layernorm_eps,
        )
        self.loss_fc = RobertaLoss()

    def forward(
        self,
        input_ids,
        attention_mask,
        tokentype_ids=None,
        lm_labels=None,
        loss_mask=None,
    ):
        """

        Args:
            input_ids (flow.LongTensor): Indices of input sequence tokens in vocabulary.
            attention_mask (flow.BoolTensor): Mask to avoid performing attention on
                padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
            tokentype_ids (flow.LongTensor, optional): Segment token indices to indicate first
                and second portions of the inputs. Indices are selected in `[0, 1]`.
                Defaults to None.
            labels (flow.LongTensor, optional): Labels for computing the masked
                language modeling loss. Indices should be in `[-1, 0, ..., config.vocab_size]`.
                Defaults to None.
            loss_mask (flow.BoolTensor, optional): Mask to avoid performing loss computing
                on ignored tokens. Tokens with indices set to `-1` are ignored (masked), the
                loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
                Defaults to None.
        """
        input_ids = input_ids.to_global(placement=dist.get_layer_placement(0))
        attention_mask = attention_mask.to_global(placement=dist.get_layer_placement(0))
        tokentype_ids = tokentype_ids.to_global(placement=dist.get_layer_placement(0))

        outputs = self.roberta(input_ids, attention_mask, tokentype_ids=tokentype_ids)
        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output, self.roberta.word_embeddings_weight())

        if lm_labels is not None:
            return self.loss_fc(prediction_scores, lm_labels, loss_mask)

        return {"prediction_scores": prediction_scores}


class RobertaForCausalLM(RobertaPreTrainedModel):
    def __init__(self, cfg):
        super().__init__()

        cfg.add_pooling_layer = False
        self.roberta = RobertaModel(cfg)
        self.lm_head = RobertaLMHead(
            cfg.vocab_size,
            cfg.hidden_size,
            init_method_normal(cfg.initializer_range),
            cfg.layernorm_eps,
        )
        self.loss_fc = RobertaLoss()

    def forward(
        self,
        input_ids,
        attention_mask,
        tokentype_ids=None,
        position_ids=None,
        labels=None,
        loss_mask=None,
    ):
        """

        Args:
            input_ids (flow.LongTensor): Indices of input sequence tokens in vocabulary.
            attention_mask (flow.BoolTensor): Mask to avoid performing attention on
                padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
            tokentype_ids (flow.LongTensor, optional): Segment token indices to indicate first
                and second portions of the inputs. Indices are selected in `[0, 1]`.
                Defaults to None.
            position_ids (flow.LongTensor, optional): Indices of positions of each input sequence
                tokens in the position embeddings. Defaults to None.
            labels (flow.LongTensor, optional): Labels for computing the masked
                language modeling loss. Indices should be in `[-1, 0, ..., config.vocab_size]`.
                Defaults to None.
            loss_mask (flow.BoolTensor, optional): Mask to avoid performing loss computing
                on ignored tokens. Tokens with indices set to `-1` are ignored (masked), the
                loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
                Defaults to None.
        """
        outputs = self.roberta(input_ids, attention_mask, position_ids, tokentype_ids)
        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output, self.roberta.word_embeddings_weight())

        if labels is not None:
            # next-token prediction task, shift prediction_scores and labels by one.
            shifted_prediction_scores = prediction_scores[:, :-1, :].contiguous()
            shifted_prediction_scores = shifted_prediction_scores.to_global(
                sbp=prediction_scores.sbp
            )
            shifted_labels = labels[:, 1:].contiguous()
            shifted_labels = shifted_labels.to_global(sbp=shifted_labels.sbp)
            lm_loss = self.loss_fc(shifted_prediction_scores, shifted_labels, loss_mask)
            return {"lm_loss": lm_loss}

        return {"prediction_scores": prediction_scores}
