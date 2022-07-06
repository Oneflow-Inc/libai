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
from libai.layers.attention import AttnMaskType
from libai.utils import distributed as dist

from .utils import init_method_normal, scaled_init_method_normal


class BertExtendedAttnMask(nn.Module):
    def forward(self, attention_mask):
        # We create a 3D attention mask from a 2D tensor mask.
        # [b, 1, s]
        attention_mask_b1s = attention_mask.unsqueeze(1)
        # [b, s, 1]
        attention_mask_bs1 = attention_mask.unsqueeze(2)
        # [b, s, s]
        attention_mask_bss = attention_mask_b1s * attention_mask_bs1
        # [b, 1, s, s]
        extended_attention_mask = attention_mask_bss.unsqueeze(1)

        return extended_attention_mask


class BertEmbeddings(nn.Module):
    def __init__(
        self,
        vocab_size,
        hidden_size,
        max_sequence_length,
        embedding_dropout_prob,
        num_tokentypes=0,
        init_method=nn.init.xavier_normal_,
        amp_enabled=False,
    ):
        super().__init__()
        self.vocab_embeddings = VocabEmbedding(
            vocab_size, hidden_size, init_method=init_method, amp_enabled=amp_enabled
        )
        self.position_embeddings = Embedding(
            max_sequence_length, hidden_size, init_method=init_method, amp_enabled=amp_enabled
        )

        # NOTE(l1aoxingyu): Set position_ids sbp sign to [B, B] initially, because position_ids is a
        # 1D-tensor from 0 to seq_length, if set to [S(0), B] at first, then position_ids
        # will split at the first dim of hierarchy.
        self.position_ids = flow.arange(
            max_sequence_length,
            dtype=flow.long,
            sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
            placement=dist.get_layer_placement(0),
        ).unsqueeze(0)

        if num_tokentypes > 0:
            self.tokentype_embeddings = Embedding(
                num_tokentypes, hidden_size, init_method=init_method, amp_enabled=amp_enabled
            )
            self.tokentype_ids = flow.zeros(
                self.position_ids.size(),
                dtype=flow.long,
                sbp=self.position_ids.sbp,
                placement=self.position_ids.placement,
            )
        else:
            self.tokentype_embeddings = None

        self.embedding_dropout = nn.Dropout(embedding_dropout_prob)

    def forward(self, input_ids, tokentype_ids=None, position_ids=None):
        seq_length = input_ids.size()[1]

        word_embeddings = self.vocab_embeddings(input_ids)
        if position_ids is None:
            # Change position_ids sbp sign: [B, B] -> [S(0), B]
            position_ids = (
                self.position_ids[:, :seq_length].expand_as(input_ids).to_global(sbp=input_ids.sbp)
            )
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

    def word_embeddings(self):
        return self.vocab_embeddings.weight


class BertLMPredictionHead(nn.Module):
    def __init__(self, hidden_size, init_method):
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
        self.layernorm = LayerNorm((hidden_size,), layer_idx=-1)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.activation_func(hidden_states)
        hidden_states = hidden_states.to_global(
            grad_sbp=dist.get_nd_sbp([flow.sbp.split(0), flow.sbp.split(2)])
        )

        # NOTE(l1aoxingyu): hidden_states shape is [B, S, H] whose sbp sign: [S(0), S(2)]
        # Change from [S(0), S(2)] -> [S(0), B] because layernorm cannot get inputs with sbp S(2)
        hidden_states = hidden_states.to_global(
            sbp=dist.get_nd_sbp([flow.sbp.split(0), flow.sbp.broadcast])
        )
        hidden_states = self.layernorm(hidden_states)
        return hidden_states


class BertPooler(nn.Module):
    """Pooler layer.

    Pool hidden states of the first token and
    add a linear transformation followed by a tanh.

    Args:
        hidden_size: hidden state feature dimension
    """

    def __init__(self, hidden_size, init_method):
        super().__init__()
        self.dense = Linear(
            hidden_size,
            hidden_size,
            bias=True,
            parallel="col",
            init_method=init_method,
            layer_idx=-1,
        )
        self.activation_func = build_activation("tanh")

    def forward(self, hidden_states):
        """Just "pool" the model by simply taking the [CLS] token corresponding
        to the first token."""
        # hidden_states: [bsz, seq_len, hidden_size]
        select_token_tensor = hidden_states[:, 0, :]
        pooled_output = self.dense(select_token_tensor)
        pooled_output = self.activation_func(pooled_output)
        return pooled_output


class BertLoss(nn.Module):
    def __init__(self, add_binary_head):
        super().__init__()
        self.add_binary_head = add_binary_head
        self.lm_loss = ParallelCrossEntropyLoss()

    def forward(self, lm_output, lm_labels, loss_mask, binary_logits, ns_labels):
        lm_labels = lm_labels.to_global(placement=lm_output.placement)
        loss_mask = loss_mask.to_global(placement=lm_output.placement)
        binary_logits = binary_logits.to_global(placement=lm_output.placement)
        ns_labels = ns_labels.to_global(placement=lm_output.placement)
        lm_loss = self.lm_loss(lm_output, lm_labels)
        loss_mask = loss_mask.float()
        # Change loss_mask.sum() sbp sign from [P, B] -> [B, B]
        # because (lm_loss * loss_mask) / loss_mask.sum() cannot accept P / P
        denominator = (
            loss_mask.sum().to_global(sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]))
            + 1e-7
        )
        masked_lm_loss = flow.sum(lm_loss.view(-1) * loss_mask.view(-1)) / denominator
        # NOTE(l1aoxingyu): Change lm loss sbp sign [P, P] -> [P, B] to add with sop loss
        # whose sbp sign: [P, B]
        masked_lm_loss = masked_lm_loss.to_global(
            sbp=dist.get_nd_sbp([flow.sbp.partial_sum, flow.sbp.broadcast])
        )

        loss_dict = {"lm_loss": masked_lm_loss}

        if self.add_binary_head:
            sop_loss = flow._C.cross_entropy(
                binary_logits, ns_labels, ignore_index=-1, reduction="none"
            ).mean()
            loss_dict["sop_loss"] = sop_loss
        return loss_dict


class BertModel(nn.Module):
    """The bare Bert Model transformer outputting raw hidden-states without
    any specific head on top.

    Args:
        vocab_size (int): The size of vocabulary file.
        hidden_size (int): The size of hidden states.
        hidden_layers (int): The number of ``TransformerLayer`` in encoder.
        num_attention_heads (int):
            The number of attention heads for each attention layer of ``TransformerLayer``.
        intermediate_size (int):
            The size of intermediate layer in feed-forward network for each ``TransformerLayer``.
        hidden_dropout_prob  (float, optional):
            The dropout ratio for the output for each TransformerLayer. Defaults to 0.0.
        attention_probs_dropout_prob  (float, optional):
            The dropout ratio for the output of each attention layer in ``TransformerLayer``.
            Defaults to 0.0.
        max_position_embeddings (int):
            Max sequence length of input, defines the shape of Position Embeddings
            in ``BertEmbedding``.
        num_tokentypes (int, optional):
            Number of segment token indices. Defaults to 2.
        add_pooling_layer (bool, optional):
            Whether or not averaging or pooling the sequence of hidden-states for the
            whole input sequence. Defaults to ``True``.
        initializer_range (float, optional):
            Sigma of the normal distribution in the initialization method. Defaults to 0.02.
        layernorm_epsilon (float, optional):
            The epsilon of LayerNorm layer. Defaults to 1e-5.
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
        bias_gelu_fusion=True,
        bias_dropout_fusion=True,
        scale_mask_softmax_fusion=True,
        apply_query_key_layer_scaling=True,
        apply_residual_post_layernorm=False,
        amp_enabled=False,
    ):
        super().__init__()
        init_method = init_method_normal(initializer_range)
        scaled_init_method = scaled_init_method_normal(initializer_range, hidden_layers)

        # Embeddings
        self.embeddings = BertEmbeddings(
            vocab_size,
            hidden_size,
            max_position_embeddings,
            hidden_dropout_prob,
            num_tokentypes,
            init_method,
            amp_enabled,
        )

        # Mask generation
        self.extended_attn_mask = BertExtendedAttnMask()

        # Encoders
        self.encoders = nn.ModuleList(
            [
                TransformerLayer(
                    hidden_size,
                    intermediate_size,
                    num_attention_heads,
                    attention_dropout_prob=attention_probs_dropout_prob,
                    output_dropout_prob=hidden_dropout_prob,
                    layernorm_epsilon=layernorm_eps,
                    bias_gelu_fusion=bias_gelu_fusion,
                    bias_dropout_fusion=bias_dropout_fusion,
                    scale_mask_softmax_fusion=scale_mask_softmax_fusion,
                    apply_query_key_layer_scaling=apply_query_key_layer_scaling,
                    init_method=init_method,
                    output_layer_init_method=scaled_init_method,
                    apply_residual_post_layernorm=apply_residual_post_layernorm,
                    attn_mask_type=AttnMaskType.padding,  # bert mask type
                    layer_idx=i,
                )
                for i in range(hidden_layers)
            ]
        )
        self.final_layernorm = LayerNorm((hidden_size,), eps=layernorm_eps, layer_idx=-1)

        self.pooler = BertPooler(hidden_size, init_method) if add_pooling_layer else None

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
            "bias_gelu_fusion": cfg.bias_gelu_fusion,
            "bias_dropout_fusion": cfg.bias_dropout_fusion,
            "scale_mask_softmax_fusion": cfg.scale_mask_softmax_fusion,
            "apply_query_key_layer_scaling": cfg.apply_query_key_layer_scaling,
            "apply_residual_post_layernorm": cfg.apply_residual_post_layernorm,
            "amp_enabled": cfg.amp_enabled,
        }

    def forward(self, input_ids, attention_mask, tokentype_ids=None):
        """

        Args:
            input_ids (flow.LongTensor): Indices of input sequence tokens in vocabulary.
            attention_mask (flow.BoolTensor): Mask to avoid performing attention
                on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

            tokentype_ids (flow.LongTensor, optional): Segment token indices to indicate first and
                second portions of the inputs. Indices are selected in `[0, 1]`. Defaults to None.
        """
        extended_attention_mask = self.extended_attn_mask(attention_mask)
        embedding_output = self.embeddings(input_ids, tokentype_ids)

        hidden_states = embedding_output
        for layer in self.encoders:
            hidden_states = layer(hidden_states, extended_attention_mask)
        encoder_output = self.final_layernorm(hidden_states)
        pooled_output = self.pooler(encoder_output) if self.pooler is not None else None
        return encoder_output, pooled_output

    def word_embeddings_weight(self):
        return self.embeddings.word_embeddings()


class BertPreTrainingHeads(nn.Module):
    def __init__(self, vocab_size, hidden_size, init_method, add_binary_head=True):
        super().__init__()
        self.predictions = BertLMPredictionHead(hidden_size, init_method)
        self.seq_relationship = Linear(
            hidden_size,
            2,
            bias=True,
            parallel="data",
            init_method=init_method,
            layer_idx=-1,
        )
        self.lm_logits = LMLogits(vocab_size, bias=True)
        self.loss_func = BertLoss(add_binary_head)

    def forward(
        self,
        sequence_output,
        pooled_output,
        word_embeddings_weight,
        ns_labels,
        lm_labels,
        loss_mask,
    ):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        prediction_scores = self.lm_logits(prediction_scores, word_embeddings_weight)

        if lm_labels is not None:
            return self.loss_func(
                prediction_scores, lm_labels, loss_mask, seq_relationship_score, ns_labels
            )
        return {
            "prediction_scores": prediction_scores,
            "seq_relationship_score": seq_relationship_score,
        }


class BertForPreTraining(nn.Module):
    """Bert Model with two heads on top as done during the pretraining: a
    `masked language modeling` head and a `next sentence prediction (classification)` head.
    """

    def __init__(self, cfg):
        super().__init__()
        self.bert = BertModel(cfg)
        self.cls_head = BertPreTrainingHeads(
            cfg.vocab_size,
            cfg.hidden_size,
            init_method_normal(cfg.initializer_range),
            cfg.add_binary_head,
        )

    def forward(
        self,
        input_ids,
        attention_mask,
        tokentype_ids=None,
        ns_labels=None,
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
            ns_labels (flow.LongTensor, optional): Labels for computing the next sequence prediction
                (classification) loss. Input should be a sequence pair (see `input_ids` docstring).
                Indices should be in `[0, 1]`:

                - 0 indicates sequence B is a continuation of sequence A,
                - 1 indicates sequence B is a random sequence.

            lm_labels (flow.LongTensor, optional): Labels for computing the masked
                language modeling loss. Indices should be in `[-1, 0, ..., config.vocab_size]`.
            loss_mask (flow.BoolTensor, optional): Mask to avoid performing loss computing
                on ignored tokens. Tokens with indices set to `-1` are ignored (masked), the
                loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """

        input_ids = input_ids.to_global(placement=dist.get_layer_placement(0))
        attention_mask = attention_mask.to_global(placement=dist.get_layer_placement(0))
        tokentype_ids = tokentype_ids.to_global(placement=dist.get_layer_placement(0))
        outputs = self.bert(input_ids, attention_mask, tokentype_ids)
        sequence_output, pooled_output = outputs[:2]

        return self.cls_head(
            sequence_output,
            pooled_output,
            self.bert.word_embeddings_weight(),
            ns_labels,
            lm_labels,
            loss_mask,
        )

    @staticmethod
    def set_pipeline_stage_id(model):
        dist_utils = dist.get_dist_util()

        # Set pipeline parallelism stage_id
        for module_block in model.modules():
            # module.origin can get the original module
            if isinstance(module_block.origin, BertEmbeddings):
                module_block.config.set_stage(
                    dist_utils.get_layer_stage_id(0), dist.get_layer_placement(0)
                )
            elif isinstance(module_block.origin, BertExtendedAttnMask):
                module_block.config.set_stage(
                    dist_utils.get_layer_stage_id(0), dist.get_layer_placement(0)
                )
            elif isinstance(module_block.origin, TransformerLayer):
                module_block.config.set_stage(
                    dist_utils.get_layer_stage_id(module_block.layer_idx),
                    dist.get_layer_placement(module_block.layer_idx),
                )
            elif isinstance(module_block.origin, BertPooler):
                module_block.config.set_stage(
                    dist_utils.get_layer_stage_id(-1), dist.get_layer_placement(-1)
                )
            elif isinstance(module_block.origin, BertPreTrainingHeads):
                module_block.config.set_stage(
                    dist_utils.get_layer_stage_id(-1), dist.get_layer_placement(-1)
                )

        # Set the last layernorm stage id
        model.bert.final_layernorm.config.set_stage(
            dist_utils.get_layer_stage_id(-1), dist.get_layer_placement(-1)
        )


class BertForClassification(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.num_labels = cfg.num_labels

        self.bert = BertModel(cfg)
        self.classifier = Linear(
            cfg.hidden_size,
            cfg.num_labels,
            bias=True,
            parallel="row",
            init_method=init_method_normal(cfg.initializer_range),
            layer_idx=-1,
        )
        classifier_dropout = (
            cfg.classifier_dropout
            if cfg.classifier_dropout is not None
            else cfg.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)

    def forward(self, input_ids, attention_mask, tokentype_ids=None, labels=None, **kwargs):
        labels = labels if labels is not None else kwargs.get("ns_labels")
        outputs = self.bert(input_ids, attention_mask, tokentype_ids)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            loss = loss.to_global(sbp=dist.get_nd_sbp([flow.sbp.partial_sum, flow.sbp.broadcast]))
            return {"cls_loss": loss}
        else:
            return {"logits": logits}
