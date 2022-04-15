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

from .utils import init_method_normal, scaled_init_method_normal


class RobertaExtendedAttnMask(nn.Module):
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

        # Convert attention mask to binary.
        extended_attention_mask = extended_attention_mask > 0.5
        return extended_attention_mask


class RobertaEmbeddings(nn.Module):
    def __init__(
        self,
        vocab_size,
        hidden_size,
        max_position_embeddings,
        hidden_dropout_prob,
        type_vocab_size=0,
        pad_token_id=1,
        init_method=nn.init.xavier_normal_,
        amp_enabled=False,
    ):
        super().__init__()
        self.word_embeddings = VocabEmbedding(
            vocab_size, hidden_size, init_method=init_method, amp_enabled=amp_enabled, padding_idx=pad_token_id
        )
        self.position_embeddings = Embedding(
            max_position_embeddings, hidden_size, init_method=init_method, amp_enabled=amp_enabled, padding_idx = pad_token_id
        )

        if type_vocab_size > 0:
            self.token_type_embeddings = Embedding(
                type_vocab_size, hidden_size, init_method=init_method, amp_enabled=amp_enabled
            )
            self.token_type_ids = flow.zeros(
                max_position_embeddings,
                dtype=flow.long,
                sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
                placement=dist.get_layer_placement(0),
            ).unsqueeze(0)
        else:
            self.token_type_embeddings = None

        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.padding_idx = pad_token_id

    def forward(self, input_ids, token_type_ids=None, position_ids=None, past_key_values_length=0):
        seq_length = input_ids.size()[1]

        word_embeddings = self.word_embeddings(input_ids)
        if position_ids is None:
            position_ids = self.create_position_ids_from_input_ids(input_ids, self.padding_idx, past_key_values_length)

        position_embeddings = self.position_embeddings(position_ids)
        embeddings = word_embeddings + position_embeddings

        if self.token_type_embeddings is not None:
            if token_type_ids is None:
                token_type_ids = (
                    self.token_type_ids[:, :seq_length]
                    .expand_as(input_ids)
                    .to_global(sbp=input_ids.sbp)
                )
            embeddings = embeddings + self.token_type_embeddings(token_type_ids)

        embeddings = self.dropout(embeddings)
        return embeddings

    def word_embeddings(self):
        return self.word_embeddings.weight

    def create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length=0):
        mask = input_ids.ne(padding_idx).int()
        incremental_indices = (flow.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
        incremental_indices = incremental_indices.long() + padding_idx
        return incremental_indices.to_global(sbp=input_ids.sbp, placement=dist.get_layer_placement(0))


class RobertaPooler(nn.Module):
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


class RobertaModel(nn.Module):
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
        type_vocab_size=2,
        add_pooling_layer=True,
        initializer_range=0.02,
        layernorm_eps=1e-5,
        pad_token_id=1,
        bias_gelu_fusion=False,
        bias_dropout_fusion=False,
        scale_mask_softmax_fusion=False,
        apply_query_key_layer_scaling=False,
        apply_residual_post_layernorm=True,
        amp_enabled=False,
    ):
        super().__init__()
        init_method = init_method_normal(initializer_range)
        scaled_init_method = scaled_init_method_normal(initializer_range, hidden_layers)

        # Embeddings
        self.embeddings = RobertaEmbeddings(
            vocab_size,
            hidden_size,
            max_position_embeddings,
            hidden_dropout_prob,
            type_vocab_size,
            pad_token_id,
            init_method,
            amp_enabled
        )

        # Mask generation
        self.extended_attn_mask = RobertaExtendedAttnMask()

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
                    layer_idx=i,
                )
                for i in range(hidden_layers)
            ]
        )
        self.final_layernorm = LayerNorm((hidden_size,), eps=layernorm_eps, layer_idx=-1)

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
            "type_vocab_size": cfg.type_vocab_size,
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

    def forward(self, input_ids, attention_mask, token_type_ids=None):
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


class RobertaLMHead(nn.Module):
    def __init__(self, vocab_size, hidden_size, init_method, layer_norm_eps):
        super().__init__()
        self.dense = Linear(
            hidden_size,
            hidden_size,
            bias=True,
            parallel='col',
            init_method=init_method,
            layer_idx=-1
        )
        self.activation_func = build_activation("gelu")
        self.layer_norm = LayerNorm((hidden_size,), eps=layer_norm_eps, layer_idx=-1)
        
        # NOTE(xzp): LMLogits as a decoder:nn.Linear(hidden_size, vocab_size), 
        # it shares the roberta.word_embeddings.weight
        self.lm_logits = LMLogits(vocab_size, bias=True)

    def forward(self, hidden_states, word_embeddings_weight):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.activation_func(hidden_states)
        hidden_states = hidden_states.to_global(
            sbp=dist.get_nd_sbp([flow.sbp.split(0), flow.sbp.broadcast])
        )
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.lm_logits(hidden_states, word_embeddings_weight)
        return hidden_states   


class RobertaForMaskedLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        cfg.add_pooling_layer = False
        self.roberta = RobertaModel(cfg)
        self.lm_head = RobertaLMHead(
            cfg.vocab_size,
            cfg.hidden_size,
            init_method_normal(cfg.initializer_range),
            cfg.layer_norm_eps
        )
        self.loss_fc = ParallelCrossEntropyLoss()
    
    def forward(
        self, 
        input_ids, 
        attention_mask, 
        token_type_ids=None, 
        position_ids=None, 
        labels=None
    ):
        outputs = self.roberta(input_ids, attention_mask, position_ids)
        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output, self.roberta.word_embeddings_weight())     # [S(0), S(0)]
        
        if labels is not None:
            masked_lm_loss = self.loss_fc(prediction_scores, labels).mean() 
            masked_lm_loss = masked_lm_loss.to_global(
                sbp=dist.get_nd_sbp([flow.sbp.partial_sum, flow.sbp.broadcast])
            )
            return {'lm_loss': masked_lm_loss}
        
        return {'prediction_scores': prediction_scores}


class RobertaForCausalLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        cfg.add_pooling_layer = False
        self.roberta = RobertaModel(cfg)
        self.lm_head = RobertaLMHead(
            cfg.vocab_size,
            cfg.hidden_size,
            init_method_normal(cfg.initializer_range),
            cfg.layer_norm_eps
        )
        self.loss_fc = ParallelCrossEntropyLoss()
    
    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids=None,
        position_ids=None,
        labels=None
    ):
        outputs = self.roberta(input_ids, attention_mask, position_ids)
        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output, self.roberta.word_embeddings_weight())

        if labels is not None:
            # next-token prediction task, shift prediction_scores and labels by one.
            shifted_prediction_scores = prediction_scores[:, :-1, :].contiguous()
            shifted_prediction_scores = shifted_prediction_scores.to_global(
                sbp=prediction_scores.sbp
            )
            shifted_labels = labels[:, 1:].contiguous()
            shifted_labels = shifted_labels.to_global(
                sbp=shifted_labels
            )
            lm_loss = self.loss_fc(shifted_prediction_scores, shifted_labels).mean()
            lm_loss = lm_loss.to_global(
                sbp=dist.get_nd_sbp([flow.sbp.partial_sum, flow.sbp.broadcast])
            )
            return {'lm_loss': lm_loss}
        
        return {'prediction_scores': prediction_scores}
