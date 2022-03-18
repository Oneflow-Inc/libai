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

import logging

import oneflow as flow
from oneflow import nn

from libai.layers import Linear
from libai.models.bert_model import BertModel
from libai.models.utils import init_method_normal
from libai.utils import distributed as dist

logger = logging.getLogger("libai." + __name__)


class ClassificationLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, classification_logits, label):
        loss = nn.CrossEntropyLoss()(classification_logits, label)
        # NOTE: Change loss sbp sign [P, P] -> [P, B] to add with sop loss
        # whose sbp sign: [P, B]
        loss = loss.to_global(sbp=dist.get_nd_sbp([flow.sbp.partial_sum, flow.sbp.broadcast]))
        return loss


class Classification(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.num_classes = cfg.num_classes
        self.language_model = BertModel(cfg)
        if cfg.pretrain_megatron_weight is not None:
            from .load_megatron_weight import load_megatron_bert

            logger.info(f"loading pretraining: {cfg.pretrain_megatron_weight}")
            load_megatron_bert(self.language_model, cfg.pretrain_megatron_weight)
            logger.info("load succeed")

        init_method = init_method_normal(cfg.initializer_range)
        self.classification_dropout = nn.Dropout(cfg.hidden_dropout_prob)
        self.classification_head = Linear(
            cfg.hidden_size,
            self.num_classes,
            bias=True,
            parallel="row",
            init_method=init_method,
            layer_idx=-1,
        )
        self.loss_func = ClassificationLoss()

    def forward(self, model_input, attention_mask, tokentype_ids=None, labels=None):

        encoder_output, pooled_output = self.language_model(
            model_input, attention_mask, tokentype_ids
        )
        classification_output = self.classification_dropout(pooled_output)
        classification_logits = self.classification_head(classification_output)

        # reshape
        classification_logits = classification_logits.view(-1, self.num_classes)

        if self.training and labels is not None:
            loss = self.loss_func(classification_logits, labels)
            return {"total_loss": loss}

        return {"prediction_scores": classification_logits}
