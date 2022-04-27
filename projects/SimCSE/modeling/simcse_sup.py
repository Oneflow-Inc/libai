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

from libai.utils import distributed as dist
from projects.SimCSE.modeling.model_utils import MLPLayer, cosine_similarity
from projects.SimCSE.utils.load_huggingface_weight import load_huggingface_bert

from .bert_for_simcse import BertForSimCSE


class Simcse_sup(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.bert = BertForSimCSE(cfg)
        self.mlp = MLPLayer(cfg)
        self.pooler_type = cfg.pooler_type

        if cfg.pretrained_model_weight is not None:
            load_huggingface_bert(
                self.bert,
                cfg.pretrained_model_weight,
                cfg["hidden_size"],
                cfg["num_attention_heads"],
                cfg["hidden_layers"],
            )

    def pooler(self, inputs, attention_mask):
        if self.pooler_type == "cls":
            return inputs[0][:, 0]

        elif self.pooler_type == "pooled":
            return inputs[1]

        elif self.pooler_type == "last-avg":
            last_hidden = inputs[0]
            return (last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(
                -1
            ).unsqueeze(-1)

        elif self.pooler_type == "first-last-avg":
            first_hidden = inputs[2][1]
            last_hidden = inputs[0]
            res = ((first_hidden + last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(
                1
            ) / attention_mask.sum(-1).unsqueeze(-1)
            return res

    def create_use_row(self, labels):
        count = 0
        use_row = []
        for row in range(labels.size(0)):
            if count % 2 == 0 and count != 0:
                count = 0
                continue
            use_row.append(row)
            count += 1
        return flow.tensor(use_row, sbp=labels.sbp, placement=labels.placement)

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        if self.training:
            bs = input_ids.size(0)
            input_ids = input_ids.view(bs * 3, -1)
            attention_mask = attention_mask.view(bs * 3, -1)
            out = self.bert(input_ids, attention_mask)
            out = self.pooler(out, attention_mask)
            out = self.mlp(out)
            labels = flow.arange(
                out.size(0),
                sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
                placement=out.placement,
            )
            use_row = self.create_use_row(labels)
            labels = (use_row - use_row % 3 * 2) + 1
            sim = cosine_similarity(out.unsqueeze(1), out.unsqueeze(0))
            sim = (
                sim
                - flow.eye(
                    out.size(0),
                    sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
                    placement=out.placement,
                )
                * 1e12
            )
            sim = flow.index_select(sim, dim=0, index=use_row)
            sim = sim / 0.05
            loss = nn.CrossEntropyLoss()(sim, labels)
            return {"loss": loss}
        else:
            bs = input_ids.size(0)
            input_ids = input_ids.view(bs * 2, -1)
            attention_mask = attention_mask.view(bs * 2, -1)
            out = self.bert(input_ids, attention_mask)
            out = self.pooler(out, attention_mask)
            self.mlp(out)
            out = out.view(bs, 2, -1)
            sent1 = out[:, 0]
            sent2 = out[:, 1]
            sim = cosine_similarity(sent1, sent2)
            sim = sim.to_global(sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]))
            return {"sim": sim.unsqueeze(1), "labels": labels}
