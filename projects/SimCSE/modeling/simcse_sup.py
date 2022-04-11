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

import numpy as np
import oneflow as flow
from oneflow import nn
import libai
from libai.utils import distributed as dist
from .bert_for_simcse import BertForSimCSE
from .load_huggingface_weight import load_huggingface_bert


def cosine_similarity(x, y, dim=-1):
    return (
        flow.sum(x * y, dim=dim)
        / (flow.linalg.norm(x, dim=dim) * flow.linalg.norm(y, dim=dim))
    )


class MLPLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.dense = libai.layers.Linear(
            cfg.hidden_size, cfg.hidden_size, bias=True, parallel="row", layer_idx=-1
        )
        self.activation = libai.layers.build_activation("tanh")

    def forward(self, features):
        x = self.dense(features)
        x = self.activation(x)
        return x


class Simcse_sup(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.bert = BertForSimCSE(cfg)
        self.mlp = MLPLayer(cfg)
        self.pooler_type = cfg.pooler_type

        if cfg.pretrained_model_weight is not None:
            load_huggingface_bert(self.bert, cfg.pretrained_model_weight, cfg['hidden_size'], cfg['num_attention_heads'], cfg['hidden_layers'])

    def pooler(self, inputs, attention_mask):
        if self.pooler_type == 'cls':
            return inputs[0][:, 0]
        
        elif self.pooler_type == 'pooled':
            return inputs[1]
        
        elif self.pooler_type == 'last-avg':
            last_hidden = inputs[0]
            return (last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
        
        elif self.pooler_type == 'first-last-avg':
            first_hidden = inputs[2][1]
            last_hidden = inputs[0]
            res = ((first_hidden + last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return res            

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        if self.training:
            bs = input_ids.size(0)
            input_ids = input_ids.view(bs*3, -1)
            attention_mask = attention_mask.view(bs*3, -1)
            out = self.bert(input_ids, attention_mask)
            out = self.pooler(out, attention_mask)
            out = self.mlp(out)
            
            labels = np.arange(out.size(0))
            use_row = np.where((labels + 1) % 3 != 0)[0]
            use_row = flow.tensor(use_row, sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]), placement=out.placement)
            labels = (use_row - use_row % 3 * 2) + 1
            labels = flow.tensor(labels, dtype=flow.long, sbp=out.sbp, placement=out.placement)
            
            sim = cosine_similarity(out.unsqueeze(1), out.unsqueeze(0))
            sim = sim - flow.eye(out.size(0), sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]), placement=out.placement) * 1e12
            
            sim = flow.index_select(sim, 0, use_row)
            sim = sim / 0.05
            
            loss = nn.CrossEntropyLoss()(sim, labels)
            loss = loss.to_global(sbp=dist.get_nd_sbp([flow.sbp.partial_sum, flow.sbp.broadcast]))
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
            return {"sim": sim, "labels":labels}