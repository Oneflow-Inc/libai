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

import csv
import random

import numpy as np
import oneflow as flow
from oneflow import nn
from oneflow.utils.data import DataLoader, Dataset
from scipy.stats import spearmanr
from tqdm import tqdm

import libai
from libai.data.structures import DistTensorData, Instance
from libai.layers import ParallelCrossEntropyLoss
from libai.tokenizer import BertTokenizer
from libai.utils import distributed as dist

from .bert_for_simcse import BertForSimCSE
from .load_megatron_weight import load_megatron_bert


class CosineSimilarity(nn.Module):
    # Calculate cosine similarity
    def __init__(self, dim, temp):
        super().__init__()
        self.dim = dim
        self.temp = temp

    def forward(self, x, y):
        return (
            flow.sum(x * y, dim=self.dim)
            / (flow.linalg.norm(x, dim=self.dim) * flow.linalg.norm(y, dim=self.dim))
            / self.temp
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


class SimCSE_Unsup_Loss(nn.Module):
    # Forward of model's training process, for wiki_datasets.
    def __init__(self, cfg):
        super().__init__()
        self.sim = CosineSimilarity(dim=-1, temp=cfg.temp)
        self.loss_fc = nn.CrossEntropyLoss()

    def forward(self, pooled_result):
        # pooled_result: [batch*2, hidden]
        labels = np.arange(pooled_result.size(0))
        labels = (labels - labels % 2 * 2) + 1
        labels = flow.tensor(
            labels, sbp=pooled_result.sbp, placement=pooled_result.placement, dtype=flow.long
        )

        sim = self.sim(pooled_result.unsqueeze(1), pooled_result.unsqueeze(0))

        eye = (
            flow.eye(
                pooled_result.shape[0],
                sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
                placement=pooled_result.placement,
            )
            * 1e12
        )

        sim = sim - eye
        loss = self.loss_fc(sim, labels)
        loss = loss.to_global(sbp=dist.get_nd_sbp([flow.sbp.partial_sum, flow.sbp.broadcast]))
        return loss


class SimCSE_Eval(nn.Module):
    # Forward of model's verification process, for sts_datasets.
    def __init__(self, cfg):
        super().__init__()
        self.sim = CosineSimilarity(dim=-1, temp=cfg.temp)

    def forward(self, pooled_result):
        # [batch*2, hidden] -> [batch, 2, hidden]
        pooled_result = pooled_result.view(-1, 2, pooled_result.size(-1))
        # sent1, sent2, [batch, hidden]
        z1 = pooled_result[:, 0]
        z2 = pooled_result[:, 1]
        # [batch]
        cos_sim = self.sim(z1, z2)
        cos_sim = cos_sim.to_global(sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]))
        return cos_sim


class SimcseModel(nn.Module):
    # simcse model
    def __init__(self, cfg):
        super().__init__()
        self.bert = BertForSimCSE(cfg)
        if cfg.pretrained_model_weight is not None:
            load_megatron_bert(self.bert, cfg.pretrained_model_weight)
        self.pooler_type = cfg.pooler_type
        self.mlp = MLPLayer(cfg)
        self.train_forward = SimCSE_Unsup_Loss(cfg)
        self.eval_forward = SimCSE_Eval(cfg)
        assert self.pooler_type in [
            "cls",
            "cls_before_pooler",
            "avg",
            "avg_top2",
            "avg_first_last",
        ], (
            "unrecognized pooling type %s" % self.pooler_type
        )

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        # [batch, 2, seq_len] -> [batch_size * 2, seq_len]
        input_ids = input_ids.view(-1, input_ids.size(-1))
        attention_mask = attention_mask.view(-1, attention_mask.size(-1))

        outputs = self.bert(input_ids, attention_mask, token_type_ids)
        last_hidden = outputs[0]  # [batch*2, seq_len, hidden]
        pooled_output = outputs[1]  # [batch*2, hidden]
        hidden_states = outputs[2]  # list

        # Select pooling mode
        if self.pooler_type in ["cls_before_pooler", "cls"]:
            if self.pooler_type == "cls":
                poolerd_result = self.mlp(last_hidden[:, 0])
            else:
                poolerd_result = last_hidden[:, 0]
        elif self.pooler_type == "avg":
            poolerd_result = (last_hidden * attention_mask.unsqueeze(-1)).sum(
                1
            ) / attention_mask.sum(-1).unsqueeze(-1)
        elif self.pooler_type == "avg_first_last":
            first_hidden = hidden_states[0]
            last_hidden = hidden_states[-1]
            poolerd_result = (
                (first_hidden + last_hidden) / 2.0 * attention_mask.unsqueeze(-1)
            ).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
        else:
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            poolerd_result = (
                (last_hidden + second_last_hidden) / 2.0 * attention_mask.unsqueeze(-1)
            ).sum(1) / attention_mask.sum(-1).unsqueeze(-1)

        if self.training:
            loss = self.train_forward(poolerd_result)
            return {"loss": loss}
        else:
            cos_sim = self.eval_forward(poolerd_result)
            return {"cos_sim": cos_sim, "labels": labels}
