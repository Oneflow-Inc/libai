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
from libai.tokenizer import BertTokenizer
from .bert import BertForSimCSE
from .load_megatron_weight import load_megatron_bert
import libai
import random
from tqdm import tqdm
from scipy.stats import spearmanr
from oneflow.utils.data import DataLoader, Dataset
from libai.utils import distributed as dist
from libai.data.structures import DistTensorData, Instance
import csv


class CosineSimilarity(nn.Module):
    def __init__(self, dim, temp):
        super().__init__()
        self.dim = dim
        self.temp = temp

    def forward(self, x, y):
        return flow.sum(x*y, dim=self.dim) / (flow.linalg.norm(x, dim=self.dim) * flow.linalg.norm(y, dim=self.dim)) / self.temp


class MLPLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.dense = libai.layers.Linear(
            cfg.hidden_size,
            cfg.hidden_size,
            bias=True,
            parallel='row',
            layer_idx=-1
        )
        self.activation = libai.layers.build_activation('tanh')
    
    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.activation(x)
        return x


class SimCSE_Unsup_Loss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.temp = cfg.temp

    def forward(self, y_pred):
        y_pred = y_pred.view(-1, 2, y_pred.size(-1))    # [batch, num_sent, hidden]
        z1, z2 = y_pred[:, 0], y_pred[:, 1]             
        sim = CosineSimilarity(dim=-1, temp=self.temp)
        cos_sim = sim(z1.unsqueeze(1), z2.unsqueeze(0))     # [batch, 1, hidden], [1, batch, hidden]
        labels = flow.arange(cos_sim.size(0), sbp=y_pred.sbp, placement=y_pred.placement).long()
        
        loss = nn.CrossEntropyLoss()(cos_sim, labels)
        loss = loss.to_global(sbp=dist.get_nd_sbp([flow.sbp.partial_sum, flow.sbp.broadcast]))
        return loss


class SimCSE_Eval(nn.Module):
    def forward(self, poolerd_result):
        poolerd_result = poolerd_result.view(-1, 2, poolerd_result.size(-1))    # [batch, num_sent, hidden]
        z1, z2 = poolerd_result[:, 0], poolerd_result[:, 1]             # [batch, hidden]
        sim = CosineSimilarity(dim=-1, temp=1)
        cos_sim = sim(z1, z2)
        return cos_sim


class SimcseModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.bert = BertForSimCSE(cfg)
        if cfg.pretrained_model_weight is not None:
            load_megatron_bert(self.bert, cfg.pretrained_model_weight)

        self.pooler_type = cfg.pooler_type
        self.mlp = MLPLayer(cfg)
        self.loss_func = SimCSE_Unsup_Loss(cfg)
        self.eval_func = SimCSE_Eval()
        assert self.pooler_type in ["cls", "cls_before_pooler", "avg", "avg_top2", "avg_first_last"], "unrecognized pooling type %s" % self.pooler_type

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        # input_ids:[2*batch_size, seq_len]
        input_ids = input_ids.view(-1, input_ids.size(-1))
        attention_mask = attention_mask.view(-1 , attention_mask.size(-1))
        token_type_ids = token_type_ids.view(-1 , token_type_ids.size(-1))

        outputs = self.bert(input_ids, attention_mask, token_type_ids)
        last_hidden = outputs[0]    # batch*2, seq_len, hidden
        pooler_output = outputs[1]  # batch*2, hidden
        hidden_states = outputs[2]
        
        if self.pooler_type in ["cls_before_pooler", "cls"]:
            if self.pooler_type == "cls":
                poolerd_result = self.mlp(last_hidden[:, 0])
            else:
                poolerd_result = last_hidden[:, 0]
        
        elif self.pooler_type == "avg":
            poolerd_result = ((last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1))
        
        elif self.pooler_type == "avg_first_last":
            first_hidden = hidden_states[0]
            last_hidden = hidden_states[-1]
            poolerd_result = ((first_hidden + last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
        
        # else self.pooler_type == "avg_top2"
        else:
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            poolerd_result = ((last_hidden + second_last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)

        if self.training:
            loss = self.loss_func(poolerd_result)
            return {"loss": loss}

        cos_sim = self.eval_func(poolerd_result)
        return {"cos_sim": cos_sim, "labels": labels}