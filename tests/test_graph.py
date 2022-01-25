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

from omegaconf import DictConfig

from configs.common.models.bert import pretrain_model as model_cfg
from configs.common.optim import optim as optim_cfg
from configs.common.optim import scheduler
from libai.config import LazyCall
from libai.models import build_graph, build_model
from libai.models.bert_model import BertForPretrainingGraph
from libai.optim import build_optimizer
from libai.scheduler import build_lr_scheduler

model = build_model(model_cfg)

optimizer = build_optimizer(optim_cfg, model)

lr_scheduler = build_lr_scheduler(scheduler, optimizer)

# Lazy config
lazy_graph_cfg = DictConfig(
    dict(
        # options for graph or eager mode
        enabled=True,
        debug=0,  # debug mode for graph
        train_graph=LazyCall(BertForPretrainingGraph)(fp16=False, is_train=True,),
        eval_graph=LazyCall(BertForPretrainingGraph)(fp16=False, is_train=False),
    )
)

# Register config
reg_graph_cfg = DictConfig(
    dict(
        enabled=True,
        debug=0,
        train_graph=dict(
            graph_name="BertForPretrainingGraph", graph_cfg=dict(fp16=False, is_train=True,),
        ),
        eval_graph=dict(
            graph_name="BertForPretrainingGraph", graph_cfg=dict(fp16=False, is_train=False),
        ),
    )
)

lazy_train_graph = build_graph(lazy_graph_cfg, model, optimizer, lr_scheduler, is_train=True)
lazy_eval_graph = build_graph(lazy_graph_cfg, model, is_train=False)

reg_train_graph = build_graph(reg_graph_cfg, model, optimizer, lr_scheduler, is_train=True)
reg_eval_graph = build_graph(reg_graph_cfg, model, is_train=False)
