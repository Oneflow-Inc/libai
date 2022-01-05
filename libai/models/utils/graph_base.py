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

from libai.layers import TransformerLayer

from libai.config import instantiate
from libai.utils.registry import Registry

GRAPH_REGISTRY = Registry("graph")
GRAPH_REGISTRY.__doc__ = """
Registry for Graph training mode.

The registered object will be called with `obj(cfg)` 
and expected to return a `nn.Graph` object.
"""


def build_graph(cfg, model, optimizer=None, lr_scheduler=None, is_train=False):
    """ Build the `nn.Graph`, defined by ``cfg.graph``.
    """
    if is_train:
        # Set train graph
        assert optimizer is not None, "optimizer must be set for train graph"
        assert lr_scheduler is not None, "lr_scheduler must be set for train graph"
        if "_target_" in cfg.train_graph:  # LazyCall
            cfg.train_graph.model = model
            cfg.train_graph.optimizer = optimizer
            cfg.train_graph.lr_scheduler = lr_scheduler
            return instantiate(cfg.train_graph)
        else:
            graph_name = cfg.train_graph.graph_name
            graph_cfg = cfg.train_graph.graph_cfg
            train_graph = GRAPH_REGISTRY.get(graph_name)(
                model, optimizer, lr_scheduler, **graph_cfg
            )
            return train_graph
    else:
        # Set eval graph
        if "_target_" in cfg.eval_graph:
            cfg.eval_graph.model = model
            return instantiate(cfg.eval_graph)
        else:
            graph_name = cfg.eval_graph.graph_name
            graph_cfg = cfg.eval_graph.graph_cfg
            eval_graph = GRAPH_REGISTRY.get(graph_name)(model, **graph_cfg)
            return eval_graph


class GraphBase(nn.Graph):
    def __init__(
        self,
        model: nn.Module,
        optimizer: flow.optim.Optimizer = None,
        lr_scheduler: flow.optim.lr_scheduler = None,
        fp16=False,
        is_train=True,
    ):
        super().__init__()

        self.model = model
        self.is_train = is_train

        if is_train:
            self.add_optimizer(optimizer, lr_sch=lr_scheduler)
            self.set_activation_checkpoint()
            self.set_pipeline_stage_id()
            if fp16:
                self.config.enable_amp(True)
                grad_scaler = flow.amp.GradScaler(
                    init_scale=2 ** 30,
                    growth_factor=2.0,
                    backoff_factor=0.5,
                    growth_interval=2000,
                )
                self.set_grad_scaler(grad_scaler)

        self.config.allow_fuse_add_to_output(True)
        self.config.allow_fuse_model_update_ops(True)
        self.config.allow_fuse_cast_scale(True)

    def set_activation_checkpoint(self):
        for module_block in self.model.modules():
            if isinstance(module_block.origin, TransformerLayer):
                module_block.config.activation_checkpointing = True

    def set_pipeline_stage_id(self):
        pass
