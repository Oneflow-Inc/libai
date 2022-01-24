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

from libai.config import instantiate
from libai.utils.registry import Registry

MODEL_ARCH_REGISTRY = Registry("model_arch")
MODEL_ARCH_REGISTRY.__doc__ = """
Registry for modeling, i.e. Bert or GPT model.

The registered object will be called with `obj(cfg)`
and expected to return a `nn.Module` object.
"""

GRAPH_REGISTRY = Registry("graph")
GRAPH_REGISTRY.__doc__ = """
Registry for Graph training mode.

The registered object will be called with `obj(cfg)`
and expected to return a `nn.Graph` object.
"""


def build_model(cfg):
    """Build the whole model architecture, defined by ``cfg.model.model_name``.
    Note that is does not load any weights from ``cfg``.
    """
    if "_target_" in cfg:  # LazyCall
        model = instantiate(cfg)
    else:
        model_name = cfg.model_name
        model = MODEL_ARCH_REGISTRY.get(model_name)(cfg.model_cfg)
    return model


def build_graph(cfg, model, optimizer=None, lr_scheduler=None, is_train=False):
    """Build the `nn.Graph`, defined by ``cfg.graph``."""
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
