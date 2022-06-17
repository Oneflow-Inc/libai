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

from libai.config import instantiate, try_get_key


def build_model(cfg):
    """Build the whole model architecture, defined by ``cfg.model``.
    Note that it does not load any weights from ``cfg``.
    """
    model = instantiate(cfg)
    return model


def build_graph(cfg, model, optimizer=None, lr_scheduler=None, is_train=False):
    """Build the `nn.Graph`, defined by ``cfg.graph``."""
    auto_parallel_conf = try_get_key(cfg, "graph.auto_parallel", default=None)
    if is_train:
        # Set train graph
        assert optimizer is not None, "optimizer must be set for train graph"
        assert lr_scheduler is not None, "lr_scheduler must be set for train graph"
        graph = cfg.graph.train_graph
        graph.model = model
        graph.optimizer = optimizer
        graph.lr_scheduler = lr_scheduler
        graph.fp16 = try_get_key(cfg, "train.amp.enabled", default=False)
        graph.activation_checkpoint = try_get_key(
            cfg, "train.activation_checkpoint.enabled", default=False
        )
        graph.zero_optim = try_get_key(cfg, "train.zero_optimization.enabled", default=False)
        graph.zero_stage = try_get_key(cfg, "train.zero_optimization.stage", default=1)
        graph.grad_acc_steps = try_get_key(cfg, "train.num_accumulation_steps", default=1)
        graph.auto_parallel_conf = auto_parallel_conf
        return instantiate(graph)
    else:
        # Set eval graph
        graph = cfg.graph.eval_graph
        graph.model = model
        graph.auto_parallel_conf = auto_parallel_conf
        return instantiate(graph)
