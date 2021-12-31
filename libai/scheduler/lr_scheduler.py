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

import math

import oneflow as flow
from oneflow.nn.optimizer.lr_scheduler import LrScheduler

from .build import SCHEDULER_REGISTRY


@SCHEDULER_REGISTRY.register()
def WarmupCosineLR(optimizer: flow.optim.Optimizer,
                   max_iters: int,
                   warmup_factor: float,
                   warmup_iters: int,
                   alpha: float = 0.0,
                   warmup_method: str = "linear",
                   **kwargs):
    cosine_decay_lr = flow.optim.lr_scheduler.CosineDecayLR(
        optimizer, decay_steps=max_iters, alpha=alpha
    )
    warmup_cosine_lr = flow.optim.lr_scheduler.WarmUpLR(
        cosine_decay_lr, warmup_factor=warmup_factor, warmup_iters=warmup_iters, warmup_method=warmup_method, **kwargs
    )
    return warmup_cosine_lr


@SCHEDULER_REGISTRY.register()
def WarmupMultiStepLR(optimizer: flow.optim.Optimizer,
                      warmup_factor: float,
                      warmup_iters: int,
                      milestones: list,
                      gamma: float = 0.1,
                      warmup_method: str = "linear",
                      **kwargs):
    multistep_lr = flow.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=gamma
    )
    warmup_multistep_lr = flow.optim.lr_scheduler.WarmUpLR(
        multistep_lr, warmup_factor=warmup_factor, warmup_iters=warmup_iters, warmup_method=warmup_method, **kwargs
    )
    return warmup_multistep_lr

