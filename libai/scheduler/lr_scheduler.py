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

from .build import SCHEDULER_REGISTRY


@SCHEDULER_REGISTRY.register()
def WarmupCosineLR(cfg, optimizer):
    cosine_decay_lr = flow.optim.lr_scheduler.CosineDecayLR(
        optimizer, cfg.decay_steps, cfg.alpha
    )
    warmup_cosine_lr = flow.optim.lr_scheduler.WarmUpLR(
        cosine_decay_lr, cfg.warmup_factor, cfg.warmup_iters, cfg.warmup_method
    )
    return warmup_cosine_lr


@SCHEDULER_REGISTRY.register()
def WarmupMultiStepLR(cfg, optimizer):
    multistep_lr = flow.optim.lr_scheduler.MultiStepLR(
        optimizer, cfg.milestones, cfg.gamma
    )
    warmup_multistep_lr = flow.optim.lr_scheduler.WarmUpLR(
        multistep_lr, cfg.warmup_factor, cfg.warmup_iters, cfg.warmup_method
    )
    return warmup_multistep_lr