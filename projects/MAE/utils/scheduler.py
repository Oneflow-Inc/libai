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

import logging
import math

import oneflow as flow
from oneflow.optim.lr_scheduler import _LRScheduler

logger = logging.getLogger(__name__)


class LayerScaleWarmupCosineDecayLR(_LRScheduler):
    def __init__(
        self,
        optimizer: flow.optim.Optimizer,
        steps: int,
        warmup_steps: int,
        warmup_factor: float,
        min_lr: float = 0.0,
        last_step: int = -1,
        verbose: bool = False,
    ):
        self.total_steps = steps
        self.decay_steps = steps - warmup_steps
        self.warmup_steps = warmup_steps
        self.warmup_factor = warmup_factor
        self.min_lr = min_lr
        super().__init__(optimizer, last_step, verbose)

    def get_lr(self, base_lr, step):
        if step < self.warmup_steps:
            progress = step / self.warmup_steps
            lr = base_lr * progress
        elif step < self.total_steps:
            progress = (step - self.warmup_steps) / self.decay_steps
            lr = self.min_lr + (base_lr - self.min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))
        else:
            lr = self.min_lr

        return lr

    def update_lrs(self, lrs):
        self._last_lr = []
        for i, (group, lr) in enumerate(zip(self.optimizer.param_groups, lrs)):
            if "lr_scale" in group:
                group["lr"] = lr * group["lr_scale"]
            else:
                group["lr"] = lr

            self._last_lr.append(lr)
            if self.verbose:
                self.print_lr(i, lr)


def warmup_layerscale_cosine_lr_scheduler(
    optimizer: flow.optim.Optimizer,
    max_iter: int,
    warmup_iter: int,
    warmup_factor: float,
    min_lr: float = 0.0,
):
    return LayerScaleWarmupCosineDecayLR(
        optimizer,
        steps=max_iter,
        warmup_steps=warmup_iter,
        warmup_factor=warmup_factor,
        min_lr=min_lr,
    )


def warmup_cosine_lr_scheduler(
    optimizer: flow.optim.Optimizer,
    max_iter: int,
    warmup_iter: int,
    warmup_factor: float = 0.0,
    warmup_method: str = "linear",
    min_lr: float = 0.0,
):
    cosine_lr = flow.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_iter - warmup_iter, eta_min=min_lr
    )
    if warmup_iter == 0:
        logger.warning("warmup iters equals to zero, return CosineLR")
        return cosine_lr

    if warmup_iter > max_iter:
        logger.warning("warmup iters is larger than the total training iters")

    warmup_cosine_lr = flow.optim.lr_scheduler.WarmupLR(
        cosine_lr,
        warmup_factor=warmup_factor,
        warmup_iters=warmup_iter,
        warmup_method=warmup_method,
    )
    return warmup_cosine_lr
