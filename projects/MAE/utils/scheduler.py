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
from oneflow.optim.lr_scheduler import CosineDecayLR

logger = logging.getLogger(__name__)


class WarmupLayerScaleCosineDecayLR(CosineDecayLR):
    def __init__(
        self,
        optimizer: flow.optim.Optimizer,
        warmup_steps: int,
        decay_steps: int,
        alpha: float = 0.0,
        last_step: int = -1,
        verbose: bool = False,
    ):
        self.warmup_steps = warmup_steps
        super().__init__(
            optimizer=optimizer,
            decay_steps=decay_steps,
            alpha=alpha,
            last_step=last_step,
            verbose=verbose,
        )

    def get_lr(self, base_lr, step):
        if step < self.warmup_steps:
            lr = base_lr * step / self.warmup_steps
            return lr
        elif step < self.decay_steps:
            cos_decay = 0.5 * (1 + math.cos(math.pi * step / self.decay_steps))
            decay_factor = (1 - self.alpha) * cos_decay + self.alpha
        else:
            decay_factor = self.alpha

        return base_lr * decay_factor

    def update_lrs(self, lrs):
        self._last_lr = []
        for i, (group, lr) in enumerate(zip(self.optimizer.param_groups, lrs)):
            if "lr_scale" in group.options:
                group.options["lr"] = lr * group.options["lr_scale"]
            else:
                group.options["lr"] = lr
            self._last_lr.append(lr)
            if self.verbose:
                self.print_lr(i, lr)


def warmup_layerscale_cosine_lr_scheduler(
    optimizer: flow.optim.Optimizer,
    max_iter: int,
    warmup_factor: float,
    warmup_iter: int,
    alpha: float = 0.0,
    warmup_method: str = "linear",
):
    layer_scale_cosine_decay_lr = WarmupLayerScaleCosineDecayLR(
        optimizer, warmup_steps=warmup_iter, decay_steps=max_iter, alpha=alpha
    )
    return layer_scale_cosine_decay_lr
