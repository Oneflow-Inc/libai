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

import oneflow as flow

logger = logging.getLogger(__name__)


def WarmupStepLR(
    optimizer: flow.optim.Optimizer,
    max_iter: int,
    warmup_factor: float,
    warmup_iter: int,
    step_size: int,
    gamma: float = 0.1,
    warmup_method: str = "linear",
):
    """Create a schedule with a learning rate that decreases following the values of the Step
    function between the initial lr set in the optimizer to 0, after a warmup period during which
    it increases linearly between 0 and the initial lr set in the optimizer.

    Args:
        optimizer (flow.optim.Optimizer): Wrapped optimizer.
        max_iter (int): Total training iters.
        warmup_factor (float): The warmup factor.
        warmup_iter (int): The number of warmup steps.
        step_size (int): Period of learning rate decay.
        gamma (float, optional): Multiplicative factor of learning rate decay. Defaults to 0.1.
        warmup_method (str, optional): The method of warmup, you can choose "linear" or "constant".
            In linear mode, the multiplication factor starts with warmup_factor in the first
            epoch and then inreases linearly to reach 1. Defaults to "linear".
    """
    step_lr = flow.optim.lr_scheduler.StepLR(
        optimizer, step_size=step_size, gamma=gamma
    )
    if warmup_iter == 0:
        logger.warning("warmup iters equals to zero, return StepLR")
        return step_lr
    warmup_step_lr = flow.optim.lr_scheduler.WarmUpLR(
        step_lr,
        warmup_factor=warmup_factor,
        warmup_iters=warmup_iter,
        warmup_method=warmup_method,
    )
    return warmup_step_lr