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

from .build import SCHEDULER_REGISTRY

logger = logging.getLogger(__name__)


@SCHEDULER_REGISTRY.register()
def WarmupCosineLR(
    optimizer: flow.optim.Optimizer,
    max_iters: int,
    warmup_factor: float,
    warmup_iters: int,
    alpha: float = 0.0,
    warmup_method: str = "linear",
):
    cosine_decay_lr = flow.optim.lr_scheduler.CosineDecayLR(
        optimizer, decay_steps=max_iters, alpha=alpha
    )
    if warmup_iters == 0:
        logger.warning("warmup iters equals to zero, return CosineLR")
        return cosine_decay_lr
    elif warmup_iters > max_iters:
        logger.warning("warmup iters is larger than the total training iters")
    warmup_cosine_lr = flow.optim.lr_scheduler.WarmUpLR(
        cosine_decay_lr,
        warmup_factor=warmup_factor,
        warmup_iters=warmup_iters,
        warmup_method=warmup_method,
    )
    return warmup_cosine_lr


@SCHEDULER_REGISTRY.register()
def WarmupCosineAnnealingLR(
    optimizer: flow.optim.Optimizer,
    max_iters: int,
    warmup_factor: float,
    warmup_iters: int,
    eta_min: float = 0.0,
    warmup_method: str = "linear",
):
    cosine_annealing_lr = flow.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_iters, eta_min=eta_min
    )
    if warmup_iters == 0:
        logger.warning("warmup iters equals to zero, return CosineAnnealingLR")
        return cosine_annealing_lr
    warmup_cosine_annealing_lr = flow.optim.lr_scheduler.WarmUpLR(
        cosine_annealing_lr,
        warmup_factor=warmup_factor,
        warmup_iters=warmup_iters,
        warmup_method=warmup_method,
    )
    return warmup_cosine_annealing_lr


@SCHEDULER_REGISTRY.register()
def WarmupMultiStepLR(
    optimizer: flow.optim.Optimizer,
    warmup_factor: float,
    warmup_iters: int,
    milestones: list,
    gamma: float = 0.1,
    warmup_method: str = "linear",
):
    multistep_lr = flow.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=gamma
    )
    if warmup_iters == 0:
        logger.warning("warmup iters equals to zero, return MultiStepLR")
        return multistep_lr
    warmup_multistep_lr = flow.optim.lr_scheduler.WarmUpLR(
        multistep_lr,
        warmup_factor=warmup_factor,
        warmup_iters=warmup_iters,
        warmup_method=warmup_method,
    )
    return warmup_multistep_lr


@SCHEDULER_REGISTRY.register()
def WarmupFixedStepLR(
    optimizer: flow.optim.Optimizer,
    warmup_factor: float,
    warmup_iters: int,
    step_size: int,
    gamma: float = 0.1,
    warmup_method: str = "linear",
):
    fixedstep_lr = flow.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    if warmup_iters == 0:
        logger.warning("warmup iters equals to zero, return FixedStepLR")
        return fixedstep_lr
    warmup_fixedstep_lr = flow.optim.lr_scheduler.WarmUpLR(
        fixedstep_lr,
        warmup_factor=warmup_factor,
        warmup_iters=warmup_iters,
        warmup_method=warmup_method,
    )
    return warmup_fixedstep_lr


@SCHEDULER_REGISTRY.register()
def WarmupExponentialLR(
    optimizer: flow.optim.Optimizer,
    gamma: float,
    warmup_factor: float,
    warmup_iters: int,
    warmup_method: str = "linear",
):
    exponential_lr = flow.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    if warmup_iters == 0:
        logger.warning("warmup iters equals to zero, return ExponentialLR")
        return exponential_lr
    warmup_exponential_lr = flow.optim.lr_scheduler.WarmUpLR(
        exponential_lr,
        warmup_factor=warmup_factor,
        warmup_iters=warmup_iters,
        warmup_method=warmup_method,
    )
    return warmup_exponential_lr


@SCHEDULER_REGISTRY.register()
def WarmupPolynomailLR(
    optimizer: flow.optim.Optimizer,
    max_iters: int,
    warmup_factor: float,
    warmup_iters: int,
    end_learning_rate: float = 0.0001,
    power: float = 1.0,
    cycle: bool = False,
    warmup_method: str = "linear",
):
    polynomial_lr = flow.optim.lr_scheduler.PolynomialLR(
        optimizer, steps=max_iters, end_learning_rate=end_learning_rate, power=power, cycle=cycle
    )
    if warmup_iters == 0:
        logger.warning("warmup iters equals to zero, return PolynomialLR")
        return polynomial_lr
    warmup_polynomial_lr = flow.optim.lr_scheduler.WarmUpLR(
        polynomial_lr,
        warmup_factor=warmup_factor,
        warmup_iters=warmup_iters,
        warmup_method=warmup_method,
    )
    return warmup_polynomial_lr
