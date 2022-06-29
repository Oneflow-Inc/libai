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


def WarmupCosineLR(
    optimizer: flow.optim.Optimizer,
    max_iter: int,
    warmup_factor: float,
    warmup_iter: int,
    alpha: float = 0.0,
    warmup_method: str = "linear",
):
    """Create a schedule with a learning rate that decreases following
    the values of the Cosine function between the initial lr set in the
    optimizer to 0, after a warmup period during which it increases linearly
    between 0 and the initial lr set in the optimizer.

    Args:
        optimizer (flow.optim.Optimizer): Wrapped optimizer.
        max_iter (int): Total training iters.
        warmup_factor (float): The warmup factor.
        warmup_iter (int): The number of warmup steps.
        alpha (float, optional): The learning rate scale factor (:math:`\\alpha`). Defaults to 0.0.
        warmup_method (str, optional): The method of warmup, you can choose "linear" or "constant".
            In linear mode, the multiplication factor starts with warmup_factor in
            the first epoch and then inreases linearly to reach 1. Defaults to "linear".
    """
    cosine_decay_lr = flow.optim.lr_scheduler.CosineDecayLR(
        optimizer, decay_steps=max_iter, alpha=alpha
    )
    if warmup_iter == 0:
        logger.warning("warmup iters equals to zero, return CosineLR")
        return cosine_decay_lr
    elif warmup_iter > max_iter:
        logger.warning("warmup iters is larger than the total training iters")
    warmup_cosine_lr = flow.optim.lr_scheduler.WarmUpLR(
        cosine_decay_lr,
        warmup_factor=warmup_factor,
        warmup_iters=warmup_iter,
        warmup_method=warmup_method,
    )
    return warmup_cosine_lr


def WarmupCosineAnnealingLR(
    optimizer: flow.optim.Optimizer,
    max_iter: int,
    warmup_factor: float,
    warmup_iter: int,
    eta_min: float = 0.0,
    warmup_method: str = "linear",
):
    """Create a schedule with a learning rate that decreases following
    the values of the Cosine Annealing function between the initial
    lr set in the optimizer to 0, after a warmup period during which
    it increases linearly between 0 and the initial lr set in the optimizer.

    Args:
        optimizer (flow.optim.Optimizer): Wrapped optimizer.
        max_iter (int): Total training iters.
        warmup_factor (float): The warmup factor.
        warmup_iter (int): The number of warmup steps.
        eta_min (float, optional): Minimum learning rate. Defaults to 0.0.
        warmup_method (str, optional): The method of warmup, you can choose "linear" or "constant".
            In linear mode, the multiplication factor starts with warmup_factor in the first epoch
            and then inreases linearly to reach 1. Defaults to "linear".
    """
    cosine_annealing_lr = flow.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_iter, eta_min=eta_min
    )
    if warmup_iter == 0:
        logger.warning("warmup iters equals to zero, return CosineAnnealingLR")
        return cosine_annealing_lr
    warmup_cosine_annealing_lr = flow.optim.lr_scheduler.WarmUpLR(
        cosine_annealing_lr,
        warmup_factor=warmup_factor,
        warmup_iters=warmup_iter,
        warmup_method=warmup_method,
    )
    return warmup_cosine_annealing_lr


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
    step_lr = flow.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
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


def WarmupMultiStepLR(
    optimizer: flow.optim.Optimizer,
    max_iter: int,
    warmup_factor: float,
    warmup_iter: int,
    milestones: list,
    gamma: float = 0.1,
    warmup_method: str = "linear",
):
    """Create a schedule with a learning rate that decreases following the values of the MultiStep
    function between the initial lr set in the optimizer to 0, after a warmup period during which
    it increases linearly between 0 and the initial lr set in the optimizer.

    Args:
        optimizer (flow.optim.Optimizer): Wrapped optimizer.
        max_iter (int): Total training iters.
        warmup_factor (float): The warmup factor.
        warmup_iter (int): The number of warmup steps.
        milestones (list): List of step indices. Must be increasing.
        gamma (float, optional): Multiplicative factor of learning rate decay. Defaults to 0.1.
        warmup_method (str, optional): The method of warmup, you can choose "linear" or "constant".
            In linear mode, the multiplication factor starts with warmup_factor in the first
            epoch and then inreases linearly to reach 1. Defaults to "linear".
    """
    multistep_lr = flow.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=gamma
    )
    if warmup_iter == 0:
        logger.warning("warmup iters equals to zero, return MultiStepLR")
        return multistep_lr
    warmup_multistep_lr = flow.optim.lr_scheduler.WarmUpLR(
        multistep_lr,
        warmup_factor=warmup_factor,
        warmup_iters=warmup_iter,
        warmup_method=warmup_method,
    )
    return warmup_multistep_lr


def WarmupExponentialLR(
    optimizer: flow.optim.Optimizer,
    max_iter: int,
    gamma: float,
    warmup_factor: float,
    warmup_iter: int,
    warmup_method: str = "linear",
):
    """Create a schedule with a learning rate that decreases following the values of
    the Exponential function between the initial lr set in the optimizer to 0,
    after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.

    Args:
        optimizer (flow.optim.Optimizer): Wrapped optimizer.
        max_iter (int): Total training iters.
        gamma (float): Multiplicative factor of learning rate decay.
        warmup_factor (float): The warmup factor.
        warmup_iter (int): The number of warmup steps.
        warmup_method (str, optional): The method of warmup, you can choose "linear" or "constant".
            In linear mode, the multiplication factor starts with warmup_factor in the first epoch
            and then inreases linearly to reach 1. Defaults to "linear".
    """
    exponential_lr = flow.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    if warmup_iter == 0:
        logger.warning("warmup iters equals to zero, return ExponentialLR")
        return exponential_lr
    warmup_exponential_lr = flow.optim.lr_scheduler.WarmUpLR(
        exponential_lr,
        warmup_factor=warmup_factor,
        warmup_iters=warmup_iter,
        warmup_method=warmup_method,
    )
    return warmup_exponential_lr


def WarmupPolynomialLR(
    optimizer: flow.optim.Optimizer,
    max_iter: int,
    warmup_factor: float,
    warmup_iter: int,
    end_learning_rate: float = 0.0001,
    power: float = 1.0,
    cycle: bool = False,
    warmup_method: str = "linear",
):
    """Create a schedule with a learning rate that decreases as a polynomial decay from
    the initial lr set in the optimizer to end lr defined by `lr_end`,
    after a warmup period during which it increases linearly from 0 to the
    initial lr set in the optimizer.


    Args:
        optimizer (flow.optim.Optimizer): Wrapped optimizer.
        max_iter (int): Total training iters.
        warmup_factor (float): The warmup factor.
        warmup_iter (int): The number of warmup steps.
        end_learning_rate (float, optional): The final learning rate. Defaults to 0.0001.
        power (float, optional): The power of polynomial. Defaults to 1.0.
        cycle (bool, optional): If cycle is True, the scheduler will decay the learning rate
            every decay steps. Defaults to False.
        warmup_method (str, optional): The method of warmup, you can choose "linear" or "constant".
            In linear mode, the multiplication factor starts with warmup_factor in the first
            epoch and then inreases linearly to reach 1. Defaults to "linear".
    """
    polynomial_lr = flow.optim.lr_scheduler.PolynomialLR(
        optimizer,
        decay_batch=max_iter,
        end_learning_rate=end_learning_rate,
        power=power,
        cycle=cycle,
    )
    if warmup_iter == 0:
        logger.warning("warmup iters equals to zero, return PolynomialLR")
        return polynomial_lr
    warmup_polynomial_lr = flow.optim.lr_scheduler.WarmUpLR(
        polynomial_lr,
        warmup_factor=warmup_factor,
        warmup_iters=warmup_iter,
        warmup_method=warmup_method,
    )
    return warmup_polynomial_lr
