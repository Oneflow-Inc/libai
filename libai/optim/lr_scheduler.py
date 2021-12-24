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


def update_train_iters(args):

    # For iteration-based training, we don't need to do anything
    if args.train_iters:
        return

    # Constant batch size with sample-based training.
    if args.rampup_batch_size is None:
        args.train_iters = args.train_samples // args.global_batch_size

    else:
        # Sample based training with rampup batch size.
        raise KeyError("not support rampup batch size right now!")
        # iterations = 0
        # consumed_samples = 0
        # # Rampup phase.
        # while consumed_samples <= int(args.rampup_batch_size[2]):
        #     update_num_microbatches(consumed_samples, consistency_check=False)
        #     consumed_samples += get_current_global_batch_size()
        #     iterations += 1
        # # Reset
        # update_num_microbatches(0, consistency_check=False)
        # # Constant phase
        # # Note that we throw away any partial last batch.
        # iterations += (args.train_samples - consumed_samples) // \
        #               args.global_batch_size
        # args.train_iters = iterations


def get_learning_rate_scheduler(args, optimizer):
    """Build the learning rate scheduler."""

    # Iteration-based training.
    if args.train_iters:
        if args.lr_decay_iters is None:
            args.lr_decay_iters = args.train_iters
        decay_steps = args.lr_decay_iters * args.global_batch_size
        if args.lr_warmup_fraction is not None:
            warmup_steps = args.lr_warmup_fraction * decay_steps
        else:
            warmup_steps = args.lr_warmup_iters * args.global_batch_size
    # Sample-based training.
    elif args.train_samples:
        # We need to set training iters for later use. Technically
        # we need to adjust the training samples too (due to last
        # batch being incomplete) but we leave it as is for now.
        update_train_iters(args)
        if args.lr_decay_samples is None:
            args.lr_decay_samples = args.train_samples
        decay_steps = args.lr_decay_samples
        if args.lr_warmup_fraction is not None:
            warmup_steps = args.lr_warmup_fraction * decay_steps
        else:
            warmup_steps = args.lr_warmup_samples
    else:
        raise Exception("either train-iters or train-samples should be provided.")

    # NOTE(l1aoxingyu): In megatron, lr scheduler update according to training samples
    # rather than training steps, we just divide the samples of each iteration to update scheduler by iter which is the oneflow lr scheduler update way
    increment = get_num_microbatches() * args.micro_batch_size * args.data_parallel_size

    lr_scheduler = PolynomialLR(
        optimizer, steps=int(decay_steps / increment), end_learning_rate=args.min_lr
    )
    lr_scheduler = flow.optim.lr_scheduler.WarmUpLR(
        lr_scheduler,
        warmup_factor=0,
        warmup_iters=int(warmup_steps / increment),
        warmup_method="linear",
    )
    return lr_scheduler


class PolynomialLR(LrScheduler):
    """This operator creates a polynomial decayed learning rate scheduler.
    The learning rate will be updated as follows:
    If cycle is `True`, the equation is:
    .. math::
        & decay\\_batch = decay\\_batch*ceil(\\frac{current\\_batch}{decay\\_batch})
        & learning\\_rate = (base\\_lr-end\\_lr)*(1-\\frac{current\\_batch}{decay\\_batch})^{pow}+end\\_lr
    If cycle is `False`, the equation is:
    .. math::
        & decay\\_batch = min(decay\\_batch, current\\_batch)
        & learning\\_rate = (base\\_lr-end\\_lr)*(1-\\frac{current\\_batch}{decay\\_batch})^{pow}+end\\_lr
    Args:
        steps (int): The decayed steps
        end_learning_rate (float, optional): The final learning rate. Defaults to 0.0001.
        power (float, optional): The power of polynomial. Defaults to 1.0.
        cycle (bool, optional): If cycle is true, the scheduler will decay the learning rate every decay steps. Defaults to False.
    For example:
        .. code-block:: python
            import oneflow as flow
           
            ... 
            polynomial_scheduler = flow.optimizer.lr_scheduler.PolynomialScheduler(optimizer,
                                                                           steps=5,
                                                                           end_learning_rate=0.00001,
                                                                           power=2)
            for epoch in range(num_epoch):
                train(...)
                polynomial_scheduler.step()
    """

    def __init__(
        self,
        optimizer,
        steps: int,
        end_learning_rate: float = 0.0001,
        power: float = 1.0,
        cycle: bool = False,
        last_step=-1,
        verbose=False,
    ):
        assert steps > 0, f"steps must greater than zero, but got {steps}"
        self.max_decay_steps = steps
        self.end_learning_rate = end_learning_rate
        self.power = power
        self.cycle = cycle
        super().__init__(optimizer, last_step, verbose)

    def get_lr(self):
        decay_batch = self.max_decay_steps
        cur_batch = self.last_step
        if self.cycle:
            decay_batch = decay_batch * math.ceil(cur_batch / decay_batch)
        else:
            cur_batch = min(cur_batch, decay_batch)
        return [
            (base_lr - self.end_learning_rate)
            * ((1 - cur_batch / decay_batch) ** (self.power))
            + self.end_learning_rate
            for base_lr in self.base_lrs
        ]

    def _generate_conf_for_graph(self, opt_confs):
        # CosineDecayLR is the same as CosineDecayConf in nn.Graph
        for opt_conf in opt_confs:
            learning_rate_decay_conf = opt_conf.mutable_learning_rate_decay()
            learning_rate_decay_conf.mutable_polynomial_conf().set_decay_batches(
                self.max_decay_steps
            )
            learning_rate_decay_conf.mutable_polynomial_conf().set_end_learning_rate(
                self.end_learning_rate
            )
            learning_rate_decay_conf.mutable_polynomial_conf().set_power(self.power)
            learning_rate_decay_conf.mutable_polynomial_conf().set_cycle(self.cycle)
