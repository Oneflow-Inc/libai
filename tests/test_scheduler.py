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
import unittest
from unittest import TestCase

import numpy as np
import oneflow as flow
import oneflow.nn as nn

from libai.scheduler import (
    WarmupCosineLR,
    WarmupExponentialLR,
    WarmupMultiStepLR,
    WarmupPolynomialLR,
    WarmupStepLR,
)


# @unittest.skip("Bugs in warmup scheduler")
class TestScheduler(TestCase):
    def test_warmup_multistep(self):
        p = nn.Parameter(flow.zeros(0))
        opt = flow.optim.SGD([p], lr=5.0)

        sched = WarmupMultiStepLR(
            optimizer=opt,
            max_iter=10,
            milestones=[10, 15, 20],
            gamma=0.1,
            warmup_factor=0.001,
            warmup_iter=5,
            warmup_method="linear",
        )

        p.sum().backward()
        opt.step()

        lrs = [0.005]
        for _ in range(30):
            sched.step()
            lrs.append(opt.param_groups[0]["lr"])
        self.assertTrue(np.allclose(lrs[:5], [0.005, 1.004, 2.003, 3.002, 4.001]))
        self.assertTrue(np.allclose(lrs[5:10], 5.0))
        self.assertTrue(np.allclose(lrs[10:15], 0.5))
        self.assertTrue(np.allclose(lrs[15:20], 0.05))
        self.assertTrue(np.allclose(lrs[20:], 0.005))

    def test_warmup_step(self):
        p = nn.Parameter(flow.zeros(0))
        opt = flow.optim.SGD([p], lr=5.0)

        sched = WarmupStepLR(
            optimizer=opt,
            max_iter=10,
            step_size=10,
            gamma=0.1,
            warmup_factor=0.001,
            warmup_iter=5,
            warmup_method="linear",
        )

        p.sum().backward()
        opt.step()

        lrs = [0.005]
        for _ in range(30):
            sched.step()
            lrs.append(opt.param_groups[0]["lr"])
        self.assertTrue(np.allclose(lrs[:5], [0.005, 1.004, 2.003, 3.002, 4.001]))
        self.assertTrue(np.allclose(lrs[5:10], 5.0))
        self.assertTrue(np.allclose(lrs[10:20], 0.5))
        self.assertTrue(np.allclose(lrs[20:30], 0.05))
        self.assertTrue(np.allclose(lrs[30:], 0.005))

    def test_warmup_cosine(self):
        p = nn.Parameter(flow.zeros(0))
        opt = flow.optim.SGD([p], lr=5.0)

        sched = WarmupCosineLR(
            optimizer=opt,
            max_iter=30,
            warmup_factor=0.001,
            warmup_iter=5,
            warmup_method="linear",
        )

        p.sum().backward()
        opt.step()
        self.assertEqual(opt.param_groups[0]["lr"], 0.005)
        lrs = [0.005]

        for _ in range(30):
            sched.step()
            lrs.append(opt.param_groups[0]["lr"])
        for idx, lr in enumerate(lrs):
            expected_cosine = 2.5 * (1.0 + math.cos(math.pi * idx / 30))
            if idx >= 5:
                self.assertAlmostEqual(lr, expected_cosine)
            else:
                self.assertNotAlmostEqual(lr, expected_cosine)

    def test_warmup_exponential(self):
        p = nn.Parameter(flow.zeros(0))
        opt = flow.optim.SGD([p], lr=5.0)

        sched = WarmupExponentialLR(
            optimizer=opt,
            max_iter=10,
            gamma=0.1,
            warmup_factor=0.001,
            warmup_iter=5,
            warmup_method="linear",
        )

        p.sum().backward()
        opt.step()
        self.assertEqual(opt.param_groups[0]["lr"], 0.005)
        lrs = [0.005]

        def _get_exponential_lr(base_lr, gamma, max_iters, warmup_iters):
            valid_values = []
            for idx in range(warmup_iters, max_iters + 1):
                valid_values.append(base_lr * (gamma ** idx))
            return valid_values

        for _ in range(30):
            sched.step()
            lrs.append(opt.param_groups[0]["lr"])
        self.assertTrue(
            np.allclose(
                lrs[:5], [0.005, 0.00401, 0.0030199999999999997, 0.00203, 0.0010399999999999997]
            )
        )
        valid_intermediate_values = _get_exponential_lr(
            base_lr=5.0, gamma=0.1, max_iters=30, warmup_iters=5
        )
        self.assertEqual(lrs[5:], valid_intermediate_values)

    def test_warmup_polynomial(self):
        p = nn.Parameter(flow.zeros(0))
        opt = flow.optim.SGD([p], lr=5.0)

        sched = WarmupPolynomialLR(
            optimizer=opt,
            max_iter=30,
            warmup_factor=0.001,
            warmup_iter=0,
            end_learning_rate=1e-4,
            power=1.0,
            cycle=False,
            warmup_method="linear",
        )

        # self.assertEqual(opt.param_groups[0]["lr"], 0.005)
        # lrs = [0.005]
        lrs = [5.0]  # lr_scheduler first invoke result

        def _get_polynomial_lr(
            base_lr, max_iters, warmup_iters, end_lr=1e-4, power=1.0, cycle=False
        ):
            valid_values = []
            decay_steps = max_iters - warmup_iters
            for step in range(max_iters - warmup_iters):
                if cycle:
                    if step == 0:
                        step = 1
                    decay_steps = decay_steps * math.ceil(step / decay_steps)
                else:
                    step = min(step, decay_steps)
                valid_values.append(
                    (base_lr - end_lr) * ((1 - step / decay_steps) ** power) + end_lr
                )
            return valid_values

        for _ in range(29):
            sched.step()  # only invoke (max_iter-1), because the first invoke is done when init
            lrs.append(opt.param_groups[0]["lr"])
        # self.assertTrue(np.allclose(lrs[:5], [0.005, 1.004, 2.003, 3.002, 4.001]))
        valid_intermediate_values = _get_polynomial_lr(base_lr=5.0, max_iters=30, warmup_iters=0)
        # self.assertEqual(lrs[5:30], valid_intermediate_values)
        self.assertEqual(lrs, valid_intermediate_values)


if __name__ == "__main__":
    unittest.main()
