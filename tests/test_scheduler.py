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
from unittest import TestCase
import unittest

import numpy as np
import oneflow as flow
import oneflow.nn as nn
from libai.scheduler import (
                            WarmupCosineLR, 
                            WarmupMultiStepLR, 
                            WarmupFixedStepLR,
                            WarmupExponentialLR,
                            WarmupCosineAnnealingLR,
                            )


class TestScheduler(TestCase):
    def test_warmup_multistep(self):
        p = nn.Parameter(flow.zeros(0))
        opt = flow.optim.SGD([p], lr=5.0)

        sched = WarmupMultiStepLR(optimizer = opt,
                                  milestones = [10, 15, 20],
                                  gamma = 0.1,
                                  warmup_factor = 0.001,
                                  warmup_iters = 5,
                                  warmup_method = "linear",)
        
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
    
    def test_warmup_cosine(self):
        p = nn.Parameter(flow.zeros(0))
        opt = flow.optim.SGD([p], lr=5.0)

        sched = WarmupCosineLR(optimizer = opt,
                               max_iters = 30,
                               warmup_factor = 0.001,
                               warmup_iters = 5,
                               warmup_method = "linear")
        
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

    def test_warmup_fixedstep(self):
        p = nn.Parameter(flow.zeros(0))
        opt = flow.optim.SGD([p], lr=5.0)

        sched = WarmupFixedStepLR(optimizer = opt,
                                  step_size = 10,
                                  gamma = 0.1,
                                  warmup_factor = 0.001,
                                  warmup_iters = 5,
                                  warmup_method = "linear")
        
        p.sum().backward()
        opt.step()
        self.assertEqual(opt.param_groups[0]["lr"], 0.005)
        lrs = [0.005]

        for _ in range(30):
            sched.step()
            lrs.append(opt.param_groups[0]["lr"])
        self.assertTrue(np.allclose(lrs[:5], [0.005, 1.004, 2.003, 3.002, 4.001]))
        self.assertTrue(np.allclose(lrs[5:10], 5.0))
        self.assertTrue(np.allclose(lrs[10:20], 0.5))
        self.assertTrue(np.allclose(lrs[20:30], 0.05))
        


if __name__ == "__main__":
    unittest.main()