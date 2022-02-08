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

"""
Unittests followed https://github.com/facebookresearch/detectron2/blob/main/tests/test_solver.py
"""

import unittest

from libai.optim.build import _expand_param_groups, reduce_param_groups


class TestOptimizer(unittest.TestCase):
    def testExpandParamGroups(self):
        params = [
            {"params": ["p1", "p2", "p3", "p4"], "lr": 1.0, "weight_decay": 3.0},
            {"params": ["p2", "p3", "p5"], "lr": 2.0, "momentum": 2.0},
            {"params": ["p1"], "weight_decay": 4.0},
        ]
        out = _expand_param_groups(params)
        gt = [
            dict(params=["p1"], lr=1.0, weight_decay=4.0),  # noqa
            dict(params=["p2"], lr=2.0, weight_decay=3.0, momentum=2.0),  # noqa
            dict(params=["p3"], lr=2.0, weight_decay=3.0, momentum=2.0),  # noqa
            dict(params=["p4"], lr=1.0, weight_decay=3.0),  # noqa
            dict(params=["p5"], lr=2.0, momentum=2.0),  # noqa
        ]
        self.assertEqual(out, gt)

    def testReduceParamGroups(self):
        params = [
            dict(params=["p1"], lr=1.0, weight_decay=4.0),  # noqa
            dict(params=["p2", "p6"], lr=2.0, weight_decay=3.0, momentum=2.0),  # noqa
            dict(params=["p3"], lr=2.0, weight_decay=3.0, momentum=2.0),  # noqa
            dict(params=["p4"], lr=1.0, weight_decay=3.0),  # noqa
            dict(params=["p5"], lr=2.0, momentum=2.0),  # noqa
        ]
        gt_groups = [
            {"lr": 1.0, "weight_decay": 4.0, "params": ["p1"]},
            {"lr": 2.0, "weight_decay": 3.0, "momentum": 2.0, "params": ["p2", "p6", "p3"]},
            {"lr": 1.0, "weight_decay": 3.0, "params": ["p4"]},
            {"lr": 2.0, "momentum": 2.0, "params": ["p5"]},
        ]
        out = reduce_param_groups(params)
        self.assertTrue(out, gt_groups)


if __name__ == "__main__":
    unittest.main()
