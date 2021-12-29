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
The following test case are mainly adopted from:
https://github.com/facebookresearch/fvcore/blob/main/tests/param_scheduler/test_scheduler_step_with_fixed_gamma.py
"""

import copy
import unittest
from typing import Any, Dict

from libai.solver.param_scheduler import StepParamScheduler


class TestStepScheduler(unittest.TestCase):
    _num_updates = 12

    def _get_valid_config(self) -> Dict[str, Any]:
        return {
            "num_updates": self._num_updates,
            "values": [0.1, 0.01, 0.001, 0.0001],
        }

    def test_invalid_config(self):
        # Invalid num epochs
        config = self._get_valid_config()

        bad_config = copy.deepcopy(config)
        bad_config["num_updates"] = -1
        with self.assertRaises(ValueError):
            StepParamScheduler(**bad_config)

        bad_config["values"] = {"a": "b"}
        with self.assertRaises(ValueError):
            StepParamScheduler(**bad_config)

        bad_config["values"] = []
        with self.assertRaises(ValueError):
            StepParamScheduler(**bad_config)

    def test_scheduler(self):
        config = self._get_valid_config()

        scheduler = StepParamScheduler(**config)
        schedule = [
            scheduler(epoch_num / self._num_updates)
            for epoch_num in range(self._num_updates)
        ]
        expected_schedule = [
            0.1,
            0.1,
            0.1,
            0.01,
            0.01,
            0.01,
            0.001,
            0.001,
            0.001,
            0.0001,
            0.0001,
            0.0001,
        ]

        self.assertEqual(schedule, expected_schedule)


if __name__ == "__main__":
    unittest.main()