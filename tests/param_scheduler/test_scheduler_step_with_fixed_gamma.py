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

from libai.solver.param_scheduler import StepWithFixedGammaParamScheduler


class TestStepWithFixedGammaScheduler(unittest.TestCase):
    _num_updates = 12

    def _get_valid_config(self):
        return {
            "base_value": 1,
            "gamma": 0.1,
            "num_decays": 3,
            "num_updates": self._num_updates,
        }

    def test_invalid_config(self):
        config = self._get_valid_config()

        # Invalid num epochs
        bad_config = copy.deepcopy(config)
        bad_config["num_updates"] = -1
        with self.assertRaises(ValueError):
            StepWithFixedGammaParamScheduler(**bad_config)

        # Invalid num_decays
        bad_config["num_decays"] = 0
        with self.assertRaises(ValueError):
            StepWithFixedGammaParamScheduler(**bad_config)

        # Invalid base_value
        bad_config = copy.deepcopy(config)
        bad_config["base_value"] = -0.01
        with self.assertRaises(ValueError):
            StepWithFixedGammaParamScheduler(**bad_config)

        # Invalid gamma
        bad_config = copy.deepcopy(config)
        bad_config["gamma"] = [2]
        with self.assertRaises(ValueError):
            StepWithFixedGammaParamScheduler(**bad_config)

    def test_scheduler(self):
        config = self._get_valid_config()

        scheduler = StepWithFixedGammaParamScheduler(**config)
        schedule = [
            scheduler(epoch_num / self._num_updates)
            for epoch_num in range(self._num_updates)
        ]
        expected_schedule = [
            1,
            1,
            1,
            0.1,
            0.1,
            0.1,
            0.01,
            0.01,
            0.01,
            0.001,
            0.001,
            0.001,
        ]

        for param, expected_param in zip(schedule, expected_schedule):
            self.assertAlmostEqual(param, expected_param)


if __name__ == "__main__":
    unittest.main()