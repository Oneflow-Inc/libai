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
https://github.com/facebookresearch/fvcore/blob/main/tests/param_scheduler/test_scheduler_cosine.py
"""

import copy
import unittest

from libai.solver.param_scheduler import CosineParamScheduler

class TestCosineScheduler(unittest.TestCase):
    _num_epochs = 10

    def _get_valid_decay_config(self):
        return {"start_value": 0.1, "end_value": 0}

    def _get_valid_decay_config_intermediate_values(self):
        return [0.0976, 0.0905, 0.0794, 0.0655, 0.05, 0.0345, 0.0206, 0.0095, 0.0024]

    def test_scheduler_as_decay(self):
        config = self._get_valid_decay_config()

        scheduler = CosineParamScheduler(**config)
        schedule = [
            round(scheduler(epoch_num / self._num_epochs), 4)
            for epoch_num in range(self._num_epochs)
        ]
        expected_schedule = [
            config["start_value"]
        ] + self._get_valid_decay_config_intermediate_values()

        self.assertEqual(schedule, expected_schedule)

    def test_scheduler_as_warmup(self):
        config = self._get_valid_decay_config()
        # Swap start and end lr to change to warmup
        tmp = config["start_value"]
        config["start_value"] = config["end_value"]
        config["end_value"] = tmp

        scheduler = CosineParamScheduler(**config)
        schedule = [
            round(scheduler(epoch_num / self._num_epochs), 4)
            for epoch_num in range(self._num_epochs)
        ]
        # Schedule should be decay reversed
        expected_schedule = [config["start_value"]] + list(
            reversed(self._get_valid_decay_config_intermediate_values())
        )

        self.assertEqual(schedule, expected_schedule)

    def test_scheduler_warmup_decay_match(self):
        decay_config = self._get_valid_decay_config()
        decay_scheduler = CosineParamScheduler(**decay_config)

        warmup_config = copy.deepcopy(decay_config)
        # Swap start and end lr to change to warmup
        tmp = warmup_config["start_value"]
        warmup_config["start_value"] = warmup_config["end_value"]
        warmup_config["end_value"] = tmp
        warmup_scheduler = CosineParamScheduler(**warmup_config)

        decay_schedule = [
            round(decay_scheduler(epoch_num / 1000), 8) for epoch_num in range(1, 1000)
        ]
        warmup_schedule = [
            round(warmup_scheduler(epoch_num / 1000), 8) for epoch_num in range(1, 1000)
        ]

        self.assertEqual(decay_schedule, list(reversed(warmup_schedule)))

if __name__ == "__main__":
    unittest.main()