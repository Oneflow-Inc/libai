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
https://github.com/facebookresearch/fvcore/blob/main/tests/param_scheduler/test_scheduler_linear.py
"""

import unittest

from libai.solver.param_scheduler import LinearParamScheduler


class TestLienarScheduler(unittest.TestCase):
    _num_epochs = 10

    def _get_valid_intermediate(self):
        return [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09]

    def _get_valid_config(self):
        return {"start_value": 0.0, "end_value": 0.1}

    def test_scheduler(self):
        config = self._get_valid_config()

        # Check as warmup
        scheduler = LinearParamScheduler(**config)
        schedule = [
            round(scheduler(epoch_num / self._num_epochs), 4)
            for epoch_num in range(self._num_epochs)
        ]
        expected_schedule = [config["start_value"]] + self._get_valid_intermediate()
        self.assertEqual(schedule, expected_schedule)

        # Check as decay
        tmp = config["start_value"]
        config["start_value"] = config["end_value"]
        config["end_value"] = tmp
        scheduler = LinearParamScheduler(**config)
        schedule = [
            round(scheduler(epoch_num / self._num_epochs), 4)
            for epoch_num in range(self._num_epochs)
        ]
        expected_schedule = [config["start_value"]] + list(
            reversed(self._get_valid_intermediate())
        )
        self.assertEqual(schedule, expected_schedule)


if __name__ == "__main__":
    unittest.main()