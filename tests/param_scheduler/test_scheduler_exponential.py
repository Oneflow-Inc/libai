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
https://github.com/facebookresearch/fvcore/blob/main/tests/param_scheduler/test_scheduler_exponential.py
"""

import unittest

from libai.solver.param_scheduler import ExponentialParamScheduler


class TestExponentialScheduler(unittest.TestCase):
    _num_epochs = 10

    def _get_valid_config(self):
        return {"start_value": 2.0, "decay": 0.1}

    def _get_valid_intermediate_values(self):
        return [1.5887, 1.2619, 1.0024, 0.7962, 0.6325, 0.5024, 0.3991, 0.3170, 0.2518]

    def test_scheduler(self):
        config = self._get_valid_config()

        scheduler = ExponentialParamScheduler(**config)
        schedule = [
            round(scheduler(epoch_num / self._num_epochs), 4)
            for epoch_num in range(self._num_epochs)
        ]
        expected_schedule = [
            config["start_value"]
        ] + self._get_valid_intermediate_values()

        self.assertEqual(schedule, expected_schedule)


if __name__ == "__main__":
    unittest.main()