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
https://github.com/facebookresearch/fvcore/blob/main/tests/param_scheduler/test_scheduler_polynomial.py
"""

import unittest

from libai.solver.param_scheduler import PolynomialDecayParamScheduler


class TestPolynomialScheduler(unittest.TestCase):
    _num_epochs = 10

    def test_scheduler(self):
        scheduler = PolynomialDecayParamScheduler(base_value=0.1, power=1)
        schedule = [
            round(scheduler(epoch_num / self._num_epochs), 2)
            for epoch_num in range(self._num_epochs)
        ]
        expected_schedule = [0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01]

        self.assertEqual(schedule, expected_schedule)


if __name__ == "__main__":
    unittest.main()