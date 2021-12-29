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
https://github.com/facebookresearch/fvcore/blob/main/tests/param_scheduler/test_scheduler_constant.py
"""

import unittest

from libai.solver.param_scheduler import ConstantParamScheduler


class TestConstantScheduler(unittest.TestCase):
    _num_epochs = 12

    def test_scheduler(self):
        scheduler = ConstantParamScheduler(0.1)
        schedule = [
            scheduler(epoch_num / self._num_epochs)
            for epoch_num in range(self._num_epochs)
        ]
        expected_schedule = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

        self.assertEqual(schedule, expected_schedule)
        # The input for the scheduler should be in the interval [0;1), open
        with self.assertRaises(RuntimeError):
            scheduler(1)


if __name__ == "__main__":
    unittest.main()