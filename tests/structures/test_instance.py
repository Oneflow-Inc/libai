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

import unittest

import oneflow as flow

from libai.data import DistTensorData, Instance


class TestInstance(unittest.TestCase):
    def test_init_args(self):
        inst = Instance(images=flow.rand(4, 5))
        inst.tokens = flow.rand(5, 10)

        self.assertTrue(inst.has("images"))
        self.assertTrue(inst.has("tokens"))

        inst.remove("images")
        self.assertFalse(inst.has("images"))

        inst.meta_tensor = DistTensorData(flow.rand(5, 6))
        self.assertTrue(inst.has("meta_tensor"))
        self.assertTrue(isinstance(inst.get("meta_tensor"), DistTensorData))

    def test_order_args(self):
        inst = Instance(a=1, b=2, c=3)
        inst.d = 4
        inst.e = 5

        inst_key = []
        for key in inst.get_fields():
            inst_key.append(key)

        self.assertEqual(inst_key, ["a", "b", "c", "d", "e"])

    def test_stack(self):
        inst_list = [
            Instance(images=flow.rand(3, 4), masks=flow.rand(4, 5), bbox=[3, 4, 5, 6])
            for _ in range(10)
        ]

        inst = Instance.stack(inst_list)

        self.assertTrue(inst.has("images"))
        self.assertTrue(inst.has("masks"))
        self.assertFalse(inst.has("tokens"))
        self.assertEqual(inst.get("images").shape, (10, 3, 4))
        self.assertEqual(inst.get("masks").shape, (10, 4, 5))
        self.assertEqual(len(inst.get("bbox")), 10)


if __name__ == "__main__":
    unittest.main()
