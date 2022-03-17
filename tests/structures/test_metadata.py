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

from libai.data import DistTensorData
from libai.utils import distributed as dist


class TestMetadata(unittest.TestCase):
    @unittest.skipIf(not flow.cuda.is_available(), "only test gpu cases")
    def test_to_global(self):
        x = flow.rand(10, 10)
        x_meta = DistTensorData(x)
        x_meta.to_global()
        x_consistent = x.to_global(
            sbp=flow.sbp.broadcast,
            placement=flow.placement("cuda", [0]),
        )

        self.assertEqual(x_meta.tensor.sbp, x_consistent.sbp)
        self.assertEqual(x_meta.tensor.placement, x_consistent.placement)
        self.assertTrue((flow.equal(x_meta.tensor, x_consistent)).sum().item() == 100)

        x_meta.to_global(
            sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.split(0)]),
            placement=dist.get_layer_placement(5),
        )
        x_consistent = x.to_global(
            sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.split(0)]),
            placement=dist.get_layer_placement(5),
        )

        self.assertEqual(x_meta.tensor.sbp, x_consistent.sbp)
        self.assertEqual(x_meta.tensor.placement, x_consistent.placement)
        self.assertTrue((flow.equal(x_meta.tensor, x_consistent)).sum().item() == 100)

    @unittest.skipIf(not flow.cuda.is_available(), "only test gpu cases")
    def test_stack(self):
        x_list = [DistTensorData(flow.rand(10, 8)) for _ in range(5)]

        x_list.append(DistTensorData(flow.rand(10, 9)))  # shape mismatch
        with self.assertRaises(Exception):
            DistTensorData.stack(x_list)
        x_list.pop(-1)

        x_list.append(DistTensorData(flow.rand(10, 8), sbp_list=["broadcast"]))  # sbp mismatch
        with self.assertRaises(Exception):
            DistTensorData.stack(x_list)
        x_list.pop(-1)

        x_list.append(DistTensorData(flow.rand(10, 8), placement_idx=2))  # placement mismatch
        with self.assertRaises(Exception):
            DistTensorData.stack(x_list)
        x_list.pop(-1)

        x_stack = DistTensorData.stack(x_list)
        self.assertTrue(x_stack.tensor.shape == (5, 10, 8))


if __name__ == "__main__":
    unittest.main()
