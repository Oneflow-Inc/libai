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

import libai.utils.distributed as dist
from configs.common.models.vit.vit_tiny_patch16_224 import model
from libai.models import build_model


class TestViTModel(unittest.TestCase):
    def test_build_vit(self):
        vit_model = build_model(model)
        self.assertTrue(isinstance(vit_model.patch_embed.proj.weight, flow.Tensor))

    @unittest.skip("No GPU in CI Environment")
    def test_vit_training_forward(self):
        input_tensor = flow.randn(1, 3, 224, 224).to_global(
            sbp=dist.get_nd_sbp([flow.sbp.split(0), flow.sbp.broadcast]),
            placement=flow.placement("cuda" if flow.cuda.is_available() else "cpu", {0: 0}),
        )
        targets = flow.zeros(1, dtype=flow.int64).to_global(
            sbp=dist.get_nd_sbp([flow.sbp.split(0), flow.sbp.broadcast]),
            placement=flow.placement("cuda" if flow.cuda.is_available() else "cpu", {0: 0}),
        )
        vit_model = build_model(model)
        vit_model.apply(dist.convert_to_distributed_default_setting)
        output_dict = vit_model(input_tensor, targets)

        self.assertEqual(list(output_dict.keys()), ["losses"])

    @unittest.skip("No GPU in CI Environment")
    def test_vit_eval_forward(self):
        input_tensor = flow.randn(1, 3, 224, 224).to_global(
            sbp=dist.get_nd_sbp([flow.sbp.split(0), flow.sbp.broadcast]),
            placement=flow.placement("cuda" if flow.cuda.is_available() else "cpu", {0: 0}),
        )
        targets = flow.zeros(1, dtype=flow.int64).to_global(
            sbp=dist.get_nd_sbp([flow.sbp.split(0), flow.sbp.broadcast]),
            placement=flow.placement("cuda" if flow.cuda.is_available() else "cpu", {0: 0}),
        )
        vit_model = build_model(model)
        vit_model.apply(dist.convert_to_distributed_default_setting)
        vit_model.eval()
        output_dict = vit_model(input_tensor, targets)

        self.assertEqual(list(output_dict.keys()), ["prediction_scores"])
        self.assertEqual(list(output_dict["prediction_scores"].shape), [1, 1000])

    @unittest.skip("No GPU in CI Environment")
    def test_vit_backward(self):
        input_tensor = flow.randn(1, 3, 224, 224).to_global(
            sbp=dist.get_nd_sbp([flow.sbp.split(0), flow.sbp.broadcast]),
            placement=flow.placement("cuda" if flow.cuda.is_available() else "cpu", {0: 0}),
        )
        targets = flow.zeros(1, dtype=flow.int64).to_global(
            sbp=dist.get_nd_sbp([flow.sbp.split(0), flow.sbp.broadcast]),
            placement=flow.placement("cuda" if flow.cuda.is_available() else "cpu", {0: 0}),
        )
        vit_model = build_model(model)
        vit_model.apply(dist.convert_to_distributed_default_setting)
        output_dict = vit_model(input_tensor, targets)
        losses = output_dict["losses"]
        losses.backward()


if __name__ == "__main__":
    unittest.main()
