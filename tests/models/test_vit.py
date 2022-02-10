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
from unittest import TestCase

import oneflow as flow

import libai.utils.distributed as dist
from configs.common.models.vit.vit_tiny_patch16_224 import model as vit_tiny_patch16_224
from libai.config import instantiate


@unittest.skip("Update CI Environments to run OneFlow on GPUs")
class TestVisionTransformer(TestCase):
    def test_vit_tiny_patch16_224(self):
        random_input = flow.randn(1, 3, 224, 224).to_consistent(
            sbp=dist.get_nd_sbp([flow.sbp.split(0), flow.sbp.broadcast]),
            placement=flow.placement("cuda" if flow.cuda.is_available() else "cpu", {0: 0}),
        )
        targets = flow.zeros(1, dtype=flow.int64).to_consistent(
            sbp=dist.get_nd_sbp([flow.sbp.split(0), flow.sbp.broadcast]),
            placement=flow.placement("cuda" if flow.cuda.is_available() else "cpu", {0: 0}),
        )
        model = instantiate(vit_tiny_patch16_224)
        model.apply(dist.convert_to_distributed_default_setting)
        output_dict = model(random_input, targets)
        loss = output_dict["losses"]
        loss.backward()


if __name__ == "__main__":
    unittest.main()
