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

from configs.common.models.gpt import pretrain_model as model
from libai.models import build_model
from libai.utils import distributed as dist


class TestGPTModel(unittest.TestCase):
    def test_gpt_build(self):
        gpt_model = build_model(model)
        self.assertTrue(isinstance(gpt_model.GPT_model.embeddings.token_embeddings.weight, flow.Tensor))

    @unittest.skip("No GPU in CI Environment")
    def test_gpt_forward(self):
        input_ids = flow.ones(
            2,
            512,
            dtype=flow.long,
            sbp=dist.get_nd_sbp([flow.sbp.split(0), flow.sbp.broadcast]),
            placement=flow.placement("cuda" if flow.cuda.is_available() else "cpu", [0]),
        )

        gpt_model = build_model(model)
        output_dict = gpt_model(input_ids)

        self.assertEqual(list(output_dict.keys()), ["prediction_scores"])
        self.assertEqual(list(output_dict["prediction_scores"].shape), [2, 512, model.cfg.vocab_size])

    @unittest.skip("No GPU in CI Environment")
    def test_gpt_backward(self):
        input_ids = flow.ones(
            2,
            512,
            dtype=flow.long,
            sbp=dist.get_nd_sbp([flow.sbp.split(0), flow.sbp.broadcast]),
            placement=flow.placement("cuda" if flow.cuda.is_available() else "cpu", [0]),
        )
        lm_labels = flow.ones(
            2,
            512,
            dtype=flow.long,
            sbp=dist.get_nd_sbp([flow.sbp.split(0), flow.sbp.broadcast]),
            placement=flow.placement("cuda" if flow.cuda.is_available() else "cpu", [0]),
        )
        gpt_model = build_model(model)
        loss_dict = gpt_model(
            input_ids, lm_labels,
        )
        losses = sum(loss_dict.values())
        losses.backward()


if __name__ == "__main__":
    unittest.main()
