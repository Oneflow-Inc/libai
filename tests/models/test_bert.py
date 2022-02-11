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

from configs.common.models.bert import pretrain_model as model
from libai.models import build_model
from libai.utils import distributed as dist


class TestBertModel(unittest.TestCase):
    def test_bert_build(self):
        bert_model = build_model(model)
        self.assertTrue(isinstance(bert_model.bert.embeddings.vocab_embeddings.weight, flow.Tensor))

    # @unittest.skip("Update CI Environments to run OneFlow on GPUs")
    def test_bert_forward(self):
        input_ids = flow.ones(
            2,
            512,
            dtype=flow.long,
            sbp=dist.get_nd_sbp([flow.sbp.split(0), flow.sbp.broadcast]),
            placement=flow.placement("cuda" if flow.cuda.is_available() else "cpu", {0: [0]}),
        )
        attention_mask = flow.zeros(
            2,
            512,
            dtype=flow.long,
            sbp=dist.get_nd_sbp([flow.sbp.split(0), flow.sbp.broadcast]),
            placement=flow.placement("cuda" if flow.cuda.is_available() else "cpu", {0: [0]}),
        )
        tokentype_ids = flow.zeros(
            2,
            512,
            dtype=flow.long,
            sbp=dist.get_nd_sbp([flow.sbp.split(0), flow.sbp.broadcast]),
            placement=flow.placement("cuda" if flow.cuda.is_available() else "cpu", {0: [0]}),
        )

        bert_model = build_model(model)
        output_dict = bert_model(input_ids, attention_mask, tokentype_ids)

        self.assertEqual(list(output_dict.keys()), ["prediction_scores", "seq_relationship_score"])
        self.assertEqual(list(output_dict["prediction_scores"].shape), [2, 512, 30522])
        self.assertEqual(list(output_dict["seq_relationship_score"].shape), [2, 2])

    def test_bert_backward(self):
        input_ids = flow.ones(
            2,
            512,
            dtype=flow.long,
            sbp=dist.get_nd_sbp([flow.sbp.split(0), flow.sbp.broadcast]),
            placement=flow.placement("cuda" if flow.cuda.is_available() else "cpu", {0: [0]}),
        )
        attention_mask = flow.zeros(
            2,
            512,
            dtype=flow.long,
            sbp=dist.get_nd_sbp([flow.sbp.split(0), flow.sbp.broadcast]),
            placement=flow.placement("cuda" if flow.cuda.is_available() else "cpu", {0: [0]}),
        )
        tokentype_ids = flow.zeros(
            2,
            512,
            dtype=flow.long,
            sbp=dist.get_nd_sbp([flow.sbp.split(0), flow.sbp.broadcast]),
            placement=flow.placement("cuda" if flow.cuda.is_available() else "cpu", {0: [0]}),
        )

        ns_labels = flow.zeros(
            2,
            dtype=flow.long,
            sbp=dist.get_nd_sbp([flow.sbp.split(0), flow.sbp.broadcast]),
            placement=flow.placement("cuda" if flow.cuda.is_available() else "cpu", {0: [0]}),
        )
        lm_labels = flow.ones(
            2,
            512,
            dtype=flow.long,
            sbp=dist.get_nd_sbp([flow.sbp.split(0), flow.sbp.broadcast]),
            placement=flow.placement("cuda" if flow.cuda.is_available() else "cpu", {0: [0]}),
        )
        loss_mask = flow.ones(
            2,
            512,
            dtype=flow.long,
            sbp=dist.get_nd_sbp([flow.sbp.split(0), flow.sbp.broadcast]),
            placement=flow.placement("cuda" if flow.cuda.is_available() else "cpu", {0: [0]}),
        )
        bert_model = build_model(model)
        loss_dict = bert_model(
            input_ids, attention_mask, tokentype_ids, ns_labels, lm_labels, loss_mask
        )
        losses = sum(loss_dict.values())
        losses.backward()


if __name__ == "__main__":
    unittest.main()
