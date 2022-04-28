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

import numpy as np
import oneflow as flow
import oneflow.unittest

from libai.inference.text_generation import TextGenerationPipeline
from libai.utils import distributed as dist


class TestTextGenerationPipeline(flow.unittest.TestCase):
    def setUp(self) -> None:
        self.texts = ["cat ", "you ", "dog ", "dragon ", "牛 ", "羊 "]

    # @flow.unittest.skip_unless_1n4d()
    def test_pipeline_with_tensor_parallel(self):
        self.pipeline = TextGenerationPipeline("configs/t5_pp_pretrain.py", 1, 4, 1)

        for _ in range(10):
            text = list(np.random.randint(0, 5, 10))
            text = "".join([self.texts[i] for i in text])
            dict1 = self.pipeline(
                text, use_cache=False, max_generate_length=15, return_type="new_text"
            )
            dict2 = self.pipeline(
                text, use_cache=True, max_generate_length=15, return_type="new_text"
            )
            if dist.is_main_process():
                assert dict1["generated_text"] == dict2["generated_text"]

    # @flow.unittest.skip_unless_1n4d()
    def test_pipeline_with_pipeline_parallel(self):
        self.pipeline = TextGenerationPipeline("configs/t5_pp_pretrain.py", 1, 1, 4)

        for _ in range(10):
            text = list(np.random.randint(0, 5, 10))
            text = "".join([self.texts[i] for i in text])
            dict1 = self.pipeline(
                text, use_cache=False, max_generate_length=15, return_type="new_text"
            )
            dict2 = self.pipeline(
                text, use_cache=True, max_generate_length=15, return_type="new_text"
            )
            if dist.is_main_process():
                assert dict1["generated_text"] == dict2["generated_text"]

    # @flow.unittest.skip_unless_1n4d()
    def test_pipeline_with_tensor_pipeline_parallel(self):
        self.pipeline = TextGenerationPipeline("configs/t5_pp_pretrain.py", 1, 2, 2)

        for _ in range(10):
            text = list(np.random.randint(0, 5, 10))
            text = "".join([self.texts[i] for i in text])
            dict1 = self.pipeline(
                text, use_cache=False, max_generate_length=15, return_type="new_text"
            )
            dict2 = self.pipeline(
                text, use_cache=True, max_generate_length=15, return_type="new_text"
            )
            if dist.is_main_process():
                assert dict1["generated_text"] == dict2["generated_text"]


if __name__ == "__main__":
    unittest.main()
