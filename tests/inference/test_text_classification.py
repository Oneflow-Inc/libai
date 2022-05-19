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


import os
import unittest

import numpy as np
import oneflow as flow
import oneflow.unittest

from libai.inference.text_classification import TextClassificationPipeline
from libai.utils import distributed as dist
from libai.utils.file_utils import get_data_from_cache

VOCAB_URL = "https://oneflow-static.oss-cn-beijing.aliyuncs.com/ci-files/dataset/libai/bert_dataset/bert-base-chinese-vocab.txt"  # noqa

VOCAB_MD5 = "65ac8a72466e859cd3c6b279ed8e532a"


class TestTextClassificationPipeline(flow.unittest.TestCase):
    def setUp(self) -> None:
        self.texts = ["cat ", "you ", "dog ", "dragon ", "牛 ", "羊 "]
        cache_dir = os.path.join(os.getenv("ONEFLOW_TEST_CACHE_DIR", "./data_test"), "bert_data")
        # prepare tokenizer
        if dist.get_local_rank() == 0:
            # download tokenzier vocab on main process of each node
            get_data_from_cache(VOCAB_URL, cache_dir, md5=VOCAB_MD5)

    @unittest.skipIf(not flow.cuda.is_available(), "only test gpu cases")
    @flow.unittest.skip_unless_1n4d()
    def test_pipeline_with_tensor_parallel(self):
        self.pipeline = TextClassificationPipeline("configs/bert_classification.py", 1, 4, 1)

        text = list(np.random.randint(0, 6, 10))
        text = "".join([self.texts[i] for i in text])
        dict1 = self.pipeline(text)
        dict2 = self.pipeline(text)
        if dist.is_main_process():
            assert dict1["score"] == dict2["score"]

    @unittest.skipIf(not flow.cuda.is_available(), "only test gpu cases")
    @flow.unittest.skip_unless_1n4d()
    def test_pipeline_with_pipeline_parallel(self):
        self.pipeline = TextClassificationPipeline("configs/bert_classification.py", 1, 1, 4)

        text = list(np.random.randint(0, 6, 10))
        text = "".join([self.texts[i] for i in text])
        dict1 = self.pipeline(text)
        dict2 = self.pipeline(text)
        if dist.is_main_process():
            assert dict1["score"] == dict2["score"]

    @unittest.skipIf(not flow.cuda.is_available(), "only test gpu cases")
    @flow.unittest.skip_unless_1n4d()
    def test_pipeline_with_tensor_pipeline_parallel(self):
        self.pipeline = TextClassificationPipeline("configs/bert_classification.py", 1, 2, 2)

        text = list(np.random.randint(0, 6, 10))
        text = "".join([self.texts[i] for i in text])
        dict1 = self.pipeline(text)
        dict2 = self.pipeline(text)
        if dist.is_main_process():
            assert dict1["score"] == dict2["score"]


if __name__ == "__main__":
    unittest.main()
