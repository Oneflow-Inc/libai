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
import oneflow.unittest

# from libai.inference.text_classification import TextClassificationPipeline


class TestTextClassificationPipeline(flow.unittest.TestCase):
    def setUp(self) -> None:
        pass

    @unittest.skipIf(not flow.cuda.is_available(), "only test gpu cases")
    @flow.unittest.skip_unless_1n4d()
    def test_pipeline_with_tensor_parallel(self):
        pass
        # TODO (Xie ZiPeng): write BertForClassification
        # self.pipeline1 = TextClassificationPipeline("")


if __name__ == "__main__":
    unittest.main()
