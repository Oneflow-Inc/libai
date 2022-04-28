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

import oneflow as flow
import oneflow.unittest

from libai.inference.image_classification import ImageClassificationPipeline
from libai.utils import distributed as dist
from libai.utils.file_utils import get_data_from_cache

IMAGE_URL = "https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/LiBai/Inference/ILSVRC2012_val_00000293.JPEG"  # noqa

IMAGE_MD5 = "65ac8a72466e859cd3c6b279ed8e532a"


class TestImageClassificationPipeline(flow.unittest.TestCase):
    def setUp(self) -> None:
        cache_dir = os.path.join(
            os.getenv("ONEFLOW_TEST_CACHE_DIR", "./data_test"), "inference_test_data"
        )
        if dist.get_local_rank() == 0:
            # download dataset on main process of each node
            get_data_from_cache(IMAGE_URL, cache_dir, md5=IMAGE_MD5)
        self.image_path = os.path.join(cache_dir, IMAGE_URL.split("/")[-1])

    @flow.unittest.skip_unless_1n4d()
    def test_pipeline_with_pipeline_parallel(self):
        self.pipeline = ImageClassificationPipeline("configs/vit_imagenet.py", 1, 1, 4)
        self.pipeline(self.image_path)

    @flow.unittest.skip_unless_1n4d()
    def test_pipeline_with_tensor_parallel(self):
        pass
        # TODO: bug occurs when tensor parallel
        # self.pipeline = ImageClassificationPipeline("configs/vit_imagenet.py", 1, 4, 1)
        # self.pipeline(self.image_path)


if __name__ == "__main__":
    unittest.main()
