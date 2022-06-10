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
import zipfile

import numpy as np
import oneflow as flow
import oneflow.unittest

from libai.inference.image_classification import ImageClassificationPipeline
from libai.utils import distributed as dist
from libai.utils.file_utils import get_data_from_cache

IMAGE_URL = "https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/LiBai/Inference/ILSVRC2012_val_00000293.JPEG"  # noqa
CONFIG_URL = "https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/LiBai/ImageNet/vit_small_patch16_224/config.yaml"  # noqa
MODEL_URL = "https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/LiBai/ImageNet/vit_small_patch16_224/model_best.zip"  # noqa

IMAGE_MD5 = "65ac8a72466e859cd3c6b279ed8e532a"
CONFIG_MD5 = "4cf8e662d76f855f4d99ce7129050e79"
MODEL_MD5 = "2bfc9cb7df5739d1a1d11db97f54d93f"


def _legacy_zip_load(filename, model_dir):
    # Note: extractall() defaults to overwrite file if exists. No need to clean up beforehand.
    # We deliberately don't handle tarfile here since our legacy serialization format was in tar.
    with zipfile.ZipFile(filename) as f:
        members = f.infolist()
        extracted_name = members[0].filename
        extracted_file = os.path.join(model_dir, extracted_name)
        if not os.path.exists(extracted_file):
            os.mkdir(extracted_file)
        f.extractall(model_dir)


class TestImageClassificationPipeline(flow.unittest.TestCase):
    def setUp(self) -> None:
        cache_dir = os.path.join(
            os.getenv("ONEFLOW_TEST_CACHE_DIR", "./data_test"), "inference_test_data"
        )
        model_path = os.path.join(cache_dir, MODEL_URL.split("/")[-1])
        if dist.get_local_rank() == 0:
            # download dataset on main process of each node
            get_data_from_cache(IMAGE_URL, cache_dir, md5=IMAGE_MD5)
            get_data_from_cache(CONFIG_URL, cache_dir, md5=CONFIG_MD5)
            get_data_from_cache(MODEL_URL, cache_dir, md5=MODEL_MD5)
            _legacy_zip_load(model_path, os.path.dirname(model_path))
        self.image_path = os.path.join(cache_dir, IMAGE_URL.split("/")[-1])
        self.config_path = os.path.join(cache_dir, CONFIG_URL.split("/")[-1])
        self.model_path = model_path.replace(".zip", "")
        assert os.path.isdir(self.model_path)

    @unittest.skipIf(not flow.cuda.is_available(), "only test gpu cases")
    @flow.unittest.skip_unless_1n4d()
    def test_smallvitpipeline_with_pipeline_parallel(self):
        self.pipeline = ImageClassificationPipeline(self.config_path, 1, 1, 4, self.model_path)
        rst = self.pipeline(self.image_path)
        if flow.env.get_rank() == 0:
            self.assertTrue(rst["label"] == "tench, Tinca tinca")
            self.assertTrue(
                np.allclose(np.array(0.7100194096565247), np.array(rst["score"]), 1e-4, 1e-4)
            )

    @unittest.skipIf(not flow.cuda.is_available(), "only test gpu cases")
    @flow.unittest.skip_unless_1n4d()
    def test_pipeline_with_pipeline_parallel(self):
        self.pipeline = ImageClassificationPipeline("configs/vit_imagenet.py", 1, 1, 4)
        self.pipeline(self.image_path)

    @unittest.skipIf(not flow.cuda.is_available(), "only test gpu cases")
    @flow.unittest.skip_unless_1n4d()
    def test_pipeline_with_tensor_parallel(self):
        pass
        # TODO: bug occurs when tensor parallel
        # self.pipeline = ImageClassificationPipeline("configs/vit_imagenet.py", 1, 4, 1)
        # self.pipeline(self.image_path)


if __name__ == "__main__":
    unittest.main()
