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
import shutil
import unittest

import numpy as np
import oneflow as flow
import oneflow.unittest
from omegaconf import DictConfig

import libai
from configs.common.models.swin.swin_tiny_patch4_window7_224 import cfg as libai_cfg
from libai.models.utils import SwinLoaderHuggerFace
from libai.utils import distributed as dist
from libai.utils.file_utils import get_data_from_cache
from libai.utils.logger import setup_logger

PRETRAINED_MODEL_URL = "http://oneflow-static.oss-cn-beijing.aliyuncs.com/ci-files/dataset/libai/model_utils_test/swin_utils/pytorch_model.bin"  # noqa
PRETRAINED_MODEL_CONFIG_URL = "http://oneflow-static.oss-cn-beijing.aliyuncs.com/ci-files/dataset/libai/model_utils_test/swin_utils/config.json"  # noqa
INIT_DATA = "http://oneflow-static.oss-cn-beijing.aliyuncs.com/ci-files/dataset/libai/model_utils_test/swin_utils/init_data.npz"  # noqa

PRETRAINED_MODEL_MD5 = "cd8c03d9cd4a9c536a5a245f663035b6"
PRETRAINED_MODEL_CONFIG_MD5 = "a8a71ed22b99323edd6a1457bede5819"
INIT_DATA_MD5 = "5fecdcd8d46bfefa310d19e084bd4815"

TEST_OUTPUT = os.path.join(os.getenv("TEST_OUTPUT", "output_unittest"), "test_swin_utils")


setup_logger(distributed_rank=dist.get_rank())


class TestSwinLoder(flow.unittest.TestCase):
    def setUp(self) -> None:
        cache_dir = os.path.join(
            os.getenv("ONEFLOW_TEST_CACHE_DIR", "./data_test"), "swin_utils_data"
        )
        self.pretrained_model_path = cache_dir
        self.init_data_path = os.path.join(cache_dir, "init_data.npz")

        # download model and data
        if dist.get_local_rank() == 0:
            # download dataset on main process of each node
            get_data_from_cache(PRETRAINED_MODEL_URL, cache_dir, md5=PRETRAINED_MODEL_MD5)
            get_data_from_cache(
                PRETRAINED_MODEL_CONFIG_URL, cache_dir, md5=PRETRAINED_MODEL_CONFIG_MD5
            )
            get_data_from_cache(INIT_DATA, cache_dir, md5=INIT_DATA_MD5)
            os.makedirs(TEST_OUTPUT, exist_ok=True)
        dist.synchronize()

        # prepare input data
        self.input_image = np.load(self.init_data_path)["arr_0"]

    @classmethod
    def tearDownClass(cls) -> None:
        if os.path.isdir(TEST_OUTPUT) and dist.get_local_rank() == 0:
            shutil.rmtree(TEST_OUTPUT)

    @flow.unittest.skip_unless_1n4d()
    def test_swin_utils_with_data_tensor_parallel(self):
        # set distributed config
        dist_cfg = DictConfig(
            dict(
                data_parallel_size=2,
                tensor_parallel_size=2,
                pipeline_parallel_size=1,
            )
        )
        dist.setup_dist_util(dist_cfg)

        # load model
        load_func = SwinLoaderHuggerFace(
            model=libai.models.SwinTransformer,
            libai_cfg=libai_cfg,
            pretrained_model_path=self.pretrained_model_path,
        )
        model = load_func.load()
        model.eval()

        input_image = flow.tensor(
            self.input_image.tolist(),
            dtype=flow.float32,
            sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
            placement=model.patch_embed.proj.weight.placement,
        )

        prediction_scores = model(input_image)["prediction_scores"]

        self.assertTrue(np.allclose(np.array(80.9373), prediction_scores.sum().data.numpy(), 1e-3))

    @flow.unittest.skip_unless_1n4d()
    def test_swin_utils_with_data_tensor_pipeline_parallel(self):
        # set distributed config
        dist_cfg = DictConfig(
            dict(
                data_parallel_size=2,
                tensor_parallel_size=1,
                pipeline_parallel_size=2,
                pipeline_num_layers=12,
            )
        )
        dist.setup_dist_util(dist_cfg)

        # load model
        load_func = SwinLoaderHuggerFace(
            model=libai.models.SwinTransformer,
            libai_cfg=libai_cfg,
            pretrained_model_path=self.pretrained_model_path,
        )
        model = load_func.load()
        model.eval()

        input_image = flow.tensor(
            self.input_image,
            dtype=flow.float32,
            sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
            placement=model.patch_embed.proj.weight.placement,
        )

        prediction_scores = model(input_image)["prediction_scores"]

        self.assertTrue(np.allclose(np.array(80.9373), prediction_scores.sum().data.numpy(), 1e-3))

    @flow.unittest.skip_unless_1n4d()
    def test_swin_utils_with_data_tensor_parallel_backward(self):
        # set distributed config
        dist_cfg = DictConfig(
            dict(
                data_parallel_size=2,
                tensor_parallel_size=2,
                pipeline_parallel_size=1,
            )
        )
        dist.setup_dist_util(dist_cfg)

        # load model
        load_func = SwinLoaderHuggerFace(
            model=libai.models.SwinTransformer,
            libai_cfg=libai_cfg,
            pretrained_model_path=self.pretrained_model_path,
            drop_rate=0.0,
            drop_path_rate=0.0,
        )
        model = load_func.load()

        input_image = flow.tensor(
            self.input_image,
            dtype=flow.float32,
            sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
            placement=model.patch_embed.proj.weight.placement,
        )

        prediction_scores = model(input_image)["prediction_scores"]
        loss = prediction_scores.sum()
        loss.backward()

        self.assertTrue(np.allclose(108775.88, model.head.weight.grad.sum().numpy(), 1e-3))
        self.assertTrue(
            np.allclose(24.320518, model.patch_embed.norm.weight.grad.sum().numpy(), 1e-2)
        )

    @flow.unittest.skip_unless_1n4d()
    def test_swin_utils_with_data_tensor_pipeline_parallel_backward(self):
        # set distributed config
        dist_cfg = DictConfig(
            dict(
                data_parallel_size=2,
                tensor_parallel_size=1,
                pipeline_parallel_size=2,
                pipeline_num_layers=12,
            )
        )
        dist.setup_dist_util(dist_cfg)

        # load model
        load_func = SwinLoaderHuggerFace(
            model=libai.models.SwinTransformer,
            libai_cfg=libai_cfg,
            pretrained_model_path=self.pretrained_model_path,
            drop_rate=0.0,
            drop_path_rate=0.0,
        )
        model = load_func.load()

        input_image = flow.tensor(
            self.input_image,
            dtype=flow.float32,
            sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
            placement=model.patch_embed.proj.weight.placement,
        )

        prediction_scores = model(input_image)["prediction_scores"]
        loss = prediction_scores.sum()
        loss.backward()

        self.assertTrue(np.allclose(108775.88, model.head.weight.grad.sum().numpy(), 1e-3))
        self.assertTrue(
            np.allclose(24.320518, model.patch_embed.norm.weight.grad.sum().numpy(), 1e-2)
        )


if __name__ == "__main__":
    unittest.main()
