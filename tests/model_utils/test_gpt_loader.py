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
from configs.common.models.gpt import cfg as libai_cfg
from libai.models.utils import GPT2LoaderHuggerFace
from libai.utils import distributed as dist
from libai.utils.file_utils import get_data_from_cache
from libai.utils.logger import setup_logger

PRETRAINED_MODEL_URL = "http://oneflow-static.oss-cn-beijing.aliyuncs.com/ci-files/dataset/libai/model_utils_test/gpt_utils/pytorch_model.bin"  # noqa
PRETRAINED_MODEL_CONFIG_URL = "http://oneflow-static.oss-cn-beijing.aliyuncs.com/ci-files/dataset/libai/model_utils_test/gpt_utils/config.json"  # noqa

PRETRAINED_MODEL_MD5 = "c086214036308afc71896da17ca0442a"
PRETRAINED_MODEL_CONFIG_MD5 = "6e1dba197b511b8759d6ad4551095a29"

TEST_OUTPUT = os.path.join(os.getenv("TEST_OUTPUT", "output_unittest"), "test_gpt_utils")


setup_logger(distributed_rank=dist.get_rank())


class TestGPT2Loader(flow.unittest.TestCase):
    def setUp(self) -> None:
        cache_dir = os.path.join(
            os.getenv("ONEFLOW_TEST_CACHE_DIR", "./data_test"), "gpt_utils_data"
        )
        self.pretrained_model_path = cache_dir

        # prepare dataset
        if dist.get_local_rank() == 0:
            # download dataset on main process of each node
            get_data_from_cache(PRETRAINED_MODEL_URL, cache_dir, md5=PRETRAINED_MODEL_MD5)
            get_data_from_cache(
                PRETRAINED_MODEL_CONFIG_URL, cache_dir, md5=PRETRAINED_MODEL_CONFIG_MD5
            )
            os.makedirs(TEST_OUTPUT, exist_ok=True)
        dist.synchronize()

        # prepare input data
        self.input_ids = [
            [101, 2009, 1005, 1055, 2986, 2651, 1012, 102],
            [101, 2028, 12314, 3377, 102, 0, 0, 0],
            [101, 2064, 2017, 3305, 2009, 102, 0, 0],
        ]

    @classmethod
    def tearDownClass(cls) -> None:
        if os.path.isdir(TEST_OUTPUT) and dist.get_local_rank() == 0:
            shutil.rmtree(TEST_OUTPUT)

    @flow.unittest.skip_unless_1n4d()
    def test_gpt_utils_with_data_tensor_parallel(self):
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
        load_func = GPT2LoaderHuggerFace(
            model=libai.models.GPTModel,
            libai_cfg=libai_cfg,
            pretrained_model_path=self.pretrained_model_path,
            bias_gelu_fusion=False,
            bias_dropout_fusion=False,
            scale_mask_softmax_fusion=True,
            apply_query_key_layer_scaling=True,
            apply_residual_post_layernorm=False,
            amp_enabled=False,
            attention_dropout_prob=0,
            output_dropout_prob=0,
        )
        model = load_func.load()
        model.eval()

        input_ids = flow.tensor(
            self.input_ids,
            dtype=flow.long,
            sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
            placement=model.embeddings.token_embeddings.weight.placement,
        )

        logits = model(input_ids)
        self.assertTrue(
            np.allclose(
                np.array(-93505072.0),
                logits.sum().data.numpy(),
            )
        )

    @flow.unittest.skip_unless_1n4d()
    def test_gpt_utils_with_data_tensor_pipeline_parallel(self):
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
        load_func = GPT2LoaderHuggerFace(
            model=libai.models.GPTModel,
            libai_cfg=libai_cfg,
            pretrained_model_path=self.pretrained_model_path,
            bias_gelu_fusion=False,
            bias_dropout_fusion=False,
            scale_mask_softmax_fusion=True,
            apply_query_key_layer_scaling=True,
            apply_residual_post_layernorm=False,
            amp_enabled=False,
            attention_dropout_prob=0,
            output_dropout_prob=0,
        )
        model = load_func.load()
        model.eval()

        input_ids = flow.tensor(
            self.input_ids,
            dtype=flow.long,
            sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
            placement=model.embeddings.token_embeddings.weight.placement,
        )

        logits = model(input_ids)
        self.assertTrue(
            np.allclose(
                np.array(-93505072.0),
                logits.sum().data.numpy(),
            )
        )


if __name__ == "__main__":
    unittest.main()
