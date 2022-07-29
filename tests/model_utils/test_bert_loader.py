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
from configs.common.models.bert import cfg as libai_cfg
from libai.models.utils import BertLoaderHuggerFace
from libai.utils import distributed as dist
from libai.utils.file_utils import get_data_from_cache
from libai.utils.logger import setup_logger

PRETRAINED_MODEL_URL = "http://oneflow-static.oss-cn-beijing.aliyuncs.com/ci-files/dataset/libai/model_utils_test/bert_utils/pytorch_model.bin"  # noqa
PRETRAINED_MODEL_CONFIG_URL = "http://oneflow-static.oss-cn-beijing.aliyuncs.com/ci-files/dataset/libai/model_utils_test/bert_utils/config.json"  # noqa

PRETRAINED_MODEL_MD5 = "ea97b42698d3b5f6d8e8011eba3d1611"
PRETRAINED_MODEL_CONFIG_MD5 = "0939b914fc32135f6c12d8ef281dbd7a"

TEST_OUTPUT = os.path.join(os.getenv("TEST_OUTPUT", "output_unittest"), "test_bert_utils")


setup_logger(distributed_rank=dist.get_rank())


class TestBertLoder(flow.unittest.TestCase):
    def setUp(self) -> None:
        cache_dir = os.path.join(
            os.getenv("ONEFLOW_TEST_CACHE_DIR", "./data_test"), "bert_utils_data"
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
        self.mask = [[1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1]]

    @classmethod
    def tearDownClass(cls) -> None:
        if os.path.isdir(TEST_OUTPUT) and dist.get_local_rank() == 0:
            shutil.rmtree(TEST_OUTPUT)

    @flow.unittest.skip_unless_1n4d()
    def test_bert_utils_with_data_tensor_parallel(self):
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
        load_func = BertLoaderHuggerFace(
            model=libai.models.BertModel,
            libai_cfg=libai_cfg,
            pretrained_model_path=self.pretrained_model_path,
            bias_gelu_fusion=False,
            bias_dropout_fusion=False,
            scale_mask_softmax_fusion=False,
            apply_query_key_layer_scaling=False,
            apply_residual_post_layernorm=True,
            amp_enabled=False,
        )
        model = load_func.load()
        model.eval()

        input_ids = flow.tensor(
            self.input_ids,
            dtype=flow.long,
            sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
            placement=model.embeddings.vocab_embeddings.weight.placement,
        )
        mask = flow.tensor(
            self.mask,
            dtype=flow.bool,
            sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
            placement=model.embeddings.vocab_embeddings.weight.placement,
        )

        last_hidden_state, _ = model(input_ids, mask)
        self.assertTrue(
            np.allclose(np.array(-214.9335), last_hidden_state.sum().data.numpy(), 1e-4, 1e-4)
        )

    @flow.unittest.skip_unless_1n4d()
    def test_bert_utils_with_data_tensor_pipeline_parallel(self):
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
        load_func = BertLoaderHuggerFace(
            model=libai.models.BertModel,
            libai_cfg=libai_cfg,
            pretrained_model_path=self.pretrained_model_path,
            bias_gelu_fusion=False,
            bias_dropout_fusion=False,
            scale_mask_softmax_fusion=False,
            apply_query_key_layer_scaling=False,
            apply_residual_post_layernorm=True,
            amp_enabled=False,
        )
        model = load_func.load()
        model.eval()

        input_ids = flow.tensor(
            self.input_ids,
            dtype=flow.long,
            sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
            placement=model.embeddings.vocab_embeddings.weight.placement,
        )
        mask = flow.tensor(
            self.mask,
            dtype=flow.bool,
            sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
            placement=model.embeddings.vocab_embeddings.weight.placement,
        )

        last_hidden_state, _ = model(input_ids, mask)
        self.assertTrue(
            np.allclose(np.array(-214.9335), last_hidden_state.sum().data.numpy(), 1e-4, 1e-4)
        )


if __name__ == "__main__":
    unittest.main()
