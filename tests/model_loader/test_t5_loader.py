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

from libai.utils import distributed as dist
from libai.utils.file_utils import get_data_from_cache
from libai.utils.logger import setup_logger
from projects.MT5.configs.mt5_base import cfg as libai_cfg
from projects.MT5.mt5_model import MT5Model
from projects.MT5.utils.mt5_loader import T5LoaderHuggerFace

PRETRAINED_MODEL_URL = "http://oneflow-static.oss-cn-beijing.aliyuncs.com/ci-files/dataset/libai/model_utils_test/t5_utils/pytorch_model.bin"  # noqa
PRETRAINED_MODEL_CONFIG_URL = "http://oneflow-static.oss-cn-beijing.aliyuncs.com/ci-files/dataset/libai/model_utils_test/t5_utils/config.json"  # noqa

PRETRAINED_MODEL_MD5 = "952862a8ba425a25739a69e5f33b0df8"
PRETRAINED_MODEL_CONFIG_MD5 = "7ebc91dc4377c01190f4116c3c1ac6cd"

TEST_OUTPUT = os.path.join(os.getenv("TEST_OUTPUT", "output_unittest"), "test_t5_utils")


setup_logger(distributed_rank=dist.get_rank())


class TestT5Loader(flow.unittest.TestCase):
    def setUp(self) -> None:
        cache_dir = os.path.join(
            os.getenv("ONEFLOW_TEST_CACHE_DIR", "./data_test"), "t5_utils_data"
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
        self.encoder_input_ids = [
            [101, 2009, 1005, 1055, 2986, 2651, 1012, 102],
            [101, 2028, 12314, 3377, 102, 0, 0, 0],
            [101, 2064, 2017, 3305, 2009, 102, 0, 0],
        ]
        self.encoder_att_mask = [
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
        ]
        self.decoder_input_ids = [
            [101, 2009, 1005, 1055, 2986],
            [101, 2028, 12314, 3377, 102],
            [101, 2064, 2017, 3305, 2009],
        ]
        self.decoder_att_mask = [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]

    @classmethod
    def tearDownClass(cls) -> None:
        if os.path.isdir(TEST_OUTPUT) and dist.get_local_rank() == 0:
            shutil.rmtree(TEST_OUTPUT)

    @flow.unittest.skip_unless_1n4d()
    def test_t5_loader_with_data_tensor_parallel(self):
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
        load_func = T5LoaderHuggerFace(
            model=MT5Model,
            libai_cfg=libai_cfg,
            pretrained_model_path=self.pretrained_model_path,
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
            embedding_dropout_prob=0.0,
            model_type="t5",
        )
        model = load_func.load()
        model.eval()

        encoder_input_ids = flow.tensor(
            self.encoder_input_ids,
            dtype=flow.long,
            sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
            placement=dist.get_layer_placement(0),
        )
        decoder_input_ids = flow.tensor(
            self.decoder_input_ids,
            dtype=flow.long,
            sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
            placement=dist.get_layer_placement(0),
        )
        encode_att_mask = flow.tensor(
            self.encoder_att_mask,
            dtype=flow.bool,
            sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
            placement=dist.get_layer_placement(0),
        )
        decoder_att_mask = flow.tensor(
            self.decoder_att_mask,
            dtype=flow.bool,
            sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
            placement=dist.get_layer_placement(0),
        )

        logits = model(
            encoder_input_ids, decoder_input_ids, encode_att_mask, decoder_att_mask, encode_att_mask
        )["logits"]
        self.assertTrue(
            np.allclose(
                np.array(-9836561.0),
                logits.sum().data.numpy(),
            )
        )

    @flow.unittest.skip_unless_1n4d()
    def test_t5_loader_with_data_tensor_pipeline_parallel(self):
        # set distributed config
        dist_cfg = DictConfig(
            dict(
                data_parallel_size=2,
                tensor_parallel_size=1,
                pipeline_parallel_size=2,
                pipeline_num_layers=24,
            )
        )
        dist.setup_dist_util(dist_cfg)

        # load model
        load_func = T5LoaderHuggerFace(
            model=MT5Model,
            libai_cfg=libai_cfg,
            pretrained_model_path=self.pretrained_model_path,
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
            embedding_dropout_prob=0.0,
            model_type="t5",
        )
        model = load_func.load()
        model.eval()

        encoder_input_ids = flow.tensor(
            self.encoder_input_ids,
            dtype=flow.long,
            sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
            placement=dist.get_layer_placement(0),
        )
        decoder_input_ids = flow.tensor(
            self.decoder_input_ids,
            dtype=flow.long,
            sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
            placement=dist.get_layer_placement(0),
        )
        encode_att_mask = flow.tensor(
            self.encoder_att_mask,
            dtype=flow.bool,
            sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
            placement=dist.get_layer_placement(0),
        )
        decoder_att_mask = flow.tensor(
            self.decoder_att_mask,
            dtype=flow.bool,
            sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
            placement=dist.get_layer_placement(0),
        )

        logits = model(
            encoder_input_ids, decoder_input_ids, encode_att_mask, decoder_att_mask, encode_att_mask
        )["logits"]
        self.assertTrue(
            np.allclose(
                np.array(-9836561.0),
                logits.sum().data.numpy(),
            )
        )


if __name__ == "__main__":
    unittest.main()
