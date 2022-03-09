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
import oneflow.unittest

import oneflow as flow

from libai.config import LazyConfig
from libai.engine import DefaultTrainer
from libai.engine.default import _check_batch_size
from libai.utils import distributed as dist
from libai.utils.file_utils import get_data_from_cache
from libai.utils.logger import setup_logger

VOCAB_URL = "https://oneflow-static.oss-cn-beijing.aliyuncs.com/ci-files/dataset/libai/bert_dataset/bert-base-chinese-vocab.txt"  # noqa
DEMO_TSV_URL = "https://oneflow-static.oss-cn-beijing.aliyuncs.com/ci-files/dataset/libai/QQP/demo.tsv"  # noqa

VOCAB_MD5 = "3b5b76c4aef48ecf8cb3abaafe960f09"
DEMO_TSV_MD5 = "e3f900e8724646234aacf758dd669d4d"
TEST_OUTPUT = "output_unittest/test_qqp"

setup_logger(distributed_rank=dist.get_rank())


class TestQQPModel(flow.unittest.TestCase):
    def setUp(self) -> None:
        cache_dir = os.path.join(os.getenv("ONEFLOW_TEST_CACHE_DIR", "./data_test"), "qqp_data")

        cfg = LazyConfig.load("projects/QQP/configs/config_qqp.py")

        # prepare dataset
        if dist.get_local_rank() == 0:
            # download dataset on main process of each node
            get_data_from_cache(VOCAB_URL, cache_dir, md5=VOCAB_MD5)
            get_data_from_cache(DEMO_TSV_URL, cache_dir, md5=DEMO_TSV_MD5)
            os.makedirs(TEST_OUTPUT, exist_ok=True)
        dist.synchronize()

        vocab_path = get_data_from_cache(VOCAB_URL, cache_dir, md5=VOCAB_MD5)
        demo_tsv_path = get_data_from_cache(DEMO_TSV_URL, cache_dir, md5=DEMO_TSV_MD5)

        # set tokenizer and data config
        cfg.tokenization.tokenizer.vocab_file = vocab_path
        cfg.dataloader.train.dataset[0].data_paths = [demo_tsv_path]
        cfg.dataloader.test[0].dataset.data_paths = [demo_tsv_path]
        # FIXME(RenTianhe): fix dataloader worker bug
        cfg.dataloader.train.num_workers = 0

        # set training config
        cfg.train.train_epoch = 0
        cfg.train.train_iter = 10
        cfg.train.eval_period = 5
        cfg.train.log_period = 1
        cfg.train.train_micro_batch_size = 2
        cfg.train.num_accumulation_steps = 1
        cfg.train.resume = False
        cfg.train.output_dir = TEST_OUTPUT     

        # set model
        cfg.model.cfg.num_attention_heads = 8
        cfg.model.cfg.hidden_size = 384
        cfg.model.cfg.hidden_layers = 4
        cfg.model.cfg.pretrain_megatron_weight = None
        cfg.train.activation_checkpoint.enabled = True
        cfg.train.amp.enabled = True

        self.cfg = cfg

    @flow.unittest.skip_unless_1n4d()
    def test_qqp_eager_with_data_tensor_parallel(self):
        # set distributed config
        self.cfg.train.dist.data_parallel_size = 2
        self.cfg.train.dist.tensor_parallel_size = 2
        # pipeline parallelism not supported in eager global now!
        self.cfg.train.dist.pipeline_parallel_size = 1

        dist.setup_dist_util(self.cfg.train.dist)
        _check_batch_size(self.cfg)

        self.cfg.graph.enabled = False
        trainer = DefaultTrainer(self.cfg)
        trainer.train()

    @flow.unittest.skip_unless_1n4d()
    def test_qqp_graph_with_data_tensor_parallel(self):
        self.cfg.train.num_accumulation_steps = 1
        # set distributed config
        self.cfg.train.dist.data_parallel_size = 2
        self.cfg.train.dist.tensor_parallel_size = 2
        self.cfg.train.dist.pipeline_parallel_size = 1

        dist.setup_dist_util(self.cfg.train.dist)
        _check_batch_size(self.cfg)

        self.cfg.graph.enabled = True
        trainer = DefaultTrainer(self.cfg)
        trainer.train()

    @flow.unittest.skip_unless_1n4d()
    @unittest.skip("There are still bugs in ZeRO")
    def test_qqp_with_zero(self):
        # set distributed config
        self.cfg.train.dist.data_parallel_size = 4
        self.cfg.train.dist.tensor_parallel_size = 1
        self.cfg.train.dist.pipeline_parallel_size = 1

        dist.setup_dist_util(self.cfg.train.dist)
        _check_batch_size(self.cfg)

        self.cfg.graph.enabled = True
        self.cfg.train.zero_optimization.enabled = True
        self.cfg.train.zero_optimization.stage = 3
        trainer = DefaultTrainer(self.cfg)
        trainer.train()


if __name__ == "__main__":
    unittest.main()
    if os.path.exists(TEST_OUTPUT):
        os.rmdir(TEST_OUTPUT)
        