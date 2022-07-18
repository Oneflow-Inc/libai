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

import oneflow as flow
import oneflow.unittest

from libai.config import LazyConfig
from libai.engine import DefaultTrainer, hooks
from libai.engine.default import _check_batch_size
from libai.utils import distributed as dist
from libai.utils.file_utils import get_data_from_cache
from libai.utils.logger import setup_logger

VOCAB_URL = "https://oneflow-static.oss-cn-beijing.aliyuncs.com/ci-files/dataset/libai/bert_dataset/bert-base-chinese-vocab.txt"  # noqa
BIN_DATA_URL = "https://oneflow-static.oss-cn-beijing.aliyuncs.com/ci-files/dataset/libai/bert_dataset/loss_compara_content_sentence.bin"  # noqa
IDX_DATA_URL = "https://oneflow-static.oss-cn-beijing.aliyuncs.com/ci-files/dataset/libai/bert_dataset/loss_compara_content_sentence.idx"  # noqa

VOCAB_MD5 = "3b5b76c4aef48ecf8cb3abaafe960f09"
BIN_DATA_MD5 = "b842467bd5ea7e52f7a612ea6b4faecc"
IDX_DATA_MD5 = "cf5963b8543f0a7a867361eb980f0372"

TEST_OUTPUT = os.path.join(os.getenv("TEST_OUTPUT", "output_unittest"), "test_t5")


setup_logger(distributed_rank=dist.get_rank())


class TestT5Model(flow.unittest.TestCase):
    def setUp(self) -> None:
        cache_dir = os.path.join(os.getenv("ONEFLOW_TEST_CACHE_DIR", "./data_test"), "bert_data")

        cfg = LazyConfig.load("configs/t5_large_pretrain.py")

        # prepare dataset
        if dist.get_local_rank() == 0:
            # download dataset on main process of each node
            get_data_from_cache(VOCAB_URL, cache_dir, md5=VOCAB_MD5)
            get_data_from_cache(BIN_DATA_URL, cache_dir, md5=BIN_DATA_MD5)
            get_data_from_cache(IDX_DATA_URL, cache_dir, md5=IDX_DATA_MD5)
            os.makedirs(TEST_OUTPUT, exist_ok=True)
        dist.synchronize()

        vocab_path = get_data_from_cache(VOCAB_URL, cache_dir, md5=VOCAB_MD5)
        data_prefix_path = get_data_from_cache(BIN_DATA_URL, cache_dir, md5=BIN_DATA_MD5)
        data_prefix = data_prefix_path[:-4]

        # set tokenizer and data config
        cfg.tokenization.tokenizer.vocab_file = vocab_path
        cfg.dataloader.train.dataset[0].data_prefix = data_prefix
        cfg.dataloader.train.dataset[0].indexed_dataset.data_prefix = data_prefix
        # FIXME(RenTianhe): fix dataloader worker bug
        cfg.dataloader.train.num_workers = 0

        # set training config
        cfg.train.train_epoch = 0
        cfg.train.train_iter = 10
        cfg.train.evaluation.enabled = True
        cfg.train.evaluation.eval_period = 10
        cfg.train.evaluation.eval_iter = 10
        cfg.train.log_period = 1
        cfg.train.train_micro_batch_size = 8
        cfg.train.num_accumulation_steps = 1
        cfg.train.resume = False
        cfg.train.output_dir = TEST_OUTPUT

        # set model
        cfg.model.cfg.num_attention_heads = 8
        cfg.model.cfg.hidden_size = 384
        cfg.model.cfg.hidden_layers = 3
        cfg.train.activation_checkpoint.enabled = True
        cfg.train.amp.enabled = True

        cfg.train.rdma_enabled = False

        self.cfg = cfg

        def build_hooks(self):
            ret = [
                hooks.IterationTimer(),
                hooks.LRScheduler(),
            ]

            if dist.is_main_process():
                # run writers in the end, so that evaluation metrics are written
                ret.append(hooks.PeriodicWriter(self.build_writers(), self.cfg.train.log_period))
            return ret

        DefaultTrainer.build_hooks = build_hooks

    @classmethod
    def tearDownClass(cls) -> None:
        if os.path.isdir(TEST_OUTPUT) and dist.get_local_rank() == 0:
            shutil.rmtree(TEST_OUTPUT)

    @flow.unittest.skip_unless_1n4d()
    def test_t5_eager_with_data_tensor_parallel(self):
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
    def test_t5_graph_with_data_tensor_parallel(self):
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
    def test_t5_graph_with_data_tensor_pipeline_parallel(self):
        self.cfg.train.num_accumulation_steps = 4
        # set distributed config
        self.cfg.train.dist.data_parallel_size = 2
        # change to 2 when 2d sbp bugfix
        self.cfg.train.dist.tensor_parallel_size = 1
        self.cfg.train.dist.pipeline_parallel_size = 2
        # encoder_layers + decoder_layers
        self.cfg.train.dist.pipeline_num_layers = 2 * self.cfg.model.cfg.hidden_layers

        dist.setup_dist_util(self.cfg.train.dist)
        _check_batch_size(self.cfg)

        self.cfg.graph.enabled = True
        trainer = DefaultTrainer(self.cfg)
        trainer.train()

    @flow.unittest.skip_unless_1n4d()
    @unittest.skip("There are still bugs in ZeRO")
    def test_t5_with_zero(self):
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
