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
from flowvision.loss.cross_entropy import SoftTargetCrossEntropy

import libai.utils.distributed as dist
from configs.common.models.swinv2.swinv2_tiny_patch4_window8_256 import model
from libai.config import LazyConfig
from libai.data.datasets import CIFAR10Dataset
from libai.engine import DefaultTrainer
from libai.engine.default import _check_batch_size
from libai.utils.file_utils import get_data_from_cache
from libai.utils.logger import setup_logger

DATA_URL = "https://oneflow-static.oss-cn-beijing.aliyuncs.com/ci-files/dataset/libai/cifar10/cifar-10-python.tar.gz"  # noqa

DATA_MD5 = "c58f30108f718f92721af3b95e74349a"

TEST_OUTPUT = os.path.join(os.getenv("TEST_OUTPUT", "output_unittest"), "test_swinv2")

setup_logger(distributed_rank=dist.get_rank())


class TestSwinV2Model(flow.unittest.TestCase):
    def setUp(self) -> None:
        cache_dir = os.path.join(os.getenv("ONEFLOW_TEST_CACHE_DIR", "./data_test"), "swinv2_data")

        cfg = LazyConfig.load("configs/swinv2_cifar100.py")

        # set model
        cfg.model = model
        cfg.model.cfg.num_classes = 10
        cfg.model.cfg.loss_func = SoftTargetCrossEntropy()

        # prepare data path
        if dist.get_local_rank() == 0:
            get_data_from_cache(DATA_URL, cache_dir, md5=DATA_MD5)
            os.makedirs(TEST_OUTPUT, exist_ok=True)
        dist.synchronize()

        data_path = get_data_from_cache(DATA_URL, cache_dir, md5=DATA_MD5)

        cfg.dataloader.train.dataset[0]._target_ = CIFAR10Dataset
        cfg.dataloader.train.dataset[0].root = "/".join(data_path.split("/")[:-1])
        cfg.dataloader.train.dataset[0].download = True
        cfg.dataloader.train.num_workers = 0

        cfg.dataloader.test[0].dataset._target_ = CIFAR10Dataset
        cfg.dataloader.test[0].dataset.train = False
        cfg.dataloader.test[0].dataset.root = "/".join(data_path.split("/")[:-1])
        cfg.dataloader.test[0].dataset.download = True
        cfg.dataloader.test[0].num_workers = 0

        # refine mixup cfg
        cfg.dataloader.train.mixup_func.num_classes = 10

        # set training config
        cfg.train.train_epoch = 0
        cfg.train.train_iter = 10
        cfg.train.evaluation.eval_period = 10
        cfg.train.evaluation.eval_iter = 10
        cfg.train.log_period = 1
        cfg.train.train_micro_batch_size = 8
        cfg.train.num_accumulation_steps = 1
        cfg.train.resume = False
        cfg.train.output_dir = TEST_OUTPUT
        cfg.train.activation_checkpoint.enabled = True
        cfg.train.amp.enabled = True
        cfg.train.rdma_enabled = False

        self.cfg = cfg

    @classmethod
    def tearDownClass(cls) -> None:
        if os.path.isdir(TEST_OUTPUT) and dist.get_local_rank() == 0:
            shutil.rmtree(TEST_OUTPUT)

    @flow.unittest.skip_unless_1n4d()
    def test_swinv2_eager_with_data_tensor_parallel(self):
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
    def test_swinv2_eager_with_pipeline_parallel(self):
        # set distributed config
        self.cfg.train.dist.data_parallel_size = 1
        self.cfg.train.dist.tensor_parallel_size = 1
        self.cfg.train.dist.pipeline_parallel_size = 4
        self.cfg.train.dist.pipeline_num_layers = sum(self.cfg.model.cfg.depths)

        dist.setup_dist_util(self.cfg.train.dist)
        _check_batch_size(self.cfg)

        self.cfg.graph.enabled = False
        trainer = DefaultTrainer(self.cfg)
        trainer.train()

    @flow.unittest.skip_unless_1n4d()
    def test_swinv2_graph_with_data_tensor_parallel(self):
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
    def test_swinv2_graph_with_data_tensor_pipeline_parallel(self):
        self.cfg.train.num_accumulation_steps = 4
        # set distributed config
        self.cfg.train.dist.data_parallel_size = 2
        # change to 2 when 2d sbp bugfix
        self.cfg.train.dist.tensor_parallel_size = 1
        self.cfg.train.dist.pipeline_parallel_size = 2
        self.cfg.train.dist.pipeline_num_layers = sum(self.cfg.model.cfg.depths)

        dist.setup_dist_util(self.cfg.train.dist)
        _check_batch_size(self.cfg)

        self.cfg.graph.enabled = True
        trainer = DefaultTrainer(self.cfg)
        trainer.train()

    @flow.unittest.skip_unless_1n4d()
    def test_swinv2_with_zero(self):
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
