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
import tempfile
import unittest

import oneflow as flow
import oneflow.unittest
from flowvision.loss.cross_entropy import SoftTargetCrossEntropy

from libai.data.datasets import CIFAR10Dataset
from libai.config import LazyConfig, LazyCall
from libai.trainer import DefaultTrainer, hooks
from libai.trainer.default import _check_batch_size
import libai.utils.distributed as dist
from libai.utils.file_utils import get_data_from_cache
from libai.utils.logger import setup_logger
from libai.models import build_model

from configs.common.models.vit.vit_small_patch16_224 import model

DATA_URL = "https://oneflow-static.oss-cn-beijing.aliyuncs.com/ci-files/dataset/libai/cifar10/cifar-10-python.tar.gz"

DATA_MD5 = "c58f30108f718f92721af3b95e74349a"

setup_logger(distributed_rank=dist.get_rank())


class TestViTModel(flow.unittest.TestCase):
    def setUp(self) -> None:
        cache_dir = os.path.join(os.getenv("ONEFLOW_TEST_CACHE_DIR", "./data_test"), "vit_data")

        cfg = LazyConfig.load("configs/vit_imagenet.py")
        
        # set model
        cfg.model = model
        cfg.model.num_classes = 10
        cfg.model.depth = 1
        cfg.model.loss_func = LazyCall(SoftTargetCrossEntropy)()

        # prepare data path
        if dist.get_local_rank() == 0:
            get_data_from_cache(DATA_URL, cache_dir, md5=DATA_MD5)
        dist.synchronize()

        data_path = get_data_from_cache(DATA_URL, cache_dir, md5=DATA_MD5)
        
        cfg.dataloader.train.dataset[0]._target_ = CIFAR10Dataset
        cfg.dataloader.train.dataset[0].root = "/".join(data_path.split("/")[:3])
        cfg.dataloader.train.dataset[0].download = True

        # refine mixup cfg
        cfg.dataloader.train.mixup_func.num_classes = 10

        del cfg.dataloader.test

        # set training config
        cfg.train.train_epoch = 0
        cfg.train.train_iter = 10
        cfg.train.eval_period = 1000  # no test now
        cfg.train.log_period = 1
        cfg.train.train_micro_batch_size = 8
        cfg.train.num_accumulation_steps = 4
        cfg.train.resume = False
        cfg.train.output_dir = tempfile.mkdtemp()
        cfg.train.recompute_grad.enabled = True
        cfg.train.amp.enabled = True

        
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

        @classmethod
        def test(cls, cfg, test_loaders, model, evaluator=None):
            return {}

        DefaultTrainer.build_hooks = build_hooks
        DefaultTrainer.test = test

    @flow.unittest.skip_unless_1n4d()
    def test_vit_eager_with_data_tensor_parallel(self):
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
    def test_vit_graph_with_data_tensor_parallel(self):
        # FIXME(l1aoxingyu): add grad_acc in nn.Graph
        # now it will make loss to inf
        self.cfg.train.num_accumulation_steps = 1

        # set distributed config
        self.cfg.train.dist.data_parallel_size = 4
        # FIXME(l1aoxingyu): set tensor_parallel_size=2 when bugfix
        self.cfg.train.dist.tensor_parallel_size = 1
        self.cfg.train.dist.pipeline_parallel_size = 1

        dist.setup_dist_util(self.cfg.train.dist)
        _check_batch_size(self.cfg)

        self.cfg.graph.enabled = True
        trainer = DefaultTrainer(self.cfg)
        trainer.train()

if __name__ == "__main__":
    unittest.main()
