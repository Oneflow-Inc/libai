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

import tempfile
import unittest

from libai.config import LazyConfig
from libai.trainer import DefaultTrainer, hooks
from libai.trainer.default import _check_batch_size
from libai.utils import distributed as dist


class TestBertModel(unittest.TestCase):
    def setUp(self) -> None:
        cfg = LazyConfig.load("configs/bert_large_pretrain.py")

        # set data path
        # prepare dataset

        cfg.tokenization.tokenizer.vocab_file = (
            "/workspace/dataset/bert_data/bert-base-chinese-vocab.txt"
        )
        cfg.dataloader.train.dataset[
            0
        ].data_prefix = "/workspace/dataset/bert_data/loss_compara_content_sentence"
        cfg.dataloader.train.dataset[
            0
        ].indexed_dataset.data_prefix = "/workspace/dataset/bert_data/loss_compara_content_sentence"

        # set training config
        cfg.train.train_epoch = 0
        cfg.train.train_iter = 100
        cfg.train.eval_period = 1000
        cfg.train.log_period = 1
        cfg.train.resume = False
        cfg.train.output_dir = tempfile.mkdtemp()

        # set distributed config
        cfg.train.dist.data_parallel_size = 1
        cfg.train.dist.tensor_parallel_size = 1
        cfg.train.dist.pipeline_parallel_size = 1
        cfg.train.dist.pipeline_num_layers = 100

        # set model
        cfg.model.cfg.num_attention_heads = 8
        cfg.model.cfg.hidden_size = 384
        cfg.model.cfg.hidden_layers = 4
        cfg.train.recompute_grad.enabled = True

        dist.setup_dist_util(cfg.train.dist)
        _check_batch_size(cfg)

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

    def test_bert_eager(self):
        self.cfg.graph.enabled = False
        trainer = DefaultTrainer(self.cfg)
        trainer.train()

    def test_bert_graph(self):
        self.cfg.graph.enabled = False
        trainer = DefaultTrainer(self.cfg)
        trainer.train()


if __name__ == "__main__":
    unittest.main()
