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


from trainer.detr_trainer import DetrEagerTrainer
from datasets.evaluation import inference_on_coco_dataset

from libai.config import instantiate, try_get_key
from libai.engine.default import DefaultTrainer
from libai.scheduler import build_lr_scheduler
from libai.utils import distributed as dist


class DetrDefaultTrainer(DefaultTrainer):
    def __init__(self, cfg):

        super().__init__(cfg)

        self.model.max_iter = cfg.train.train_iter
        self._trainer = DetrEagerTrainer(
            self.model, self.train_loader, self.optimizer, cfg.train.num_accumulation_steps
        )

    @classmethod
    def build_evaluator(cls, cfg):
        evaluator = instantiate(cfg.train.evaluation.evaluator)
        return evaluator

    @classmethod
    def test(cls, cfg, test_loaders, model, evaluator=None):
        test_batch_size = cfg.train.test_micro_batch_size * dist.get_data_parallel_size()
        evaluator = cls.build_evaluator(cfg) if not evaluator else evaluator
        results = inference_on_coco_dataset(
            model,
            test_loaders[0],
            test_batch_size,
            cfg.train.evaluation.eval_iter,
            cls.get_batch,
            evaluator,
        )

        return results

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`libai.scheduler.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        assert (
            try_get_key(cfg, "train.scheduler") is not None
        ), "cfg.train must contain `scheduler` namespace"
        if cfg.train.train_epoch:
            assert cfg.train.train_epoch > cfg.train.scheduler.step_size
            cfg.train.scheduler.step_size = (
                cfg.train.scheduler.step_size / cfg.train.train_epoch
            ) * cfg.train.train_iter
        return build_lr_scheduler(cfg.train.scheduler, optimizer)
