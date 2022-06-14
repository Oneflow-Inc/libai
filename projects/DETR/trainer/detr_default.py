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


import logging
from collections import OrderedDict

import oneflow as flow

from libai.data import Instance
from libai.data.structures import Instance
from libai.engine.default import DefaultTrainer
from libai.config import instantiate, try_get_key
from libai.data import Instance
from libai.models import  build_model
from libai.utils import distributed as dist

from trainer.detr_trainer import DetrEagerTrainer
from datasets.coco_eval import inference_on_coco_dataset
from modeling.backbone import FrozenBatchNorm2d
from utils.distributed import convert_to_distributed_default_setting


class DetrDefaultTrainer(DefaultTrainer):

    def __init__(self, cfg):

        super().__init__(cfg)

        self.model.max_iter = cfg.train.train_iter

        self._trainer = DetrEagerTrainer(
            self.model, self.train_loader, self.optimizer, cfg.train.num_accumulation_steps
        )

    @classmethod
    def get_batch(cls, data: Instance):
        """
        Convert batched local tensor to distributed tensor for model step running.
        """
        if isinstance(data, flow.utils.data._utils.worker.ExceptionWrapper):
            data.reraise()
            
        images = data.get_fields()["images"]
        labels = data.get_fields()["labels"]
        
        tensors = images[0] 
        tensors.to_global()
        
        mask = images[1] 
        mask.to_global()
        
        images = (tensors, mask)
        
        for i in range(len(labels)):
            for k, v in labels[i].items():
                labels[i][k] = v.to(device="cuda:0")
                
        ret_dict = {
            "images": images,
            "labels": labels
        }

        return ret_dict 
    
    @classmethod
    def build_evaluator(cls, cfg):
        evaluator = instantiate(cfg.train.evaluation.evaluator)
        return evaluator

    @classmethod
    def test(cls, cfg, test_loaders, model, evaluator=None):

        test_batch_size = cfg.train.test_micro_batch_size * dist.get_data_parallel_size()
        evaluator = cls.build_evaluator(cfg) if not evaluator else evaluator
        inference_on_coco_dataset(
            model,
            test_loaders[0],
            test_batch_size,
            cfg.train.evaluation.eval_iter,
            cls.get_batch,
            evaluator,
        )
            
    @classmethod
    def build_model(cls, cfg):
        """
        Returns:
            flow.nn.Module:

        It now calls :func:`libai.models.build_model`.
        Overwrite it if you'd like a different model.
        """
        assert try_get_key(cfg, "model") is not None, "cfg must contain `model` namespace"
        # Set model fp16 option because of embedding layer `white_identity` manual
        # insert for amp training if provided.
        if try_get_key(cfg.model, "cfg.amp_enabled") is not None:
            cfg.model.cfg.amp_enabled = cfg.train.amp.enabled and cfg.graph.enabled
        # In case some model define without cfg keyword.
        elif try_get_key(cfg.model, "amp_enabled") is not None:
            cfg.model.amp_enabled = cfg.train.amp.enabled and cfg.graph.enabled
        model = build_model(cfg.model)
        logger = logging.getLogger(__name__)
        logger.info("Model:\n{}".format(model))

        model.apply(convert_to_distributed_default_setting)

        return model
