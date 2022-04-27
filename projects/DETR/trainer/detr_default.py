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
from typing import Callable, Optional
import logging
import math
import os
import time
from collections import OrderedDict
from omegaconf import OmegaConf
from termcolor import colored

import oneflow as flow

from libai.data import Instance
from libai.config.configs.common.data.coco import NestedTensor
from libai.data.structures import DistTensorData, Instance
from libai.engine.default import DefaultTrainer
from libai.config import LazyConfig, instantiate, try_get_key
from libai.data import Instance
from libai.engine import hooks
from libai.engine.trainer import EagerTrainer, GraphTrainer, TrainerBase
from libai.evaluation import print_csv_format
from libai.models import build_graph, build_model
from libai.optim import build_optimizer
from libai.scheduler import build_lr_scheduler
from libai.tokenizer import build_tokenizer
from libai.utils import distributed as dist
from libai.utils.checkpoint import Checkpointer
from libai.utils.events import CommonMetricPrinter, JSONWriter, TensorboardXWriter
from libai.utils.logger import setup_logger

from trainer.detr_trainer import DetrEagerTrainer
from datasets.coco_eval import inference_on_coco_dataset


class DetrDefaultTrainer(DefaultTrainer):

    def __init__(self, cfg):

        super().__init__(cfg)

        self.model.max_iter = cfg.train.train_iter

        self._trainer = DetrEagerTrainer(
            self.model, self.train_loader, self.optimizer, cfg.train.num_accumulation_steps
        )

    @classmethod
    def get_batch(cls, data: Instance, mixup_func: Optional[Callable] = None):
        """
        Convert batched local tensor to distributed tensor for model step running.

        If you want to do something with batched data before model, (e.g. mixup),
        you can rewrite this function.
        """
        if isinstance(data, flow.utils.data._utils.worker.ExceptionWrapper):
            data.reraise()

        # TODO: impl the mixup_func
        # if mixup_func is not None:
        #     images, labels = mixup_func(
        #         data.get("images").tensor.cuda(),
        #         data.get("labels").tensor.cuda(),
        #     )
        #     data.get("images").tensor = images
        #     data.get("labels").tensor = labels

        images, labels = data

        tensors = DistTensorData(images.tensors, placement_idx=-1)
        tensors.to_global()

        mask = DistTensorData(images.mask, placement_idx=-1)
        mask.to_global()

        images = NestedTensor(tensors,mask)
        # TDOO: refine the code. to DistTensorData func should impl in class CocoDetection
        for i in range(len(labels)):
            for k,v in labels[i].items():
                labels[i][k] = DistTensorData(flow.tensor(v), placement_idx=-1)
                labels[i][k].to_global()
            
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
        """
        Evaluate the given model. The given model is expected to already contain
        weights to evaluate.

        Args:
            cfg (CfgNode):
            test_loaders: list [dataloader1, dataloader2, ...]
            model (nn.Graph):
            evaluators (list[DatasetEvaluator] or None): if None, will call
                :meth:`build_evaluator`. Otherwise, must have the same length as
                ``cfg.DATASETS.TEST``.

        Returns:
            dict: a dict of result metrics
        """
        logger = logging.getLogger(__name__)
        # TODO: support multi evaluator
        # if isinstance(evaluators, DatasetEvaluator):
        #     evaluators = [evaluators]
        test_batch_size = cfg.train.test_micro_batch_size * dist.get_data_parallel_size()
        evaluator = cls.build_evaluator(cfg) if not evaluator else evaluator

        results = OrderedDict()
        for idx, data_loader in enumerate(test_loaders):
            # When evaluators are passed in as arguments,
            # implicitly assume that evaluators can be created before data_loader.
            dataset_name = type(data_loader.dataset).__name__
            import pdb
            pdb.set_trace()
            results_i = inference_on_coco_dataset(
                model,
                data_loader,
                test_batch_size,
                cfg.train.evaluation.eval_iter,
                cls.get_batch,
                evaluator,
            )
            results[dataset_name] = results_i
            if dist.is_main_process():
                assert isinstance(
                    results_i, dict
                ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                    results_i
                )
                logger.info(
                    "Evaluation results for {} in csv format:".format(
                        colored(dataset_name, "green")
                    )
                )
                print_csv_format(results_i)

        if len(results) == 1:
            results = list(results.values())[0]
        return results