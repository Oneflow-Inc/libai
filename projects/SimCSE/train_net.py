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


import copy
import os
import sys
from collections import OrderedDict

import numpy as np
from scipy.stats import spearmanr

import oneflow as flow

from libai.config import LazyConfig, default_argument_parser, try_get_key
from libai.evaluation import DatasetEvaluator
from libai.trainer import DefaultTrainer, default_setup
from libai.utils import distributed as dist
from libai.utils.checkpoint import Checkpointer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))


def spearman_target(cos_sim, labels):
    return spearmanr(cos_sim, labels).correlation


class SimcseEvaluator(DatasetEvaluator):
    def __init__(self, cfg):
        self.cfg = cfg
        self._predictions = []

    def reset(self):
        self._predictions = []

    def process(self, inputs, outputs):
        # outputs: model's out
        # inputs: model's input
        cos_sim = outputs["cos_sim"]  # [batch]
        labels = outputs["labels"]  # [batch]
        ids_model = outputs["ids"]
        ids = inputs["input_ids"]
        self._predictions.append({"cos_sim": cos_sim, "labels": labels, "input_ids": ids, "model_ids":ids_model})

    def evaluate(self):
        if not dist.is_main_process():
            return {}
        else:
            predictions = self._predictions

        sim_array = np.array([])
        label_array = np.array([])
        ids_array = np.array([])
        ids_model = np.array([])

        for prediction in predictions:
            sim_array = np.append(sim_array, np.array(prediction["cos_sim"].cpu().numpy()))
            label_array = np.append(label_array, np.array(prediction["labels"].cpu().numpy()))
            ids_array = np.append(ids_array, np.array(prediction["input_ids"].cpu().numpy()))
            ids_model = np.append(ids_array, np.array(prediction["model_ids"].cpu().numpy()))
        
        np.savetxt('/home/xiezipeng/libai/projects/SimCSE/result/ids_array_test.txt', ids_array)
        np.savetxt('/home/xiezipeng/libai/projects/SimCSE/result/ids_array_model.txt', ids_model)


        self._results = spearman_target(sim_array, label_array)
        return {"spearman": self._results}


class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg):
        return SimcseEvaluator(cfg)


def main(args):
    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    default_setup(cfg, args)

    if args.fast_dev_run:
        cfg.train.train_epoch = 0
        cfg.train.train_iter = 20
        cfg.train.eval_period = 10
        cfg.train.log_period = 1

    if args.eval_only:
        tokenizer = None
        if try_get_key(cfg, "tokenization.setup", default=False):
            tokenizer = Trainer.build_tokenizer(cfg)
        model = Trainer.build_model(cfg)
        Checkpointer(model, save_dir=cfg.train.output_dir).resume_or_load(
            cfg.train.load_weight, resume=args.resume
        )
        graph = Trainer.build_graph(cfg, model, is_train=False)
        test_loader = Trainer.build_test_loader(cfg, tokenizer)
        res = Trainer.test(cfg, test_loader, graph)  # noqa
        return

    trainer = Trainer(cfg)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    main(args)
