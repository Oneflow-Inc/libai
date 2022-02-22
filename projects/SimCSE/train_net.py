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
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import copy
from libai.config import LazyConfig, default_argument_parser, try_get_key
from libai.trainer import DefaultTrainer, default_setup
from libai.utils.checkpoint import Checkpointer
from libai.utils import distributed as dist
from libai.evaluation import DatasetEvaluator
from scipy.stats import spearmanr


def spearman_target(labels, cos_sim):
    return spearmanr(labels.cpu().numpy(), cos_sim.cpu().numpy()).correlation


class SimcseEvaluator(DatasetEvaluator):
    def __init__(self, cfg):
        self.cfg = cfg
        self._predictions = []

    def reset(self):
        self._predictions = []

    def process(self, inputs, outputs):
        cos_sim = outputs["cos_sim"]
        labels = inputs["labels"]                # [batch]  

        # measure spearman
        res = spearman_target(labels, cos_sim)

        self._predictions.append(
            {"spearman": res}
        )

    def evaluate(self):
        if not dist.is_main_process():
            return {}
        else:
            predictions = self._predictions
        total_spearman = 0
        total_samples = len(predictions)
        for prediction in predictions:
            total_spearman += prediction["spearman"]

        self._results = total_spearman / total_samples

        return {"spearman":self._results}
        

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
