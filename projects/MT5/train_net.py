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
import os
import random
import sys

import numpy as np
import oneflow as flow

from libai.config import LazyConfig, default_argument_parser, try_get_key
from libai.engine import DefaultTrainer, default_setup
from libai.utils.checkpoint import Checkpointer
from libai.utils.events import JSONWriter, TensorboardXWriter
from projects.MT5.utils.mt5_metrc_printer import MT5MetricPrinter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))


logger = logging.getLogger("libai." + __name__)


class Mt5Trainer(DefaultTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)

    def build_writers(self):
        """
        Build a list of writers to be used. By default it contains
        writers that write metrics to the screen,
        a json file, and a tensorboard event file respectively.
        If you'd like a different list of writers, you can overwrite it in
        your trainer.

        Returns:
            list[EventWriter]: a list of :class:`EventWriter` objects.

        It is now implemented by:

        .. code-block:: python

            return [
                MT5MetricPrinter(self.global_batch_size, self.max_iter),
                JSONWriter(os.path.join(self.cfg.train.output_dir, "metrics.json")),
                TensorboardXWriter(self.cfg.train.output_dir),
            ]
        """
        # Assume the default print/log frequency.
        return [
            # It may not always print what you want to see, since it prints "common" metrics only.
            MT5MetricPrinter(self.global_batch_size, self.max_iter, self.cfg.train.log_period),
            JSONWriter(os.path.join(self.cfg.train.output_dir, "metrics.json")),
            TensorboardXWriter(self.cfg.train.output_dir),
        ]


def main(args):
    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    default_setup(cfg, args)

    seed_for_rank = cfg.train.seed + flow.env.get_rank()
    flow.manual_seed(seed_for_rank)
    flow.cuda.manual_seed(seed_for_rank)
    np.random.seed(seed_for_rank)
    random.seed(seed_for_rank)

    if args.fast_dev_run:
        cfg.train.train_epoch = 0
        cfg.train.train_iter = 20
        cfg.train.evaluation.eval_period = 10
        cfg.train.log_period = 1

    if args.eval_only:
        tokenizer = None
        if try_get_key(cfg, "tokenization") is not None:
            tokenizer = Mt5Trainer.build_tokenizer(cfg)
        model = Mt5Trainer.build_model(cfg)
        Checkpointer(model, save_dir=cfg.train.output_dir).resume_or_load(
            cfg.train.load_weight, resume=args.resume
        )
        if try_get_key(cfg, "train.graph.enabled", default=False):
            model = Mt5Trainer.build_graph(cfg, model, is_train=False)
        test_loader = Mt5Trainer.build_test_loader(cfg, tokenizer)
        if len(test_loader) == 0:
            logger.info("No dataset in dataloader.test, please set dataset for dataloader.test")
        _ = Mt5Trainer.test(cfg, test_loader, model)
        return

    trainer = Mt5Trainer(cfg)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    main(args)
