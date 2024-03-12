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
import argparse
import sys

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)
from libai.config import LazyConfig
from libai.engine import DefaultTrainer, default_setup
from libai.remat import remat_argument_parser, RematTrainer
from libai.utils import distributed as dist

logger = logging.getLogger("libai." + __name__)


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
        cfg.train.train_iter = 10
        cfg.train.evaluation.enabled = False
        cfg.train.log_period = 1
        cfg.train.input_placement_device = "cuda+remat"

    trainer = RematTrainer(cfg)
    return trainer.train()


if __name__ == "__main__":
    dist.set_device_type("cuda+remat")
    args = remat_argument_parser().parse_args()
    flow.remat.set_budget(f"{args.threshold}MB")
    flow.remat.set_small_pieces_optimization(False)
    main(args)
