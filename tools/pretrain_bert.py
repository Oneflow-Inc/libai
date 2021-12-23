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
from libai.config import LazyConfig, instantiate, default_argument_parser

from libai.trainer import DefaultTrainer, default_setup


def setup():
    """
    Create configs and perform basic setups.
    """
    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)

    default_setup(cfg)
    return cfg


def do_train(cfg):
    model = instantiate(cfg.model)
    print(model)

    for i in range(cfg.train.max_iter):
        print(i)


def main(args):
    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    default_setup(cfg, args)

    do_train(cfg)


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    main(args)
