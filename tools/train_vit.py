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

import sys

import oneflow as flow

sys.path.append(".")
from libai.config import LazyConfig, default_argument_parser
from libai.utils import distributed as dist
from libai.utils.checkpoint import Checkpointer

from libai.trainer import DefaultTrainer, default_setup


def get_batch(data_iterator):
    # use fake data for testing
    data = flow.randn(16, 3, 384, 384).to("cuda")
    label = flow.randn(16, 1000)
    data = data.to_consistent(
        sbp=flow.sbp.split(0), placement=flow.env.all_device_placement("cuda")
    )
    label = label.to_consistent(
        sbp=flow.sbp.split(0), placement=flow.env.all_device_placement("cuda")
    )
    return (data, label)


class Trainer(DefaultTrainer):
    @classmethod
    def build_train_valid_test_loader(cls, cfg):
        # TODO: switch to real data
        return [], [], []

    def run_step(self):
        return super().run_step(get_batch)


def main(args):
    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    default_setup(cfg, args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        Checkpointer(model, save_dir=cfg.train.output_dir).resume_or_load(
            cfg.train.load_weight, resume=args.resume
        )
        graph = Trainer.build_graph(cfg, model, is_train=False)
        res = Trainer.test(cfg, graph)

    trainer = Trainer(cfg)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    main(args)