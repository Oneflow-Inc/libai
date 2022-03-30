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


from ast import arg
import sys
import logging

sys.path.append(".")

from libai.config import LazyConfig, default_argument_parser, try_get_key
from libai.engine import default_setup, DefaultTrainer
from libai.utils.checkpoint import Checkpointer
from trainer.MoCo_v3_trainer import MoCoEagerTrainer


logger = logging.getLogger(__name__)


class MoCoPretrainingTrainer(DefaultTrainer):

    def __init__(self, cfg):

        super().__init__(cfg)

        self.model.max_iter = cfg.train.train_iter

        self._trainer = MoCoEagerTrainer(
                self.model, self.train_loader, self.optimizer, cfg.train.num_accumulation_steps
            )

DefaultTrainer = MoCoPretrainingTrainer


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
            tokenizer = DefaultTrainer.build_tokenizer(cfg)
        model = DefaultTrainer.build_model(cfg)
        Checkpointer(model, save_dir=cfg.train.output_dir).resume_or_load(
            cfg.train.load_weight, resume=args.resume
        )
        if try_get_key(cfg, "train.graph.enabled", default=False):
            model = DefaultTrainer.build_graph(cfg, model, is_train=False)
        test_loader = DefaultTrainer.build_test_loader(cfg, tokenizer)
        _ = DefaultTrainer.test(cfg, test_loader, model)
        return

    trainer = DefaultTrainer(cfg)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    main(args)
