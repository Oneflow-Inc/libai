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
import sys

from trainer.moco_trainer import MoCoEagerTrainer

from libai.config import LazyConfig, default_argument_parser, try_get_key
from libai.engine import DefaultTrainer, default_setup
from libai.utils.checkpoint import Checkpointer

sys.path.append(".")
logger = logging.getLogger(__name__)


class MoCoPretrainingTrainer(DefaultTrainer):
    def __init__(self, cfg):

        super().__init__(cfg)

        self.model.max_iter = cfg.train.train_iter

        self._trainer = MoCoEagerTrainer(
            self.model, self.train_loader, self.optimizer, cfg.train.num_accumulation_steps
        )


def main(args):
    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)

    if try_get_key(cfg, "graph.enabled") is True:
        raise NotImplementedError(
            "LiBai MOCO only support eager global mode now, please set cfg.graph.enabled=False"
        )

    default_setup(cfg, args)

    if args.fast_dev_run:
        cfg.train.train_epoch = 0
        cfg.train.train_iter = 20
        cfg.train.eval_period = 10
        cfg.train.log_period = 1

    if args.eval_only:
        tokenizer = None
        if try_get_key(cfg, "tokenization.setup", default=False):
            tokenizer = MoCoPretrainingTrainer.build_tokenizer(cfg)
        model = MoCoPretrainingTrainer.build_model(cfg)
        Checkpointer(model, save_dir=cfg.train.output_dir).resume_or_load(
            cfg.train.load_weight, resume=args.resume
        )
        if try_get_key(cfg, "train.graph.enabled", default=False):
            model = MoCoPretrainingTrainer.build_graph(cfg, model, is_train=False)
        test_loader = MoCoPretrainingTrainer.build_test_loader(cfg, tokenizer)
        _ = MoCoPretrainingTrainer.test(cfg, test_loader, model)
        return

    trainer = MoCoPretrainingTrainer(cfg)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    main(args)
