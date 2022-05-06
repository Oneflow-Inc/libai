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

from utils.weight_convert import load_torch_checkpoint

from libai.config import LazyConfig, default_argument_parser, try_get_key
from libai.engine import DefaultTrainer, default_setup
from libai.utils.checkpoint import Checkpointer

sys.path.append(".")
logger = logging.getLogger(__name__)


class Trainer(DefaultTrainer):
    @classmethod
    def build_model(cls, cfg):
        if try_get_key(cfg, "graph.enabled") is True:
            raise NotImplementedError(
                "LiBai MAE only support eager global mode now, please set cfg.graph.enabled=False"  # noqa
            )

        model = super().build_model(cfg)
        if try_get_key(cfg, "finetune") is not None:
            if cfg.finetune.enable is True:
                logger.info("Loading pretrained weight for finetuning")
                assert cfg.finetune.weight_style in ["oneflow", "pytorch"]
                if cfg.finetune.weight_style == "oneflow":
                    Checkpointer(model).load(cfg.finetune.path)
                elif cfg.finetune.weight_style == "pytorch":
                    model = load_torch_checkpoint(model, cfg, path=cfg.finetune.path, strict=False)
                else:
                    raise NotImplementedError(
                        "Only support loading oneflow & pytorch pretrained weight now."
                    )
        return model


def main(args):
    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    default_setup(cfg, args)

    if args.fast_dev_run:
        cfg.train.train_epoch = 0
        cfg.train.checkpointer.period = 5
        cfg.train.train_iter = 10
        cfg.train.evaluation.eval_period = 10
        cfg.train.log_period = 1

    if args.eval_only:
        tokenizer = None
        if try_get_key(cfg, "tokenization.setup", default=False):
            tokenizer = Trainer.build_tokenizer(cfg)
        model = Trainer.build_model(cfg)
        Checkpointer(model, save_dir=cfg.train.output_dir).resume_or_load(
            cfg.train.load_weight, resume=args.resume
        )
        if try_get_key(cfg, "train.graph.enabled", default=False):
            model = Trainer.build_graph(cfg, model, is_train=False)
        test_loader = Trainer.build_test_loader(cfg, tokenizer)
        if len(test_loader) == 0:
            logger.info("No dataset in dataloader.test, please set dataset for dataloader.test")
        _ = Trainer.test(cfg, test_loader, model)
        return

    trainer = Trainer(cfg)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    main(args)
