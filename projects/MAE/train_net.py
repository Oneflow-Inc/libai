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
import logging

sys.path.append(".")

import oneflow as flow

from libai.config import LazyConfig, default_argument_parser, try_get_key
from libai.engine import DefaultTrainer, default_setup
from libai.utils.checkpoint import Checkpointer
from libai.optim import build_optimizer
from utils.weight_convert import load_torch_checkpoint
from utils.lr_decay import param_groups_lrd


logger = logging.getLogger(__name__)


class Trainer(DefaultTrainer):
    @classmethod
    def build_model(cls, cfg):
        assert try_get_key(cfg, "graph.enabled") == False, "LiBai MAE only support eager global mode now, please set cfg.graph.enabled=False"

        model = super().build_model(cfg)
        if try_get_key(cfg, "finetune") is not None:
            if cfg.finetune.enable == True:
                logger.info("Loading pretrained weight for finetuning")
                assert cfg.finetune.weight_style in ["oneflow", "pytorch"]
                if cfg.finetune.weight_style == "oneflow":
                    model.load_state_dict(flow.load(cfg.finetune.path))
                else:
                    model = load_torch_checkpoint(model, path=cfg.finetune.path)
        return model
    
    @classmethod
    def build_optimizer(cls, cfg, model):
        if try_get_key(cfg, "train.layer_decay") is not None:
            param_groups = param_groups_lrd(
                model, 
                weight_decay=cfg.optim.weight_decay, 
                no_weight_decay_list=model.no_weight_decay(), 
                layer_decay=cfg.train.layer_decay
            )
            optim = flow.optim.AdamW(
                parameters=param_groups,
                lr=cfg.optim.lr,
                weight_decay=cfg.optim.weight_decay,
                betas=cfg.optim.betas,
                eps=cfg.optim.eps,
                do_bias_correction=cfg.optim.do_bias_correction
            )
            return optim
        return build_optimizer(cfg.optim, model)


DefaultTrainer = Trainer

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
        graph = DefaultTrainer.build_graph(cfg, model, is_train=False)
        test_loader = DefaultTrainer.build_test_loader(cfg, tokenizer)
        res = DefaultTrainer.test(cfg, test_loader, graph)  # noqa
        return

    trainer = DefaultTrainer(cfg)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    main(args)
