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

sys.path.append(".")

import oneflow as flow

from libai.config import LazyConfig, default_argument_parser, try_get_key
from libai.engine import DefaultTrainer, default_setup
from libai.utils.checkpoint import Checkpointer
from libai.optim import build_optimizer
from utils.weight_convert import load_torch_checkpoint
from utils.lr_decay import get_layer_wise_lrd_overrides



class Trainer(DefaultTrainer):
    @classmethod
    def build_model(cls, cfg):
        model = super().build_model(cfg)
        # if try_get_key(cfg, "finetune") is not None:
        #     model.load_state_dict(flow.load(cfg.finetune.path))
        # model = load_torch_checkpoint(model, path="/home/rentianhe/code/OneFlow-Models/libai/mae_finetuned_vit_base.pth")
        return model
    
    @classmethod
    def build_optimizer(cls, cfg, model):
        param_overrides = None
        # TODO: add lr_scale in optim param_groups
        # if try_get_key(cfg, "train.layer_decay") is not None:
        #     param_overrides = get_layer_wise_lrd_overrides(model, cfg.optim.lr, cfg.train.layer_decay)
        # cfg.optim.parameters.overrides = param_overrides
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
