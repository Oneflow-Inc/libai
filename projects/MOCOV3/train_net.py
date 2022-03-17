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

import oneflow as flow
import sys
import logging
sys.path.append(".")

from libai.config import LazyConfig, default_argument_parser, try_get_key
from libai.engine import default_setup
from libai.engine import DefaultTrainer
from libai.utils.checkpoint import Checkpointer
from utils.weight_convert import load_torch_checkpoint, load_torch_checkpoint_linear


logger = logging.getLogger(__name__)

class Trainer(DefaultTrainer):
    @classmethod
    def build_model(cls, cfg):
        model = super().build_model(cfg)

        if try_get_key(cfg, "finetune") is not None:
            if cfg.finetune.enable == True:

                logger.info("freeze all layers but the last head")
                linear_keyword = "head"
                for name, param in model.named_parameters():
                    if name not in ['%s.weight' % linear_keyword, '%s.bias' % linear_keyword]:
                        param.requires_grad = False

                logger.info("Loading pretrained weight for finetuning")
                assert cfg.finetune.weight_style in ["oneflow", "pytorch"]
                if cfg.finetune.weight_style == "oneflow":
                    model.load_state_dict(flow.load(cfg.finetune.path))
                else:
                    # model = load_torch_checkpoint(model, path=cfg.finetune.path, strict=False, linear_keyword=linear_keyword)
                    model = load_torch_checkpoint_linear(model, path=cfg.linearProb.path, strict=True)
                # if cfg.linearProb.enable == True:
                #     logger.info("Loading pretrained weight for linearprob")
                #     assert cfg.linearProb.weight_style in ["oneflow", "pytorch"]
                #     if cfg.linearProb.weight_style == "oneflow":
                #         model.load_state_dict(flow.load(cfg.linearProb.path))
                #     else:
                #         model = load_torch_checkpoint_linear(model, path=cfg.linearProb.path, strict=False)
                # else:
                #     logger.info("init the head layer")
                #     getattr(model, linear_keyword).weight.data.normal_(mean=0.0, std=0.01)
                #     getattr(model, linear_keyword).bias.data.zeros_()
            else:
                model.initialization()
        return model

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
        if try_get_key(cfg, "train.graph.enabled", default=False):
            model = DefaultTrainer.build_graph(cfg, model, is_train=False)
        test_loader = DefaultTrainer.build_test_loader(cfg, tokenizer)
        _ = DefaultTrainer.test(cfg, test_loader, model)
        return

    trainer = DefaultTrainer(cfg)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    args.eval_only = True
    main(args)
