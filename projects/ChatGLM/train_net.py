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
import random

import numpy as np
import oneflow as flow

import libai.utils.distributed as dist
from libai.config import LazyConfig, default_argument_parser, try_get_key
from libai.engine import DefaultTrainer, default_setup
from libai.utils.checkpoint import Checkpointer
from projects.ChatGLM.utils.chatglm_loader import (
    ChatGLMLoaderHuggerFace,
    ChatGLMLoraLoaderHuggerFace,
)


def build_model(cfg):
    if cfg.cfg.lora_enable:
        model_loader = ChatGLMLoraLoaderHuggerFace(
            cfg,
            cfg.cfg,
            cfg.cfg.pretrained_model_path,
            lora_cfg=cfg.cfg.lora_cfg,
            lora_pretrained_model_path=cfg.cfg.lora_pretrained_model_path,
        )
        model = model_loader.load()
    else:
        model_loader = ChatGLMLoaderHuggerFace(
            cfg,
            cfg.cfg,
            cfg.cfg.pretrained_model_path,
        )
        model = model_loader.load()
    return model


class ChatGLMTrainer(DefaultTrainer):
    @classmethod
    def build_model(cls, cfg):
        assert try_get_key(cfg, "model") is not None, "cfg must contain `model` namespace"
        # Set model fp16 option because of embedding layer `white_identity` manual
        # insert for amp training if provided.
        if try_get_key(cfg.model, "cfg.amp_enabled") is not None:
            cfg.model.cfg.amp_enabled = cfg.train.amp.enabled and cfg.graph.enabled
        # In case some model define without cfg keyword.
        elif try_get_key(cfg.model, "amp_enabled") is not None:
            cfg.model.amp_enabled = cfg.train.amp.enabled and cfg.graph.enabled
        model = build_model(cfg.model)
        logger = logging.getLogger(__name__)
        logger.info("Model:\n{}".format(model))
        model._apply(dist.convert_to_distributed_default_setting)
        return model


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
            tokenizer = DefaultTrainer.build_tokenizer(cfg)
        model = DefaultTrainer.build_model(cfg)
        Checkpointer(model, save_dir=cfg.train.output_dir).resume_or_load(
            cfg.train.load_weight, resume=args.resume
        )
        if try_get_key(cfg, "graph.enabled", default=False):
            model = DefaultTrainer.build_graph(cfg, model, is_train=False)
        test_loader = DefaultTrainer.build_test_loader(cfg, tokenizer)
        if len(test_loader) == 0:
            logger = logging.getLogger(__name__)
            logger.info("No dataset in dataloader.test, please set dataset for dataloader.test")
        _ = DefaultTrainer.test(cfg, test_loader, model)
        return

    trainer = ChatGLMTrainer(cfg)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    main(args)
