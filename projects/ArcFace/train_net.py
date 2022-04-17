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
import sys
import math

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
import oneflow as flow
from libai.config import LazyConfig, default_argument_parser, try_get_key
from libai.engine import DefaultTrainer, default_setup
from libai.utils.checkpoint import Checkpointer
from libai.config import instantiate
from libai.utils import distributed as dist

logger = logging.getLogger("libai." + __name__)


class Trainer(DefaultTrainer):

    @classmethod
    def auto_scale_hyperparams(cls, cfg, data_loader):
        logger = logging.getLogger(__name__)
        log_info = ""

        # Get or set default iteration cfg
        train_iter = try_get_key(cfg, "train.train_iter", default=0)
        train_epoch = try_get_key(cfg, "train.train_epoch", default=0)
        warmup_ratio = try_get_key(cfg, "train.warmup_ratio", default=0)
        assert (
            warmup_ratio < 1 and warmup_ratio >= 0
        ), "warmup_ratio must be in [0, 1) that presents the ratio of warmup iter to the train iter"

        # Automatically scale iteration num depend on the settings
        # The total iters in one epoch is `len(dataset) / global_batch_size`
        cfg.train.train_iter = max(
            math.ceil(
                len(data_loader) * train_epoch / cfg.train.global_batch_size),
            train_iter,
        )
        cfg.train.warmup_iter = math.ceil(cfg.train.train_iter *
                                          cfg.train.warmup_ratio)
        if not cfg.graph.enabled:
            # In eager mode, dataloader only get micro-batch-size each iter,
            # which is mini-batch-size // num_accumulation, so scale `train_iter`
            # and `warmup_iter` to be consistent with static graph mode.
            cfg.train.train_iter *= cfg.train.num_accumulation_steps
            cfg.train.warmup_iter *= cfg.train.num_accumulation_steps
        log_info += "Auto-scaling the config to train.train_iter={}, train.warmup_iter={}".format(
            cfg.train.train_iter, cfg.train.warmup_iter)

        # Automatically scale the milestones
        if try_get_key(cfg, "train.scheduler.milestones"):
            if len([
                    milestone for milestone in cfg.train.scheduler.milestones
                    if milestone < 0 or milestone >= 1
            ]):
                raise ValueError(
                    "milestones should be a list of increasing ratio in [0, 1), but got {}"
                    .format(cfg.train.scheduler.milestones))
            cfg.train.scheduler.milestones = [
                int(milestone * cfg.train.train_iter)
                for milestone in cfg.train.scheduler.milestones
            ]
            log_info += f", scheduler milestones={cfg.train.scheduler.milestones}"
        logger.info(log_info)

        # Global scheduler cfg
        cfg.train.scheduler.warmup_iter = cfg.train.warmup_iter
        cfg.train.scheduler.max_iter = cfg.train.train_iter

    @classmethod
    def build_train_loader(cls, cfg, *args, **kwargs):
        logger = logging.getLogger(__name__)
        logger.info("Prepare training, validating, testing set")
        # if cfg.graph.enabled:
        #     # In static graph mode, data will be sliced in nn.Graph automatically,
        #     # so dataloader will get mini-batch-size.
        #     cfg.dataloader.train.train_batch_size = (
        #         cfg.train.train_micro_batch_size *
        #         cfg.train.num_accumulation_steps)
        # else:
        #     # In eager mode, gradient accumulation will act like PyTorch, so dataloader
        #     # will get micro-batch-size
        #     cfg.dataloader.train.train_batch_size = cfg.train.train_micro_batch_size

        train_loader = instantiate(cfg.dataloader)

        return train_loader, train_loader, train_loader

    @classmethod
    def get_batch(cls, data, *args, **kwargs):
        imgs, labels = data
        dist.synchronize()
        imgs = imgs.to_global(sbp=flow.sbp.broadcast,
                              placement=flow.env.all_device_placement("cuda"))
        imgs = imgs.to_global(sbp=dist.get_nd_sbp(
            [flow.sbp.split(0), flow.sbp.broadcast]),
                              placement=dist.get_layer_placement(0))

        labels = labels.to_global(
            sbp=flow.sbp.broadcast,
            placement=flow.env.all_device_placement("cuda"))
        labels = labels.to_global(sbp=dist.get_nd_sbp(
            [flow.sbp.split(0), flow.sbp.broadcast]),
                                  placement=dist.get_layer_placement(-1))
        return {"images": imgs, "labels": labels}


def main(args):
    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    default_setup(cfg, args)

    trainer = Trainer(cfg)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    main(args)
