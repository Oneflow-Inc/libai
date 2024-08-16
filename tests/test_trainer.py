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
from omegaconf import OmegaConf
from oneflow.utils.data import DataLoader, TensorDataset

sys.path.append(".")
from libai.config import LazyCall, default_argument_parser
from libai.engine import DefaultTrainer, default_setup
from libai.optim import get_default_optimizer_params
from libai.scheduler import WarmupMultiStepLR
from tests.layers.test_trainer_model import build_graph, build_model


def setup(args):
    """
    Create configs and perform basic setups.
    """

    cfg = OmegaConf.create()

    cfg.train = dict(
        output_dir="./demo_output",
        train_micro_batch_size=32,
        test_micro_batch_size=32,
        dist=dict(
            data_parallel_size=1,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            pipeline_num_layers=4,
        ),
        start_iter=0,
        train_iter=20,
        train_epoch=1,
        warmup_ratio=0.05,
        lr_warmup_fraction=0.01,
        lr_decay_iter=6000,
        eval_period=1000,
        log_period=1,
        checkpointer=dict(period=100),
        nccl_fusion_threshold_mb=16,
        nccl_fusion_max_ops=24,
        scheduler=LazyCall(WarmupMultiStepLR)(
            warmup_factor=0.001,
            # alpha=0.01,
            warmup_method="linear",
            milestones=[0.1, 0.2],
        ),
    )

    cfg.optim = LazyCall(flow.optim.AdamW)(
        parameters=LazyCall(get_default_optimizer_params)(
            # parameters.model is meant to be set to the model object, before
            # instantiating the optimizer.
            clip_grad_max_norm=1.0,
            clip_grad_norm_type=2.0,
            weight_decay_norm=0.0,
            weight_decay_bias=0.0,
        ),
        lr=1e-4,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        do_bias_correction=True,
    )

    cfg.graph = dict(
        enabled=True,
    )

    default_setup(cfg, args)
    return cfg


class DemoTrainer(DefaultTrainer):
    @classmethod
    def build_model(cls, cfg):
        """
        Returns:
            flow.nn.Module:
        It now calls :func:`libai.layers.build_model`.
        Overwrite it if you'd like a different model.
        """
        model = build_model(cfg)
        return model

    @classmethod
    def build_graph(cls, cfg, model, optimizer=None, lr_scheduler=None, is_train=True):
        return build_graph(cfg, model, optimizer, lr_scheduler)

    @classmethod
    def get_batch(cls, data):
        return [
            flow.randn(
                32,
                512,
                sbp=flow.sbp.split(0),
                placement=flow.placement("cuda", [0]),
            )
        ]

    @classmethod
    def build_train_loader(cls, cfg, tokenizer=None):
        return (
            DataLoader(
                TensorDataset(flow.randn(1000)), batch_size=cfg.train.train_micro_batch_size
            ),
            None,
            None,
        )

    @classmethod
    def build_test_loader(cls, cfg):
        return []


def main(args):
    cfg = setup(args)

    trainer = DemoTrainer(cfg)

    # trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    main(args)
