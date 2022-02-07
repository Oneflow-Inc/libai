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
from omegaconf import OmegaConf

from libai.config import LazyCall, default_argument_parser
from libai.data.build import build_nlp_test_loader, build_nlp_train_val_test_loader
from libai.data.structures import Instance
from libai.optim import get_default_optimizer_params
from libai.scheduler import WarmupCosineLR
from libai.trainer import DefaultTrainer, default_setup
from tests.data.datasets.demo_dataset import DemoNlpDataset
from tests.layers.test_trainer_model import build_graph, build_model


def setup(args):
    """
    Create configs and perform basic setups.
    """

    cfg = OmegaConf.create()

    cfg.train = dict(
        output_dir="./demo_output",
        train_micro_batch_size=32,
        test_micro_batch_size=16,
        dist=dict(
            data_parallel_size=1,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            pipeline_num_layers=4,
        ),
        start_iter=0,
        train_iter=20,
        lr_warmup_fraction=0.01,
        lr_decay_iter=6000,
        log_period=1,
        checkpointer=dict(period=100),
        nccl_fusion_threshold_mb=16,
        nccl_fusion_max_ops=24,
    )

    cfg.optim = LazyCall(flow.optim.AdamW)(
        parameters=LazyCall(get_default_optimizer_params)(
            # parameters.model is meant to be set to the model object,
            # before instantiating the optimizer.
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

    cfg.scheduler = LazyCall(WarmupCosineLR)(
        max_iters=2000,
        alpha=0.001,
        warmup_factor=0.001,
        warmup_iters=1000,
        warmup_method="linear",
    )

    cfg.dataloader = OmegaConf.create()

    cfg.dataloader.train = LazyCall(build_nlp_train_val_test_loader)(
        dataset=[
            LazyCall(DemoNlpDataset)(
                data_root="train1",
            ),
            LazyCall(DemoNlpDataset)(
                data_root="train2",
            ),
        ],
        splits=[[949.0, 50.0, 1.0], [900.0, 99.0, 1.0]],
        weights=[0.5, 0.5],
        num_workers=4,
    )

    cfg.dataloader.test = [
        LazyCall(build_nlp_test_loader)(
            dataset=LazyCall(DemoNlpDataset)(
                data_root="test1",
            )
        ),
        LazyCall(build_nlp_test_loader)(
            dataset=LazyCall(DemoNlpDataset)(
                data_root="test2",
            )
        ),
    ]

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


def main(args):
    cfg = setup(args)

    trainer = DemoTrainer(cfg)

    print(len(trainer.train_loader), len(trainer.test_loader))
    for sample in trainer.train_loader:
        assert isinstance(sample, Instance)
        print(f"train sample shape {sample.input.tensor.shape} {sample.label.tensor.shape}")
        break

    for loader in trainer.test_loader:
        print(f"test loader: {len(loader)}")
        for sample in loader:
            assert isinstance(sample, Instance)
            print(f"train sample shape {sample.input.tensor.shape} {sample.label.tensor.shape}")
            break


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    main(args)
