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

import oneflow as flow
from oneflow import nn

from libai.layers import TransformerLayer
from libai.utils import distributed as dist

logger = logging.getLogger(__name__)


class GraphBase(nn.Graph):
    def __init__(
        self,
        model: nn.Module,
        optimizer: flow.optim.Optimizer = None,
        lr_scheduler: flow.optim.lr_scheduler = None,
        fp16=False,
        activation_checkpoint=False,
        grad_acc_steps=1,
        zero_optim=False,
        zero_stage=0,
        is_train=True,
        auto_parallel_conf=None,
    ):
        super().__init__()

        self.model = model
        self.is_train = is_train

        if is_train:
            self.add_optimizer(optimizer, lr_sch=lr_scheduler)
            if fp16:
                self.config.enable_amp(True)
                grad_scaler = flow.amp.GradScaler(
                    init_scale=2 ** 30,
                    growth_factor=2.0,
                    backoff_factor=0.5,
                    growth_interval=2000,
                )
                self.set_grad_scaler(grad_scaler)

            if grad_acc_steps > 1:
                self.config.set_gradient_accumulation_steps(grad_acc_steps)

            if activation_checkpoint:
                self.set_activation_checkpoint()

            if zero_optim:
                self.config.enable_zero(True, stage=zero_stage)

            self.set_pipeline_stage_id()

        self.config.allow_fuse_add_to_output(True)
        self.config.allow_fuse_model_update_ops(True)
        self.config.allow_fuse_cast_scale(True)

        # auto_parallel
        if auto_parallel_conf is not None and auto_parallel_conf.enabled:
            try:
                self.config.enable_auto_parallel(True)
                self.config.enable_auto_parallel_prune_parallel_cast_ops(
                    auto_parallel_conf.prune_parallel_cast_ops
                )
                self.config.set_auto_parallel_computation_cost_ratio(0.05)
                self.config.set_auto_parallel_wait_time(1.65e4)
                self.config.enable_auto_parallel_mainstem_algo(auto_parallel_conf.mainstem_algo)
                self.config.enable_auto_parallel_sbp_collector(auto_parallel_conf.sbp_collector)
            except RuntimeWarning:
                import warnings

                warnings.warn(
                    "The version of oneflow don't support auto_parallel.\n"
                    "Please reinstall the oneflow for auto_parallel:\n"
                    "python3 -m pip install --pre oneflow -f https://staging.oneflow.info/branch/release-auto_parallel-v0.1/[PLATFORM]"  # noqa
                )

        # Enable cuda stream for computation and communication as the same stream.
        # This will reduce memory when using model parallelism.
        dist_util = dist.get_dist_util()
        if dist_util.is_tensor_model_parallel() or dist_util.is_pipeline_model_parallel():
            flow.boxing.nccl.enable_use_compute_stream(True)

    def build(self, **kwargs):
        if self.is_train:
            logger.info(
                "Start compling the train graph which may take some time. "
                "Please wait for a moment ..."
            )
            loss_dict = self.model(**kwargs)
            losses = sum(loss_dict.values())
            losses.backward()
            # set loss_dict on rank0
            # Consider if it's 2d mesh, ranks should be [[0]] instead of [0]
            loss_dict = {
                k: v.to_global(
                    placement=flow.placement(
                        "cpu", ranks=[0] if v.placement.ranks.ndim == 1 else [[0]]
                    ),
                    sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
                )
                for k, v in loss_dict.items()
            }
            return loss_dict
        else:
            logger.info(
                "Start compling the eval graph which may take some time. "
                "Please wait for a moment ..."
            )
            return self.model(**kwargs)

    def set_activation_checkpoint(self):
        if hasattr(type(self.model.origin), "set_activation_checkpoint"):
            type(self.model.origin).set_activation_checkpoint(self.model)
        else:
            for module_block in self.model.modules():
                if isinstance(module_block.origin, TransformerLayer):
                    module_block.config.activation_checkpointing = True

    def set_pipeline_stage_id(self):
        if hasattr(type(self.model.origin), "set_pipeline_stage_id"):
            type(self.model.origin).set_pipeline_stage_id(self.model)
