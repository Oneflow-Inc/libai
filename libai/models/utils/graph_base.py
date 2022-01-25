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
from oneflow import nn

from libai.layers import TransformerLayer


class GraphBase(nn.Graph):
    def __init__(
        self,
        model: nn.Module,
        optimizer: flow.optim.Optimizer = None,
        lr_scheduler: flow.optim.lr_scheduler = None,
        fp16: bool = False,
        is_train: bool = True,
    ):
        super().__init__()

        self.model = model
        self.is_train = is_train

        if is_train:
            self.add_optimizer(optimizer, lr_sch=lr_scheduler)
            self.set_activation_checkpoint()
            self.set_pipeline_stage_id()
            if fp16:
                self.config.enable_amp(True)
                grad_scaler = flow.amp.GradScaler(
                    init_scale=2 ** 30,
                    growth_factor=2.0,
                    backoff_factor=0.5,
                    growth_interval=2000,
                )
                self.set_grad_scaler(grad_scaler)

        self.config.allow_fuse_add_to_output(True)
        self.config.allow_fuse_model_update_ops(True)
        self.config.allow_fuse_cast_scale(True)

    def build(self, **kwargs):
        if self.is_train:
            loss_dict = self.model(**kwargs)
            losses = sum(loss_dict.values())
            losses.backward()
            return loss_dict
        else:
            return self.model(**kwargs)

    def set_activation_checkpoint(self):
        for module_block in self.model.modules():
            if isinstance(module_block.origin, TransformerLayer):
                module_block.config.activation_checkpointing = True

    def set_pipeline_stage_id(self):
        if hasattr(type(self.model.origin), "set_pipeline_stage_id"):
            type(self.model.origin).set_pipeline_stage_id(self.model)
