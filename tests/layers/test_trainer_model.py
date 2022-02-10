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


class demo_model(nn.Module):
    def __init__(self, input_dim=512, out_dim=64):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, out_dim)
        self.linear2 = nn.Linear(out_dim, out_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        loss = self.get_loss(x)
        return loss

    def get_loss(self, x):
        return x.sum()


def build_model(cfg):
    model = demo_model()
    placement = flow.env.all_device_placement("cuda")
    model = model.to_global(placement=placement, sbp=flow.sbp.broadcast)
    return model


def build_graph(cfg, model, optimizer, lr_scheduler, fp16=False):
    class GraphModel(nn.Graph):
        def __init__(self, model, optimizer, lr_scheduler, fp16=False):
            super().__init__()
            self.model = model
            self.add_optimizer(optimizer, lr_sch=lr_scheduler)
            self.config.allow_fuse_add_to_output(True)
            self.config.allow_fuse_model_update_ops(True)
            if fp16:
                self.config.enable_amp(True)
                grad_scaler = flow.amp.GradScaler(
                    init_scale=2 ** 30,
                    growth_factor=2.0,
                    backoff_factor=0.5,
                    growth_interval=2000,
                )
                self.set_grad_scaler(grad_scaler)

        def build(self, x):
            loss = self.model(x)
            loss.backward()
            return loss

    if optimizer:
        return GraphModel(model, optimizer, lr_scheduler, fp16=fp16)
    else:
        return None
