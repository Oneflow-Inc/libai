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

from libai.models.utils.graph_base import GraphBase


class demo_model(nn.Module):
    def __init__(self, input_dim=512, out_dim=3):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, input_dim // 2)
        self.linear2 = nn.Linear(input_dim // 2, out_dim)

    def forward(self, x, label=None):
        x = x.to(dtype=flow.float32)
        x = self.linear1(x)
        x = self.linear2(x)
        if label is None:
            return x
        loss = self.get_loss(x)
        return loss

    def get_loss(self, x):
        return x.sum()


def build_model(cfg):
    model = demo_model()
    placement = flow.env.all_device_placement("cuda")
    model = model.to_global(placement=placement, sbp=flow.sbp.broadcast)
    return model


class GraphModel(GraphBase):
    def build(self, x, label=None):
        if self.is_train:
            loss = self.model(x, label)
            loss.backward()
            return loss
        else:
            return self.model(x)
