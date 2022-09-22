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
import oneflow.nn as nn


class SoftTargetCrossEntropy(nn.Module):
    def __init__(self):
        super(SoftTargetCrossEntropy, self).__init__()

    def forward(self, x: flow.Tensor, target: flow.Tensor) -> flow.Tensor:
        pred = flow.log_softmax(x, dim=-1)
        loss = -target * pred
        # sum and mean should be calculated with float32
        # amp_white_identity ensure -target * pred using float16
        # amp_black_identity ensure sum and mean using float32
        loss = flow._C.amp_white_identity(loss)
        loss = flow._C.amp_black_identity(loss)
        loss = flow.sum(loss, dim=-1)
        return loss.mean()
