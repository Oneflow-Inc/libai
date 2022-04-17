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


def drop_path(x, drop_prob: float = 0.5, training: bool = False, generator: flow.Generator = None):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = (
        flow.rand(*shape, dtype=x.dtype, generator=generator, sbp=x.sbp, placement=x.placement)
        + keep_prob
    )
    random_tensor = random_tensor.floor()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None, seed=0):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.generator = flow.Generator()
        self.generator.manual_seed(seed)

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.generator)
