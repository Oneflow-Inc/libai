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
import oneflow.nn.init as init
from libai import distribute as dist

class LayerNorm(flow.nn.Module):
    """Layer normalization. This is same as nn.LayerNorm but add placement and sbp attribution.

    Arguments:
        layer_idx: the layer index, which determines the placement.
        normalized_shape: input shape from an expected input of size.
        eps: a value added to the denominator for numerical stability. Default: 1e-5.
    """
    def __init__(self, layer_idx, normalized_shape, eps=1e-5):
        super().__init__()
        self.normalized_shape = (normalized_shape,)
        self.epsilon = eps

        self.weight = flow.nn.Parameter(
            flow.empty(
                normalized_shape,
                dtype=flow.float32,
                placement=dist.get_layer_placement(layer_idx),
                sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
            )
        )
        flow.nn.init.ones_(self.weight)

        self.bias = flow.nn.Parameter(
            flow.empty(
                normalized_shape,
                dtype=flow.float32,
                placement=dist.get_layer_placement(layer_idx),
                sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
            )
        )
        flow.nn.init.zeros_(self.bias)

    def forward(self, x):
        assert x.shape[-len(self.normalized_shape) :] == self.normalized_shape
        begin_norm_axis = x.ndim - len(self.normalized_shape)
        begin_params_axis = x.ndim - len(self.normalized_shape)
        y = flow._C.layer_norm_affine(
            x,
            self.weight,
            self.bias,
            begin_norm_axis=begin_norm_axis,
            begin_params_axis=begin_params_axis,
            epsilon=self.epsilon,
        )
        return y

