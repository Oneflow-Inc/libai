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

from libai.layers import DropPath, Linear, build_activation
from libai.utils import distributed as dist
from projects.ConvNeXT.modeling.layer_norm import ConvNextLayerNorm


class ConvNextLayer(nn.Module):
    def __init__(
        self, dim, eps=1e-6, drop_path=0, layer_scale_init_value=1e-6, layer_idx=0
    ) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim).to_global(
            sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
            placement=dist.get_layer_placement(layer_idx),
        )
        self.layernorm = ConvNextLayerNorm(dim, eps=eps, layer_idx=layer_idx)
        self.pwconv1 = Linear(dim, 4 * dim, parallel="col", layer_idx=layer_idx)
        self.act = build_activation("gelu")
        self.pwconv2 = Linear(4 * dim, dim, parallel="row", layer_idx=layer_idx)

        layer_scale_parameter = (
            flow.ones(
                (dim),
                placement=dist.get_layer_placement(layer_idx),
                sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
            )
            * layer_scale_init_value
        )

        self.layer_scale_parameter = (
            nn.Parameter(layer_scale_parameter, requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.layer_idx = layer_idx

    def forward(self, hidden_states):
        input = hidden_states
        x = self.dwconv(hidden_states)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.layernorm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.layer_scale_parameter is not None:
            x = self.layer_scale_parameter * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class ConvNextStage(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=2,
        stride=2,
        depth=2,
        drop_path_rates=None,
        layer_idx=0,
    ):
        super().__init__()
        if in_channels != out_channels or stride > 1:
            self.downsampling_layer = nn.Sequential(
                ConvNextLayerNorm(
                    in_channels, eps=1e-6, data_format="channels_first", layer_idx=layer_idx
                ),
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=kernel_size, stride=stride
                ).to_global(
                    sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
                    placement=dist.get_layer_placement(layer_idx),
                ),
            )
        else:
            self.downsampling_layer = nn.Identity()
        drop_path_rates = drop_path_rates or [0.0] * depth
        self.layers = nn.Sequential(
            *[
                ConvNextLayer(dim=out_channels, drop_path=drop_path_rates[j], layer_idx=layer_idx)
                for j in range(depth)
            ]
        )
        self.layer_idx = layer_idx

    def forward(self, hidden_states):
        hidden_states = self.downsampling_layer(hidden_states)
        hidden_states = self.layers(hidden_states)
        return hidden_states


class ConvNextEncoder(nn.Module):
    def __init__(self, hidden_sizes, depths, num_stages, drop_path_rate):
        super().__init__()
        self.stages = nn.ModuleList()
        drop_path_rates = [
            x.tolist() for x in flow.linspace(0, drop_path_rate, sum(depths)).split(list(depths))
        ]
        prev_chs = hidden_sizes[0]
        for i in range(num_stages):
            out_chs = hidden_sizes[i]
            stage = ConvNextStage(
                in_channels=prev_chs,
                out_channels=out_chs,
                stride=2 if i > 0 else 1,
                depth=depths[i],
                drop_path_rates=drop_path_rates[i],
                layer_idx=i,
            )
            self.stages.append(stage)
            prev_chs = out_chs

    def forward(
        self,
        hidden_states,
    ):
        for i, layer_module in enumerate(self.stages):
            hidden_states = layer_module(hidden_states)

        return hidden_states
