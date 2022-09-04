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

from libai.utils import distributed as dist


class Embedding(nn.Module):
    def __init__(self, in_channels, N_freqs, logscale=True):
        """
        Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)
        in_channels: number of input channels (3 for both xyz and direction)
        """
        super(Embedding, self).__init__()
        self.N_freqs = N_freqs
        self.in_channels = in_channels
        self.funcs = [flow.sin, flow.cos]
        self.out_channels = in_channels * (len(self.funcs) * N_freqs + 1)

        if logscale:
            freq_bands = 2 ** flow.linspace(0, N_freqs - 1, N_freqs)
        else:
            freq_bands = flow.linspace(1, 2 ** (N_freqs - 1), N_freqs).cuda()
        self.register_buffer(
            "freq_bands",
            freq_bands.to_global(
                placement=dist.get_layer_placement(0),
                sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
            ),
            persistent=False
        )

    def forward(self, x):
        """
        Embeds x to (x, sin(2^k x), cos(2^k x), ...)
        Different from the paper, "x" is also in the output
        See https://github.com/bmild/nerf/issues/12

        Inputs:
            x (Tensor): (B, self.in_channels)

        Outputs:
            out (Tensor): (B, self.out_channels)
        """
        out = [x]
        for freq in self.freq_bands:
            for func in self.funcs:
                m = func(freq * x)
                out += [m]

        return flow.cat(out, -1)


class NeRF(nn.Module):
    def __init__(self, D=8, W=256, in_channels_xyz=63, in_channels_dir=27, skips=[4]):
        """
        D: number of layers for density (sigma) encoder
        W: number of hidden units in each layer
        in_channels_xyz: number of input channels for xyz (3+3*10*2=63 by default)
        in_channels_dir: number of input channels for direction (3+3*4*2=27 by default)
        skips: add skip connection in the Dth layer
        """
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.in_channels_xyz = in_channels_xyz
        self.in_channels_dir = in_channels_dir
        self.skips = skips

        # xyz encoding layers
        for i in range(D):
            if i == 0:
                layer = nn.Linear(in_channels_xyz, W)
            elif i in skips:
                layer = nn.Linear(W + in_channels_xyz, W)
            else:
                layer = nn.Linear(W, W)
            layer = nn.Sequential(layer, nn.ReLU(True))
            setattr(self, f"xyz_encoding_{i+1}", layer)
        self.xyz_encoding_final = nn.Linear(W, W)

        # direction encoding layers
        self.dir_encoding = nn.Sequential(nn.Linear(W + in_channels_dir, W // 2), nn.ReLU(True))

        # output layers
        self.sigma = nn.Linear(W, 1)
        self.rgb = nn.Sequential(nn.Linear(W // 2, 3), nn.Sigmoid())
        self.reset_parameters(self)

    def reset_parameters(self, modules):
        for module in modules.modules():
            if isinstance(module, nn.Linear):
                nn.init.constant_(module.weight.data, 0.1)
                if module.bias is not None:
                    nn.init.constant_(module.weight.data, 0.01)

    def forward(self, x, sigma_only=False):
        """
        Encodes input (xyz+dir) to rgb+sigma (not ready to render yet).
        For rendering this ray, please see rendering.py

        Inputs:
            x (Tensor): (B, self.in_channels_xyz(+self.in_channels_dir))
               the embedded vector of position and direction
            sigma_only (bool): whether to infer sigma only. If True,
                        x is of shape (B, self.in_channels_xyz)

        Outputs:
            if sigma_ony:
                sigma (Tensor): (B, 1) sigma
            else:
                out (Tensor): (B, 4), rgb and sigma
        """
        if not sigma_only:
            input_xyz, input_dir = flow.split(
                x, [self.in_channels_xyz, self.in_channels_dir], dim=-1
            )
        else:
            input_xyz = x

        xyz_ = input_xyz
        for i in range(self.D):
            if i in self.skips:
                xyz_ = flow.cat([input_xyz, xyz_], -1)
            xyz_ = getattr(self, f"xyz_encoding_{i+1}")(xyz_)

        sigma = self.sigma(xyz_)
        if sigma_only:
            return sigma

        xyz_encoding_final = self.xyz_encoding_final(xyz_)

        dir_encoding_input = flow.cat([xyz_encoding_final, input_dir], -1)
        dir_encoding = self.dir_encoding(dir_encoding_input)
        rgb = self.rgb(dir_encoding)

        out = flow.cat([rgb, sigma], -1)

        return out
