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
import oneflow.nn.functional as F

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
            persistent=False,
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


class NeRF(nn.Module):  # a alignment point with nerf_pytorch
    def __init__(
        self, D=8, W=256, input_ch=63, input_ch_views=27, output_ch=5, skips=[4], use_viewdirs=True
    ):
        """
        D: number of layers for density (sigma) encoder
        W: number of hidden units in each layer
        input_ch: number of input channels for xyz (3+3*10*2=63 by default)
        input_ch_views: number of input channels for direction (3+3*4*2=27 by default)
        output_ch: number of output channels
        skips: add skip connection in the Dth layer
        """
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs

        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)]
            + [
                nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W)
                for i in range(D - 1)
            ]
        )
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W // 2)])
        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W // 2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x, sigma_only=False):
        """
        Encodes input (xyz+dir) to rgb+sigma (not ready to render yet).
        For rendering this ray, please see rendering.py

        Inputs:
            x (Tensor): (B, self.in_channels_xyz+self.in_channels_dir)
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
            input_pts, input_views = flow.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        else:
            input_pts = x
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = flow.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            if sigma_only:
                return alpha
            feature = self.feature_linear(h)
            h = flow.cat([feature, input_views], -1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h).sigmoid()  # sigmoid
            outputs = flow.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)
        return outputs
