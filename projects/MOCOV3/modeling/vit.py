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


# --------------------------------------------------------
# ViT Model
# References:
# moco-v3: https://github.com/facebookresearch/moco-v3/blob/main/vits.py
# --------------------------------------------------------


import math
from functools import reduce
from operator import mul

import oneflow as flow
import oneflow.nn as nn
from flowvision.layers.weight_init import trunc_normal_
from utils.load_checkpoint import load_checkpoint

from libai.layers import Linear, PatchEmbedding
from libai.models import vision_transformer


class VisionTransformer(vision_transformer.VisionTransformer):
    """Vision Transformer for MOCO
    LiBai impl of: `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    """

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        global_pool=False,
        num_classes=1000,
        loss_func=None,
        linear_prob=None,
        weight_style="pytorch",
        stop_grad_conv1=False,
    ):
        super(VisionTransformer, self).__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            num_classes=num_classes,
            loss_func=loss_func,
        )
        self.global_pool = global_pool

        # weight init
        if linear_prob:
            load_checkpoint(self, linear_prob, weight_style, num_heads, embed_dim)
            self.head.weight.data.normal_(mean=0.0, std=0.01)
            self.head.bias.data.zeros_()
        else:
            trunc_normal_(self.pos_embed, std=0.02)
            trunc_normal_(self.cls_token, std=0.02)
            self.apply(self._init_weights)

            self.stop_grad_conv1 = stop_grad_conv1
            self.embed_dim = embed_dim
            self.initialization()

    def initialization(self):

        # Use fixed 2D sin-cos position embedding
        self.build_2d_sincos_position_embedding()

        # weight initialization
        for name, m in self.named_modules():
            if isinstance(m, Linear):
                if "query_key_value" in name:
                    val = math.sqrt(6.0 / float(m.weight.shape[0] // 3 + m.weight.shape[1]))
                    nn.init.uniform_(m.weight, -val, val)
                else:
                    nn.init.xavier_uniform_(m.weight)

                nn.init.zeros_(m.bias)
        nn.init.normal_(self.cls_token, std=1e-6)

        if isinstance(self.patch_embed, PatchEmbedding):
            # xavier_uniform initialization
            val = math.sqrt(
                6.0 / float(3 * reduce(mul, self.patch_embed.patch_size, 1) + self.embed_dim)
            )
            nn.init.uniform_(self.patch_embed.proj.weight, -val, val)
            nn.init.zeros_(self.patch_embed.proj.bias)

            if self.stop_grad_conv1:
                self.patch_embed.proj.weight.requires_grad = False
                self.patch_embed.proj.bias.requires_grad = False

    def build_2d_sincos_position_embedding(self, temperature=10000.0):
        sbp = self.pos_embed.sbp
        placement = self.pos_embed.placement

        h, w = self.patch_embed.grid_size
        grid_w = flow.arange(w, dtype=flow.float32).to_global(sbp=sbp, placement=placement)
        grid_h = flow.arange(h, dtype=flow.float32).to_global(sbp=sbp, placement=placement)
        grid_w, grid_h = flow.meshgrid(grid_w, grid_h)
        assert (
            self.embed_dim % 4 == 0
        ), "Embed dimension must be divisible by 4 for 2D sin-cos position embedding"
        pos_dim = self.embed_dim // 4
        omega = (flow.arange(pos_dim, dtype=flow.float32) / pos_dim).to_global(
            sbp=sbp, placement=placement
        )
        omega = 1.0 / flow.tensor(temperature).to_global(sbp=sbp, placement=placement) ** omega
        out_w = flow.einsum("m,d->md", grid_w.flatten(), omega)
        out_h = flow.einsum("m,d->md", grid_h.flatten(), omega)
        pos_emb = flow.cat(
            [flow.sin(out_w), flow.cos(out_w), flow.sin(out_h), flow.cos(out_h)], dim=1
        )[None, :, :]
        pe_token = flow.zeros([1, 1, self.embed_dim], dtype=flow.float32).to_global(
            sbp=sbp, placement=placement
        )
        self.pos_embed = nn.Parameter(flow.cat([pe_token, pos_emb], dim=1))
        self.pos_embed.requires_grad = False

    def forward_head(self, x):
        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.norm(x)
            outcome = self.head(outcome)
        else:
            x = self.norm(x)
            outcome = x[:, 0]
            outcome = self.head(outcome)
        return outcome
