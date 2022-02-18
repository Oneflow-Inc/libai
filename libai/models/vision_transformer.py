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
from flowvision.layers.weight_init import trunc_normal_

import libai.utils.distributed as dist
from libai.config.config import configurable
from libai.layers import LayerNorm, Linear, PatchEmbedding, TransformerLayer

from .build import MODEL_ARCH_REGISTRY


@MODEL_ARCH_REGISTRY.register()
class VisionTransformer(nn.Module):
    """Vision Transformer
    LiBai impl of: `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    """

    @configurable
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4.0,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        num_classes=1000,
        loss_func=None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        ffn_size = int(embed_dim * mlp_ratio)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(
            flow.zeros(
                1,
                1,
                embed_dim,
                sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
                placement=dist.get_layer_placement(0),
            )
        )
        self.pos_embed = nn.Parameter(
            flow.zeros(
                1,
                num_patches + 1,
                embed_dim,
                sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
                placement=dist.get_layer_placement(0),
            )
        )

        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [
            x.item() for x in flow.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        self.blocks = nn.Sequential(
            *[
                TransformerLayer(
                    hidden_size=embed_dim,
                    ffn_hidden_size=ffn_size,
                    num_attention_heads=num_heads,
                    attention_dropout_prob=attn_drop_rate,
                    output_dropout_prob=drop_rate,
                    drop_path_prob=dpr[i],
                    layer_idx=i,
                )
                for i in range(depth)
            ]
        )
        self.norm = LayerNorm(embed_dim)
        self.head = Linear(embed_dim, num_classes)

        # Loss func
        self.loss_func = nn.CrossEntropyLoss() if loss_func is None else loss_func

        # weight init
        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @classmethod
    def from_config(self, cfg):
        return {
            "img_size": cfg.img_size,
            "patch_size": cfg.patch_size,
            "in_chans": cfg.in_chans,
            "embed_dim": cfg.embed_dim,
            "depth": cfg.depth,
            "num_heads": cfg.num_heads,
            "mlp_ratio": cfg.mlp_ratio,
            "drop_rate": cfg.drop_rate,
            "attn_drop_rate": cfg.attn_drop_rate,
            "drop_path_rate": cfg.drop_path_rate,
            "num_classes": cfg.num_classes,
            "loss_func": cfg.loss_func,
        }

    def forward_features(self, x):
        # patch embedding
        x = self.patch_embed(x)

        cls_token = self.cls_token.expand(
            x.shape[0], -1, -1
        )  # stole cls_tokens impl from Phil Wang, thanks
        cls_token = cls_token.to_global(sbp=flow.sbp.split(0), placement=cls_token.placement)
        x = flow.cat((cls_token, x), dim=1)

        # position embedding
        pos_embed = self.pos_embed.expand(x.shape[0], -1, -1)
        pos_embed = pos_embed.to_global(sbp=flow.sbp.split(0), placement=pos_embed.placement)
        x = self.pos_drop(x + self.pos_embed)

        # transformer block
        x = self.blocks(x)
        x = self.norm(x)

        return x[:, 0]

    def forward(self, images, labels=None):
        x = self.forward_features(images)
        x = self.head(x)

        if labels is not None and self.training:
            losses = self.loss_func(x, labels)
            return {"losses": losses}
        else:
            return {"prediction_scores": x}
