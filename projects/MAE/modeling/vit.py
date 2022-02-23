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
# mae: https://github.com/facebookresearch/mae/blob/main/models_vit.py
# --------------------------------------------------------

from functools import partial
import pdb

import oneflow as flow
import oneflow.nn as nn
from libai.layers.layer_norm import LayerNorm

import libai.models.vision_transformer


class VisionTransformer(libai.models.vision_transformer.VisionTransformer):
    """Vision Transformer with support for global average pooling
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
        norm_layer=LayerNorm,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        global_pool=False,
        num_classes=1000,
        loss_func=None,):
        super(VisionTransformer, self).__init__(
            img_size, patch_size, in_chans, embed_dim, depth, num_heads, mlp_ratio,
            drop_rate, attn_drop_rate, drop_path_rate, num_classes, loss_func)

        self.global_pool = global_pool
        if self.global_pool:
            self.fc_norm = norm_layer(embed_dim)

            # TODO: fix del error
            # del self.norm  # remove the original norm
            self.norm = nn.Identity()
    
    @classmethod
    def from_config(cls, cfg):
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
            "global_pool": cfg.global_pool,
            "num_classes": cfg.num_classes,
        }

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_token = self.cls_token.expand(
            B, -1, -1
        )  # stole cls_tokens impl from Phil Wang, thanks
        cls_token = cls_token.to_global(sbp=x.sbp, placement=cls_token.placement)
        x = flow.cat((cls_token, x), dim=1)

        # position embedding
        pos_embed = self.pos_embed.expand(B, -1, -1)
        pos_embed = pos_embed.to_global(sbp=x.sbp, placement=pos_embed.placement)
        x = self.pos_drop(x + pos_embed)

        # transformer block
        x = self.blocks(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome