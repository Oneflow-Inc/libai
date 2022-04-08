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
# ResMLP Model
# References:
# resmlp: https://github.com/facebookresearch/deit/blob/main/resmlp_models.py
# --------------------------------------------------------

import oneflow as flow
import oneflow.nn as nn
from flowvision.layers.weight_init import trunc_normal_

import libai.utils.distributed as dist
from libai.layers import (
    LayerNorm,
    Linear, 
    PatchEmbedding, 
    DropPath, 
    MLP,
)


class Affine(nn.Module):
    def __init__(self, dim, layer_idx):
        super().__init__()
        self.alpha = nn.Parameter(
            flow.ones(dim), 
            sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
            placement=dist.get_layer_placement(layer_idx)
        )
        self.beta = nn.Parameter(
            flow.ones(dim), 
            sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
            placement=dist.get_layer_placement(layer_idx)
        )
    
    def forward(self, x):
        return self.alpha * x + self.beta


class layers_scale_mlp_blocks(nn.Module):

    def __init__(
        self, 
        dim, 
        drop=0., 
        drop_path=0., 
        init_values=1e-4, 
        num_patches=196,
        layer_idx = 0
    ):
        super().__init__()
        self.norm1 = Affine(dim, layer_idx=layer_idx)
        self.attn = Linear(num_patches, num_patches)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = Affine(dim, layer_idx=layer_idx)
        self.mlp = MLP(hidden_size=dim, ffn_hidden_size=(4.0 * dim), layer_idx=layer_idx)
        self.gamma_1 = nn.Parameter(
            init_values * flow.ones(dim, sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]), placement=dist.get_layer_placement(layer_idx)), 
            requires_grad=True
            )
        self.gamma_2 = nn.Parameter(
            init_values * flow.ones(dim, sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]), placement=dist.get_layer_placement(layer_idx)), 
            requires_grad=True
            )
    
    def forward(self, x):
        x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x).transpose(1,2)).transpose(1,2))
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x 


class ResMLP(nn.Module):
    def __init__(
        self, 
        img_size=224, 
        patch_size=16, 
        in_chans=3, 
        num_classes=1000, 
        embed_dim=768, 
        depth=12, 
        drop_rate=0.,
        drop_path_rate=0.,
        init_scale=1e-4,
        loss_func=None,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim

        self.patch_embed = PatchEmbedding(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim
        )

        num_patches = self.patch_embed.num_patches
        dpr = [
            drop_path_rate for i in range(depth)
        ]  # stochastic depth decay rule

        self.blocks = nn.ModuleList([
            layers_scale_mlp_blocks(
                dim=embed_dim, drop=drop_rate, drop_path=dpr[i],
                init_values=init_scale, num_patches=num_patches,
                layer_idx=i
            ) for i in range(depth)
        ])

        self.norm = Affine(embed_dim, layer_idx=-1)
        self.head = Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        
        # Loss func
        self.loss_func = nn.CrossEntropyLoss() if loss_func is None else loss_func
        
        # Weight init
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward_features(self, x):
        x = self.patch_embed(x)

        for i, blk in enumerate(self.blocks):
            x = blk(x)
    
        return x
    
    def forward_head(self, x):
        B = x.shape[0]
        x = self.norm(x)
        x = x.mean(dim=1).reshape(B, 1, -1)
        return self.head(x[:, 0])
    
    def forward(self, images, labels=None):
        """

        Args:
            images (flow.Tensor): training samples.
            labels (flow.LongTensor, optional): training targets

        Returns:
            dict:
                A dict containing :code:`loss_value` or :code:`logits`
                depending on training or evaluation mode.
                :code:`{"losses": loss_value}` when training,
                :code:`{"prediction_scores": logits}` when evaluating.
        """
        x = self.forward_features(images)
        x = self.forward_head(x)
        
        if labels is not None and self.training:
            losses = self.loss_func(x, labels)
            return {"losses": losses}
        else:
            return {"prediction_scores": x}