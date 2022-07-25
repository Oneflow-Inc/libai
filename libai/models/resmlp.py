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
from libai.config import configurable
from libai.layers import MLP, DropPath, LayerNorm, Linear, PatchEmbedding


class Affine(nn.Module):
    def __init__(self, dim, *, layer_idx=0):
        super().__init__()
        self.alpha = nn.Parameter(
            flow.ones(
                dim,
                placement=dist.get_layer_placement(layer_idx),
                sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
            )
        )
        self.beta = nn.Parameter(
            flow.zeros(
                dim,
                placement=dist.get_layer_placement(layer_idx),
                sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
            ),
        )

        self.layer_idx = layer_idx

    def forward(self, x):
        x = x.to_global(placement=dist.get_layer_placement(self.layer_idx))
        return self.alpha * x + self.beta


class layers_scale_mlp_blocks(nn.Module):
    def __init__(
        self, dim, drop=0.0, drop_path=0.0, init_values=1e-4, num_patches=196, *, layer_idx=0
    ):
        super().__init__()
        self.norm1 = Affine(dim, layer_idx=layer_idx)
        self.attn = Linear(num_patches, num_patches, layer_idx=layer_idx)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = Affine(dim, layer_idx=layer_idx)
        self.mlp = MLP(hidden_size=dim, ffn_hidden_size=int(4.0 * dim), layer_idx=layer_idx)
        self.gamma_1 = nn.Parameter(
            init_values
            * flow.ones(
                dim,
                sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
                placement=dist.get_layer_placement(layer_idx),
            ),
            requires_grad=True,
        )
        self.gamma_2 = nn.Parameter(
            init_values
            * flow.ones(
                dim,
                sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
                placement=dist.get_layer_placement(layer_idx),
            ),
            requires_grad=True,
        )

        self.layer_idx = layer_idx

    def forward(self, x):
        x = x.to_global(placement=dist.get_layer_placement(self.layer_idx))
        x = x + self.drop_path(
            self.gamma_1 * self.attn(self.norm1(x).transpose(1, 2)).transpose(1, 2)
        )
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class ResMLP(nn.Module):
    """ResMLP in LiBai.

    LiBai's implementation of:
    `ResMLP: Feedforward networks for image classification with data-efficient training
    <https://arxiv.org/abs/2105.03404>`_

    Args:
        img_size (int, tuple(int)): input image size
        patch_size (int, tuple(int)): patch size
        in_chans (int): number of input channels
        embed_dim (int): embedding dimension
        depth (int): depth of transformer
        drop_rate (float): dropout rate
        drop_path_rate (float): stochastic depth rate
        init_scale (float): the layer scale ratio
        num_classes (int): number of classes for classification head
        loss_func (callable, optional): loss function for computing the total loss
                                        between logits and labels

    """

    @configurable
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        drop_rate=0.0,
        drop_path_rate=0.0,
        init_scale=1e-4,
        num_classes=1000,
        loss_func=None,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim

        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        num_patches = self.patch_embed.num_patches
        dpr = [drop_path_rate for i in range(depth)]  # stochastic depth decay rule

        self.blocks = nn.ModuleList(
            [
                layers_scale_mlp_blocks(
                    dim=embed_dim,
                    drop=drop_rate,
                    drop_path=dpr[i],
                    init_values=init_scale,
                    num_patches=num_patches,
                    layer_idx=i,
                )
                for i in range(depth)
            ]
        )

        self.norm = Affine(embed_dim, layer_idx=-1)
        self.head = (
            Linear(embed_dim, num_classes, layer_idx=-1) if num_classes > 0 else nn.Identity()
        )

        # loss func
        self.loss_func = nn.CrossEntropyLoss() if loss_func is None else loss_func

        # weight init
        self.apply(self._init_weights)

    @classmethod
    def from_config(cls, cfg):
        return {
            "img_size": cfg.img_size,
            "patch_size": cfg.patch_size,
            "in_chans": cfg.in_chans,
            "embed_dim": cfg.embed_dim,
            "depth": cfg.depth,
            "drop_rate": cfg.drop_rate,
            "drop_path_rate": cfg.drop_path_rate,
            "init_scale": cfg.init_scale,
            "num_classes": cfg.num_classes,
            "loss_func": cfg.loss_func,
        }

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

        # layer scale mlp blocks
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

    @staticmethod
    def set_pipeline_stage_id(model):
        dist_utils = dist.get_dist_util()

        # Set pipeline parallelism stage_id
        for module_block in model.modules():
            # module.origin can get the original module
            if isinstance(module_block.origin, PatchEmbedding):
                module_block.config.set_stage(
                    dist_utils.get_layer_stage_id(0), dist.get_layer_placement(0)
                )
            elif isinstance(module_block.origin, layers_scale_mlp_blocks):
                module_block.config.set_stage(
                    dist_utils.get_layer_stage_id(module_block.layer_idx),
                    dist.get_layer_placement(module_block.layer_idx),
                )

        # Set norm and head stage id
        model.norm.config.set_stage(dist_utils.get_layer_stage_id(-1), dist.get_layer_placement(-1))
        model.head.config.set_stage(dist_utils.get_layer_stage_id(-1), dist.get_layer_placement(-1))
        model.loss_func.config.set_stage(
            dist_utils.get_layer_stage_id(-1), dist.get_layer_placement(-1)
        )

    @staticmethod
    def set_activation_checkpoint(model):
        for module_block in model.modules():
            if isinstance(module_block.origin, layers_scale_mlp_blocks):
                module_block.config.activation_checkpointing = True
