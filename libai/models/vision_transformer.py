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


class VisionTransformer(nn.Module):
    """Vision Transformer in LiBai.

    LiBai's implementation of:
    `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale
    <https://arxiv.org/abs/2010.11929>`_

    Args:
        img_size (int, tuple(int)): input image size
        patch_size (int, tuple(int)): patch size
        in_chans (int): number of input channels
        embed_dim (int): embedding dimension
        depth (int): depth of transformer
        num_heads (int): number of attention heads
        mlp_ratio (int): ratio of mlp hidden dim to embedding dim
        drop_rate (float): dropout rate
        attn_drop_rate (float): attention dropout rate
        drop_path_rate (float): stochastic depth rate
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
        self.img_size = img_size
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
        self.norm = LayerNorm(embed_dim, layer_idx=-1)
        self.head = Linear(embed_dim, num_classes, layer_idx=-1)

        # loss func
        self.loss_func = nn.CrossEntropyLoss() if loss_func is None else loss_func

        # weight init
        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def no_weight_decay(self):
        return {"pos_embed", "cls_token"}

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
            "num_classes": cfg.num_classes,
            "loss_func": cfg.loss_func,
        }

    def forward_features(self, x):
        # patch embedding
        x = self.patch_embed(x)

        cls_token = self.cls_token.expand(
            x.shape[0], -1, -1
        )  # stole cls_tokens impl from Phil Wang, thanks
        cls_token = cls_token.to_global(sbp=x.sbp, placement=cls_token.placement)
        x = flow.cat((cls_token, x), dim=1)

        # position embedding
        pos_embed = self.pos_embed.expand(x.shape[0], -1, -1)
        pos_embed = pos_embed.to_global(sbp=x.sbp, placement=pos_embed.placement)
        x = self.pos_drop(x + pos_embed)

        # transformer block
        x = self.blocks(x)
        return x

    def forward_head(self, x):
        x = self.norm(x)
        outcome = x[:, 0]
        outcome = self.head(outcome)
        return outcome

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
            elif isinstance(module_block.origin, TransformerLayer):
                module_block.config.set_stage(
                    dist_utils.get_layer_stage_id(module_block.layer_idx),
                    dist.get_layer_placement(module_block.layer_idx),
                )

        # Set pos_embed and cls_token stage id
        model.pos_embed.config.set_stage(
            dist_utils.get_layer_stage_id(0), dist.get_layer_placement(0)
        )
        model.cls_token.config.set_stage(
            dist_utils.get_layer_stage_id(0), dist.get_layer_placement(0)
        )
        model.pos_drop.config.set_stage(
            dist_utils.get_layer_stage_id(0), dist.get_layer_placement(0)
        )
        model.norm.config.set_stage(dist_utils.get_layer_stage_id(-1), dist.get_layer_placement(-1))
        model.head.config.set_stage(dist_utils.get_layer_stage_id(-1), dist.get_layer_placement(-1))
        model.loss_func.config.set_stage(
            dist_utils.get_layer_stage_id(-1), dist.get_layer_placement(-1)
        )
