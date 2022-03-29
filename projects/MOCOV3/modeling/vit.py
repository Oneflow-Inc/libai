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


import oneflow as flow
import oneflow.nn as nn
from flowvision.layers.weight_init import trunc_normal_

import libai.utils.distributed as dist
from libai.config.config import configurable
from libai.layers import LayerNorm, Linear, PatchEmbedding, TransformerLayer
from utils.weight_convert import load_torch_checkpoint_finetune

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
        embed_dim=384,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        global_pool=False,
        num_classes=1000,
        loss_func=None,
        finetune=None,
        weight_style="pytorch"
    ):
        super().__init__()
        self.global_pool = global_pool
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

        # Loss func
        self.loss_func = nn.CrossEntropyLoss() if loss_func is None else loss_func

        # weight init
        if finetune:
            self.load_checkpoint(finetune, weight_style, num_heads, embed_dim)
        else:
            trunc_normal_(self.pos_embed, std=0.02)
            trunc_normal_(self.cls_token, std=0.02)
            self.apply(self._init_weights)

    def load_checkpoint(self, finetune, weight_style, num_heads, embed_dim):
        linear_keyword = "head"
        for name, param in self.named_parameters():
            if name not in ['%s.weight' % linear_keyword, '%s.bias' % linear_keyword]:
                param.requires_grad = False

        if weight_style == "pytorch":
            params = load_torch_checkpoint_finetune(num_heads, embed_dim, path=finetune)
        else:
            params = flow.load(finetune)

        self.head.weight.data.normal_(mean=0.0, std=0.01)
        self.head.bias.data.zeros_()

        self.load_state_dict(params, strict=False)

    def _init_weights(self, m):
        if isinstance(m, Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

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

    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

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
            outcome = self.norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]
        
        return outcome

    def forward(self, images, labels=None):
        x = self.forward_features(images)
        x = self.head(x)

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
                module_block.config.stage_id = dist_utils.get_layer_stage_id(0)
            elif isinstance(module_block.origin, TransformerLayer):
                module_block.config.stage_id = dist_utils.get_layer_stage_id(module_block.layer_idx)

        # Set pos_embed and cls_token stage id
        model.pos_embed.config.stage_id = dist_utils.get_layer_stage_id(0)
        model.cls_token.config.stage_id = dist_utils.get_layer_stage_id(0)
        model.pos_drop.config.stage_id = dist_utils.get_layer_stage_id(0)
        model.norm.config.stage_id = dist_utils.get_layer_stage_id(-1)
        model.head.config.stage_id = dist_utils.get_layer_stage_id(-1)
        model.loss_func.config.stage_id = dist_utils.get_layer_stage_id(-1)



