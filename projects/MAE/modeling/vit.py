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


import libai.models.vision_transformer


class VisionTransformer(libai.models.vision_transformer.VisionTransformer):
    """Vision Transformer for MAE
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

    def no_weight_decay(self):
        return {"pos_embed", "cls_token"}

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
