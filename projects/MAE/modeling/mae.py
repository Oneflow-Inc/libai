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
# MAE Model
# References:
# mae: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------


import oneflow as flow
import oneflow.nn as nn

import libai.utils.distributed as dist
from libai.config import configurable
from libai.layers import LayerNorm, Linear, PatchEmbedding, TransformerLayer

from .pos_embed import get_2d_sincos_pos_embed


class MaskedAutoencoderViT(nn.Module):
    """Masked Autoencoder with VisionTransformer backbone"""

    @configurable
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.0,
        norm_layer=LayerNorm,
        norm_pix_loss=False,
        mask_ratio=0.75,
    ):
        super().__init__()

        self.mask_ratio = mask_ratio
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_chans, embed_dim)
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
        self.blocks = nn.ModuleList(
            [
                TransformerLayer(
                    hidden_size=embed_dim,
                    ffn_hidden_size=int(embed_dim * mlp_ratio),
                    num_attention_heads=num_heads,
                    layer_idx=i,
                )
                for i in range(depth)
            ]
        )
        # TODO: set norm layer placement stage id
        self.norm = norm_layer(embed_dim, layer_idx=depth)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = Linear(embed_dim, decoder_embed_dim, bias=True, layer_idx=depth)

        self.mask_token = nn.Parameter(
            flow.zeros(
                1,
                1,
                decoder_embed_dim,
                sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
                placement=dist.get_layer_placement(depth),
            )
        )

        self.decoder_pos_embed = nn.Parameter(
            flow.zeros(
                1,
                num_patches + 1,
                decoder_embed_dim,
                sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
                placement=dist.get_layer_placement(depth),
            )
        )

        self.decoder_blocks = nn.ModuleList(
            [
                TransformerLayer(
                    hidden_size=decoder_embed_dim,
                    ffn_hidden_size=int(decoder_embed_dim * mlp_ratio),
                    num_attention_heads=decoder_num_heads,
                    layer_idx=(i + depth),
                )
                for i in range(decoder_depth)
            ]
        )

        self.decoder_norm = norm_layer(decoder_embed_dim, layer_idx=-1)
        self.decoder_pred = Linear(
            decoder_embed_dim, patch_size ** 2 * in_chans, bias=True, layer_idx=-1
        )  # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** 0.5), cls_token=True
        )
        self.pos_embed.data.copy_(
            flow.from_numpy(pos_embed)
            .float()
            .unsqueeze(0)
            .to_global(
                sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
                placement=self.pos_embed.placement,
            )
        )

        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1],
            int(self.patch_embed.num_patches ** 0.5),
            cls_token=True,
        )
        self.decoder_pos_embed.data.copy_(
            flow.from_numpy(decoder_pos_embed)
            .float()
            .unsqueeze(0)
            .to_global(
                sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
                placement=self.decoder_pos_embed.placement,
            )
        )

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        flow.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        flow.nn.init.normal_(self.cls_token, std=0.02)
        flow.nn.init.normal_(self.mask_token, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, Linear):
            # we use xavier_uniform following official JAX ViT:
            flow.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, Linear) and m.bias is not None:
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
            "decoder_embed_dim": cfg.decoder_embed_dim,
            "decoder_depth": cfg.decoder_depth,
            "decoder_num_heads": cfg.decoder_num_heads,
            "mlp_ratio": cfg.mlp_ratio,
            "norm_layer": cfg.norm_layer,
            "norm_pix_loss": cfg.norm_pix_loss,
            "mask_ratio": cfg.mask_ratio,
        }

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(imgs.shape[0], 3, h, p, w, p)
        # TODO: replace permute with flow.einsum
        # (n c h p w q) -> (n h w p q c)
        x = x.permute(0, 2, 4, 3, 5, 1)
        x = x.reshape(imgs.shape[0], h * w, p ** 2 * 3)
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(x.shape[0], h, w, p, p, 3)
        # TODO: replace permute with flow.einsum
        # (n h w p q c) -> (n c h p w q)
        x = x.permute(0, 5, 1, 3, 2, 4)
        imgs = x.reshape(x.shape[0], 3, h * p, h * p)
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))

        noise = flow.rand(N, L, sbp=x.sbp, placement=x.placement)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = flow.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = flow.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = flow.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = flow.ones([N, L], sbp=x.sbp, placement=x.placement)
        mask[:, :len_keep] = 0

        # unshuffle to get binary mask
        mask = flow.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = flow.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = flow.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = flow.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])
        )  # unshuffle
        x = flow.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, images):
        latent, mask, ids_restore = self.forward_encoder(images, self.mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        loss = self.forward_loss(images, pred, mask)
        if self.training:
            return {"losses": loss}
        else:
            return {
                "losses": loss,
                "pred": pred,
                "mask": mask,
            }
