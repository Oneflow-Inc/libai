import math
from operator import mul
from functools import reduce

import oneflow as flow
import oneflow.nn as nn

from flowvision.layers.weight_init import trunc_normal_

import libai.utils.distributed as dist
from libai.config.config import configurable
from libai.layers import LayerNorm, Linear, PatchEmbedding
from .transformer_layer import TransformerLayer


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



class VisionTransformerMoCo(VisionTransformer):
    def __init__(self, stop_grad_conv1=False, **kwargs):
        super().__init__(**kwargs)

        # vit_small
        self.img_size=224 
        self.patch_size=16  
        self.in_chans=3 
        self.embed_dim=384  # if build_2d_sincos_position_embedding 192 else 384
        self.mlp_ratio=4.0 
        self.depth=12 
        self.num_heads=12 
        self.drop_rate=0.0
        self.attn_drop_rate=0.0
        self.drop_path_rate=0.0
        self.qkv_bias=True

        self.stop_grad_conv1 = stop_grad_conv1

    def initialization(self):
        # Use fixed 2D sin-cos position embedding
        self.build_2d_sincos_position_embedding()

        # weight initialization
        for name, m in self.named_modules():
            if isinstance(m, Linear): # libai
                if 'query_key_value' in name:
                    val = math.sqrt(6. / float(m.weight.shape[0] // 3 + m.weight.shape[1])) # shape may be wrong in oneflow (the transpose issue)
                    nn.init.uniform_(m.weight, -val, val)
                else:
                    nn.init.xavier_uniform_(m.weight)

                nn.init.zeros_(m.bias)
        nn.init.normal_(self.cls_token, std=1e-6)

        if isinstance(self.patch_embed, PatchEmbedding):
            # xavier_uniform initialization
            val = math.sqrt(6. / float(3 * reduce(mul, self.patch_embed.patch_size, 1) + self.embed_dim))
            nn.init.uniform_(self.patch_embed.proj.weight, -val, val)
            nn.init.zeros_(self.patch_embed.proj.bias)

            if self.stop_grad_conv1:
                self.patch_embed.proj.weight.requires_grad = False
                self.patch_embed.proj.bias.requires_grad = False

    def build_2d_sincos_position_embedding(self, temperature=10000.):
        sbp = self.pos_embed.sbp
        placement = self.pos_embed.placement

        h, w = self.patch_embed.grid_size
        grid_w = flow.arange(w, dtype=flow.float32).cuda().to_global(sbp=sbp, placement=placement)
        grid_h = flow.arange(h, dtype=flow.float32).cuda().to_global(sbp=sbp, placement=placement)
        grid_w, grid_h = flow.meshgrid(grid_w, grid_h)
        assert self.embed_dim % 4 == 0, 'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = self.embed_dim // 4
        omega = (flow.arange(pos_dim, dtype=flow.float32) / pos_dim).cuda().to_global(sbp=sbp, placement=placement)
        omega = 1. / flow.tensor(temperature).cuda().to_global(sbp=sbp, placement=placement)**omega  
        # out_w = flow.mul(grid_w.flatten().unsqueeze(1), omega.unsqueeze(0))  
        # out_h = flow.mul(grid_h.flatten().unsqueeze(1), omega.unsqueeze(0))  
        out_w = flow.einsum('m,d->md', [grid_w.flatten(), omega])
        out_h = flow.einsum('m,d->md', [grid_h.flatten(), omega])
        pos_emb = flow.cat([flow.sin(out_w), flow.cos(out_w), flow.sin(out_h), flow.cos(out_h)], dim=1)[None, :, :]
        # assert self.num_tokens == 1, 'Assuming one and only one token, [cls]'  # num_token=1 in libai impl
        pe_token = flow.zeros([1, 1, self.embed_dim], dtype=flow.float32).cuda().to_global(sbp=sbp, placement=placement)
        self.pos_embed = nn.Parameter(flow.cat([pe_token, pos_emb], dim=1))
        self.pos_embed.requires_grad = False
