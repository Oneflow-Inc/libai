from libai.config import LazyCall
from libai.models import MoCo_ViT
from libai.models.vision_transformer import VisionTransformer
import oneflow as flow
import oneflow.nn as nn
import math
from operator import mul
from functools import reduce
from libai.layers import PatchEmbedding


class VisionTransformerMoCo(VisionTransformer):
    def __init__(self, stop_grad_conv1=False, **kwargs):
        super().__init__(**kwargs)

        # vit_small
        self.img_size=224 
        self.patch_size=16  
        self.in_chans=3 
        self.embed_dim=384 
        self.mlp_ratio=4.0 
        self.depth=12 
        self.num_heads=12 
        self.drop_rate=0.0
        self.attn_drop_rate=0.0
        self.drop_path_rate=0.0
        self.qkv_bias=True

        # Use fixed 2D sin-cos position embedding
        self.build_2d_sincos_position_embedding()
        # weight initialization
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                if 'qkv' in name:
                    # treat the weights of Q, K, V separately
                    val = math.sqrt(6. / float(m.weight.shape[0] // 3 + m.weight.shape[1]))
                    nn.init.uniform_(m.weight, -val, val)
                else:
                    nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.normal_(self.cls_token, std=1e-6)

        if isinstance(self.patch_embed, PatchEmbedding):
            # xavier_uniform initialization
            val = math.sqrt(6. / float(3 * reduce(mul, self.patch_embed.patch_size, 1) + self.embed_dim[0]))
            nn.init.uniform_(self.patch_embed.proj.weight, -val, val)
            nn.init.zeros_(self.patch_embed.proj.bias)

            if stop_grad_conv1:
                self.patch_embed.proj.weight.requires_grad = False
                self.patch_embed.proj.bias.requires_grad = False

    def build_2d_sincos_position_embedding(self, temperature=10000.):
        h, w = self.patch_embed.grid_size
        grid_w = flow.arange(w, dtype=flow.float32)
        grid_h = flow.arange(h, dtype=flow.float32)
        grid_w, grid_h = flow.meshgrid(grid_w, grid_h)
        assert self.embed_dim[0] % 4 == 0, 'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = self.embed_dim[0] // 4
        omega = flow.arange(pos_dim, dtype=flow.float32) / pos_dim
        omega = 1. / flow.tensor(temperature)**omega  # torch supprts the ** op between tensor and float, but oneflow does not.  (temperature**omega) -> omega.pow(temperature)
        out_w = flow.mul(grid_w.flatten().unsqueeze(1), omega.unsqueeze(0))  #  out_w = flow.einsum('m,d->md', [grid_w.flatten(), omega])
        out_h = flow.mul(grid_h.flatten().unsqueeze(1), omega.unsqueeze(0))  #  out_h = flow.einsum('m,d->md', [grid_h.flatten(), omega])
        pos_emb = flow.cat([flow.sin(out_w), flow.cos(out_w), flow.sin(out_h), flow.cos(out_h)], dim=1)[None, :, :]

        # assert self.num_tokens == 1, 'Assuming one and only one token, [cls]'  # not self.num_tokens in our impl
        pe_token = flow.zeros([1, 1, self.embed_dim[0]], dtype=flow.float32).to(pos_emb.device)
        self.pos_embed = nn.Parameter(flow.cat([pe_token, pos_emb], dim=1))
        self.pos_embed.requires_grad = False


model = LazyCall(MoCo_ViT)(
            base_encoder=LazyCall(VisionTransformer)(), 
            momentum_encoder=LazyCall(VisionTransformer)(),
            dim=256, 
            mlp_dim=4096, 
            T=.2
)
