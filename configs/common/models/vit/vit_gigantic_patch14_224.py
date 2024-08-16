from libai.config import LazyCall
from libai.models import VisionTransformer

from .vit_tiny_patch16_224 import cfg


cfg.patch_size = 14
cfg.embed_dim = 1664
cfg.mlp_ratio = 64 / 13
cfg.depth = 48
cfg.num_heads = 16

model = LazyCall(VisionTransformer)(cfg=cfg)
