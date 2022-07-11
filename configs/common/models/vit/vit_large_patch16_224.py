from libai.config import LazyCall
from libai.models import VisionTransformer

from .vit_tiny_patch16_224 import cfg


cfg.patch_size = 16
cfg.embed_dim = 1024
cfg.depth = 24
cfg.num_heads = 16

model = LazyCall(VisionTransformer)(cfg=cfg)
