from libai.config import LazyCall
from libai.models import VisionTransformer

from .vit_tiny_patch16_224 import cfg


cfg.patch_size = 16
cfg.embed_dim = 1280
cfg.depth = 32
cfg.num_heads = 16

model = LazyCall(VisionTransformer)(cfg=cfg)
