from libai.config import LazyCall
from libai.models import VisionTransformer

from .vit_tiny_patch16_224 import cfg


cfg.patch_size = 16
cfg.embed_dim = 768
cfg.num_heads = 12

model = LazyCall(VisionTransformer)(cfg=cfg)
