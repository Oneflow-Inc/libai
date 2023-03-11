from libai.config import LazyCall
from libai.models import VisionTransformer

from .vit_tiny_patch16_224 import cfg


cfg.patch_size = 16
cfg.embed_dim = 512
cfg.num_heads = 16

model = LazyCall(VisionTransformer)(cfg=cfg)
