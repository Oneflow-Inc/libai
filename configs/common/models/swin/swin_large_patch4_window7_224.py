from libai.config import LazyCall
from libai.models import SwinTransformer

from .swin_tiny_patch4_window7_224 import cfg


cfg.embed_dim = 192
cfg.depths = [2, 2, 18, 2]
cfg.num_heads = [6, 12, 24, 48]
cfg.drop_path_rate = 0.1

model = LazyCall(SwinTransformer)(cfg=cfg)
