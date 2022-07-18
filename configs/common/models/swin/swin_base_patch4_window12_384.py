from libai.config import LazyCall
from libai.models import SwinTransformer

from .swin_tiny_patch4_window7_224 import cfg


cfg.img_size = 384
cfg.embed_dim = 128
cfg.depths = [2, 2, 18, 2]
cfg.num_heads = [4, 8, 16, 32]
cfg.drop_path_rate = 0.1

model = LazyCall(SwinTransformer)(cfg=cfg)
