from libai.config import LazyCall
from libai.models import SwinTransformer
from .swin_tiny_patch4_window7_224 import cfg


cfg.img_size = 256
cfg.num_heads = [4, 8, 16, 32]
cfg.window_size = 8

model = LazyCall(SwinTransformer)(cfg=cfg)
