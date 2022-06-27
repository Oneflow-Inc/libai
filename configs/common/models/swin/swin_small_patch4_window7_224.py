from libai.config import LazyCall
from libai.models import SwinTransformer
from .swin_tiny_patch4_window7_224 import cfg


cfg.depths = [2, 2, 18, 2]
cfg.drop_path_rate = 0.3

model = LazyCall(SwinTransformer)(cfg=cfg)
