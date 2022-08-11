from libai.config import LazyCall
from libai.models import SwinTransformerV2
from .swinv2_tiny_patch4_window8_256 import cfg

cfg.window_size = 16

model = LazyCall(SwinTransformerV2)(cfg=cfg)
