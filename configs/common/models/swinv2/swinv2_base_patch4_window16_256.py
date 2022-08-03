from libai.config import LazyCall
from libai.models import SwinTransformerV2
from .swinv2_tiny_patch4_window8_256 import cfg

cfg.window_size = 16
cfg.depths = [2, 2, 18, 2]
cfg.num_heads = [4, 8, 16, 32]
cfg.drop_path_rate = 0.5

model = LazyCall(SwinTransformerV2)(cfg=cfg)
