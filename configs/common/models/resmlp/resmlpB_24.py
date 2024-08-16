from libai.config import LazyCall
from libai.models import ResMLP

from .resmlp_12 import cfg


cfg.patch_size = 8
cfg.embed_dim = 768
cfg.depth = 24
cfg.init_scale = 1e-6

model = LazyCall(ResMLP)(cfg=cfg)
