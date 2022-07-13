from libai.config import LazyCall
from libai.models import ResMLP

from .resmlp_12 import cfg


cfg.depth = 24
cfg.init_scale = 1e-5

model = LazyCall(ResMLP)(cfg=cfg)
