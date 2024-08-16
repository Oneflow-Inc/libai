from omegaconf import DictConfig
from libai.config import LazyCall
from libai.models import ResMLP


cfg = dict(
    img_size=224,
    patch_size=16,
    in_chans=3,
    embed_dim=384,
    depth=12,
    drop_rate=0.0,
    drop_path_rate=0.05,
    init_scale=0.1,
    num_classes=1000,
    loss_func=None,
)

cfg = DictConfig(cfg)

model = LazyCall(ResMLP)(cfg=cfg)
