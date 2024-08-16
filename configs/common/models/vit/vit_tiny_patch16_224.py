from omegaconf import DictConfig
from libai.config import LazyCall
from libai.models import VisionTransformer


cfg = dict(
    img_size=224,
    patch_size=16,
    in_chans=3,
    embed_dim=192,
    depth=12,
    num_heads=3,
    mlp_ratio=4.0,
    drop_rate=0.0,
    attn_drop_rate=0.0,
    drop_path_rate=0.0,
    num_classes=1000,
    loss_func=None,
)

cfg = DictConfig(cfg)

model = LazyCall(VisionTransformer)(cfg=cfg)
