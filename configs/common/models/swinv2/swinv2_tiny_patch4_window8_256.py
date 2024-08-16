from omegaconf import DictConfig
from libai.config import LazyCall
from libai.models import SwinTransformerV2

cfg = dict(
    img_size=256,
    patch_size=4,
    in_chans=3,
    num_classes=1000,
    embed_dim=96,
    depths=[2, 2, 6, 2],
    num_heads=[3, 6, 12, 24],
    window_size=8,
    mlp_ratio=4.0,
    qkv_bias=True,
    drop_rate=0.0,
    drop_path_rate=0.2,
    ape=False,
    patch_norm=True,
    pretrained_window_sizes=[0, 0, 0, 0],
    loss_func=None,
)

cfg = DictConfig(cfg)

model = LazyCall(SwinTransformerV2)(cfg=cfg)
