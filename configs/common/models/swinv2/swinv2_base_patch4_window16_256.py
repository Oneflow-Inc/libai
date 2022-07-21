from libai.config import LazyCall
from libai.models import SwinTransformerV2

model = LazyCall(SwinTransformerV2)(
    img_size=256,
    patch_size=4,
    in_chans=3,
    num_classes=1000,
    embed_dim=96,
    depths=[2, 2, 18, 2],
    num_heads=[4, 8, 16, 32],
    window_size=16,
    mlp_ratio=4.0,
    qkv_bias=True,
    drop_rate=0.0,
    drop_path_rate=0.5,
    ape=False,
    patch_norm=True,
    use_checkpoint=False,
    pretrained_window_sizes=[0, 0, 0, 0],
)
