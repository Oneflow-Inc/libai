from libai.config import LazyCall

from libai.models import SwinTransformer


swin_large_patch4_window12_384_model = LazyCall(SwinTransformer)(
    img_size=384,
    patch_size=4,
    in_chans=3,
    num_classes=1000,
    embed_dim=192,
    depths=[2, 2, 18, 2],
    num_heads=[6, 12, 24, 48],
    window_size=12,
    mlp_ratio=4.0,
    qkv_bias=True,
    qk_scale=None,
    drop_rate=0.0,
    drop_path_rate=0.1,
    ape=False,
    patch_norm=True,
    use_checkpoint=False,
)
