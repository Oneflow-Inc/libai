from libai.config import LazyCall

from libai.models import SwinTransformer


swin_mlp_tiny_c12_patch4_window8_256_model = LazyCall(SwinTransformer)(
    img_size=256,
    patch_size=4,
    in_chans=3,
    num_classes=1000,
    embed_dim=96,
    depths=[2, 2, 6, 2],
    num_heads=[8, 16, 32, 64],
    window_size=8,
    mlp_ratio=4.0,
    qkv_bias=True,
    qk_scale=None,
    drop_rate=0.0,
    drop_path_rate=0.2,
    ape=False,
    patch_norm=True,
    use_checkpoint=False,
)
