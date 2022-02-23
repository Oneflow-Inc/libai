from libai.config import LazyCall

from libai.models import SwinTransformer


swin_base_patch4_window7_224_model = LazyCall(SwinTransformer)(
    img_size=224,
    patch_size=4,
    in_chans=3,
    num_classes=1000,
    embed_dim=128,
    depths=[2, 2, 18, 2],
    num_heads=[4, 8, 16, 32],
    window_size=7,
    mlp_ratio=4.0,
    qkv_bias=True,
    qk_scale=None,
    drop_rate=0.0,
    drop_path_rate=0.5,
    ape=False,
    patch_norm=True,
    use_checkpoint=False,
)
