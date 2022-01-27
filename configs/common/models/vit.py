from libai.config import LazyCall

from libai.models import VisionTransformer

cfg = dict(
    img_size=224,
    patch_size=16,
    in_chans=3,
    embed_dim=384,
    mlp_ratio=4.0,
    depth=12,
    num_heads=6,
    qkv_bias=True,
    drop_rate=0.1,
    attn_drop_rate=0.1,
    drop_path_rate=0.1,
    num_classes=1000,
)

vit_model = LazyCall(VisionTransformer)(cfg=cfg)
