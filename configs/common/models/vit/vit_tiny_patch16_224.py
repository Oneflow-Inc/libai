from libai.config import LazyCall

from libai.models import VisionTransformer


model = LazyCall(VisionTransformer)(
    img_size=224,
    patch_size=16,
    in_chans=3,
    embed_dim=192,
    mlp_ratio=4.0,
    depth=12,
    num_heads=3,
    drop_rate=0.0,
    attn_drop_rate=0.0,
    drop_path_rate=0.1,
    global_pool="avg_pool",
    num_classes=1000,
)
