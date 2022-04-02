from libai.config import LazyCall
from modeling.vit import VisionTransformer

model = LazyCall(VisionTransformer)(
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        drop_path_rate=0.1,
        global_pool=False,
)