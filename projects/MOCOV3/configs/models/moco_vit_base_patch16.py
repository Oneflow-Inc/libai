from libai.config import LazyCall
from modeling.moco import MoCo_ViT
from modeling.vit import VisionTransformer


base_encoder = LazyCall(VisionTransformer)(
    img_size=224,
    patch_size=16,
    in_chans=3,
    embed_dim=768,
    depth=12,
    num_heads=12,
    mlp_ratio=4,
    drop_path_rate=0.1,
    global_pool=False,
    stop_grad_conv1=True,
)

momentum_encoder = LazyCall(VisionTransformer)(
    img_size=224,
    patch_size=16,
    in_chans=3,
    embed_dim=768,
    depth=12,
    num_heads=12,
    mlp_ratio=4,
    drop_path_rate=0.1,
    global_pool=False,
    stop_grad_conv1=True,
)


model = LazyCall(MoCo_ViT)(
    base_encoder=base_encoder,
    momentum_encoder=momentum_encoder,
    dim=256,
    mlp_dim=4096,
    T=0.2,
    m=0.99,
)
