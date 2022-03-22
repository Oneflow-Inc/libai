from libai.config import LazyCall

from modeling.MoCo_v3 import MoCo_ViT
from modeling.vit_moco import VisionTransformerMoCo


base_encoder = LazyCall(VisionTransformerMoCo)(
                img_size=224,
                patch_size=16,
                in_chans=3,
                embed_dim=768,
                depth=12,
                num_heads=12,
                mlp_ratio=4,
                drop_path_rate=0.1,
                global_pool=False,
                stop_grad_conv1=False
            )

momentum_encoder = LazyCall(VisionTransformerMoCo)(
                img_size=224,
                patch_size=16,
                in_chans=3,
                embed_dim=768,
                depth=12,
                num_heads=12,
                mlp_ratio=4,
                drop_path_rate=0.1,
                global_pool=False,
                stop_grad_conv1=False
            )


model = LazyCall(MoCo_ViT)(
            base_encoder=base_encoder, 
            momentum_encoder=momentum_encoder,
            dim=256, 
            mlp_dim=4096, 
            T=.2,
            m = 0.99
)