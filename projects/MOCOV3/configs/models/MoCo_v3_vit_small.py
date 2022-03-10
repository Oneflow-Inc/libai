from projects.MOCOV3.modeling.MoCo_v3 import MoCo_ViT
from libai.config import LazyCall
from projects.MOCOV3.modeling.vit_moco import VisionTransformerMoCo


model = LazyCall(MoCo_ViT)(
            base_encoder=LazyCall(VisionTransformerMoCo)(), 
            momentum_encoder=LazyCall(VisionTransformerMoCo)(),
            dim=256, 
            mlp_dim=4096, 
            T=.2,
            m = 0.99
)
