from libai.config import LazyCall

from libai.models import VisionTransformer

cfg = dict(
    img_size=224,
    patch_size=16,
    hidden_dim=768,
    mlp_dim=3072,
    num_heads=12,
    num_layers=12,
    num_classes=1000,
    attn_dropout=0.0,
    dropout=0.1,
)

vit_model = LazyCall(VisionTransformer)(cfg=cfg)