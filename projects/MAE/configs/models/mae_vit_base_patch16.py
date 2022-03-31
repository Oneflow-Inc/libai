from functools import partial

from libai.config import LazyCall
from libai.layers import LayerNorm

from modeling.mae import MaskedAutoencoderViT


model = LazyCall(MaskedAutoencoderViT)(
    img_size=224,
    patch_size=16,
    in_chans=3,
    embed_dim=768,
    depth=12,
    num_heads=12,
    decoder_embed_dim=512,
    decoder_depth=8,
    decoder_num_heads=16,
    mlp_ratio=4,
    norm_layer=partial(LayerNorm, eps=1e-6),
    norm_pix_loss=False,
    mask_ratio=0.75,
)
