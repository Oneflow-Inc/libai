from omegaconf import DictConfig
from libai.config import LazyCall
from modeling.mix_transformer import MixVisionTransformer

cfg = dict(
    img_size=224,
    patch_size=[7,3,3,3],
    in_chans=3,
    num_classes=150,
    embed_dims=[32, 64, 160, 256],
    num_heads=[1, 2, 5, 8],
    mlp_ratios=[4, 4, 4, 4],
    qkv_bias=True,
    qk_scale=False,
    drop_rate=0.0,
    attn_drop_rate=0.0,
    drop_path_rate=0.1,
    depths=[2, 2, 2, 2],
    sr_ratios=[8, 4, 2, 1],
    loss_func=None,
    decoder_in_channels=[32, 64, 160, 256],
    decoder_embedding_dim=256,
    decoder_dropout_prob=0.1
)

cfg = DictConfig(cfg)

model = LazyCall(MixVisionTransformer)(cfg=cfg)
