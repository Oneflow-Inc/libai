from libai.config import LazyCall

from modeling.mix_transformer import MixVisionTransformer
from mit_b0 import cfg

cfg.embed_dims=[64, 128, 320, 512]
cfg.deptps=[3, 4, 18, 3]
cfg.decoder_in_channels=[64, 128, 320, 512]
cfg.decoder_embedding_dim=768


model = LazyCall(MixVisionTransformer)(cfg=cfg)
