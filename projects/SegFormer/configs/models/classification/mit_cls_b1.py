from projects.SegFormer.configs.models.classification.mit_cls_b0 import cfg

cfg.embed_dims=[64, 128, 320, 512]
cfg.decoder_in_channels=[64, 128, 320, 512]
