from .vit_tiny_patch16_224 import model

model.patch_size = 14
model.embed_dim = 1664
model.mlp_ratio = 64 / 13
model.depth = 48
model.num_heads = 16
