from .vit_tiny_patch16_224 import model

model.patch_size = 14
model.embed_dim = 1408
model.mlp_ratio = 48 / 11
model.depth = 40
model.num_heads = 16
