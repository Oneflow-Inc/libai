from .vit_base_patch16 import model


model.patch_size = 14
model.embed_dim = 1280
model.depth = 32
model.num_heads = 16
model.drop_path_rate = 0.2
