from .vit_base_patch16 import model


model.embed_dim = 384
model.depth = 12
model.num_heads = 12
model.drop_path_rate = 0.0
