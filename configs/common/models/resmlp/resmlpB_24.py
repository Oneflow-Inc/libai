from .resmlp_12 import model

model.patch_size = 8
model.embed_dim = 768
model.depth = 24
model.init_scale = 1e-6
