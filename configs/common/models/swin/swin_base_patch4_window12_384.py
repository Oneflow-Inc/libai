from .swin_tiny_patch4_window7_224 import model

model.img_size = 384
model.embed_dim = 128
model.depths = [2, 2, 18, 2]
model.num_heads = [4, 8, 16, 32]
model.drop_path_rate = 0.1
