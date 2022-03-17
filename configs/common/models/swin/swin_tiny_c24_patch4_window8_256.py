from .swin_tiny_patch4_window7_224 import model

model.img_size = 256
model.num_heads = [4, 8, 16, 32]
model.window_size = 8
