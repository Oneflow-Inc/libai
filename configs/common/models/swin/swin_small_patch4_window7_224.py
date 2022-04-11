from .swin_tiny_patch4_window7_224 import model

model.depths = [2, 2, 18, 2]
model.drop_path_rate = 0.3
