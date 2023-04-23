from omegaconf import DictConfig
from libai.config import LazyCall
from projects.ConvNeXT.modeling.convnext_model import ConvNextForImageClassification


cfg = dict(
    num_channels=3,
    patch_size=4,
    num_stages=4,
    hidden_sizes=[96, 192, 384, 768],
    depths=[3, 3, 9, 3],
    layer_norm_eps=1e-12,
    drop_path_rate=0.0,
    image_size=224,
    num_labels=1000,
    initializer_range=0.02,
    problem_type=None,
)

cfg = DictConfig(cfg)
model = LazyCall(ConvNextForImageClassification)(cfg=cfg)
