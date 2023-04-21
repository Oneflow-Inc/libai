from omegaconf import DictConfig
from libai.config import LazyCall
from projects.ConvNeXT.modeling.convnext_model import ConvNextModel


cfg = dict(
    num_channels=3,
    patch_size=4,
    num_stages=4,
    hidden_sizes=None,
    depths=None,
    layer_norm_eps=1e-12,
    drop_path_rate=0.0,
    image_size=224,
    num_labels=None,
    problem_type=None,
)

cfg = DictConfig(cfg)

model = LazyCall(ConvNextModel)(cfg=cfg)
