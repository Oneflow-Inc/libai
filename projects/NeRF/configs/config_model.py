from omegaconf import DictConfig
from libai.config import LazyCall

from projects.NeRF.modeling.System import NerfSystem


cfg = dict(
    D=8,
    W=256,
    in_channels_xyz=63,
    in_channels_dir=27,
    skips=[4],
    N_samples=64,
    use_disp=False,
    perturb=1.0,
    noise_std=0.0,
    N_importance=128,
    chunk=64 * 1204,
    dataset_type="Blender",
)

cfg = DictConfig(cfg)

model = LazyCall(NerfSystem)(cfg=cfg)
