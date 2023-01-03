from libai.config import LazyCall
from omegaconf import OmegaConf
from libai.data import build_criteo_dataloader

dataloader = OmegaConf.create()

dataloader.train = LazyCall(build_criteo_dataloader)(
    data_path = "/path/to/train",
    batch_size = 55296,
    shuffle=True,
)

dataloader.validation = LazyCall(build_criteo_dataloader)(
    data_path = "/path/to/val",
    batch_size = 55296,
    shuffle=False,
)

dataloader.test = LazyCall(build_criteo_dataloader)(
    data_path = "/path/to/test",
    batch_size = 55296,
    shuffle=False,
)
