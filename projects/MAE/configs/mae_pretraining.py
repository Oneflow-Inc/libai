from omegaconf import OmegaConf

from configs.common.data.imagenet import dataloader
from configs.common.optim import optim
from configs.common.train import train
from configs.common.models.graph import graph

from models.mae_vit_base_patch16 import model


# Refine data path to imagenet
dataloader.train.dataset[0].root = "/path/to/imagenet"
dataloader.test[0].dataset.root = "/path/to/imagenet"