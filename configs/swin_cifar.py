from .vit_imagenet import train, optim, dataloader, graph

from .common.models.swin import swin_model as model

dataloader.train.dataset[0].root = "/workspace/datasets/"
dataloader.test[0].dataset.root = "/workspace/datasets/"

train.log_period = 20
