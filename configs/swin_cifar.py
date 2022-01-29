from .vit_imagenet import train, optim, dataloader, graph

from .common.models.swin import swin_model as model

model.num_classes = 100

optim.parameters.clip_grad_max_norm = None
optim.parameters.clip_grad_norm_type = None
optim.parameters.overrides = {
    "absolute_pos_embed": {"weight_decay": 0.0},
    "relative_position_bias_table": {"weight_decay": 0.0},
}

dataloader.train.dataset[0].root = "/workspace/datasets/"
dataloader.test[0].dataset.root = "/workspace/datasets/"

train.log_period = 1
