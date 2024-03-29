from libai.config import LazyCall
from projects.ConvNeXT.configs.convnext import model
from configs.common.models.graph import graph
from configs.common.train import train
from configs.common.optim import optim
from configs.common.data.imagenet import dataloader

from flowvision.data import Mixup

# Refine data path to imagenet
dataloader.train.dataset[0].root = "/data/dataset/ImageNet/extract"
dataloader.test[0].dataset.root = "/data/dataset/ImageNet/extract"

# Add Mixup Func
dataloader.train.mixup_func = LazyCall(Mixup)(
    mixup_alpha=0.8,
    cutmix_alpha=1.0,
    prob=1.0,
    switch_prob=0.5,
    mode="batch",
    num_classes=model.cfg.num_labels,
)

# Refine model cfg for vit training on cifar100
model.cfg.num_lables = 1000

# Add Mixup Func
dataloader.train.mixup_func = LazyCall(Mixup)(
    mixup_alpha=0.8,
    cutmix_alpha=1.0,
    prob=1.0,
    switch_prob=0.5,
    mode="batch",
    num_classes=model.cfg.num_labels,
)

# Refine optimizer cfg for vit model
optim.lr = 1e-3  # 5e-4 * 1024 (batchsize) / 512
optim.eps = 1e-8
optim.weight_decay = 0.05
optim.params.clip_grad_max_norm = None
optim.params.clip_grad_norm_type = None
optim.params.overrides = {"pos_embed": {"weight_decay": 0.0}, "cls_token": {"weight_decay": 0.0}}

# Refine train cfg for vit model
train.train_micro_batch_size = 128
train.test_micro_batch_size = 128
train.train_epoch = 300
train.warmup_ratio = 5 / 300
train.evaluation.eval_period = 1000
train.log_period = 1

# Scheduler
train.scheduler.warmup_factor = 0.001
train.scheduler.alpha = 0.01
train.scheduler.warmup_method = "linear"

# Set fp16 ON
train.amp.enabled = True

# Distributed Settings
# train.dist.pipeline_num_layers = 4
train.dist.data_parallel_size = 1
train.dist.tensor_parallel_size = 1
train.dist.pipeline_parallel_size = 1
