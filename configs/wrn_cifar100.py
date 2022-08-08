from libai.config import LazyCall
from .common.models.graph import graph
from .common.train import train
from .common.data.cifar100 import dataloader

from flowvision import transforms
import oneflow.nn as nn

CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

# Add Mixup Func
dataloader.train.mixup_func = None
dataloader.train.dataset[0].transform = LazyCall(transforms.Compose)(
    transforms=[
        LazyCall(transforms.RandomHorizontalFlip)(),
        LazyCall(transforms.ToTensor)(),
        LazyCall(transforms.Normalize)(mean=CIFAR100_TRAIN_MEAN, std=CIFAR100_TRAIN_STD),
    ]
)
dataloader.test[0].dataset.transform = LazyCall(transforms.Compose)(
    transforms=[
        LazyCall(transforms.ToTensor)(),
        LazyCall(transforms.Normalize)(
            mean=CIFAR100_TRAIN_MEAN,
            std=CIFAR100_TRAIN_STD,
        ),
    ]
)

# Refine model cfg for vit training on cifar100

from omegaconf import DictConfig
from libai.config import LazyCall
from libai.models import WideResNet

cfg = dict(
    depth=40,
    num_classes=100,
    widen_factor=2,
)

cfg = DictConfig(cfg)

model = LazyCall(WideResNet)(cfg=cfg)

model.cfg.num_classes = 100
model.cfg.loss_func = nn.CrossEntropyLoss()

# Refine optimizer cfg for swinv2 model
import oneflow as flow
from libai.optim import get_default_optimizer_params

optim = LazyCall(flow.optim.SGD)(
    params=LazyCall(get_default_optimizer_params)(
        # params.model is meant to be set to the model object,
        # before instantiating the optimizer.
    ),
    lr=1e-1,
    weight_decay=1e-4,
    momentum=True,
)


# Refine train cfg for swin model
train.train_micro_batch_size = 32
train.num_accumulation_steps = 8
train.test_micro_batch_size = 32
train.train_epoch = 300
train.warmup_ratio = 20 / 300
train.evaluation.eval_period = 1562
train.log_period = 10

# Scheduler
train.scheduler.warmup_factor = 5e-7
train.scheduler.alpha = 0.0
train.scheduler.warmup_method = "linear"

# parallel strategy settings
train.dist.data_parallel_size = 1
train.dist.tensor_parallel_size = 1
train.dist.pipeline_parallel_size = 1
train.dist.pipeline_num_layers = model.cfg.depth
train.output_dir = "./output"

train.rdma_enabled = False
# Set fp16 ON
train.amp.enabled = False
train.activation_checkpoint.enabled = False
# train.zero_optimization.enabled = True
# train.zero_optimization.stage = 1
graph.enabled = False
