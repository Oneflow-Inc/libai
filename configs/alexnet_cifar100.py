from libai.config import LazyCall
from .common.models.alexnet import model
from .common.models.graph import graph
from .common.train import train
from .common.optim import optim
from .common.data.cifar100 import dataloader

from flowvision.data import Mixup
from flowvision.loss.cross_entropy import SoftTargetCrossEntropy

# Add Mixup Func
# dataloader.train.mixup_func = LazyCall(Mixup)(
#     mixup_alpha=0.8,
#     cutmix_alpha=1.0,
#     prob=1.0,
#     switch_prob=0.5,
#     mode="batch",
#     num_classes=100,
# )

dataloader.train.mixup_func = None
# Refine model cfg for vit training on cifar100
model.num_classes = 100
# model.loss_func = LazyCall(SoftTargetCrossEntropy)()
model.loss_func = None

# Refine optimizer cfg for swin model
optim.lr = 5e-4
optim.eps = 1e-8
optim.weight_decay = 0.05
optim.params.clip_grad_max_norm = 1.0
optim.params.clip_grad_norm_type = 2.0

# Refine train cfg for swin model
train.train_micro_batch_size = 16
train.num_accumulation_steps = None
train.test_micro_batch_size = 16
train.train_epoch = 300
train.warmup_ratio = 20 / 300
train.evaluation.eval_period = 200
train.log_period = 20
# train.load_weight=""


# Scheduler
train.scheduler.warmup_factor = 5e-7
train.scheduler.alpha = 0.0
train.scheduler.warmup_method = "linear"

# different parallel strategy settings

train.dist.data_parallel_size=1
train.dist.tensor_parallel_size=1
train.dist.pipeline_parallel_size=2
train.dist.pipeline_num_layers=2
train.output_dir="./output_2"

# Set fp16 ON
train.amp.enabled = False
graph.enabled = False


