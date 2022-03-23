from libai.config import LazyCall
from .common.models.swin.swin_tiny_patch4_window7_224 import model
from .common.models.graph import graph
from .common.train import train
from .common.optim import optim
from .common.data.cifar100 import dataloader, train_aug

from flowvision.data import transforms_imagenet_train, Mixup
from flowvision.loss.cross_entropy import SoftTargetCrossEntropy

train_aug = LazyCall(transforms_imagenet_train)(
    img_size=224,
    color_jitter=0.4,
    auto_augment="rand-m9-mstd0.5-inc1",
    re_prob=0.25,
    re_mode="pixel",
    re_count=1,
    interpolation="bicubic",
)

dataloader.train.dataset[0].transform = train_aug

# Add Mixup Func
dataloader.train.mixup_func = LazyCall(Mixup)(
    mixup_alpha=0.8,
    cutmix_alpha=1.0,
    cutmix_minmax=None,
    prob=1.0,
    switch_prob=0.5,
    mode="batch",
    label_smoothing=0.1,
    num_classes=100,
)

# Refine model cfg for vit training on cifar100
model.num_classes = 100
model.loss_func = LazyCall(SoftTargetCrossEntropy)()

# Refine optimizer cfg for swin model
optim.lr = 5e-4
optim.eps = 1e-8
optim.weight_decay = 0.05
optim.params.clip_grad_max_norm = 1.0

# Refine train cfg for swin model
train.train_micro_batch_size = 32
train.test_micro_batch_size = 32
train.train_epoch = 300
train.warmup_ratio = 20 / 300
train.evaluation.eval_period = 200
train.log_period = 1

# Scheduler
train.scheduler.warmup_factor = 5e-7
train.scheduler.alpha = 0.0
train.scheduler.warmup_method = "linear"

# Set fp16 ON
train.amp.enabled = False
graph.enabled = False
