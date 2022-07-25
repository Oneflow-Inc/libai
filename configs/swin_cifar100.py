from libai.config import LazyCall
from .common.models.swin.swin_tiny_patch4_window7_224 import model
from .common.models.graph import graph
from .common.train import train
from .common.optim import optim
from .common.data.cifar100 import dataloader

from flowvision.data import Mixup
from flowvision.loss.cross_entropy import SoftTargetCrossEntropy

# Add Mixup Func
dataloader.train.mixup_func = LazyCall(Mixup)(
    mixup_alpha=0.8,
    cutmix_alpha=1.0,
    prob=1.0,
    switch_prob=0.5,
    mode="batch",
    num_classes=100,
)

# Refine model cfg for vit training on cifar100
model.cfg.num_classes = 100
model.cfg.loss_func = SoftTargetCrossEntropy()

# Refine optimizer cfg for swin model
optim.lr = 5e-4
optim.eps = 1e-8
optim.weight_decay = 0.05
optim.params.clip_grad_max_norm = None
optim.params.clip_grad_norm_type = None

# Refine train cfg for swin model
train.train_micro_batch_size = 32
train.num_accumulation_steps = 1
train.test_micro_batch_size = 32
train.train_epoch = 300
train.warmup_ratio = 20 / 300
train.evaluation.eval_period = 200
train.log_period = 20

# Scheduler
train.scheduler.warmup_factor = 5e-7
train.scheduler.alpha = 0.0
train.scheduler.warmup_method = "linear"

# parallel strategy settings
train.dist.data_parallel_size = 8
train.dist.tensor_parallel_size = 1
train.dist.pipeline_parallel_size = 1
train.dist.pipeline_num_layers = sum(model.cfg.depths)
train.output_dir = "./output"

# Set fp16 ON
train.amp.enabled = False
train.activation_checkpoint.enabled = False
# train.zero_optimization.enabled = True
# train.zero_optimization.stage = 1
graph.enabled = False
