from omegaconf import OmegaConf

from libai.config import LazyCall, get_config
from configs.common.data.imagenet import dataloader
from .models.vit_base_patch16 import model
from ..utils.scheduler import warmup_layerscale_cosine_lr_scheduler

from flowvision.data import Mixup
from flowvision.loss.cross_entropy import SoftTargetCrossEntropy

# Path to the weight for fine-tune
finetune = OmegaConf.create()
finetune.enable = True  # only load weight if enable is True
finetune.weight_style = "pytorch"  # Set "oneflow" for loading oneflow weights, set "pytorch" for loading torch weights
# finetune.path = "/path/to/pretrained_mae_weight"
finetune.path = "/home/rentianhe/code/OneFlow-Models/libai/mae_pretrain_vit_base.pth"

# Get train, optim and graph configs
train = get_config("common/train.py").train
optim = get_config("common/optim.py").optim
graph = get_config("common/models/graph.py").graph

# Refine data path to imagenet
dataloader.train.dataset[0].root = "/path/to/imagenet"
dataloader.test[0].dataset.root = "/path/to/imagenet"
dataloader.train.dataset[0].root = "/dataset/extract"
dataloader.test[0].dataset.root = "/dataset/extract"

# Graph training
graph.enabled = False


# Add Mixup Func
dataloader.train.mixup_func = LazyCall(Mixup)(
    mixup_alpha=0.8,
    cutmix_alpha=1.0,
    prob=1.0,
    switch_prob=0.5,
    mode="batch",
    label_smoothing=0.1,
    num_classes=1000,
)

# Refine model cfg for vit training on imagenet
model.num_classes = 1000
model.depth = 3
model.loss_func = LazyCall(SoftTargetCrossEntropy)()

# Refine training settings for MAE finetune
train.train_micro_batch_size = 64
train.train_epoch = 100
train.warmup_ratio = 5 / 100
train.log_period = 1
train.eval_period = 1000

# Set layer decay for MAE fine-tune
train.layer_decay = 0.75

# Base learning in MAE is set to 1.5e-4
# The actually learning rate should be computed by linear scaling rule: lr = base_lr * batch_size / 256
# In LiBai, you should refine the actually learning rate due to your on settings
# Here we use 8 GPUs, 128 batch_size per GPU for training, batch_size equals to 1024
base_lr = 1e-3
actual_lr = base_lr * (train.train_micro_batch_size * 8 / 256)


# Refine optim settings
optim.params.clip_grad_max_norm = None
optim.params.clip_grad_norm_type = None
optim.params.weight_decay_norm = None
optim.params.weight_decay_bias = None
optim.lr = actual_lr
optim.weight_decay = 0.05
optim.betas = (0.9, 0.999)


# Refine scheduler
train.scheduler._target_ = warmup_layerscale_cosine_lr_scheduler
train.scheduler.warmup_factor = 0.001
train.scheduler.alpha = 0.
train.scheduler.warmup_method = "linear"


# Set fp16 ON
train.amp.enabled = True