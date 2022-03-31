from omegaconf import OmegaConf

from flowvision.data import Mixup
from flowvision.loss.cross_entropy import SoftTargetCrossEntropy

from libai.config import LazyCall, get_config
from .models.vit_base_patch16 import model
from ..utils.scheduler import warmup_layerscale_cosine_lr_scheduler
from ..utils.lr_decay import param_groups_lrd


# Path to the weight for fine-tune
finetune = OmegaConf.create()
finetune.enable = True  # only load weight if enable is True
finetune.weight_style = (
    "oneflow"  # Set "oneflow" for loading oneflow weights, set "pytorch" for loading torch weights
)
finetune.path = "/path/to/pretrained_mae_weight"


# Get train, optim and graph configs
train = get_config("common/train.py").train
optim = get_config("common/optim.py").optim
graph = get_config("common/models/graph.py").graph
dataloader = get_config("common/data/imagenet.py").dataloader


# Refine data path to imagenet
dataloader.train.dataset[0].root = "/path/to/imagenet"
dataloader.test[0].dataset.root = "/path/to/imagenet"


# Graph training
graph.enabled = False


# Refine model cfg for vit training on imagenet
model.num_classes = 1000
model.loss_func = LazyCall(SoftTargetCrossEntropy)()


# Add Mixup Func
dataloader.train.mixup_func = LazyCall(Mixup)(
    mixup_alpha=0.8,
    cutmix_alpha=1.0,
    prob=1.0,
    switch_prob=0.5,
    mode="batch",
    label_smoothing=0.1,
    num_classes=model.num_classes,
)

# Refine training settings for MAE finetune
train.train_micro_batch_size = 128
train.test_micro_batch_size = 32
train.train_epoch = 100
train.warmup_ratio = 5 / 100
train.log_period = 1
train.evaluation.eval_period = 1000

# Set layer decay for MAE fine-tune
train.layer_decay = 0.75

# Base learning in MAE is set to 1.5e-4
# The actually learning rate should be computed by linear scaling rule as follows:
# lr = base_lr * batch_size / 256
# In LiBai, you should refine the actually learning rate due to your on settings
# Here we use 8 GPUs, 128 batch_size per GPU for training, batch_size equals to 1024
base_lr = 1e-3
actual_lr = base_lr * (train.train_micro_batch_size * 8 / 256)


# Refine optim settings
optim.params._target_ = param_groups_lrd
optim.params.weight_decay = 0.05
optim.params.layer_decay = 0.75
optim.lr = actual_lr
optim.weight_decay = 0.05
optim.betas = (0.9, 0.999)

del optim.params.clip_grad_max_norm
del optim.params.clip_grad_norm_type
del optim.params.weight_decay_norm
del optim.params.weight_decay_bias


# Refine scheduler
train.scheduler._target_ = warmup_layerscale_cosine_lr_scheduler
train.scheduler.warmup_factor = 0.001
train.scheduler.alpha = 0.0
train.scheduler.warmup_method = "linear"


# Distributed Settings
train.dist.pipeline_num_layers = model.depth
train.dist.data_parallel_size = 1
train.dist.tensor_parallel_size = 1
train.dist.pipeline_parallel_size = 1
