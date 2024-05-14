import os.path as osp
import sys
from omegaconf import OmegaConf

from flowvision.data import Mixup
from flowvision.loss.cross_entropy import SoftTargetCrossEntropy

from libai.config import LazyCall, get_config

# add necessary paths to PYTHONPATH
PROJECTS_ROOT = osp.abspath(osp.join(__file__, '../../../'))
sys.path.insert(0, PROJECTS_ROOT)
sys.path.insert(0, osp.join(PROJECTS_ROOT, 'MAE'))

from MAE.configs.models.vit_base_patch16 import model      # use ViT-Base (Patch Size: 16) as the model
from MAE.utils.scheduler import warmup_layerscale_cosine_lr_scheduler
from MAE.utils.lr_decay import param_groups_lrd

from MAE_Finetune_Example.dataset.stanford_cars import dataloader  # use a custom DataLoader


# Specify the path of the weight for finetuning
finetune = OmegaConf.create()
finetune.enable = True  # only load weight if enable is True
finetune.weight_style = (
    "pytorch"  # Set "oneflow" for loading oneflow weights, set "pytorch" for loading torch weights
)
finetune.path = "MAE_Finetune_Example/pretrained_weights/mae_pretrain_vit_base.pth"


# Get train, optim and graph configs
train = get_config("common/train.py").train
optim = get_config("common/optim.py").optim
graph = get_config("common/models/graph.py").graph


# Specifiy the root directory of the dataset
dataloader.train.dataset[0].root = "/home/StanfordCars"
dataloader.test[0].dataset.root = "/home/StanfordCars"


# Graph training
graph.enabled = False


model.num_classes = 196
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
train.train_micro_batch_size = 64
train.test_micro_batch_size = 16
train.train_epoch = 300
train.warmup_ratio = 5 / 100
train.log_period = 1
train.evaluation.eval_period = 1000


# Set layer decay for MAE fine-tune
train.layer_decay = 0.75


# Base learning in MAE is set to 1.5e-4
# The actually learning rate should be computed by linear scaling rule as follows:
# lr = base_lr * batch_size / 256
# In LiBai, you should refine the actually learning rate due to your on settings
base_lr = 1e-3
actual_lr = base_lr * (train.train_micro_batch_size * 1 / 256)


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
