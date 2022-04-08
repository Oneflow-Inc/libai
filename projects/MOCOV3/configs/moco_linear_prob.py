from oneflow.optim import SGD
from flowvision.transforms import transforms

from libai.config import get_config, LazyCall
from .models.vit_small_patch16 import model
from ..transform.linear_prob_transform import train_aug


dataloader = get_config("common/data/imagenet.py").dataloader
train = get_config("common/train.py").train
graph = get_config("common/models/graph.py").graph
optim = get_config("common/optim.py").optim

# Path to the weight for fine-tune
model.linear_prob = "path/to/pretrained_weight"
model.weight_style = "oneflow"

# Refine data path to imagenet
dataloader.train.dataset[0].root = "/path/to/imagenet/"
dataloader.test[0].dataset.root = "/path/to/imagenet/"

# Add augmentation Func
dataloader.train.dataset[0].transform = LazyCall(transforms.Compose)(transforms=train_aug)

# Refine train cfg for moco v3 model
train.train_micro_batch_size = 128
train.test_micro_batch_size = 32
train.train_epoch = 90
train.log_period = 1
train.evaluation.eval_period = 1000

optim._target_ = SGD
optim.params.clip_grad_max_norm = None
optim.params.clip_grad_norm_type = None
optim.params.weight_decay_norm = None
optim.params.weight_decay_bias = None

del optim.betas
del optim.eps
del optim.do_bias_correction

# Refine optimizer cfg for moco v3 model
# Reference:
# https://github.com/facebookresearch/moco-v3/blob/main/CONFIG.md
# https://github.com/facebookresearch/moco-v3/blob/main/main_lincls.py
base_lr = 3.0
actual_lr = base_lr * (train.train_micro_batch_size * 8 / 256)
optim.lr = actual_lr
optim.weight_decay = 0.0
optim.momentum = 0.9

# Scheduler
train.scheduler.warmup_iter = 0
train.scheduler.alpha = 0

graph.enabled = False
