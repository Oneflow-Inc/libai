from libai.config import get_config
from .optim.optim_finetune import optim

from .models.vit_small_patch16 import model

dataloader = get_config("common/data/imagenet.py").dataloader
train = get_config("common/train.py").train
graph = get_config("common/models/graph.py").graph

# Path to the weight for fine-tune
model.finetune = "projects/MOCOV3/output/vit-s-300ep.pth.tar"
model.weight_style = "pytorch"

# Refine data path to imagenet
dataloader.train.dataset[0].root = "/dataset/extract"
dataloader.test[0].dataset.root = "/dataset/extract"

# Refine optimizer cfg for moco v3 model
# Reference:
# https://github.com/facebookresearch/moco-v3/blob/main/CONFIG.md
# https://github.com/facebookresearch/moco-v3/blob/main/main_lincls.py

base_lr = 3.
actual_lr = base_lr * (train.train_micro_batch_size * 8 / 256)

optim.lr = actual_lr
optim.weight_decay = 0.
optim.momentum = .9

# Refine train cfg for moco v3 model
train.train_micro_batch_size = 128
train.test_micro_batch_size = 32
train.train_epoch = 100
train.warmup_ratio = 5 / 100
train.log_period = 1
train.evaluation.eval_period = 1000

# Scheduler
train.scheduler.warmup_factor = 0.001
train.scheduler.alpha = 1.5e-4
train.scheduler.warmup_method = "linear"

graph.enabled = False
