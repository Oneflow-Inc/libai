from flowvision import transforms

from libai.config import get_config, LazyCall

from .models.MoCo_v3_vit_small_patch16 import model
from transform.pretraining_transform import TwoCropsTransform, augmentation1, augmentation2

dataloader = get_config("common/data/imagenet.py").dataloader
train = get_config("common/train.py").train
graph = get_config("common/models/graph.py").graph
optim = get_config("common/optim.py").optim

# Refine data path to imagenet
dataloader.train.dataset[0].root = "/dataset/extract"
dataloader.test[0].dataset.root = "/dataset/extract"

dataloader.train.mixup_func = None

# Add augmentation Func
dataloader.train.dataset[0].transform=LazyCall(TwoCropsTransform)(
                                               base_transform1=LazyCall(transforms.Compose)(transforms=augmentation1),
                                               base_transform2=LazyCall(transforms.Compose)(transforms=augmentation2))


# the momentum of MOCOV3
model.m = 10

# Refine optimizer cfg for moco v3 model
optim.lr = 1.5e-4
optim.eps = 1e-8
optim.weight_decay = .1

# Refine train cfg for moco v3 model
train.train_micro_batch_size=64
train.test_micro_batch_size= 64
train.train_iter = 1
train.train_epoch = 10
train.warmup_ratio = 5 / 10
train.eval_period = 5
train.log_period  =1

# Scheduler
train.scheduler.warmup_factor = 0.001
train.scheduler.alpha = 1.5e-4
train.scheduler.warmup_method = "linear"

graph.enabled = False