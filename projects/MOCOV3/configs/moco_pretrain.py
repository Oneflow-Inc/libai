from flowvision import transforms

from libai.config import get_config, LazyCall

from .models.moco_vit_small_patch16 import model
from transform.pretrain_transform import TwoCropsTransform, augmentation1, augmentation2


dataloader = get_config("common/data/imagenet.py").dataloader
train = get_config("common/train.py").train
graph = get_config("common/models/graph.py").graph
optim = get_config("common/optim.py").optim

# Refine data path to imagenet
dataloader.train.dataset[0].root = "/path/to/imagenet/"
dataloader.test[0].dataset.root = "/path/to/imagenet/"

# Add augmentation Func
dataloader.train.dataset[0].transform = LazyCall(TwoCropsTransform)(
    base_transform1=LazyCall(transforms.Compose)(transforms=augmentation1),
    base_transform2=LazyCall(transforms.Compose)(transforms=augmentation2),
)

# the momentum of MOCOV3
model.m = 0.99

# the temperature coefficient of MOCOV3
model.T = 0.2

# Refine train cfg for moco v3 model
train.train_micro_batch_size = 32
train.test_micro_batch_size = 32
train.train_epoch = 300
train.warmup_ratio = 40 / 300
train.eval_period = 5
train.log_period = 1

# Refine optimizer cfg for moco v3 model
base_lr = 1.5e-4
actual_lr = base_lr * (train.train_micro_batch_size * 8 / 256)
optim.lr = actual_lr
optim.weight_decay = 0.1

# Scheduler
train.scheduler.warmup_factor = 0.001
train.scheduler.alpha = 1.5e-4
train.scheduler.warmup_method = "linear"

graph.enabled = False
