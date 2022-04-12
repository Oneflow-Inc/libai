from flowvision import transforms

from libai.config import get_config, LazyCall

from models.detr_res50 import model, criterion, postprocessors


dataloader = get_config("common/data/coco.py").dataloader
train = get_config("common/train.py").train
graph = get_config("common/models/graph.py").graph
optim = get_config("common/optim.py").optim

print(dataloader)

# Refine data path to imagenet
dataloader.train.dataset[0].root = "/DATA/disk1/datasets/mscoco_2017/test2017"
dataloader.test[0].dataset.root = "/DATA/disk1/datasets/mscoco_2017/test2017"


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
