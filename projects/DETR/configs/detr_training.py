from ast import arg
import sys

from flowvision import transforms

from libai.config import get_config, LazyCall

sys.path.append("projects/DETR")


from .models.configs_detr import model, postprocessors


dataloader = get_config("common/data/coco.py").dataloader
train = get_config("common/train.py").train
graph = get_config("common/models/graph.py").graph
optim = get_config("common/optim.py").optim

# Refine data path to imagenet
dataloader.train.dataset[0].img_folder= "/dataset/mscoco_2017/train2017"
dataloader.train.dataset[0].ann_file = "/dataset/mscoco_2017/annotations/instances_train2017.json"


dataloader.test[0].dataset.img_folder = "/dataset/mscoco_2017/val2017"
dataloader.test[0].dataset.ann_file = "/dataset/mscoco_2017/annotations/instances_val2017.json"


# Refine train cfg for detr model
train.train_micro_batch_size = 2
train.test_micro_batch_size = 2
train.train_epoch = 300
train.warmup_ratio = 40 / 300
train.eval_period = 5
train.log_period = 1

# Refine optimizer cfg for detr model
base_lr = 1.5e-4
actual_lr = base_lr * (train.train_micro_batch_size * 8 / 256)
optim.lr = actual_lr
optim.weight_decay = 0.1

# Scheduler
train.scheduler.warmup_factor = 0.001
train.scheduler.alpha = 1.5e-4
train.scheduler.warmup_method = "linear"

graph.enabled = False
