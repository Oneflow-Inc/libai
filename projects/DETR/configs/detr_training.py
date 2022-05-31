from statistics import mode
from libai.config import get_config, LazyCall
from libai.data.datasets.coco import CocoDetection
from .models.configs_detr import model, postprocessors
from ..datasets.coco_eval import CocoEvaluator, get_coco_api_from_dataset
from libai.config.configs.common.data.coco import make_coco_transforms


dataloader = get_config("common/data/coco.py").dataloader
train = get_config("common/train.py").train
graph = get_config("common/models/graph.py").graph
optim = get_config("common/optim.py").optim

# Refine data path to imagenet
dataloader.train.dataset[0].img_folder= "/dataset/coco/train2017"
dataloader.train.dataset[0].ann_file = "/dataset/coco/annotations/instances_train2017.json"
dataloader.train.dataset_mixer = None

dataloader.test[0].dataset.img_folder = "/dataset/coco/val2017"
dataloader.test[0].dataset.ann_file = "/dataset/coco/annotations/instances_val2017.json"


# For inference
# train.load_weight = "projects/DETR/checkpoint/detr-r50-e632da11.pth"

# Refine train cfg for detr model
train.train_micro_batch_size = 2
train.test_micro_batch_size = 2
train.train_epoch = 10
# train.warmup_ratio = 40 / 300
train.eval_period = 5
train.log_period = 1

train.checkpointer["period"]=100

# *TODO: refine it
coco_detection = LazyCall(CocoDetection)(img_folder="/dataset/coco/val2017", 
                               ann_file="/dataset/coco//annotations/instances_val2017.json", 
                               return_masks=False, transforms=make_coco_transforms("val"))

train.evaluation.evaluator = LazyCall(CocoEvaluator)(coco_detection=coco_detection)


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

# model_parallel
# train.dist.data_parallel_size = 1
# train.dist.tensor_parallel_size = 2
