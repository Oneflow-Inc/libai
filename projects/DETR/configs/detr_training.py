from statistics import mode
from libai.config import get_config, LazyCall
from ..datasets.coco_detection import CocoDetection
from ..datasets.coco_dataloader import dataloader, make_coco_transforms
from .models.configs_detr_resnet50 import model, postprocessors
from ..datasets.coco_eval import CocoEvaluator, get_coco_api_from_dataset


# dataloader = get_config("common/data/coco.py").dataloader
train = get_config("common/train.py").train
graph = get_config("common/models/graph.py").graph
optim = get_config("common/optim.py").optim

# Refine data path to imagenet
path_train_img = "/dataset/coco/train2017"
path_train_ann = "/dataset/coco/annotations/instances_train2017.json"

path_val_img = "/dataset/coco/val2017"
path_val_ann = "/dataset/coco/annotations/instances_val2017.json"

dataloader.train.dataset[0].img_folder= path_train_img
dataloader.train.dataset[0].ann_file = path_train_ann
dataloader.train.dataset_mixer = None

dataloader.test[0].dataset.img_folder = path_val_img
dataloader.test[0].dataset.ann_file = path_val_ann


# For inference
# train.load_weight = "projects/DETR/checkpoint/detr-r50-e632da11.pth"

# Refine train cfg for detr model
train.train_micro_batch_size = 2
train.test_micro_batch_size = 2
train.train_iter=200
train.evaluation.eval_period = 10

coco_detection = LazyCall(CocoDetection)(
    img_folder = path_val_img, 
    ann_file = path_val_ann, 
    return_masks = False,
    transforms = make_coco_transforms("val")
    )
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
train.dist.data_parallel_size = 1
train.dist.tensor_parallel_size = 2
