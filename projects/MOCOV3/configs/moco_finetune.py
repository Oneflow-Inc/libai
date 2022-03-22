from omegaconf import OmegaConf

from libai.config import get_config, LazyCall

from .models.vit_small_patch16 import model
from projects.MOCOV3.configs.build_finetune_model import build_model

dataloader = get_config("common/data/imagenet.py").dataloader
train = get_config("common/train.py").train
graph = get_config("common/models/graph.py").graph
optim = get_config("common/optim.py").optim

# Path to the weight for fine-tune
finetune = OmegaConf.create()
finetune.enable = True  # True for finetune and False for inference
finetune.weight_style = "pytorch"  # Set "oneflow" for loading oneflow weights, set "pytorch" for loading torch weights
finetune.finetune_path = "projects/MOCOV3/output/vit-s-300ep.pth.tar"
finetune.inference_path = "projects/MOCOV3/output/linear-vit-s-300ep.pth.tar"

model = LazyCall(build_model)(finetune=finetune, model=model)

# Refine data path to imagenet
dataloader.train.dataset[0].root = "/dataset/extract"
dataloader.test[0].dataset.root = "/dataset/extract"

dataloader.train.mixup_func = None

# Refine optimizer cfg for moco v3 model
optim.lr = .1
optim.weight_decay = 0

# Refine train cfg for moco v3 model
train.train_micro_batch_size=32
train.test_micro_batch_size= 32
train.train_epoch = 90
train.warmup_ratio = 5 / 90
train.eval_period = 1
train.log_period  =1

# Scheduler
train.scheduler.warmup_factor = 0.001
train.scheduler.alpha = 1.5e-4
train.scheduler.warmup_method = "linear"

# Set fp16 ON
train.amp.enabled = True

graph.enabled = False
