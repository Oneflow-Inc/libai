from operator import mod
import random
from omegaconf import OmegaConf
import oneflow as flow
from flowvision import transforms

from configs.common.data.imagenet import dataloader
from configs.common.optim import optim
from configs.common.train import train
from configs.common.models.graph import graph
from .models.vit import model
# train = get_config("common/train.py").train
# optim = get_config("common/optim.py").optim
# graph = get_config("common/models/graph.py").graph

# Path to the weight for fine-tune
finetune = OmegaConf.create()
finetune.enable = True  # only load weight if enable is True
finetune.weight_style = "pytorch"  # Set "oneflow" for loading oneflow weights, set "pytorch" for loading torch weights
finetune.path = "projects/MOCOV3/output/vit-s-300ep.pth.tar"

linearProb = OmegaConf.create()
linearProb.enable = False  
linearProb.weight_style = "pytorch" 
linearProb.path = "projects/MOCOV3/output/linear-vit-s-300ep.pth.tar"

# Refine data path to imagenet
dataloader.train.dataset[0].root = "/dataset/extract"
dataloader.test[0].dataset.root = "/dataset/extract"

dataloader.train.mixup_func = None

# Refine optimizer cfg for moco v3 model
optim.lr = 1.5e-4
optim.eps = 1e-8
optim.weight_decay = .1
# Refine train cfg for moco v3 model
train["train_micro_batch_size"] = 128
train["test_micro_batch_size"] = 128
train["train_epoch"] = 100
train["warmup_ratio"] = 40 / 300
train["eval_period"] = 10
train["log_period"]  =1

# Scheduler
train["scheduler"]["warmup_factor"] = 0.001
train["scheduler"]["alpha"] = 1.5e-4
train["scheduler"]["warmup_method"] = "linear"

# Set fp16 ON
# train.amp.enabled = True
train['amp']['enabled']=True

# graph.enabled = False
graph['enabled'] = False
