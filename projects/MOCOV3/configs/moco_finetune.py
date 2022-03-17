from operator import mod
import random
from omegaconf import OmegaConf
import oneflow as flow
from flowvision import transforms

from libai.config import LazyCall, get_config
# from projects.MOCOV3.configs.models.MoCo_v3_vit_small import model
from configs.common.data.imagenet import dataloader
# from projects.MOCOV3.transform.finetune_transform import train_augmentation, test_augmentation
from configs.common.optim import optim
from configs.common.train import train
from configs.common.models.graph import graph
from .models.vit import model
# train = get_config("common/train.py").train
# optim = get_config("common/optim.py").optim
# graph = get_config("common/models/graph.py").graph

# Path to the weight for fine-tune
# finetune = OmegaConf.create()
# finetune.enable = True  # only load weight if enable is True
# finetune.weight_style = "pytorch"  # Set "oneflow" for loading oneflow weights, set "pytorch" for loading torch weights
# # finetune.path = "/path/to/pretrained_mae_weight"
# finetune.path = "projects/MOCOV3/output/vit-s-300ep.pth.tar"

# Refine data path to imagenet
dataloader.train.dataset[0].root = "/dataset/extract"
dataloader.test[0].dataset.root = "/dataset/extract"

dataloader.train.mixup_func = None

# train["load_weight"] = "/dataset/czq_home/projects/libai/projects/MOCOV3/output" 
# train["linear_prob_path"] = "projects/MOCOV3/output/linear-vit-s-300ep.pth.tar"
# train["output_dir"] = "projects/MOCOV3/output/"

print("set hyper-parameters")
# Refine optimizer cfg for moco v3 model
optim.lr = 1.5e-4
optim.eps = 1e-8
optim.weight_decay = .1
# Refine train cfg for moco v3 model
train["train_micro_batch_size"] = 64
train["test_micro_batch_size"] = 64
train["train_epoch"] = 1
train["warmup_ratio"] = 40 / 300
train["eval_period"] = 1000
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
