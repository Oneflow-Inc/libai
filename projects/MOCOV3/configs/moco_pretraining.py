from flowvision import transforms

from libai.config import LazyCall

# from projects.MOCOV3.configs.models.MoCo_v3_vit_small_patch16 import model
from projects.MOCOV3.configs.models.MoCo_v3_vit_base_patch16 import model
from projects.MOCOV3.transform.pretraining_transform import TwoCropsTransform, augmentation1, augmentation2
from configs.common.data.imagenet import dataloader
from configs.common.optim import optim
from configs.common.train import train
from configs.common.models.graph import graph


# Refine data path to imagenet
dataloader.train.dataset[0].root = "/dataset/extract"
dataloader.test[0].dataset.root = "/dataset/extract"

dataloader.train.mixup_func = None

# Add augmentation Func
dataloader.train.dataset[0].transform=LazyCall(TwoCropsTransform)(
                                               base_transform1=LazyCall(transforms.Compose)(transforms=augmentation1),
                                               base_transform2=LazyCall(transforms.Compose)(transforms=augmentation2))


# Refine optimizer cfg for moco v3 model
optim.lr = 1.5e-4
optim.eps = 1e-8
optim.weight_decay = .1
# Refine train cfg for moco v3 model
train["train_micro_batch_size"] = 64
train["test_micro_batch_size"] = 64
train["train_epoch"] = 300
train["warmup_ratio"] = 40 / 300
train["eval_period"] = 1000
train["log_period"]  =1
train["pretrain_path"] = "projects/MOCOV3/output/vit-s-300ep.pth.tar" 
train["linear_prob_path"] = "projects/MOCOV3/output/linear-vit-s-300ep.pth.tar"


# Scheduler
train["scheduler"]["warmup_factor"] = 0.001
train["scheduler"]["alpha"] = 1.5e-4
train["scheduler"]["warmup_method"] = "linear"

# Set fp16 ON
# train.amp.enabled = True
train['amp']['enabled']=True

# graph.enabled = False
graph['enabled'] = False