from operator import mod
import random
from omegaconf import OmegaConf
import oneflow as flow
from flowvision import transforms

from libai.config import LazyCall, get_config
# from projects.MOCOV3.configs.models.MoCo_v3_vit_small import model
from configs.common.data.imagenet import dataloader
from projects.MOCOV3.trainsform.finetune_transform import train_augmentation, test_augmentation
from configs.common.optim import optim
from configs.common.train import train
from configs.common.models.graph import graph
from projects.MOCOV3.modeling.vit_moco import VisionTransformerMoCo
from utils.weight_convert_tools import load_torch_checkpoint
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

# # Add augmentation Func
# dataloader.train.dataset[0].transform=LazyCall(transforms.Compose)(transforms=train_augmentation)
# dataloader.test.dataset[0].transform=LazyCall(transforms.Compose)(transforms=test_augmentation)

# freeze all layers but the last head

# linear_keyword = "head"
# for name, param in VisionTransformerMoCo().named_parameters():
#     if name not in ['%s.weight' % linear_keyword, '%s.bias' % linear_keyword]:
#         param.requires_grad = False

# # init the head layer
# getattr(VisionTransformerMoCo(), linear_keyword).weight.data.normal_(mean=0.0, std=0.01)
# getattr(VisionTransformerMoCo(), linear_keyword).bias.data.zeros_()


model = LazyCall(VisionTransformerMoCo)()

# train["pretrain_path"] = "projects/MOCOV3/output/vit-s-300ep.pth.tar" 
# train["linear_prob_path"] = "projects/MOCOV3/output/linear-vit-s-300ep.pth.tar"

# print("=> loading checkpoint '{}'".format(train["pretrain_path"]))
# pretrain_state_dict = load_torch_checkpoint(model,train["pretrain_path"])

# # rename moco pre-trained keys
# for k in list(pretrain_state_dict.keys()):
#     # retain only base_encoder up to before the embedding layer
#     if k.startswith('module.base_encoder') and not k.startswith('module.base_encoder.%s' % linear_keyword):
#         # remove prefix
#         pretrain_state_dict[k[len("module.base_encoder."):]] = pretrain_state_dict[k]
#     # delete renamed or unused k
#     del pretrain_state_dict[k]

# model = model.load_state_dict(pretrain_state_dict, strict=False)

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

# Scheduler
train["scheduler"]["warmup_factor"] = 0.001
train["scheduler"]["alpha"] = 1.5e-4
train["scheduler"]["warmup_method"] = "linear"

# Set fp16 ON
# train.amp.enabled = True
train['amp']['enabled']=True

# graph.enabled = False
graph['enabled'] = False

print("no prob on moco_fintune")