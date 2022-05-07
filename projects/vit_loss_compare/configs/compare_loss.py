from datetime import date
from libai.config import LazyCall, get_config
from libai.scheduler.lr_scheduler import WarmupMultiStepLR
from libai.optim import get_default_optimizer_params
from libai.data.datasets.cifar import CIFAR10Dataset

import oneflow as flow
from oneflow.nn import CrossEntropyLoss
from flowvision.transforms import transforms
from flowvision.transforms import InterpolationMode
from flowvision.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from data.build import build_image_train_loader

model = get_config("common/models/vit/vit_tiny_patch16_224.py").model
graph = get_config("common/models/graph.py").graph
train = get_config("common/train.py").train
train.seed = 0  # 固定seed为0
dataloader = get_config("common/data/imagenet.py").dataloader

dataloader.train._target_ = build_image_train_loader
dataloader.train.dataset[0]._target_ = CIFAR10Dataset
dataloader.train.dataset[0].download = True
# Remove test dataset
del dataloader.test

graph.enabled = True

# 数据相关的设置
no_augmentation_transform = LazyCall(transforms.Compose)(
    transforms=[
        LazyCall(transforms.Resize)(
            size=(224, 224),
            interpolation=InterpolationMode.BILINEAR,
        ),
        LazyCall(transforms.CenterCrop)(
            size=(224, 224),
        ),
        LazyCall(transforms.Normalize)(
            mean=IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_DEFAULT_STD,
        ),
        LazyCall(transforms.ToTensor)(),
    ]
)
dataloader.train.dataset[0].transform = no_augmentation_transform

dataloader.train.dataset[0].root = "./"


# 模型设置: 关闭dropout等任何随机性的部分
model.num_classes = 10
model.loss_func = LazyCall(CrossEntropyLoss)()  # 使用最简单的Loss
model.embed_dim = 192
model.num_heads = 3
model.drop_rate = 0.0
model.attn_drop_rate = 0.0
model.drop_path_rate = 0.0

# 将clip grad相关的参数设置为None
optim = LazyCall(flow.optim.AdamW)(
    parameters=LazyCall(get_default_optimizer_params)(
        # parameters.model is meant to be set to the model object,
        # before instantiating the optimizer.
        clip_grad_max_norm=None,
        clip_grad_norm_type=None,
        weight_decay_norm=None,
        weight_decay_bias=None,
    ),
    lr=1e-4,
    weight_decay=1e-8,
    betas=(0.9, 0.999),
    do_bias_correction=True,
)

# 对齐batchsize, 总的train_iter等数据
train.train_micro_batch_size = 32
train.test_micro_batch_size = 128
train.train_iter = 1000
train.log_period = 1

# 将scheduler的milestones设大, 以达到constant LR的目的
train.scheduler = LazyCall(WarmupMultiStepLR)(
    max_iter=1000, warmup_iter=0, warmup_factor=0.0001, milestones=[0.999999]
)

# Set fp16 ON
train.amp.enabled = False


today = date.today()
train.output_dir = f"loss_compare/vit_loss_compare/{today}"
