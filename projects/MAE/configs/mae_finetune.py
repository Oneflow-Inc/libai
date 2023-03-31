# coding=utf-8
# Copyright 2021 The OneFlow Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from omegaconf import OmegaConf
from flowvision.data import Mixup

# from flowvision.loss.cross_entropy import SoftTargetCrossEntropy
from libai.config import LazyCall, get_config
from modeling.cross_entropy import SoftTargetCrossEntropy
from configs.models.vit_base_patch16 import model
from utils.scheduler import (
    warmup_layerscale_cosine_lr_scheduler,
    warmup_cosine_lr_scheduler,
)
from utils.lr_decay import param_groups_lrd


# Get train, optim and graph configs
train = get_config("common/train.py").train
optim = get_config("common/optim.py").optim
graph = get_config("common/models/graph.py").graph
dataloader = get_config("common/data/imagenet.py").dataloader


# number devices
n_gpus = 8

# Graph training
graph.enabled = True

# Refine model cfg for vit training on imagenet
model.num_classes = 1000
model.loss_func = LazyCall(SoftTargetCrossEntropy)()

# Path to the weight for fine-tune
finetune = OmegaConf.create()
finetune.enable = True  # only load weight if enable is True
finetune.weight_style = (
    "oneflow"  # Set "oneflow" for loading oneflow weights, set "pytorch" for loading torch weights
)
finetune.path = "/path/to/pretrained_mae_weight"


# Refine data path to imagenet
dataloader.train.dataset[0].root = "/path/to/imagenet"
dataloader.test[0].dataset.root = "/path/to/imagenet"

# Add Mixup Func
dataloader.train.mixup_func = LazyCall(Mixup)(
    mixup_alpha=0.8,
    cutmix_alpha=1.0,
    prob=1.0,
    switch_prob=0.5,
    mode="batch",
    label_smoothing=0.1,
    num_classes=model.num_classes,
)


# Refine training settings for MAE finetune
train.train_micro_batch_size = 32
train.num_accumulation_steps = 4
train.test_micro_batch_size = 32
effective_batch_size = train.train_micro_batch_size * train.num_accumulation_steps * n_gpus

train.train_epoch = 100
train.warmup_ratio = 5 / 100
train.log_period = 20
train.evaluation.eval_after_n_epoch = 1
train.checkpointer.save_model_after_n_epoch = 1

# Set layer decay for MAE fine-tune
train.layer_decay = 0.65

# AMP
train.amp.enabled = True


# Base learning in MAE is set to 1.5e-4
# The actually learning rate should be computed by linear scaling rule as follows:
# lr = base_lr * batch_size / 256
# In LiBai, you should refine the actually learning rate due to your on settings
# Here we use 8 GPUs, 128 batch_size per GPU for training, batch_size equals to 1024
base_lr = 5e-4
actual_lr = base_lr * effective_batch_size / 256

# Refine optim settings
optim.params._target_ = param_groups_lrd
optim.params.weight_decay = 0.05
optim.params.layer_decay = 0.65
optim.lr = actual_lr

del optim.params.clip_grad_max_norm
del optim.params.clip_grad_norm_type
del optim.params.weight_decay_norm
del optim.params.weight_decay_bias
del optim.weight_decay

# Refine scheduler
if graph.enabled:
    train.scheduler = LazyCall(warmup_cosine_lr_scheduler)(
        warmup_factor=0.0,
        min_lr=1e-6,
    )
else:
    train.scheduler = LazyCall(warmup_layerscale_cosine_lr_scheduler)(
        warmup_factor=0.0,
        min_lr=1e-6,
    )


# Distributed Settings
train.dist.pipeline_num_layers = model.depth
train.dist.data_parallel_size = n_gpus
train.dist.tensor_parallel_size = 1
train.dist.pipeline_parallel_size = 1


eval_only = False
