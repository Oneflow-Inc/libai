from flowvision.transforms import transforms, InterpolationMode
from flowvision.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from libai.config import LazyCall, get_config
from configs.models.mae_vit_base_patch16 import model
from data.pretraining_imagenet import PretrainingImageNetDataset
from utils.lr_decay import param_groups_weight_decay
from utils.scheduler import warmup_cosine_lr_scheduler


train = get_config("common/train.py").train
optim = get_config("common/optim.py").optim
graph = get_config("common/models/graph.py").graph
dataloader = get_config("common/data/imagenet.py").dataloader


# MAE Graph training for faster speed
graph.enabled = True

# Refine data path to imagenet
dataloader.train.dataset[0].root = "/path/to/imagenet"
dataloader.train.dataset[0]._target_ = PretrainingImageNetDataset

# No test data for pretraining
del dataloader.test

# Refine data transform to MAE's default settings
transform_train = LazyCall(transforms.Compose)(
    transforms=[
        LazyCall(transforms.RandomResizedCrop)(
            size=(224, 224),
            scale=(0.2, 1.0),
            interpolation=InterpolationMode.BICUBIC,
        ),
        LazyCall(transforms.RandomHorizontalFlip)(),
        LazyCall(transforms.ToTensor)(),
        LazyCall(transforms.Normalize)(
            mean=IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_DEFAULT_STD,
        ),
    ]
)
dataloader.train.dataset[0].transform = transform_train


# number devices
n_gpus = 8

# Refine training settings for MAE
train.train_micro_batch_size = 64
train.num_accumulation_steps = 8
effective_batch_size = train.train_micro_batch_size * train.num_accumulation_steps * n_gpus

train.train_epoch = 800
train.warmup_ratio = 40 / 800
train.log_period = 20
train.checkpointer.save_model_after_n_epoch = 20

# enable activation checkpointing
# train.activation_checkpoint.enabled = True

# set rdma enabled when num nodes > 1
# train.rdma_enabled = False


# Base learning in MAE is set to 1.5e-4
# The actually learning rate should be computed by linear scaling rule as follows:
# lr = base_lr * batch_size / 256
# In LiBai, you should refine the actually learning rate due to your on settings
# Here we use 8 GPUs, 128 batch_size per GPU for training, batch_size equals to 1024
base_lr = 1.5e-4
actual_lr = base_lr * effective_batch_size / 256

# Refine optim settings
optim.params._target_ = param_groups_weight_decay
optim.params.weight_decay = 0.05
optim.lr = actual_lr
optim.betas = (0.9, 0.95)

del optim.params.clip_grad_max_norm
del optim.params.clip_grad_norm_type
del optim.params.weight_decay_norm
del optim.params.weight_decay_bias
del optim.weight_decay

# Refine scheduler
# Default scheduler in LiBai training config is WarmupCosineLR
train.scheduler = LazyCall(warmup_cosine_lr_scheduler)(
    warmup_factor=0.0,
    min_lr=0.0,
)


# AMP
train.amp.enabled = True


# Distributed Settings
train.dist.data_parallel_size = n_gpus
train.dist.tensor_parallel_size = 1
train.dist.pipeline_parallel_size = 1
# train.dist.pipeline_num_layers = model.depth
