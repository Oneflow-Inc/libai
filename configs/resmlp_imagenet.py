from libai.config import LazyCall
from .common.models.resmlp.resmlp_12 import model
from .common.models.graph import graph
from .common.train import train
from .common.optim import optim
from .common.data.imagenet import dataloader

import oneflow as flow
import flowvision.transforms as transforms
from flowvision.transforms import InterpolationMode
from flowvision.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from flowvision.data import Mixup
from flowvision.loss.cross_entropy import SoftTargetCrossEntropy

# Refine output dir
train.output_dir = "./output_resmlp"

# Refine data path to imagenet
dataloader.train.dataset[0].root = "/path/to/imagenet"
dataloader.test[0].dataset.root = "/path/to/imagenet"

# Refine test data augmentation for resmlp model
resmlp_test_aug = LazyCall(transforms.Compose)(
    transforms=[
        LazyCall(transforms.Resize)(
            size=int(224 / 0.9),
            interpolation=InterpolationMode.BICUBIC,
        ),
        LazyCall(transforms.CenterCrop)(
            size=224,
        ),
        LazyCall(transforms.ToTensor)(),
        LazyCall(transforms.Normalize)(
            mean=IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_DEFAULT_STD,
        ),
    ]
)
dataloader.test[0].dataset.transform = resmlp_test_aug

# Refine model cfg for resmlp training on imagenet
model.cfg.num_classes = 1000
model.cfg.loss_func = SoftTargetCrossEntropy()

# Add Mixup Func
dataloader.train.mixup_func = LazyCall(Mixup)(
    mixup_alpha=0.8,
    cutmix_alpha=1.0,
    prob=1.0,
    switch_prob=0.5,
    mode="batch",
    num_classes=model.cfg.num_classes,
)

# Refine optimizer cfg for resmlp model
optim._target_ = flow.optim.LAMB  # use lamb optimizer
optim.lr = 5e-3  # default batch size equals to 256 * 8 = 2048
optim.eps = 1e-8
optim.weight_decay = 0.2
optim.params.clip_grad_max_norm = None
optim.params.clip_grad_norm_type = None
optim.params.overrides = {
    "alpha": {"weight_decay": 0.0},
    "beta": {"weight_decay": 0.0},
    "gamma_1": {"weight_decay": 0.0},
    "gamma_2": {"weight_decay": 0.0},
}

# Refine train cfg for resmlp model
train.train_micro_batch_size = 256
train.test_micro_batch_size = 64
train.train_epoch = 400
train.warmup_ratio = 5 / 400
train.evaluation.eval_period = 1000
train.log_period = 1

# Scheduler
train.scheduler.warmup_factor = 0.001
train.scheduler.alpha = 0.01
train.scheduler.warmup_method = "linear"

# Set fp16 ON
train.amp.enabled = True

# Distributed Settings
train.dist.pipeline_num_layers = model.cfg.depth
train.dist.data_parallel_size = 1
train.dist.tensor_parallel_size = 1
train.dist.pipeline_parallel_size = 1
