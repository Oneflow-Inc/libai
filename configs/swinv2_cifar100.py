from libai.config import LazyCall

from .common.models.swinv2.swinv2_tiny_patch4_window8_256 import model
from .common.models.graph import graph
from .common.train import train
from .common.optim import optim
from .common.data.cifar100 import dataloader

from flowvision import transforms
from flowvision.transforms import InterpolationMode
from flowvision.transforms.functional import str_to_interp_mode
from flowvision.data import Mixup
from flowvision.loss.cross_entropy import SoftTargetCrossEntropy

CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

# Add Mixup Func
dataloader.train.mixup_func = LazyCall(Mixup)(
    mixup_alpha=0.8,
    cutmix_alpha=1.0,
    prob=1.0,
    switch_prob=0.5,
    mode="batch",
    num_classes=100,
)
dataloader.train.dataset[0].transform = LazyCall(transforms.Compose)(
    transforms=[
        LazyCall(transforms.RandomResizedCrop)(
            size=(256, 256),
            scale=(0.08, 1.0),
            ratio=(3.0 / 4.0, 4.0 / 3.0),
            interpolation=str_to_interp_mode("bicubic"),
        ),
        LazyCall(transforms.RandomHorizontalFlip)(),
        LazyCall(transforms.ToTensor)(),
        LazyCall(transforms.Normalize)(mean=CIFAR100_TRAIN_MEAN, std=CIFAR100_TRAIN_STD),
    ]
)
dataloader.test[0].dataset.transform = LazyCall(transforms.Compose)(
    transforms=[
        LazyCall(transforms.Resize)(
            size=256,
            interpolation=InterpolationMode.BICUBIC,
        ),
        LazyCall(transforms.CenterCrop)(
            size=256,
        ),
        LazyCall(transforms.ToTensor)(),
        LazyCall(transforms.Normalize)(
            mean=CIFAR100_TRAIN_MEAN,
            std=CIFAR100_TRAIN_STD,
        ),
    ]
)

# Refine model cfg for vit training on cifar100
model.cfg.num_classes = 100
model.cfg.loss_func = SoftTargetCrossEntropy()

# Refine optimizer cfg for swinv2 model
optim.lr = 5e-4
optim.eps = 1e-8
optim.weight_decay = 0.05


def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin


def set_weight_decay(model, skip_list=(), skip_keywords=()):
    has_decay = []
    no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if (
            len(param.shape) == 1
            or name.endswith(".bias")
            or (name in skip_list)
            or check_keywords_in_name(name, skip_keywords)
        ):
            no_decay.append(param)
        else:
            has_decay.append(param)
    return [{"params": has_decay}, {"params": no_decay, "weight_decay": 0.0}]


optim.params = LazyCall(set_weight_decay)(
    model=model,
    skip_list=("absolute_pos_embed"),
    skip_keywords=("cpb_mlp", "logit_scale", "relative_position_bias_table"),
)
# Refine train cfg for swin model
train.train_micro_batch_size = 32
train.num_accumulation_steps = 8
train.test_micro_batch_size = 32
train.train_epoch = 300
train.warmup_ratio = 20 / 300
train.evaluation.eval_period = 1562
train.log_period = 10

# Scheduler
train.scheduler.warmup_factor = 5e-7
train.scheduler.alpha = 0.0
train.scheduler.warmup_method = "linear"

# parallel strategy settings
train.dist.data_parallel_size = 1
train.dist.tensor_parallel_size = 1
train.dist.pipeline_parallel_size = 1
train.dist.pipeline_num_layers = sum(model.cfg.depths)
train.output_dir = "./output"

train.rdma_enabled = False
# Set fp16 ON
train.amp.enabled = False
train.activation_checkpoint.enabled = False
# train.zero_optimization.enabled = True
# train.zero_optimization.stage = 1
graph.enabled = False
