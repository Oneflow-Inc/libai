from flowvision.data import Mixup
from flowvision.loss.cross_entropy import SoftTargetCrossEntropy

from libai.config import LazyCall, get_config
from ..modeling.vision_wrapper import VisionModel

# Get train, optim and graph configs
train = get_config("common/train.py").train
optim = get_config("common/optim.py").optim
graph = get_config("common/models/graph.py").graph
dataloader = get_config("common/data/imagenet.py").dataloader

# Add model for training
model = LazyCall(VisionModel)(
    model_name="vit_tiny_patch16_224",
    pretrained=False,
    num_classes=1000,
    loss_func=LazyCall(SoftTargetCrossEntropy)(),
)

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
    num_classes=model.num_classes,
)

# Refine optimizer cfg for vit model
optim.lr = 1e-3
optim.eps = 1e-8
optim.weight_decay = 0.05

# Refine train cfg for vit model
train.train_micro_batch_size = 128
train.test_micro_batch_size = 128
train.train_epoch = 300
train.warmup_ratio = 5 / 300
train.evaluation.eval_period = 1000
train.log_period = 1

# Scheduler
train.scheduler.warmup_factor = 0.001
train.scheduler.alpha = 0.01
train.scheduler.warmup_method = "linear"

# Set fp16 ON
train.amp.enabled = True

# Set checkpointing on
train.activation_checkpoint.enabled = True
