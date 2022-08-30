from pyexpat import model
from flowvision.data import Mixup
from flowvision.loss.cross_entropy import SoftTargetCrossEntropy

from configs.common.data.imagenet import dataloader
from configs.common.models.graph import graph
from configs.common.optim import optim
from configs.common.train import train
from libai.config import LazyCall
from projects.SegFormer.configs.models.classification.mit_cls_b0 import cfg
from projects.SegFormer.modeling.segformer_model import SegformerForImageClassification


# Refine data path to imagenet
dataloader.train.dataset[0].root = "/data/dataset/ImageNet/extract"
dataloader.test[0].dataset.root = "/data/dataset/ImageNet/extract"

# Add Mixup Func
dataloader.train.mixup_func = LazyCall(Mixup)(
    mixup_alpha=0.8,
    cutmix_alpha=1.0,
    prob=1.0,
    switch_prob=0.5,
    mode="batch",
    num_classes=1000,
)

# Refine model cfg for segformer pretraining on imagenet
model = LazyCall(SegformerForImageClassification)(cfg=cfg)
model.cfg.num_classes = 1000
model.cfg.loss_func = SoftTargetCrossEntropy()
# Refine optimizer cfg for segformer model
optim.lr = 5e-4
optim.eps = 1e-8
optim.weight_decay = 0.05

# Refine train cfg for vit model
train.train_micro_batch_size = 128
train.test_micro_batch_size = 128

train.rdma_enabled = False

train.dist.data_parallel_size=4
train.dist.tensor_parallel_size=1
train.dist.pipeline_parallel_size = 1

train.train_epoch = 300
train.warmup_ratio = 20 / 300
train.eval_period = 1000
train.log_period = 1

# Scheduler
train.scheduler.warmup_factor = 0.001
train.scheduler.alpha = 0.01
train.scheduler.warmup_method = "linear"

# Set fp16 ON
train.amp.enabled = True
