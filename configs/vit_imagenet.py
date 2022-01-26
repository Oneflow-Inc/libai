from libai.config import LazyCall
from libai.scheduler import WarmupCosineLR
from libai.scheduler.lr_scheduler import WarmupMultiStepLR
from .common.models.vit import vit_model as model
from .common.train import train
from .common.optim import optim
from .common.data.cv_data import dataloader


from libai.models import VisionTransformerGraph

# Refine optimizer cfg for vit model
optim.lr = 0.0001
optim.weight_decay = 1e-8

# Set pipeline layers for paralleleism
train.dist.pipeline_num_layers = model.cfg.depth

# Refine train cfg for vit model
train.train_epoch = 300
train.warmup_ratio = 20/300
train.eval_period = 5000
train.log_period = 1

# Set fp16 ON
train.amp.enabled = False

# fmt: off
graph = dict(
    # options for graph or eager mode
    enabled=True,
    train_graph=LazyCall(VisionTransformerGraph)(
        fp16=train.amp.enabled,
        is_train=True,
    ),
    eval_graph=LazyCall(VisionTransformerGraph)(
        fp16=train.amp.enabled, 
        is_train=False,),
    debug=-1,
)
# fmt: on
