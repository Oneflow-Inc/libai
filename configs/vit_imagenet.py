from libai.config import LazyCall
from .common.models.vision_transformer import vit_model as model
from .common.train import train
from .common.optim import optim
from .common.data.cifar import dataloader


from libai.models import VisionTransformerGraph

# Refine optimizer cfg for vit model
optim.lr = 5e-4
optim.eps = 1e-8
optim.weight_decay = 0.05

# Refine train cfg for vit model
train.train_micro_batch_size = 128
train.test_micro_batch_size = 128
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

# fmt: off
graph = dict(
    # options for graph or eager mode
    enabled=True,
    train_graph=LazyCall(VisionTransformerGraph)(
        is_train=True,
    ),
    eval_graph=LazyCall(VisionTransformerGraph)(
        is_train=False,),
    debug=-1,
)
# fmt: on
