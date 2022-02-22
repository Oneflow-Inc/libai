from libai.config.lazy import LazyCall

from libai.scheduler import WarmupMultiStepLR

from .t5_model import pretrain_model as model
from .t5_dataset import dataloader, tokenization
from .graph import graph
from .train import train
from .optim import optim
# model = get_config("./common/models/t5.py").pretrain_model
# graph = get_config("./common/models/graph.py").graph
# train = get_config("./common/train.py").train
# optim = get_config("./common/optim.py").optim

train.eval_iter = 10


# Set all dropout to 0.
model.cfg.hidden_dropout_prob = 0.0
model.cfg.attention_probs_dropout_prob = 0.0

# Set matched model arguments
model.cfg.hidden_layers = 5
model.cfg.hidden_size = 384
model.cfg.intermediate_size = 1536
model.cfg.num_attention_heads = 16
model.cfg.max_position_embeddings = 512

train.train_iter = 1000
train.micro_batch_size = 16
train.log_period = 20
train.warmup_ratio = 0.01


# Set a constant lr scheduler after warmup
optim.lr = 0.0001
train.scheduler = LazyCall(WarmupMultiStepLR)(warmup_factor=0.1, milestones=[0.99])


for ds in dataloader.train.dataset:
    ds.max_num_samples = train.train_iter * train.micro_batch_size