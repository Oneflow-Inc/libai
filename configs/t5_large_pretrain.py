from libai.config import LazyCall
from .common.models.t5 import pretrain_model as model
from .common.train import train
from .common.optim import optim
from .common.data.t5_dataset import dataloader, tokenization

from .common.models.graph import graph

# T5-large model config
model.cfg.num_attention_heads = 12
model.cfg.hidden_size = 384
model.cfg.hidden_layers = 6

train.train_micro_batch_size = 16
train.recompute_grad.enabled = True
train.output_dir = "./demo_output"
