from .common.models.bert import pretrain_model as model
from .common.train import train
from .common.optim import optim

# Bert-large model config
model.cfg.num_attention_heads = 16
model.cfg.hidden_size = 1024

# Set pipeline layers for paralleleism
train.dist.pipeline_num_layers = model.cfg.hidden_layers
