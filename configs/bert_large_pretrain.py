from .common.models.bert import pretrain_model as model
from .common.models.graph import graph
from .common.train import train
from .common.optim import optim, scheduler
from .common.data.bert_dataset import dataloader, tokenization

# Bert-large model config
model.cfg.num_attention_heads = 16
model.cfg.hidden_size = 768
model.cfg.hidden_layers = 8

train.train_micro_batch_size = 16

# Set fp16 ON
# train.amp.enabled = True

graph.enabled = True  # graph mode
