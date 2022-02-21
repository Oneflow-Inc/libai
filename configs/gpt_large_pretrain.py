from libai.config import LazyCall
from .common.models.gpt import pretrain_model as model
from .common.train import train
from .common.optim import optim
from .common.data.gpt_dataset import dataloader, tokenization

from .common.models.graph import graph

# Bert-large model config
model.cfg.embedding_dropout_prob = 0.1
model.cfg.attention_dropout_prob = 0.1
model.cfg.num_attention_heads = 16
model.cfg.hidden_size = 384
model.cfg.ffn_hidden_size = 1536
model.cfg.num_layers = 6
model.cfg.max_seq_length = 1024

for ds in dataloader.train.dataset:
    ds.max_seq_length = model.cfg.max_seq_length

optim.lr = 1.5e-4

train.train_micro_batch_size = 4
train.recompute_grad.enabled = True
train.output_dir = "./demo_output"

