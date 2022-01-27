from libai.config import LazyCall
from .common.models.bert import pretrain_model as model
from .common.train import train
from .common.optim import optim
from .common.data.bert_dataset import dataloader, tokenization

from libai.models import BertForPretrainingGraph

# Bert-large model config
model.cfg.num_attention_heads = 16
model.cfg.hidden_size = 768
model.cfg.hidden_layers = 8

train.train_micro_batch_size = 16

train.amp.enabled = False
train.recompute_grad.enabled = False

# LazyCall
graph = dict(
    # options for graph or eager mode
    enabled=True,
    debug=-1,  # debug mode for graph
    train_graph=LazyCall(BertForPretrainingGraph)(
        is_train=True,
    ),
    eval_graph=LazyCall(BertForPretrainingGraph)(is_train=False),
)
