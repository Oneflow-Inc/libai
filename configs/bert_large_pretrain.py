from libai.config import LazyCall
from .common.models.bert import pretrain_model as model
from .common.train import train
from .common.optim import optim, scheduler
from .common.data.nlp_data import data

from libai.models.utils import GraphBase

# Bert-large model config
model.cfg.num_attention_heads = 16
model.cfg.hidden_size = 768
model.cfg.hidden_layers = 8

train.micro_batch_size = 16

# Set fp16 ON
train.amp.enabled = True

# fmt: off
# LazyCall
graph = dict(
    # options for graph or eager mode
    enabled=True,
    debug=-1, # debug mode for graph
    train_graph=LazyCall(GraphBase)(
        fp16=train.amp.enabled,
        is_train=True,
    ),
    eval_graph=LazyCall(GraphBase)(
        fp16=train.amp.enabled, 
        is_train=False
    ),
)
# fmt: on
