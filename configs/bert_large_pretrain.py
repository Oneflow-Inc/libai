from libai.config import LazyCall as L
from .common.models.bert import pretrain_model as model
from .common.train import train
from .common.optim import optim, lr_scheduler
from .common.data.nlp_data import data
from libai.models import BertForPretrainingGraph

# Bert-large model config
model.cfg.num_attention_heads = 16
model.cfg.hidden_size = 768
model.cfg.hidden_layers = 8

# Set pipeline layers for paralleleism
train.dist.pipeline_num_layers = model.cfg.hidden_layers
train.dist.tensor_parallel_size = 1
train.dist.pipeline_parallel_size = 1

train.micro_batch_size = 16

train.amp.enabled = True

graph = dict(
    # options for graph or eager mode
    enabled=True,
    train=L(BertForPretrainingGraph)(
        fp16=train.amp.enabled,
        is_eval=False,
    ),
    eval=L(BertForPretrainingGraph)(fp16=train.amp.enabled, is_eval=True,),
)
