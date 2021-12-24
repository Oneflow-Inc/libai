from libai.config import LazyCall as L
from .common.models.bert import pretrain_model as model
from .common.train import train
from .common.optim import optim, lr_scheduler
from .common.data.nlp_data import data
from libai.models import BertForPretrainingGraph

# Bert-large model config
model.cfg.num_attention_heads = 16
model.cfg.hidden_size = 1024

# Set pipeline layers for paralleleism
train.dist.pipeline_num_layers = model.cfg.hidden_layers

graph = dict(
    # options for graph or eager mode
    enabled=False,
    train=L(BertForPretrainingGraph)(
        fp16=train.amp.enabled, is_eval=False, num_accumulation_steps=1,
    ),
    eval=L(BertForPretrainingGraph)(fp16=train.amp.enabled, is_eval=True,),
)
