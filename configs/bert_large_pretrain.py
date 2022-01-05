from libai.config import LazyCall
from .common.models.bert import pretrain_model as model
from .common.train import train
from .common.optim import optim, scheduler
from .common.data.nlp_data import data

from libai.models import BertForPretrainingGraph

# Bert-large model config
model.cfg.num_attention_heads = 16
model.cfg.hidden_size = 768
model.cfg.hidden_layers = 8

# Set pipeline layers for paralleleism
train.dist.pipeline_num_layers = model.cfg.hidden_layers

train.micro_batch_size = 16

# Set fp16 ON
train.amp.enabled = True

# fmt: off
# LazyCall
graph = dict(
    # options for graph or eager mode
    enabled=True,
    debug=0, # debug mode for graph
    train_graph=LazyCall(BertForPretrainingGraph)(
        fp16=train.amp.enabled,
        is_train=True,
    ),
    eval_graph=LazyCall(BertForPretrainingGraph)(
        fp16=train.amp.enabled, 
        is_train=False,),
)

# Register
# graph = dict(
#     enabled=True,
#     debug=0,
#     train_graph = dict(
#         graph_name="BertForPretrainingGraph",
#         graph_cfg = dict(
#             fp16=train.amp.enabled,
#             is_train=True,
#         )
#     ),
#     eval_graph = dict(
#         graph_name="BertForPretrainingGraph",
#         graph_cfg = dict(
#             fp16=train.amp.enabled,
#             is_train=False
#         )
#     )
# )
# fmt: on
