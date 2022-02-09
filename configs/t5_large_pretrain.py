from libai.config import LazyCall
from .common.models.t5 import pretrain_model as model
from .common.train import train
from .common.optim import optim
from .common.data.t5_dataset import dataloader, tokenization

from libai.models import T5ForPretrainingGraph

# Bert-large model config
model.cfg.num_attention_heads = 12
model.cfg.hidden_size = 384
model.cfg.hidden_layers = 6

train.train_micro_batch_size = 16
train.recompute_grad.enabled = True
train.output_dir = "./debug1"


# Set fp16 ON
# train.amp.enabled = True

# LazyCall
graph = dict(
    # options for graph or eager mode
    enabled=True,
    debug=1,  # debug mode for graph
    train_graph=LazyCall(T5ForPretrainingGraph)(
        fp16=train.amp.enabled,
        is_train=True,
    ),
    eval_graph=LazyCall(T5ForPretrainingGraph)(fp16=train.amp.enabled, is_train=False),
)
