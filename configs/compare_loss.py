from .common.models.bert import pretrain_model as model
from .common.train import train
from .common.optim import optim, scheduler
from .common.data.nlp_data import data

from libai.config import LazyCall
from libai.models import BertForPretrainingGraph
from libai.scheduler import WarmupMultiStepLR

# Set all dropout to 0.
model.cfg.hidden_dropout_prob = 0.0
model.cfg.attention_probs_dropout_prob = 0.0

# Set matched model arguments
model.cfg.hidden_layers = 5
model.cfg.hidden_size = 384
model.cfg.intermediate_size = 1536
model.cfg.num_attention_heads = 16
model.cfg.max_position_embeddings = 512

train.train_iter = 1000
train.micro_batch_size = 16
train.log_period = 1

optim.lr = 0.0001

# Set a constant lr scheduler after warmup
scheduler._target_ = WarmupMultiStepLR
scheduler.warmup_iters = 10
scheduler.milestones = [1000000]
del scheduler.max_iters

data.seq_length = 512
data.dataset_type = "standard_bert"
data.tokenizer_type = "BertCNWWMTokenizer"

# fmt: off
graph = dict(
    # options for graph or eager mode
    enabled=True,
    debug=-1,  # debug mode for graph
    train_graph=LazyCall(BertForPretrainingGraph)(
        fp16=train.amp.enabled,
        is_train=True,
    ),
    eval_graph=LazyCall(BertForPretrainingGraph)(
        fp16=train.amp.enabled,
        is_train=False
    ),
)
# fmt: on
