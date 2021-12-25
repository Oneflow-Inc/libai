
import oneflow as flow
from libai.config import LazyCall as L
from .common.models.bert import pretrain_model as model
from .common.train import train
from .common.optim import optim, lr_scheduler
from .common.data.nlp_data import data
from libai.models import BertForPretrainingGraph

# Set all dropout to 0.
model.cfg.hidden_dropout_prob = 0.1
model.cfg.attention_probs_dropout_prob = 0.1
model.cfg.bias_dropout_fusion = True

# Set matched model arguments
model.cfg.hidden_layers = 5
model.cfg.hidden_size = 384
model.cfg.num_attention_heads = 16
model.cfg.max_position_embeddings = 512

train.dist.pipeline_num_layers = model.cfg.hidden_layers

train.train_iter = 1000
train.micro_batch_size = 16
train.log_period = 1

optim.lr = 0.0001

# Set a constant lr scheduler after warmup
lr_scheduler.lrsch_or_optimizer._target_ = flow.optim.lr_scheduler.StepLR
lr_scheduler.lrsch_or_optimizer.step_size = 10000
del lr_scheduler.lrsch_or_optimizer.steps
del lr_scheduler.lrsch_or_optimizer.end_learning_rate

data.seq_length = 512
data.dataset_type = "standard_bert"
data.tokenizer_type = "BertCNWWMTokenizer"

# fmt: off
graph = dict(
    # options for graph or eager mode
    enabled=True,
    train=L(BertForPretrainingGraph)(
        fp16=train.amp.enabled,
        is_eval=False,
    ),
    eval=L(BertForPretrainingGraph)(
        fp16=train.amp.enabled, 
        is_eval=True,),
)
# fmt: on
