from .common.models.t5 import pretrain_model as model
from .common.train import train
from .common.optim import optim, scheduler
from .common.data.nlp_data import data

from libai.config import LazyCall
from libai.models import T5ForPretrainingGraph
from libai.scheduler import WarmupMultiStepLR

# Set all dropout to 0.
model.cfg.hidden_dropout_prob = 0.0
model.cfg.attention_probs_dropout_prob = 0.0
model.cfg.bias_dropout_fusion = True

# Set matched model arguments
model.cfg.hidden_layers = 6
model.cfg.hidden_size = 384
model.cfg.intermediate_size = 1536
model.cfg.num_attention_heads = 16
model.cfg.max_position_embeddings = 512

train.dist.pipeline_num_layers = model.cfg.hidden_layers

train.train_iter = 1000
train.micro_batch_size = 16
train.log_period = 1

optim.lr = 0.0001

# Set a constant lr scheduler after warmup
scheduler._target_ = WarmupMultiStepLR
scheduler.milestones = [1000000]
del scheduler.max_iters

data.seq_length = 512
data.max_seq_length_dec = 128
# data.dataset_type = "standard_bert"
data.dataset_type = "t5"
# data.tokenizer_type = "BertCNWWMTokenizer"
data.tokenizer_type = "BertWordPieceLowerCase"
data.data_path = ['/home/wang/data/t5/dataset/loss_compara_content_sentence']
data.vocab_file = '/home/wang/data/t5/dataset/bert-base-chinese-vocab.txt'

# fmt: off
graph = dict(
    # options for graph or eager mode
    enabled=True,
    debug=-1, # debug mode for graph
    train_graph=LazyCall(T5ForPretrainingGraph)(
        fp16=train.amp.enabled,
        is_train=True,
    ),
    eval_graph=LazyCall(T5ForPretrainingGraph)(
        fp16=train.amp.enabled, 
        is_train=False
    ),
)
# fmt: on
