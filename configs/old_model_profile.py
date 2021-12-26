from libai.config import LazyCall as L
from .common.models.bert import pretrain_model as model
from .common.train import train
from .common.optim import optim, lr_scheduler
from .common.data.nlp_data import data
from libai.models import BertForPretrainingGraph

model.cfg.hidden_dropout_prob = 0.1
model.cfg.attention_probs_dropout_prob = 0.1

model.cfg.bias_dropout_fusion = True
model.cfg.bias_gelu_fusion = True

# Set matched old model arguments
model.cfg.hidden_layers = 5
model.cfg.hidden_size = 384
model.cfg.intermediate_size = 1536
model.cfg.num_attention_heads = 16
model.cfg.max_position_embeddings = 512

train.dist.pipeline_num_layers = model.cfg.hidden_layers

train.train_iter = 1000
train.micro_batch_size = 16

train.enable_use_compute_stream = False

optim.lr = 0.0001

data.seq_length = 512
data.dataset_type = "standard_bert"
data.tokenizer_type = "BertCNWWMTokenizer"

# fmt: off
graph = dict(
    # options for graph or eager mode
    enabled=True,
    train=L(BertForPretrainingGraph)(
        is_eval=False,
    ),
    eval=L(BertForPretrainingGraph)(
        is_eval=True,
    )
)
# fmt: on
