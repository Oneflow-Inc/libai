from .common.models.t5 import pretrain_model as model
from .common.train import train
from .common.optim import optim, scheduler
from .common.data.nlp_data import data

from libai.config import LazyCall
from libai.models import T5ForPretrainingGraph
from libai.scheduler import WarmupMultiStepLR

# Set all dropout to 0.
model.cfg.hidden_dropout_prob = 0.0
model.cfg.bias_dropout_fusion = False
model.cfg.bias_gelu_fusion=False

# Set matched model arguments
model.cfg.num_layers=6
model.cfg.vocab_size=50304
model.cfg.hidden_size=384
model.cfg.ffn_hidden_size=1536
model.cfg.num_attention_heads=16
model.cfg.max_seq_length=512
model.cfg.embedding_dropout_prob=0.1
model.cfg.attention_dropout_prob=0.1
model.cfg.apply_query_key_layer_scaling=False

train.dist.pipeline_num_layers = model.cfg.num_layers
train.train_iter = 1000
train.micro_batch_size = 16
train.log_period = 1
train.load_weight = "./tests/t5_test/flow_t5.f/"

optim.lr = 0.0001

# Set a constant lr scheduler after warmup
scheduler._target_ = WarmupMultiStepLR
scheduler.milestones = [1000000]
scheduler.warmup_iters = 0
del scheduler.max_iters

data.seq_length = 512
data.max_seq_length_dec = 128
# data.dataset_type = "standard_bert"
data.dataset_type = "t5"
# data.tokenizer_type = "BertCNWWMTokenizer"
data.tokenizer_type = "GPT2BPETokenizer"
data.data_path = ['/workspace/data/libai_dataset/loss_compara_content_sentence']
data.vocab_file = '/workspace/data/gpt_dataset/vocab_file.txt'
data.num_workers = 1
data.vocab_extra_ids = 100
data.mmap_warmup = False

# fmt: off
graph = dict(
    # options for graph or eager mode
    # enabled=True
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
