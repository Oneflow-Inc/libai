from datetime import date
from omegaconf import OmegaConf
from libai.config import LazyCall, get_config
from tokenizer.tokenizer import _BertCNWWMTokenizer

from libai.scheduler import WarmupMultiStepLR

model = get_config("common/models/bert.py").pretrain_model
graph = get_config("common/models/graph.py").graph
train = get_config("common/train.py").train
optim = get_config("common/optim.py").optim
dataloader = get_config("common/data/bert_dataset.py").dataloader

from data.data_samplers import MegatronPretrainingSampler

dataloader.train.train_sampler = LazyCall(MegatronPretrainingSampler)()

tokenization = OmegaConf.create()
tokenization.tokenizer = LazyCall(_BertCNWWMTokenizer)(
    lower_case=False,
    vocab_extra_ids=0,
)
tokenization.append_eod = False
tokenization.make_vocab_size_divisible_by = 128


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
train.global_batch_size = 16
train.train_micro_batch_size = None
train.num_accumulation_steps = 1
train.log_period = 20
train.warmup_ratio = 0.01
train.evaluation.enabled = False
train.eval_iter = 10

train.dist.pipeline_num_layers = model.cfg.hidden_layers

# Set a constant lr scheduler after warmup
optim.lr = 0.0001
train.scheduler = LazyCall(WarmupMultiStepLR)(warmup_factor=0.1, milestones=[0.99])

today = date.today()
train.output_dir = f"loss_align/bert_loss_compare/{today}_base"
