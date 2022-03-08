from libai.config import LazyCall, get_config

from .common.models.t5 import pretrain_model as model
from .common.train import train
from .common.optim import optim
from .common.data.t5_dataset import dataloader, tokenization
# from projects.idea_t5.configs.t5_dataset import dataloader, tokenization

from .common.models.graph import graph

vocab_file = "./data_test/bert_data/bert-base-chinese-vocab.txt"
data_prefix = "./data_test/bert_data/loss_compara_content_sentence"

tokenization.tokenizer.vocab_file = vocab_file
dataloader.train.dataset[0].data_prefix = data_prefix
dataloader.train.dataset[0].indexed_dataset.data_prefix = data_prefix

train.eval_iter = 10

# Set all dropout to 0.
model.cfg.hidden_dropout_prob = 0.1
model.cfg.attention_probs_dropout_prob = 0.1
model.cfg.embedding_dropout_prob = 0.1
model.cfg.bias_gelu_fusion = True
model.cfg.bias_dropout_fusion = True

# Set matched model arguments
model.cfg.hidden_layers = 6
model.cfg.hidden_size = 384
model.cfg.intermediate_size = 1536
model.cfg.num_attention_heads = 12
model.cfg.max_position_embeddings = 512

train.train_iter = 1000
train.micro_batch_size = 16
train.train_micro_batch_size = 16
train.log_period = 10
train.warmup_ratio = 0.01

train.amp.enabled = False
train.activation_checkpoint.enabled = False
train.log_period = 1


train.dist.tensor_parallel_size = 1
train.dist.pipeline_parallel_size = 1

train.dist.pipeline_num_layers = 2 * model.cfg.hidden_layers

# Set a constant lr scheduler after warmup
optim.lr = 0.0001

train.output_dir = "output/t5_profile"

# for ds in dataloader.train.dataset:
    # ds.max_num_samples = train.train_iter * train.micro_batch_size
