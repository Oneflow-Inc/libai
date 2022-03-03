from .common.models.t5 import pretrain_model as model
from .common.train import train
from .common.optim import optim
from .common.data.t5_dataset import dataloader, tokenization

from .common.models.graph import graph

vocab_file = "./data_test/bert_data/bert-base-chinese-vocab.txt"
data_prefix = "./data_test/bert_data/loss_compara_content_sentence"

tokenization.tokenizer.vocab_file = vocab_file
dataloader.train.dataset[0].data_prefix = data_prefix
dataloader.train.dataset[0].indexed_dataset.data_prefix = data_prefix

# T5-large model config for megatron profile
model.cfg.hidden_layers = 12
model.cfg.hidden_size = 768
model.cfg.num_attention_heads = 12
model.cfg.intermediate_size = 1536
model.cfg.max_position_embeddings = 512
model.cfg.hidden_dropout_prob = 0.1
model.cfg.attention_probs_dropout_prob = 0.1
model.cfg.embedding_dropout_prob = 0.1
model.cfg.bias_gelu_fusion = True
model.cfg.bias_dropout_fusion = True

optim.lr = 1e-4
optim.weight_decay = 1e-2

train.train_micro_batch_size = 16
train.train_iter = 1000
train.warmup_ratio = .01

# fp16
train.amp.enabled = False
# gradient checkpointing
train.recompute_grad.enabled = False

train.dist.pipeline_num_layers = 2 * model.cfg.hidden_layers

train.output_dir = "./output/t5_output"
