
from .common.models.bert import pretrain_model as model
from .common.models.graph import graph
from .common.train import train
from .common.optim import optim
from .common.data.bert_dataset import dataloader, tokenization

vocab_file = "./data_test/bert_data/bert-base-chinese-vocab.txt"
data_prefix = "./data_test/bert_data/loss_compara_content_sentence"

tokenization.tokenizer.vocab_file = vocab_file
dataloader.train.dataset[0].data_prefix = data_prefix
dataloader.train.dataset[0].indexed_dataset.data_prefix = data_prefix

# Bert-large model config
model.cfg.hidden_layers = 24
model.cfg.num_attention_heads = 16
model.cfg.hidden_size = 1024

train.train_micro_batch_size = 4

# 8 gpus
# train.global_batch_size = 32

train.dist.tensor_parallel_size = 1
train.dist.pipeline_parallel_size = 1

train.dist.pipeline_num_layers = model.cfg.hidden_layers

# fp16
train.amp.enabled = True
train.activation_checkpoint.enabled = False

train.output_dir = "output/bert_profile"
