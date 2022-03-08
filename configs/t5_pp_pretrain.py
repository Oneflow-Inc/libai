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

# T5-large model config
model.cfg.num_attention_heads = 12
model.cfg.hidden_size = 384
model.cfg.hidden_layers = 6

train.activation_checkpoint.enabled = True

train.train_micro_batch_size = 16

train.dist.tensor_parallel_size = 1
train.dist.pipeline_parallel_size = 1
# encoder_layers + decoder_layers
train.dist.pipeline_num_layers = 2 * model.cfg.hidden_layers

train.output_dir = "./output/t5_pipeline_output"
