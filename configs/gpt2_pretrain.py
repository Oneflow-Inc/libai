from libai.config import LazyCall
from libai.evaluation import PPLEvaluator
from .common.models.gpt import pretrain_model as model
from .common.train import train
from .common.optim import optim
from .common.data.gpt_dataset import dataloader, tokenization

from .common.models.graph import graph

vocab_file = "./data_test/gpt_data/gpt2-vocab.json"
merge_files = "./data_test/gpt_data/gpt2-merges.txt"
data_prefix = "./data_test/gpt_data/loss_compara_content_sentence"

tokenization.tokenizer.vocab_file = vocab_file
tokenization.tokenizer.merges_file = merge_files
dataloader.train.dataset[0].data_prefix = data_prefix
dataloader.train.dataset[0].indexed_dataset.data_prefix = data_prefix

# GPT-2 model config
model.cfg.embedding_dropout_prob = 0.1
model.cfg.attention_dropout_prob = 0.1
model.cfg.num_attention_heads = 16
model.cfg.hidden_size = 384
model.cfg.ffn_hidden_size = 1536
model.cfg.num_layers = 6
model.cfg.max_seq_length = 1024

train.input_placement_device = "cpu"

train.dist.pipeline_num_layers = model.cfg.num_layers

for ds in dataloader.train.dataset:
    ds.max_seq_length = model.cfg.max_seq_length

optim.lr = 1.5e-4

train.train_micro_batch_size = 4
train.amp.enabled = True

train.evaluation.evaluator = LazyCall(PPLEvaluator)()

train.output_dir = "./output/gpt2_output"
