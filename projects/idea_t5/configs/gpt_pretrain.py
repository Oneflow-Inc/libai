from libai.config import LazyCall, get_config

from libai.scheduler import WarmupMultiStepLR

from .gpt_dataset import dataloader, tokenization

model = get_config("./common/models/gpt.py").pretrain_model
graph = get_config("./common/models/graph.py").graph
train = get_config("./common/train.py").train
optim = get_config("./common/optim.py").optim

vocab_file = "./data_test/gpt_data/gpt2-vocab.json"
merge_files = "./data_test/gpt_data/gpt2-merges.txt"
data_prefix = "./data_test/gpt_data/loss_compara_content_sentence"

tokenization.tokenizer.vocab_file = vocab_file
tokenization.tokenizer.merges_file = merge_files
dataloader.train.dataset[0].data_prefix = data_prefix
dataloader.train.dataset[0].indexed_dataset.data_prefix = data_prefix

train.eval_iter = 100

# GPT-2 model config
model.cfg.embedding_dropout_prob = 0.1
model.cfg.attention_dropout_prob = 0.1
model.cfg.num_attention_heads = 16
model.cfg.hidden_size = 384
model.cfg.ffn_hidden_size = 1536
model.cfg.num_layers = 6
model.cfg.max_seq_length = 1024

train.train_iter = 10000
train.micro_batch_size = 4
train.train_micro_batch_size = 4
train.log_period = 20
train.warmup_ratio = 0.01

# Set a constant lr scheduler after warmup
optim.lr = 0.0001
train.scheduler = LazyCall(WarmupMultiStepLR)(warmup_factor=0.1, milestones=[0.99])

train.amp.enabled = True
train.activation_checkpoint.enabled = False

for ds in dataloader.train.dataset:
    ds.max_seq_length = model.cfg.max_seq_length
    ds.max_num_samples = train.train_iter * train.micro_batch_size

train.output_dir = "./output/gpt_output_megatron"
