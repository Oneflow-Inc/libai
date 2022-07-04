from libai.config import LazyCall
from libai.evaluation import PPLEvaluator
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
model.cfg.num_attention_heads = 16
model.cfg.hidden_size = 768
model.cfg.hidden_layers = 8

train.input_placement_device = "cpu"

train.dist.pipeline_num_layers = model.cfg.hidden_layers

train.train_micro_batch_size = 16

train.amp.enabled = True

for ds in dataloader.train.dataset:
    ds.max_seq_length = model.cfg.max_position_embeddings

train.evaluation.evaluator = LazyCall(PPLEvaluator)()

train.output_dir = "output/bert_output"
