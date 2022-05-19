from libai.config import LazyCall
from libai.models.bert_model import BertForClassification
from libai.config.configs.common.models.bert import cfg
from .common.models.graph import graph
from .common.train import train
from .common.data.bert_dataset import tokenization

vocab_file = "./data_test/bert_data/bert-base-chinese-vocab.txt"

cfg["num_labels"] = 2
cfg["classifier_dropout"] = 0.1

model = LazyCall(BertForClassification)(cfg=cfg)
tokenization.tokenizer.vocab_file = vocab_file

model.cfg.vocab_size = 21128
model.cfg.intermediate_size = 3072
model.cfg.num_attention_heads = 12
model.cfg.hidden_layers = 12
model.cfg.hidden_size = 768

train.amp.enabled = True
train.activation_checkpoint.enabled = True
train.dist.pipeline_num_layers = model.cfg.hidden_layers

train.output_dir = "output/bert_classification_output"
