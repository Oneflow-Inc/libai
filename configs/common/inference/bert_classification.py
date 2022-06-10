from libai.config import LazyCall
from libai.models.bert_model import BertForClassification
from ..models.bert import cfg as bert_cfg
from ..models.graph import graph
from ..train import train
from ..data.bert_dataset import tokenization

vocab_file = "./data_test/bert_data/bert-base-chinese-vocab.txt"

bert_cfg["num_labels"] = 2
bert_cfg["classifier_dropout"] = 0.1

model = LazyCall(BertForClassification)(cfg=bert_cfg)
tokenization.tokenizer.vocab_file = vocab_file

model.cfg.vocab_size = 21128
model.cfg.intermediate_size = 3072
model.cfg.num_attention_heads = 12
model.cfg.hidden_layers = 12
model.cfg.hidden_size = 768

train.amp.enabled = True
train.activation_checkpoint.enabled = True
train.dist.pipeline_num_layers = model.cfg.hidden_layers
