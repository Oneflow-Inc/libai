import imp
from libai.config import LazyCall
from libai.evaluation import ClsEvaluator
from libai.models.bert_model import BertForClassification
from libai.config.configs.common.models.bert import cfg
from .common.models.graph import graph
from .common.train import train
from .common.optim import optim
from .common.data.bert_dataset import tokenization

vocab_file = "./data_test/bert_data/bert-base-chinese-vocab.txt"

model = LazyCall(BertForClassification)(cfg=cfg, num_labels=2)
tokenization.tokenizer.vocab_file = vocab_file

# Bert-base model config
model.cfg.vocab_size = 21128
model.cfg.intermediate_size = 3072
model.cfg.num_attention_heads = 12
model.cfg.hidden_size = 768
model.cfg.hidden_layers = 12

train.train_micro_batch_size = 16

train.amp.enabled = True
train.activation_checkpoint.enabled = True

train.output_dir = "output/bert_classification_output"
