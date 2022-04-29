from libai.config import LazyCall
from libai.evaluation import PPLEvaluator
from .common.models.bert import pretrain_model as model
from .common.train import train
from .common.optim import optim
from .common.data.bert_dataset import dataloader, tokenization

from .common.models.graph import graph

# vocab_file = "./data_test/bert_data/bert-base-chinese-vocab.txt"
# data_prefix = "./data_test/bert_data/loss_compara_content_sentence"

vocab_file = '/cognitive_comp/ganruyi/Megatron/vocab/bert-base-chinese-vocab.txt'
data_prefix= '/cognitive_comp/ganruyi/experiments/oneflow_bert_3.9B/wudao180G_bert_text_sentence'


tokenization.tokenizer.vocab_file = vocab_file
dataloader.train.dataset[0].data_prefix = data_prefix
dataloader.train.dataset[0].indexed_dataset.data_prefix = data_prefix

# Bert-large model config
model.cfg.num_attention_heads = 40
model.cfg.hidden_size = 2560
model.cfg.hidden_layers = 48

train.train_micro_batch_size = 16
train.dist.pipeline_parallel_size = 4
# encoder_layers + decoder_layers
train.dist.pipeline_num_layers = model.cfg.hidden_layers

train.zero_optimization.enabled = True
# enable zero for level-1
train.zero_optimization.stage = 1
train.amp.enabled = True
train.activation_checkpoint.enabled = False
train.train_iter = 200

train.evaluation.evaluator = LazyCall(PPLEvaluator)()
train.evaluation.eval_iter = 1000

train.output_dir = "output/bert_3.9B_output"