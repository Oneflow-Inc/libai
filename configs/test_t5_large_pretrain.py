from libai.config import LazyCall
from libai.evaluation import PPLEvaluator
from .common.models.t5 import pretrain_model as model
from .common.train import train
from .common.optim import optim
from .common.data.test_t5_dataset import dataloader

from .common.models.graph import graph

# T5-large model config
model.cfg.num_attention_heads = 8
model.cfg.vocab_size = 8
model.cfg.hidden_size = 8
model.cfg.hidden_layers = 6
model.cfg.scale_mask_softmax_fusion = False
model.cfg.bias_dropout_fusion = False
model.cfg.bias_gelu_fusion = False

graph.debug = 1

train.input_placement_device = "cpu"

train.dist.data_parallel_size=10
train.dist.tensor_parallel_size=1
train.dist.pipeline_parallel_size=1
train.dist.pipeline_num_layers = 2 * model.cfg.hidden_layers

train.train_micro_batch_size =100 
train.amp.enabled = True

train.evaluation.evaluator = LazyCall(PPLEvaluator)()

train.output_dir = "./output/t5_output"
