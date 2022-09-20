from libai.config import LazyCall
from libai.evaluation import PPLEvaluator
from .common.models.t5 import pretrain_model as model
from .common.train import train
from .common.optim import optim
from .common.data.test_t5_dataset import dataloader

from .common.models.graph import graph

# T5-large model config
model.cfg.num_attention_heads = 12
model.cfg.hidden_size = 384
model.cfg.hidden_layers = 6

train.input_placement_device = "cpu"

train.dist.pipeline_num_layers = 2 * model.cfg.hidden_layers

train.train_micro_batch_size = 4
train.amp.enabled = True

train.evaluation.evaluator = LazyCall(PPLEvaluator)()

train.output_dir = "./output/t5_output"
