from projects.SegFormer.configs.models.mit_b0 import model
from projects.SegFormer.configs.data.cityscapes import dataloader
from configs.common.models.graph import graph
from configs.common.train import train
from configs.common.optim import optim


optim.lr = 0.00006
optim.weight_decay = 0.0001

model.cfg.num_classes = 19

dataloader.train.dataset[0].root = "/dataset/cityscapes"
dataloader.test[0].dataset.root = "/dataset/cityscapes"

train.output_dir = "./output"

# Refine train cfg for segformer model
train.train_micro_batch_size = 4
train.num_accumulation_steps = 1
train.test_micro_batch_size = 4

train.dist.data_parallel_size=2
train.dist.tensor_parallel_size=2
train.dist.pipeline_parallel_size = 1
# train.dist.pipeline_num_layers=8

train.train_epoch = 100
train.warmup_ratio = 20 / 300
train.eval_period = 1000
train.log_period = 1

# Scheduler
train.scheduler.warmup_factor = 0.001
train.scheduler.alpha = 0.01
train.scheduler.warmup_method = "linear"

# Set fp16 ON
train.amp.enabled = True

train.activation_checkpoint.enabled = False
graph.enabled = False