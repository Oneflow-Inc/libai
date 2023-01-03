from omegaconf import DictConfig
from libai.config import LazyCall

from .common.models.dlrm import dlrm_model as model
from .common.data.criteo_dataset import dataloader

from .common.models.ctr_graph import graph
from .common.ctr_train import train
from .common.optim import sgd_optim as optim


data_dir = "/RAID0/xiexuan/criteo1t_dlrm_parquet"
dataloader.train.data_path = f"{data_dir}/train"
dataloader.validation.data_path = f"{data_dir}/val"
dataloader.test.data_path = f"{data_dir}/test"

train.amp.enabled = False
train.model_save_dir = ""
train.train_iter = 75000
train.log_peroid = 10
#train.evaluation.enabled = False
train.output_dir = "output"

optim.lr = 24
#test = None

graph.enabled = True
