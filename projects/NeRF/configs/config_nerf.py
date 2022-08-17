from omegaconf import OmegaConf

import oneflow as flow
import oneflow.nn as nn

from libai.data.datasets import BlenderDataset, LLFFDataset
from libai.data.build import build_image_train_loader, build_image_test_loader
from libai.config import LazyCall, get_config
from libai.optim import get_default_optimizer_params
from libai.scheduler.lr_scheduler import WarmupCosineAnnealingLR, WarmupMultiStepLR

from projects.NeRF.optimizers import Ranger, RAdam
from projects.NeRF.configs.models.config_model import model
from projects.NeRF.evaluation.nerf_evaluator import NerfEvaluator

# TODO: Create a unified interface to Nerf datasets



def get_nerf_dataset(dataset_type="Blender"):
    """
    Args:
        dataset_type: Blender or LLFF
    """
    assert dataset_type in ["Blender", "LLFF"], "The Nerf dataset must be one of Blender and LLFF"
    if dataset_type == "Blender":
        return BlenderDataset
    else:
        return LLFFDataset


graph = get_config("common/models/graph.py").graph
graph.enabled = False
train = get_config("common/train.py").train

# Refine train cfg for Nerf System

train.train_micro_batch_size = 1024
train.test_micro_batch_size = 1
train.dataset_type = "Blender"
train.blender_dataset_path = "/home/Bigdata/Nerf/nerf_synthetic/chair"
train.llff_dataset_path = "/path/to/llff"
train.train_epoch = 16 if train.dataset_type == "Blender" else 30
train.warmup_ratio = 0 / train.train_epoch
train.evaluation.eval_period = 1000
train.log_period = 50
train.optim_type = "adam"
train.lr_scheduler_type = "steplr"


# Redefining model config

model.cfg.dataset_type = train.dataset_type
model.cfg.loss_func = nn.MSELoss()

# Redefining evaluator

train.evaluation = dict(
    enabled=True,
    # evaluator for calculating top-k acc
    evaluator=LazyCall(NerfEvaluator)(
        img_wh=(400, 400) if train.dataset_type == "Blender" else (504, 378)
    ),
    eval_period=train.evaluation.eval_period,
    eval_iter=1e5,  # running steps for validation/test
    # Metrics to be used for best model checkpoint.
    eval_metric="Psnr",
    eval_mode="max",
)

# Refine optimizer cfg for Nerf System
# NOTE: In theory, both datasets used by Nerf
# are optimized using the Adam optimizer, but
# since the borrowed code base also implemen-
# ts three other optimizer configurations, l-
# ibai also implements the corresponding opt-
# imizer.
if train.optim_type == "adam":
    optimizer = flow.optim.Adam
    lr = 5e-4
elif train.optim_type == "sgd":
    optimizer = flow.optim.SGD
    lr = 5e-1
elif train.optim_type == "radam":
    optimizer = RAdam
    lr = 5e-4
elif train.optim_type == "ranger":
    optimizer = Ranger
    lr = 5e-4
else:
    raise NotImplementedError("Nerf does not support this type of optimizer!")

optim = LazyCall(optimizer)(
    params=LazyCall(get_default_optimizer_params)(
        # params.model is meant to be set to the model object,
        # before instantiating the optimizer.
        clip_grad_max_norm=None,
        clip_grad_norm_type=None,
        weight_decay_norm=None,
        weight_decay_bias=None,
    ),
    lr=lr,
    weight_decay=0,
)

if train.optim_type == "sgd":
    optim.momentum = 0.9

if train.lr_scheduler_type == "steplr":
    scheduler = WarmupMultiStepLR
elif train.optim_type == "cosine":
    scheduler = WarmupCosineAnnealingLR
else:
    raise NotImplementedError("Nerf does not support this type of scheduler!")

train.scheduler = LazyCall(scheduler)(
    # In DefaultTrainer we will automatically set `max_iter`
    # and `warmup_iter` by the given train cfg.
    warmup_factor=0.001,
    warmup_method="linear",
)

if train.lr_scheduler_type == "steplr":
    if train.dataset_type == "Blender":
        milestones = [2/16, 4/16, 8/16]
    else:
        milestones = [10/30, 20/30]
    train.scheduler.milestones = milestones
    train.scheduler.gamma = 0.5
elif train.optim_type == "cosine":
    train.scheduler.eta_min = 1e-8

train.warmup_ratio = (
    train.warmup_ratio
    if train.warmup_ratio > 0 and train.optim_type not in ["radam", "ranger"]
    else 0.0
)

# Set fp16 ON
train.amp.enabled = True

dataset = LazyCall(get_nerf_dataset)(dataset_type=train.dataset_type)

dataloader = OmegaConf.create()

dataloader.train = LazyCall(build_image_train_loader)(
    dataset=[
        LazyCall(dataset)(
            split="train",
            img_wh=(400, 400) if dataset.dataset_type == "Blender" else (504, 378),
            root_dir=train.blender_dataset_path if dataset.dataset_type == "Blender" else train.llff_dataset_path,
            spheric_poses=None if dataset.dataset_type == "Blender" else False,
            val_num=None if dataset.dataset_type == "Blender" else 1,  # Number of your GPUs
        )
    ],
    num_workers=4,
    train_batch_size=train.train_micro_batch_size,
    test_batch_size=1,
)

dataloader.test = [
    LazyCall(build_image_test_loader)(
        dataset=LazyCall(dataset)(
            split="val",
            img_wh=(400, 400) if dataset.dataset_type == "Blender" else (504, 378),
            root_dir=train.blender_dataset_path if dataset.dataset_type == "Blender" else train.llff_dataset_path,
            spheric_poses=None if dataset.dataset_type == "Blender" else False,
            val_num=None if dataset.dataset_type == "Blender" else 1,  # Number of your GPUs
        ),
        num_workers=4,
        test_batch_size=1,
    )
]

# Distributed Settings
depth = None
train.dist.pipeline_num_layers = depth
train.dist.data_parallel_size = 1
train.dist.tensor_parallel_size = 1
train.dist.pipeline_parallel_size = 1
