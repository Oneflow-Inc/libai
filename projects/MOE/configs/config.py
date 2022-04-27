from omegaconf import OmegaConf

import flowvision as vision
from libai.config import get_config
from libai.config import LazyCall
from libai.data.build import build_image_test_loader, build_image_train_loader
from libai.moe.moe import MoE

from projects.MOE.model.mlp import MLP

train = get_config("common/train.py").train
optim = get_config("common/optim.py").optim
graph = get_config("common/models/graph.py").graph




data_root = "./data"
transform = vision.transforms.Compose(
    [vision.transforms.ToTensor(),
        vision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = vision.datasets.CIFAR10(root=data_root, train=True,
                                        download=True, transform=transform)
testset = vision.datasets.CIFAR10(root=data_root, train=True,
                                        download=True, transform=transform)

dataloader = OmegaConf.create()

dataloader.train = LazyCall(build_image_train_loader)(
        dataset=trainset,
        test_batch_size=16,
        num_workers=4,
        collate_fn = lambda x: x.view(x.shape[0],-1),
    ),

dataloader.test = LazyCall(build_image_test_loader)(
        dataset=testset,
        test_batch_size=16,
        num_workers=4,
        collate_fn = lambda x: x.view(x.shape[0],-1),
    ),


model_cfg = dict(
        expert=MLP,
        input_size=3072,
        output_size=10,
        num_experts=10,
        noisy_gating=True,
        k=4,
        device = "cuda",
)

model = LazyCall(MoE)(**model_cfg)

train.update(
    dict(
        recompute_grad=dict(enabled=True),
        output_dir="output/benchmark/",
        train_micro_batch_size=16,
        test_micro_batch_size=16,
        train_epoch=1,
        eval_period=500,
        log_period=50,
        dist=dict(
            data_parallel_size=1,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
        ),
    )
)
