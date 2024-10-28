import flowvision as vision

from omegaconf import OmegaConf
from libai.config import get_config
from libai.config import LazyCall
from libai.data.build import build_image_test_loader, build_image_train_loader
from libai.layers import Linear
from libai.data.datasets import CIFAR10Dataset

from projects.MOE.model.moe import MoE

train = get_config("common/train.py").train
optim = get_config("common/optim.py").optim
graph = get_config("common/models/graph.py").graph
graph.enabled = False


data_root = "./projects/MOE/data"
transform = vision.transforms.Compose(
    [
        vision.transforms.ToTensor(),
        vision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
)

dataloader = OmegaConf.create()

dataloader.train = LazyCall(build_image_train_loader)(
    dataset=[
        LazyCall(CIFAR10Dataset)(
            root=data_root,
            train=True,
            download=True,
            transform=transform,
        ),
    ],
    num_workers=1,
)

dataloader.test = [
    LazyCall(build_image_test_loader)(
        dataset=LazyCall(CIFAR10Dataset)(
            root=data_root,
            train=False,
            download=True,
            transform=transform,
        ),
        num_workers=1,
    ),
]

model_cfg = dict(
    expert=Linear(in_features=3072, out_features=10),
    input_size=3072,
    output_size=10,
    num_experts=10,
    noisy_gating=True,
    k=4,
    device="cuda",
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
