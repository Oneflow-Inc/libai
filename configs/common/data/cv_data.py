from omegaconf import OmegaConf

from libai.data.datasets import ImageNetDataset, CIFAR100Dataset
from libai.data.build import build_image_train_loader, build_image_test_loader
from libai.config import LazyCall

from .transform import default_train_transform as train_aug_cfg
from .transform import default_test_transform as test_aug_cfg


dataloader = OmegaConf.create()

dataloader.train = LazyCall(build_image_train_loader)(
    dataset=[
        LazyCall(CIFAR100Dataset)(
            root="./", train=True, download=True, transform=train_aug_cfg
        ),
    ],
)


dataloader.test = [
    LazyCall(build_image_test_loader)(
        dataset=LazyCall(CIFAR100Dataset)(
            root="./", train=False, download=True, transform=test_aug_cfg
        ),
    )
]
