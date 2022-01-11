from omegaconf import OmegaConf

from libai.config import LazyCall
from libai.data.datasets import ImageNetDataset
from libai.data.build import build_image_train_loader, build_image_test_loader

from .transform import default_train_transform as train_aug
from .transform import default_test_transform as test_aug

dataloader = OmegaConf.create()

dataloader.train = LazyCall(build_image_train_loader)(
    dataset= # LazyCall(ImageNetDataset)(root="...", transform=train_aug),
        [
            LazyCall(ImageNetDataset)(root="...", transform=train_aug),
            LazyCall(ImageNetDataset)(root="...", transform=train_aug),
        ],
    weight=[0.5, 0.5],
    batch_size=128,
    num_workers=4,
)

# train_loader = instantiate(dataloader.train)

dataloader.test = [
    LazyCall(build_image_test_loader)(
        dataset=LazyCall(ImageNetDataset)(root="...", transform=test_aug),
        batch_size=128,
    ),
    LazyCall(build_image_test_loader)(
        dataset=LazyCall(CIFAR100)(root="...", transform=test_aug),
        batch_size=128,
    ),
]

# for test_cfg in dataloader.test:
#     test_loader = instantiate(test_cfg)
