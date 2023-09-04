from omegaconf import OmegaConf

from libai.config import LazyCall
from flowvision.data.constants import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
)
from libai.data.build import build_image_test_loader, build_image_train_loader
from projects.SegFormer.dataset.cityscapes import CityScapes
from projects.SegFormer.dataset.transform import (
    Compose,
    Normalize,
    RandomCrop,
    RandomHorizontalFlip,
    Resize,
    ToTensor,
)

train_aug = LazyCall(Compose)(
    transforms=[
        LazyCall(Resize)(
            size=(2048, 1024),
        ),
        LazyCall(RandomCrop)(
            size=(1024, 1024),
        ),
        LazyCall(RandomHorizontalFlip)(p=0.5),
        LazyCall(ToTensor)(),
        LazyCall(Normalize)(
            mean=IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_DEFAULT_STD,
        )
    ]
)


test_aug = LazyCall(Compose)(
    transforms=[
        LazyCall(Resize)(
            size=(2048, 1024),
        ),
        LazyCall(RandomCrop)(
            size=(1024, 1024),
        ),
        LazyCall(ToTensor)(),
        LazyCall(Normalize)(
            mean=IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_DEFAULT_STD,
        ),
    ]
)


dataloader = OmegaConf.create()
dataloader.train = LazyCall(build_image_train_loader)(
    dataset=[
        LazyCall(CityScapes)(
            root="./dataset",
            split="train",
            transforms=train_aug,
        ),
    ],
    num_workers=16,
    mixup_func=None,
)


dataloader.test = [
    LazyCall(build_image_test_loader)(
        dataset=LazyCall(CityScapes)(
            root="./dataset",
            split="test",
            transforms=test_aug,
        ),
        num_workers=16,
    )
]
