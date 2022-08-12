from posixpath import split
from omegaconf import OmegaConf
from flowvision import transforms
from flowvision.transforms import InterpolationMode
from flowvision.transforms.functional import str_to_interp_mode
from flowvision.data.constants import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
)
from flowvision.data.auto_augment import rand_augment_transform
from flowvision.data.random_erasing import RandomErasing

from libai.config import LazyCall
from projects.SegFormer.dataset.cityscapes import CityScapes
from libai.data.build import build_image_train_loader, build_image_test_loader

train_aug = LazyCall(transforms.Compose)(
    transforms=[
        LazyCall(transforms.RandomResizedCrop)(
            size=(512, 1024),
            interpolation=InterpolationMode.BICUBIC,
        ),
        LazyCall(transforms.RandomHorizontalFlip)(p=0.5),
        LazyCall(transforms.ToTensor)(),
        LazyCall(transforms.Normalize)(
            mean=IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_DEFAULT_STD,
        )
    ]
)


test_aug = LazyCall(transforms.Compose)(
    transforms=[
        LazyCall(transforms.Resize)(
            size=(512, 1024),
            interpolation=InterpolationMode.BICUBIC,
        ),
        LazyCall(transforms.ToTensor)(),
        LazyCall(transforms.Normalize)(
            mean=IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_DEFAULT_STD,
        ),
    ]
)


dataloader = OmegaConf.create()
dataloader.train = LazyCall(build_image_train_loader)(
    dataset=[
        LazyCall(CityScapes)(
            root="/dataset/cityscapes",
            split="train",
            transform=train_aug,
            target_transform=train_aug,
        ),
    ],
    num_workers=4,
    mixup_func=None,
)


dataloader.test = [
    LazyCall(build_image_test_loader)(
        dataset=LazyCall(CityScapes)(
            root="/dataset/cityscapes",
            split="test",
            transform=test_aug,
            target_transform=test_aug,
        ),
        num_workers=4,
    )
]
