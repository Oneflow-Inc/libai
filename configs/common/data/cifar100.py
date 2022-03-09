from omegaconf import OmegaConf
from flowvision import transforms
from flowvision.data.mixup import Mixup
from flowvision.transforms import InterpolationMode
from flowvision.transforms.functional import str_to_interp_mode

from libai.data.datasets import CIFAR100Dataset
from libai.data.build import build_image_train_loader, build_image_test_loader
from libai.config import LazyCall
from flowvision.data.auto_augment import rand_augment_transform
from flowvision.data.random_erasing import RandomErasing
from flowvision.data.constants import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
)

# mean and std of cifar100 dataset
CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

train_aug = LazyCall(transforms.Compose)(
    transforms=[
        LazyCall(transforms.RandomResizedCrop)(
            size=224,
            scale=(0.08, 1.0),
            ratio=(3.0 / 4.0, 4.0 / 3.0),
            interpolation=str_to_interp_mode("bicubic"),
        ),
        LazyCall(transforms.RandomHorizontalFlip)(p=0.5),
        LazyCall(rand_augment_transform)(
            config_str="rand-m9-mstd0.5-inc1",
            hparams=dict(
                translate_const=int(224 * 0.45),
                img_mean=tuple([min(255, round(255 * x)) for x in IMAGENET_DEFAULT_MEAN]),
                interpolation=str_to_interp_mode("bicubic"),
            ),
        ),
        LazyCall(transforms.Normalize)(
            mean=IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_DEFAULT_STD,
        ),
        LazyCall(transforms.ToTensor)(),
        LazyCall(RandomErasing)(
            probability=0.25,
            mode="pixel",
            max_count=1,
            num_splits=0,
            device="cpu",
        ),
    ]
)


test_aug = LazyCall(transforms.Compose)(
    transforms=[
        LazyCall(transforms.Resize)(
            size=256,
            interpolation=InterpolationMode.BICUBIC,
        ),
        LazyCall(transforms.CenterCrop)(
            size=224,
        ),
        LazyCall(transforms.ToTensor)(),
        LazyCall(transforms.Normalize)(
            mean=IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_DEFAULT_STD,
        ),
    ]
)


# Dataloader config
dataloader = OmegaConf.create()
dataloader.train = LazyCall(build_image_train_loader)(
    dataset=[
        LazyCall(CIFAR100Dataset)(
            root="./",
            train=True,
            download=True,
            transform=train_aug,
            dataset_name="cifar100 train set",
        ),
    ],
    num_workers=4,
    mixup_func=LazyCall(Mixup)(
        mixup_alpha=0.8,
        cutmix_alpha=1.0,
        prob=1.0,
        switch_prob=0.5,
        mode="batch",
        num_classes=100,
    ),
)

dataloader.test = [
    LazyCall(build_image_test_loader)(
        dataset=LazyCall(CIFAR100Dataset)(
            root="./",
            train=False,
            download=True,
            transform=test_aug,
            dataset_name="cifar100 test set",
        ),
        num_workers=4,
    )
]
