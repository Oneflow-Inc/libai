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
from libai.data.datasets import ImageNetDataset
from libai.data.build import build_image_train_loader, build_image_test_loader

train_aug = LazyCall(transforms.Compose)(
    transforms=[
        LazyCall(transforms.RandomResizedCrop)(
            size=224,
            scale=(0.08, 1.0),
            ratio=(3.0 / 4.0, 4.0 / 3.0),
            interpolation=InterpolationMode.BICUBIC,
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
        LazyCall(transforms.ToTensor)(),
        LazyCall(transforms.Normalize)(
            mean=IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_DEFAULT_STD,
        ),
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


dataloader = OmegaConf.create()
dataloader.train = LazyCall(build_image_train_loader)(
    dataset=[
        LazyCall(ImageNetDataset)(
            root="./dataset",
            train=True,
            transform=train_aug,
        ),
    ],
    num_workers=4,
    mixup_func=None,
)


dataloader.test = [
    LazyCall(build_image_test_loader)(
        dataset=LazyCall(ImageNetDataset)(
            root="./dataset",
            train=False,
            transform=test_aug,
        ),
        num_workers=4,
    )
]
