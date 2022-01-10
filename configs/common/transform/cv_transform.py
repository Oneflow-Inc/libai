import math

import oneflow as flow
from flowvision import transforms
from flowvision.data.constants import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    DEFAULT_CROP_PCT,
)
from flowvision.data.auto_augment import (
    rand_augment_transform,
    augment_and_mix_transform,
    auto_augment_transform,
)
from flowvision.transforms.functional import str_to_interp_mode, str_to_pil_interp
from flowvision.data.random_erasing import RandomErasing

from libai.config import LazyCall


default_train_transform = LazyCall(transforms.Compose)(
    transforms = [
        LazyCall(transforms.RandomResizedCrop)(
            size=224,
            scale=(0.08, 1.0),
            ratio=(3.0 / 4.0, 4.0 / 3.0),
            interpolation=str_to_interp_mode("bicubic")
        ),
        LazyCall(transforms.RandomHorizontalFlip)(
            p=0.5
        ),
        LazyCall(rand_augment_transform)(
            config_str="rand-m9-mstd0.5-inc1",
            hparams=dict(
                translate_const=int(224 * 0.45),
                img_mean=tuple([min(255, round(255 * x)) for x in IMAGENET_DEFAULT_MEAN]),
                interpolation=str_to_interp_mode("bicubic")
            )
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
        )
    ]
)