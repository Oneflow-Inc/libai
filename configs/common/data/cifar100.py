from omegaconf import OmegaConf
from flowvision import transforms
from flowvision.data.mixup import Mixup
from flowvision.transforms import InterpolationMode
from flowvision.transforms.functional import str_to_interp_mode

from libai.data.datasets import CIFAR100Dataset
from libai.data.build import build_image_train_loader, build_image_test_loader
from libai.config import LazyCall

# mean and std of cifar100 dataset
CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

train_aug = LazyCall(transforms.Compose)(
    transforms=[
        LazyCall(transforms.RandomResizedCrop)(
            size=(224, 224),
            scale=(0.08, 1.0),
            ratio=(3.0 / 4.0, 4.0 / 3.0),
            interpolation=str_to_interp_mode("bicubic"),
        ),
        LazyCall(transforms.RandomHorizontalFlip)(),
        LazyCall(transforms.ToTensor)(),
        LazyCall(transforms.Normalize)(mean=CIFAR100_TRAIN_MEAN, std=CIFAR100_TRAIN_STD),
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
            mean=CIFAR100_TRAIN_MEAN,
            std=CIFAR100_TRAIN_STD,
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
        ),
        num_workers=4,
    )
]
