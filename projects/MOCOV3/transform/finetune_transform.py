from flowvision import transforms
from flowvision.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from libai.config import LazyCall

train_augmentation = [
    LazyCall(transforms.RandomResizedCrop)(size=224),
    LazyCall(transforms.RandomHorizontalFlip)(),
    LazyCall(transforms.ToTensor)(),
    LazyCall(transforms.Normalize)(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
]

test_augmentation = [
    LazyCall(transforms.Resize)(size=256),
    LazyCall(transforms.CenterCrop)(size=224),
    LazyCall(transforms.ToTensor)(),
    LazyCall(transforms.Normalize)(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
]