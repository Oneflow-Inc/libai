from libai.data.datasets import ImageNetDataset, CIFAR10Dataset, CIFAR100Dataset

from libai.config import LazyCall

imagenet = LazyCall(ImageNetDataset)(
    root="./dataset/imagenet",
    train=True,
)

cifar10 = LazyCall(CIFAR10Dataset)(
    root = "./dataset/cifar10",
    train=True,
    download=True,
)

cifar100 = LazyCall(CIFAR100Dataset)(
    root="./dataset/cifar100",
    train=True,
    download=True
)