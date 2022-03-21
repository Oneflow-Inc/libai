import random
from PIL import ImageFilter, ImageOps

import oneflow as flow
from flowvision import transforms
from flowvision.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from libai.config import LazyCall


class GaussianBlur(object):
    """Gaussian blur augmentation from SimCLR: https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class Solarize(object):
    """Solarize augmentation from BYOL: https://arxiv.org/abs/2006.07733"""

    def __call__(self, x):
        return ImageOps.solarize(x)



# follow BYOL's augmentation recipe: https://arxiv.org/abs/2006.07733
augmentation1 = [
    LazyCall(transforms.RandomResizedCrop)(size=224, scale=(.2, 1.)),
    LazyCall(transforms.RandomApply)(transforms=[
        LazyCall(transforms.ColorJitter)(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)  # not strengthened
    ], p=0.8),
    # transforms.RandomGrayscale(p=0.2),
    LazyCall(transforms.RandomApply)(transforms=[GaussianBlur(sigma=[.1, 2.])], p=1.0), 
    LazyCall(transforms.RandomHorizontalFlip)(),
    LazyCall(transforms.ToTensor)(),
    LazyCall(transforms.Normalize)(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
]

augmentation2 = [
    LazyCall(transforms.RandomResizedCrop)(size=224, scale=(.2, 1.)),
    LazyCall(transforms.RandomApply)(transforms=[
        LazyCall(transforms.ColorJitter)(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)  # not strengthened
    ], p=0.8),
    # transforms.RandomGrayscale(p=0.2), 
    LazyCall(transforms.RandomApply)(transforms=[GaussianBlur(sigma=[.1, 2.])], p=1.0), 
    LazyCall(transforms.RandomApply)(transforms=[Solarize()], p=0.2),
    LazyCall(transforms.RandomHorizontalFlip)(),
    LazyCall(transforms.ToTensor)(),
    LazyCall(transforms.Normalize)(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
]



class TwoCropsTransform:
    """Take two random crops of one image"""

    def __init__(self, base_transform1, base_transform2):
        self.base_transform1 = base_transform1
        self.base_transform2 = base_transform2

    def __call__(self, x):
        im1 = self.base_transform1(x)
        im2 = self.base_transform2(x)
        # return [im1, im2]
        return flow.cat((im1, im2), dim=0)
