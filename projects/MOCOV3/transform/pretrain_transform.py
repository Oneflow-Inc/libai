# coding=utf-8
# Copyright 2021 The OneFlow Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import random

import oneflow as flow
from flowvision import transforms
from flowvision.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from PIL import ImageFilter, ImageOps

from libai.config import LazyCall


class GaussianBlur(object):
    """Gaussian blur augmentation from SimCLR: https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[0.1, 2.0]):
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
    LazyCall(transforms.RandomResizedCrop)(size=224, scale=(0.2, 1.0)),
    LazyCall(transforms.RandomApply)(
        transforms=[
            LazyCall(transforms.ColorJitter)(
                brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
            )  # not strengthened
        ],
        p=0.8,
    ),
    # TODO: Add RandomGrayscale
    # LazyCall(transforms.RandomGrayscale)(p=0.2),
    LazyCall(transforms.RandomApply)(transforms=[LazyCall(GaussianBlur)(sigma=[0.1, 2.0])], p=1.0),
    LazyCall(transforms.RandomHorizontalFlip)(),
    LazyCall(transforms.ToTensor)(),
    LazyCall(transforms.Normalize)(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
]

augmentation2 = [
    LazyCall(transforms.RandomResizedCrop)(size=224, scale=(0.2, 1.0)),
    LazyCall(transforms.RandomApply)(
        transforms=[
            LazyCall(transforms.ColorJitter)(
                brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
            )  # not strengthened
        ],
        p=0.8,
    ),
    # TODO: Add RandomGrayscale
    # LazyCall(transforms.RandomGrayscale)(p=0.2),
    LazyCall(transforms.RandomApply)(transforms=[LazyCall(GaussianBlur)(sigma=[0.1, 2.0])], p=1.0),
    LazyCall(transforms.RandomApply)(transforms=[LazyCall(Solarize)()], p=0.2),
    LazyCall(transforms.RandomHorizontalFlip)(),
    LazyCall(transforms.ToTensor)(),
    LazyCall(transforms.Normalize)(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
]


class TwoCropsTransform:
    """Take two random crops of one image"""

    def __init__(self, base_transform1, base_transform2):
        self.base_transform1 = base_transform1
        self.base_transform2 = base_transform2

    def __call__(self, x):
        im1 = self.base_transform1(x)
        im2 = self.base_transform2(x)
        return flow.cat((im1, im2), dim=0)
