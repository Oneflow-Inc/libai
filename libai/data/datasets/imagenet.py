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

import os
from typing import Optional, Callable

import oneflow as flow
from oneflow.utils.data import DataLoader

from flowvision import datasets, transforms
from flowvision.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from flowvision.data import Mixup
from flowvision.data import create_transform
from flowvision.transforms.functional import str_to_interp_mode

from libai.data.structures import DistTensorData, Instance


class ImageNetDataset(datasets.ImageFolder):
    """ImageNet Dataset
    """

    def __init__(self, 
                 root: str,
                 train: bool = True, 
                 transform: Optional[Callable] = None,
                 **kwargs):
        prefix = "train" if train else "val"
        root = os.path.join(root, prefix)
        super(ImageNetDataset, self).__init__(root=root, 
                                              transform=transform,
                                              **kwargs)
    
    def __getitem__(self, index: int):
        sample, target = super().__getitem__(index)
        data_sample = Instance(
            images = DistTensorData(sample, placement_idx=0),
            targets = DistTensorData(target, placement_idx=-1)
        )
        return data_sample
        

# def build_imagenet_dataset(is_train, cfg):
#     transform = build_transform(is_train, cfg)
#     prefix = "train" if is_train else "val"
#     root = os.path.join(cfg.data.data_path, prefix)
#     dataset = ImageNetDataset(root, transform=transform)
#     if is_train:
#         assert len(dataset) == 1281167, "The whole train set of ImageNet contains 1281167 images but got {} instead.".format(len(dataset))
#     else:
#         assert len(dataset) == 50000, "The whole val set of ImageNet contains 50000 images but got {} instead.".format(len(dataset))
#     nb_classes = 1000    
#     return dataset, nb_classes