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
from typing import Callable, Optional

import oneflow as flow
from flowvision import datasets

from libai.data.structures import DistTensorData, Instance


class ImageNetDataset(datasets.ImageFolder):
    r"""`ImageNet <http://image-net.org/>`_ 2012 Classification Dataset in LiBai.

    Args:

        root (string): Root directory of the ImageNet Dataset.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
    """

    def __init__(
        self, root: str, train: bool = True, transform: Optional[Callable] = None, **kwargs
    ):
        prefix = "train" if train else "val"
        root = os.path.join(root, prefix)
        super(ImageNetDataset, self).__init__(root=root, transform=transform, **kwargs)

    def __getitem__(self, index: int):
        sample, target = super().__getitem__(index)
        data_sample = Instance(
            images=DistTensorData(sample, placement_idx=0),
            labels=DistTensorData(flow.tensor(target, dtype=flow.long), placement_idx=-1),
        )
        return data_sample
