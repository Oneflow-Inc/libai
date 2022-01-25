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

from typing import Callable, Optional

import oneflow as flow
from flowvision import datasets

from libai.data.structures import DistTensorData, Instance


class CIFAR10Dataset(datasets.CIFAR10):
    """CIFAR10 Dataset"""

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        download: bool = False,
        **kwargs
    ):
        super(CIFAR10Dataset, self).__init__(
            root=root, train=train, transform=transform, download=download, **kwargs
        )

    def __getitem__(self, index: int):
        img, target = super().__getitem__(index)
        print(img)
        data_sample = Instance(
            images=DistTensorData(img, placement_idx=0),
            targets=DistTensorData(flow.tensor(target, dtype=flow.long), placement_idx=-1),
        )
        return data_sample


class CIFAR100Dataset(datasets.CIFAR100):
    """CIFAR100 Dataset"""

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        download: bool = False,
        **kwargs
    ):
        super(CIFAR100Dataset, self).__init__(
            root=root, train=train, transform=transform, download=download, **kwargs
        )

    def __getitem__(self, index: int):
        img, target = super().__getitem__(index)
        data_sample = Instance(
            images=DistTensorData(img, placement_idx=0),
            targets=DistTensorData(flow.tensor(target, dtype=flow.long), placement_idx=-1),
        )
        return data_sample
