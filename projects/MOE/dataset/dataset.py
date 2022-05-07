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

import flowvision as vision
import oneflow as flow

from libai.data.structures import DistTensorData, Instance


class CIFAR_Dataset(flow.utils.data.Dataset):
    def __init__(self, root, train, download, transform) -> None:
        super().__init__()
        self.data = vision.datasets.CIFAR10(
            root=root, train=train, download=download, transform=transform
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        tensors = {}
        d = self.data[index]
        inputs = d[0]
        inputs = inputs.view(-1)
        labels = flow.tensor(d[1], dtype=flow.long)
        tensors["x"] = DistTensorData(inputs, placement_idx=-1)
        tensors["labels"] = DistTensorData(labels)
        return Instance(**tensors)
