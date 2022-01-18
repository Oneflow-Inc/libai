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
"""dataset for bert."""

import oneflow as flow

from libai.data.structures import DistTensorData, Instance


class DemoNlpDataset(flow.utils.data.Dataset):
    def __init__(
        self, data_root="",
    ):
        self.data_root = data_root
        self.dataset = list(range(10000))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = Instance(
            input=DistTensorData(
                flow.ones((32, 128), dtype=flow.long), placement_idx=0
            ),
            label=DistTensorData(flow.ones((32,), dtype=flow.long), placement_idx=-1),
        )
        return sample

    @property
    def supports_prefetch(self):
        return False

    def prefetch(self, indices):
        self.dataset.prefetch(indices)
